"""Complex-baseband QPSK over a few-tap fading ISI channel, with learned and
classical trellis detectors. Standalone (does not use the BPSK framework).

Signal model (per word of N data symbols):
    y[t] = sum_{k=0}^{L-1} h_k s[t-k] + n[t],   n ~ CN(0, N0),  E|s|^2 = 1
L-1 known zero-index symbols are prepended (known start state) and appended
(known end state), so y has N + L - 1 samples.

Taps: complex Gaussian with an exponential power-delay profile (decay per tap
`pdp_decay`), total power 1 -- a beamformed mmWave link with a few residual
paths. They evolve word to word as AR(1) with correlation `rho`, so a receiver
has to track them. A frame is `words` words; word 0 is a pilot, the rest data.

Trellis: branch class c = sum_k d_k M^k over the symbol indices
d_0 = s[t] (current), ..., d_{L-1} = s[t-L+1]; C = M^L classes,
S = M^(L-1) states (the most recent L-1 symbols). Full Viterbi with
traceback (max-sum over per-sample class scores).

Receivers (every one gets the same pilot word and the same correct-word gate
as the repo's ViterbiNet online training: after a data word, a frame adapts
only if that word's SER <= ser_thresh, i.e. an ECC/CRC would have accepted it,
and then it uses that word's transmitted symbols):
    classic_csi   exact taps and N0 (the bound)
    le_oracle     symbol-by-symbol MMSE linear equalizer over the same W-sample
                  window with the exact taps and N0, QPSK slicer (linear bound)
    le_ls         the same equalizer fitted by LS to the pilot symbols, then
                  re-solved by LS on each data word's own decisions
    classic_ls    LS channel estimate from the pilot; re-estimated by LS on each
                  accepted data word (decision-directed tracking)
    affine        per-sample scores W^T [Re y, Im y] + b  (3C parameters) --
                  the exact form of the Gaussian log-likelihood, learned
    mlp           ViterbiNet-style MLP on [Re y, Im y]
    sym_affine    4-class symbol classifier: linear map from a window of 2L-1
                  complex samples around each symbol to the 4 QPSK classes,
                  argmax decision, no trellis ((4L-2)*4+4 parameters)
    sym_mlp       same window, small MLP, argmax, no trellis
    sym_eq        4-class symbol classifier with the structure of a linear
                  equalizer: z = w^H window + b (complex), class scores
                  a * Re(conj(c_m) z), argmax, no trellis (4W+3 parameters).
                  Pilot: 4-class cross-entropy. Tracking between pilots:
                  decision-directed LMS, |z - c_dhat|^2 -- cross-entropy on
                  its own argmax labels has ~zero gradient on confident
                  correct symbols, so it cannot follow a drifting channel
                  (sym_affine/sym_mlp SER climbs 0.06 -> 0.28 over a frame)
    tied          'structured affine': the affine scores with their class means
                  tied to L learnable complex taps + a learnable noise scale
                  (2L+1 real parameters), trained by the same gradient steps
    tied_mlp      'structured MLP': the tied Gaussian branch score plus a small
                  MLP g([Re r, Im r, |r|^2]) of the residual r = y - mu_c, shared
                  by all C branches (output layer zero-initialised, so it starts
                  as `tied` and learns only a non-Gaussian correction).
    The free per-branch models (affine, mlp) need samples of all C = 64 branch
    classes; one 120-symbol pilot leaves ~10 unseen, so they stay at 0.15-0.5
    SER even on a static channel. Tying the classes to L taps fixes that.
Learned receivers: offline training on random channels at the operating SNR,
then per frame: `pilot_iters` Adam steps on the pilot and `online_iters` steps
on each data word (its decisions, or its true symbols if gated), starting from
the offline weights.
"""
import math
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
CDT = torch.complex64


class Config:
    M = 4                # QPSK
    L = 3                # channel taps (memory L-1)
    N = 120              # data symbols per word
    words = 25           # per frame, the first pilot_words words are pilots
    pilot_words = 1
    pdp_decay = 0.5      # tap power ~ decay^k
    rho = 0.99           # AR(1) tap correlation word to word (used when doppler == 0)
    doppler = 0.0        # > 0: fast fading instead -- every tap is a complex Jakes
                         # (Rayleigh) process with normalized Doppler f_D*T_symbol,
                         # varying sample by sample and continuous across words
    jakes_sinusoids = 16
    ser_thresh = 0.02
    adapt = 'dd'         # 'dd': adapt on the receiver's own decisions after every data word;
                         # 'gated': adapt on true symbols only if word SER <= ser_thresh (repo-style ECC gate)
    offline_steps = 300
    offline_words = 16
    offline_lr = 1e-3
    online_lr = 5e-2
    pilot_iters = 200    # learned receivers: steps on the pilot (once per frame)
    dd_lr = None         # lr for the per-word tracking steps; default online_lr
    mlp_hidden = (100, 58)
    sym_hidden = (32, 16)  # 4-class symbol-classifier MLP
    sym_window = None      # samples per symbol classifier window; default 2L-1
    tied_hidden = 16       # hidden units of the tied_mlp residual MLP

    @property
    def W(self):
        return self.sym_window or 2 * self.L - 1

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.S = self.M ** (self.L - 1)
        self.C = self.M ** self.L
        self.const = torch.exp(1j * (math.pi / 4 + math.pi / 2 * torch.arange(self.M))).to(CDT)
        pdp = torch.tensor([self.pdp_decay ** k for k in range(self.L)])
        self.tap_std = (pdp / pdp.sum()).sqrt()
        c = torch.arange(self.C)
        self.digits = torch.stack([(c // self.M ** k) % self.M for k in range(self.L)], 1)  # (C, L)


# ---------------------------------------------------------------- channel ---
def random_taps(cfg, B):
    z = torch.complex(torch.randn(B, cfg.L), torch.randn(B, cfg.L)) / math.sqrt(2)
    return (z * cfg.tap_std).to(CDT)


def evolve_taps(cfg, h):
    return (cfg.rho * h + math.sqrt(1 - cfg.rho ** 2) * random_taps(cfg, h.shape[0])).to(CDT)


def jakes_taps(cfg, B, n):
    """(B, n, L) complex taps: independent unit-power Jakes processes (sum of
    sinusoids with random arrival angles and phases), scaled by the PDP."""
    Ms = cfg.jakes_sinusoids
    t = torch.arange(n, dtype=torch.float64).view(1, n, 1, 1)
    alpha = torch.rand(B, 1, cfg.L, Ms, dtype=torch.float64) * 2 * math.pi
    phi = torch.rand(B, 1, cfg.L, Ms, dtype=torch.float64) * 2 * math.pi
    g = torch.exp(1j * (2 * math.pi * cfg.doppler * torch.cos(alpha) * t + phi)).sum(-1) / math.sqrt(Ms)
    return (g * cfg.tap_std.double()).to(CDT)


def transmit(cfg, h, N0):
    """h: (B, L) block taps, or (B, T, L) per-sample taps (fast fading).
    Returns data symbol indices (B, N), y (B, T), classes (B, T)."""
    B, L = h.shape[0], h.shape[-1]
    d = torch.randint(0, cfg.M, (B, cfg.N))
    pad = torch.zeros(B, L - 1, dtype=torch.long)
    full = torch.cat([pad, d, pad], 1)                        # (B, N + 2(L-1))
    win = full.unfold(1, L, 1).flip(-1)                       # (B, T, L): [s[t], s[t-1], ...]
    s = cfg.const[win]
    y = (s * (h[:, None, :] if h.dim() == 2 else h)).sum(-1)
    y = y + math.sqrt(N0 / 2) * torch.complex(torch.randn_like(y.real), torch.randn_like(y.real))
    classes = (win * (cfg.M ** torch.arange(L))).sum(-1)
    return d, y.to(CDT), classes


def class_means(cfg, h):
    if h.dim() == 3:                                                  # per-sample taps
        return (cfg.const[cfg.digits][None, None] * h[:, :, None, :]).sum(-1)   # (B, T, C)
    return (cfg.const[cfg.digits][None] * h[:, None, :]).sum(-1)    # (B, C)


def features(y):
    return torch.stack([y.real, y.imag], -1)                  # (B, T, 2)


def sym_features(cfg, y):
    """Window y[t-(L-1) .. t+(L-1)] around each data symbol t (its own energy
    spans y[t..t+L-1]; the previous L-1 symbols' ISI sits in y[t-L+1..t]).
    y: (B, N+L-1) -> (B, N, 2W)."""
    back = (cfg.W - 1) // 2
    need = cfg.N - 1 + cfg.W - back                           # last index needed + 1
    yp = F.pad(torch.view_as_real(y).transpose(1, 2), [back, max(0, need - y.shape[1])])
    win = yp.unfold(2, cfg.W, 1)[:, :, :cfg.N]                 # (B, 2, N, W)
    return win.permute(0, 2, 1, 3).reshape(y.shape[0], cfg.N, 2 * cfg.W)


# ---------------------------------------------------------------- viterbi ---
def viterbi(cfg, scores):
    """scores: (B, T, C) log-scores (higher = better). Returns (B, N) symbol indices."""
    B, T, _ = scores.shape
    S, M = cfg.S, cfg.M
    metric = torch.full((B, S), -1e30)
    metric[:, 0] = 0.0
    prev_of = torch.arange(cfg.C) // M                        # previous state for class c
    back = torch.empty(B, T, S, dtype=torch.long)
    for t in range(T):
        cand = metric[:, prev_of] + scores[:, t]              # (B, C)
        cand = cand.view(B, M, S)                             # c = s + S*j
        metric, j = cand.max(1)
        back[:, t] = j
    state = torch.zeros(B, dtype=torch.long)                  # known zero tail
    out = torch.empty(B, T, dtype=torch.long)
    ar = torch.arange(B)
    for t in range(T - 1, -1, -1):
        out[:, t] = state % M                                 # current symbol of this state
        c = state + S * back[ar, t, state]
        state = c // M
    return out[:, :cfg.N]


# ------------------------------------------------- batched learned models ---
class BatchedNet:
    """F independent copies of one small net (per-frame weights), with a
    masked Adam so frames that don't adapt this word are left untouched."""

    def __init__(self, cfg, kind, F_):
        self.cfg, self.kind = cfg, kind
        self.params = []
        if kind in ('tied', 'tied_mlp'):
            # class means tied to L learnable complex taps + one learnable
            # inverse-noise scale: 2L+1 real parameters
            self.params = [(0.01 * torch.randn(F_, 1, cfg.L)).requires_grad_(),
                           (0.01 * torch.randn(F_, 1, cfg.L)).requires_grad_(),
                           torch.zeros(F_, 1, 1).requires_grad_()]
            if kind == 'tied_mlp':
                # shared residual MLP 3 -> H -> 1; zero output layer: starts as `tied`
                H = cfg.tied_hidden
                W1 = torch.empty(F_, 3, H)
                for f in range(F_):
                    torch.nn.init.kaiming_uniform_(W1[f].T, a=math.sqrt(5))
                self.params += [W1.requires_grad_(), torch.empty(F_, 1, H).uniform_(-1 / math.sqrt(3), 1 / math.sqrt(3)).requires_grad_(),
                                torch.zeros(F_, H, 1).requires_grad_(), torch.zeros(F_, 1, 1).requires_grad_()]
            self.sym = cfg.const[cfg.digits]                   # (C, L) complex
            self.reset_opt()
            return
        if kind == 'sym_eq':
            # complex linear equalizer taps (centre tap 1), complex bias, log class-score scale
            wr = torch.zeros(F_, 1, cfg.W); wr[:, :, (cfg.W - 1) // 2] = 1.0
            self.params = [wr.requires_grad_(), torch.zeros(F_, 1, cfg.W).requires_grad_(),
                           torch.zeros(F_, 1, 1).requires_grad_(), torch.zeros(F_, 1, 1).requires_grad_(),
                           torch.zeros(F_, 1, 1).requires_grad_()]
            self.reset_opt()
            return
        if kind.startswith('sym'):
            # 4-class symbol classifier on a window of samples, no trellis
            dims = [2 * cfg.W] + (list(cfg.sym_hidden) if kind == 'sym_mlp' else []) + [cfg.M]
        else:
            dims = [2] + (list(cfg.mlp_hidden) if kind == 'mlp' else []) + [cfg.C]
        for a, b in zip(dims, dims[1:]):
            W = torch.empty(F_, a, b)
            for f in range(F_):
                torch.nn.init.kaiming_uniform_(W[f].T, a=math.sqrt(5))
            bound = 1 / math.sqrt(a)
            self.params += [W.requires_grad_(), torch.empty(F_, 1, b).uniform_(-bound, bound).requires_grad_()]
        self.reset_opt()

    def n_params(self):
        return sum(p[0].numel() for p in self.params)   # per frame

    def equalize(self, x):                                    # sym_eq: (F, n, 2W) -> complex (F, n)
        W = self.cfg.W
        yw = torch.complex(x[..., :W], x[..., W:])
        w = torch.complex(self.params[0], self.params[1])
        return (yw * w).sum(-1) + torch.complex(self.params[2], self.params[3])[..., 0]

    def forward(self, x):                                     # x: (F, n, 2)
        if self.kind == 'sym_eq':
            z = self.equalize(x)
            c = self.cfg.const
            return torch.exp(self.params[4]) * 4 * (z[..., None] * c.conj()).real   # (F, n, M)
        if self.kind == 'tied':
            h = torch.complex(self.params[0], self.params[1])[:, 0]     # (F, L)
            mu = (self.sym[None] * h[:, None, :]).sum(-1)               # (F, C)
            a = torch.exp(self.params[2])                              # (F, 1, 1)
            corr = x[..., :1] * mu.real[:, None, :] + x[..., 1:] * mu.imag[:, None, :]
            return a * (2 * corr - (mu.abs() ** 2)[:, None, :])
        if self.kind == 'tied_mlp':
            h = torch.complex(self.params[0], self.params[1])[:, 0]
            mu = (self.sym[None] * h[:, None, :]).sum(-1)               # (F, C)
            a = torch.exp(self.params[2])                              # (F, 1, 1)
            y = torch.complex(x[..., 0], x[..., 1])                    # (F, n)
            r = y[:, :, None] - mu[:, None, :]                         # (F, n, C)
            feat = torch.stack([r.real, r.imag, r.abs() ** 2], -1)     # (F, n, C, 3)
            Fn, n, C, _ = feat.shape
            W1, b1, W2, b2 = self.params[3:]
            hid = torch.relu(torch.baddbmm(b1, feat.reshape(Fn, n * C, 3), W1))
            g = torch.baddbmm(b2, hid, W2).reshape(Fn, n, C)
            return -a[..., 0:1] * (r.abs() ** 2) + g
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            x = torch.baddbmm(b, x, W)
            if i < n_layers - 1:
                x = torch.sigmoid(x) if i == 0 else torch.relu(x)
        return x

    def prepare(self, y, d, cls):
        """(inputs, labels) for one word: per-sample features + branch classes
        for trellis front ends, symbol windows + symbol indices for 4-class ones."""
        if self.kind.startswith('sym'):
            return sym_features(self.cfg, y), d
        return features(y), cls

    def scores(self, y):                                      # y: (F, T) -> (F, T, C)
        with torch.no_grad():
            return F.log_softmax(self.forward(features(y)), -1)

    def decide(self, y):                                      # 4-class: (F, N) symbol indices
        with torch.no_grad():
            return self.forward(sym_features(self.cfg, y)).argmax(-1)

    def reset_opt(self):
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = torch.zeros(self.params[0].shape[0])

    def step(self, x, classes, lr, mask=None, iters=1, loss_kind='ce'):
        """iters Adam steps on (inputs x, labels) for the frames in mask.
        loss_kind='lms' (sym_eq only): |z - c_label|^2 instead of cross-entropy."""
        F_ = x.shape[0]
        mask = torch.ones(F_, dtype=torch.bool) if mask is None else mask
        if iters == 0 or not mask.any():
            return
        mf = mask.float()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(iters):
            if loss_kind == 'lms':
                loss = (self.equalize(x) - self.cfg.const[classes]).abs().reshape(-1) ** 2
            else:
                logits = self.forward(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), classes.reshape(-1), reduction='none')
            loss = (loss.view(F_, -1).mean(1) * mf).sum()
            grads = torch.autograd.grad(loss, self.params, allow_unused=True)   # lms leaves the scale unused
            grads = [torch.zeros_like(p) if g is None else g for p, g in zip(self.params, grads)]
            self.t = self.t + mf
            with torch.no_grad():
                for p, g, m, v in zip(self.params, grads, self.m, self.v):
                    sh = (-1,) + (1,) * (p.dim() - 1)
                    k = mf.view(sh)
                    m.mul_(1 - k * (1 - b1)).add_(k * (1 - b1) * g)
                    v.mul_(1 - k * (1 - b2)).add_(k * (1 - b2) * g * g)
                    tt = self.t.clamp(min=1).view(sh)
                    mhat = m / (1 - b1 ** tt)
                    vhat = v / (1 - b2 ** tt)
                    p.sub_(k * lr * mhat / (vhat.sqrt() + eps))

    def broadcast_from(self, other, F_):
        """Copy frame 0 of `other` (offline model) to F_ frames."""
        self.params = [p.detach()[0:1].repeat(F_, 1, 1).clone().requires_grad_() for p in other.params]
        self.reset_opt()


def offline_train(cfg, kind, N0, steps=None):
    net = BatchedNet(cfg, kind, 1)
    steps = cfg.offline_steps if steps is None else steps
    for _ in range(steps):
        h = random_taps(cfg, cfg.offline_words)
        d, y, cls = transmit(cfg, h, N0)
        x, lab = net.prepare(y, d, cls)
        # one frame, many words: flatten words into the sample axis
        net.step(x.reshape(1, -1, x.shape[-1]), lab.reshape(1, -1), cfg.offline_lr)
    return net


# ----------------------------------------------------------- classic LS ---
def ls_estimate(cfg, y, classes):
    """Per-frame LS taps from known symbols. y, classes: (F, T)."""
    win = cfg.digits[classes]                                  # (F, T, L)
    X = cfg.const[win]                                        # (F, T, L)
    Xh = X.conj().transpose(1, 2)
    h = torch.linalg.solve(Xh @ X + 1e-6 * torch.eye(cfg.L, dtype=CDT), Xh @ y[..., None])[..., 0]
    resid = y - (X * h[:, None, :]).sum(-1)
    N0 = (resid.abs() ** 2).mean(1).clamp(min=1e-6)
    return h.to(CDT), N0


def classic_scores(cfg, y, h, N0):
    mu = class_means(cfg, h)                                  # (F, C) or (F, T, C)
    mu = mu[:, None, :] if mu.dim() == 2 else mu
    return -(y[:, :, None] - mu).abs() ** 2 / N0.view(-1, 1, 1)


# ------------------------------------------------- linear equalizers ---
def eq_windows(cfg, y):
    """(F, N, W+1) complex windows around each symbol, + a bias column."""
    x = sym_features(cfg, y)
    X = torch.complex(x[..., :cfg.W], x[..., cfg.W:])
    return torch.cat([X, torch.ones_like(X[..., :1])], -1)


def le_ls_fit(X, s, lam=1e-3):
    Xh = X.conj().transpose(1, 2)
    return torch.linalg.solve(Xh @ X + lam * torch.eye(X.shape[-1], dtype=CDT), Xh @ s[..., None])[..., 0]


def le_oracle(cfg, h, N0):
    """MMSE weights (with a zero bias) for d[i] from y[i-back .. i-back+W-1]."""
    W, L = cfg.W, cfg.L
    back = (W - 1) // 2
    H = torch.zeros(h.shape[0], W, W + L - 1, dtype=CDT)
    for m in range(W):
        for k in range(L):
            H[:, m, m + L - 1 - k] = h[:, k]
    R = H @ H.conj().transpose(1, 2) + N0 * torch.eye(W, dtype=CDT)
    w = torch.linalg.solve(R, H[:, :, back + L - 1][..., None])[..., 0].conj()
    return torch.cat([w, torch.zeros_like(w[:, :1])], -1)


def slicer(cfg, z):
    return (z[..., None] - cfg.const).abs().argmin(-1)


# -------------------------------------------------------------- evaluate ---
def run_snr(cfg, snr_db, frames, receivers, online_iters, seed=0):
    """Returns {receiver: (symbol_errors, symbols)} over the data words."""
    torch.manual_seed(seed)
    N0 = 10 ** (-snr_db / 10)
    nets = {}
    for r in receivers:
        if r.startswith(('affine', 'mlp', 'tied', 'sym')):
            kind = r.split('@')[0]
            off = offline_train(cfg, kind, N0)
            net = BatchedNet(cfg, kind, frames)
            net.broadcast_from(off, frames)
            nets[r] = net
    if cfg.doppler > 0:
        # reseed after offline training so every receiver set sees the same
        # channel and data (paired comparison across separate run_snr calls)
        torch.manual_seed(seed + 7919)
        T = cfg.N + cfg.L - 1
        taps_all = jakes_taps(cfg, frames, cfg.words * T)
        h = taps_all[:, :T]
    else:
        h = random_taps(cfg, frames)
    stats = {r: [0, 0] for r in receivers}
    ls_state = {}
    pil = []
    for w in range(cfg.words):
        if cfg.doppler > 0:
            h = taps_all[:, w * T:(w + 1) * T]                # (F, T, L), varies within the word
        elif w > 0:
            h = evolve_taps(cfg, h)
        d, y, cls = transmit(cfg, h, N0)
        if w < cfg.pilot_words:                               # pilot words
            pil.append((d, y, cls))
            if w == cfg.pilot_words - 1:
                for r in receivers:
                    if r == 'classic_ls':
                        ls_state[r] = ls_estimate(cfg, torch.cat([p[1] for p in pil], 1),
                                                  torch.cat([p[2] for p in pil], 1))
                    elif r == 'le_ls':
                        ls_state[r] = le_ls_fit(torch.cat([eq_windows(cfg, p[1]) for p in pil], 1),
                                                cfg.const[torch.cat([p[0] for p in pil], 1)])
                    elif r in nets:
                        xs, labs = zip(*[nets[r].prepare(yy, dd, cc) for dd, yy, cc in pil])
                        nets[r].step(torch.cat(xs, 1), torch.cat(labs, 1), cfg.online_lr, iters=cfg.pilot_iters)
            continue
        for r in receivers:
            if r == 'classic_csi':
                sc = classic_scores(cfg, y, h, torch.full((frames,), N0))
            elif r == 'classic_ls':
                sc = classic_scores(cfg, y, *ls_state[r])
            elif r.startswith('le_'):
                w_eq = le_oracle(cfg, h if h.dim() == 2 else h[:, h.shape[1] // 2], N0) if r == 'le_oracle' else ls_state[r]
                X_eq = eq_windows(cfg, y)
            elif not r.startswith('sym'):
                sc = nets[r].scores(y)
            if r.startswith('le_'):
                dhat = slicer(cfg, (X_eq * w_eq[:, None, :]).sum(-1))
            else:
                dhat = nets[r].decide(y) if r.startswith('sym') else viterbi(cfg, sc)
            err = (dhat != d).float().mean(1)                 # per-frame word SER
            stats[r][0] += int((dhat != d).sum())
            stats[r][1] += d.numel()
            if cfg.adapt == 'dd':
                ok = torch.ones(frames, dtype=torch.bool)
                pad = torch.zeros(frames, cfg.L - 1, dtype=torch.long)
                full = torch.cat([pad, dhat, pad], 1)
                cls = (full.unfold(1, cfg.L, 1).flip(-1) * (cfg.M ** torch.arange(cfg.L))).sum(-1)
            else:
                ok = err <= cfg.ser_thresh
            if r == 'le_ls':
                lab_s = cfg.const[dhat if cfg.adapt == 'dd' else d]
                ls_state[r] = torch.where(ok[:, None], le_ls_fit(X_eq, lab_s), ls_state[r])
            elif r == 'classic_ls':
                h_new, n_new = ls_estimate(cfg, y, cls)
                h_old, n_old = ls_state[r]
                ls_state[r] = (torch.where(ok[:, None], h_new, h_old), torch.where(ok, n_new, n_old))
            elif r in nets:
                lab_d = dhat if cfg.adapt == 'dd' else d
                x, lab = nets[r].prepare(y, lab_d, cls)
                nets[r].step(x, lab, cfg.dd_lr or cfg.online_lr, mask=ok, iters=online_iters[r],
                             loss_kind='lms' if nets[r].kind == 'sym_eq' else 'ce')
    return {r: tuple(v) for r, v in stats.items()}, {r: nets[r].n_params() for r in nets}
