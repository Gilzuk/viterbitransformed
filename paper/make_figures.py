"""Vector figures and table numbers for the two IEEE papers, from the repo CSVs.

    python3 paper/make_figures.py        (run from the repository root)

Writes paper/figures/*.pdf and paper/figures/numbers.tex (LaTeX macros for the
tables/text), so every number in the papers traces back to Results/metrics/.

BPSK bookkeeping corrections (applied here, uniformly to every BPSK receiver):
  * run_mc_sweep.py averages the per-word SER over all 125 words of a repetition
    with the 5 pilot words entered as 0, so the stored ser_mean is 120/125 of the
    per-data-word SER. We report ser_mean * 125/120.
  * bits_run / errors_observed in the CSV use a nominal 16 bits per word; each of
    the 120 data words per repetition actually carries 120 information bits
    (14,400 bits per repetition). Zero-error points are reported with the
    rule-of-three bound 3 / (n_reps * 14,400).
QPSK results (Code/qpsk_sim.py) count data symbols directly and need no correction.
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, 'Results', 'metrics')
OUT = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT, exist_ok=True)

DATA_SCALE = 125 / 120           # pilot words averaged in as zero-error
BITS_PER_REP = 120 * 120         # 120 data words x 120 information bits

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'lines.markersize': 3.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.grid': True,
    'grid.linewidth': 0.4, 'grid.color': '#dddddd', 'legend.frameon': False, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02})
COL, DCOL = 3.45, 7.0  # IEEE single / double column widths (in)
# Okabe-Ito colour-blind-safe palette
K, BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, YELLOW = ('#000000', '#0072B2', '#E69F00', '#009E73',
                                                    '#D55E00', '#56B4E9', '#CC79A7', '#F0E442')


def read(name):
    return list(csv.DictReader(open(os.path.join(MET, name))))


# --------------------------------------------------------------- BPSK sweep ---
sweep = read('mc_sweep_validation.csv')


SNR_MAX_PLOT = 15   # SNR plots stop at 15 dB; reported results stop at 14 dB


def bpsk_curve(model, snr_max=17):
    """[(snr, ser, is_bound)] with the corrections above."""
    out = []
    for r in sorted((r for r in sweep if r['model'] == model and int(r['snr']) <= snr_max), key=lambda r: int(r['snr'])):
        if r['censored'] == '1' or float(r['ser_mean']) == 0:
            out.append((int(r['snr']), 3 / (int(r['n_reps']) * BITS_PER_REP), True))
        else:
            out.append((int(r['snr']), float(r['ser_mean']) * DATA_SCALE, False))
    return out


def plot_curve(ax, model, label, color, marker, ls='-', bounds=True, **kw):
    pts = bpsk_curve(model, SNR_MAX_PLOT)
    meas = [(s, v) for s, v, b in pts if not b]
    if meas:
        ax.plot(*zip(*meas), ls, color=color, marker=marker, label=label, **kw)
    cens = [(s, v) for s, v, b in pts if b]
    if cens and bounds:
        ax.plot(*zip(*cens), 'v', color=color, mfc='none', ms=4, mew=0.8, **{k: v for k, v in kw.items() if k == 'zorder'})


def snr_axes(ax, ylim=(1e-8, 0.4)):
    ax.set_yscale('log'); ax.set_ylim(*ylim); ax.set_xlim(-0.5, SNR_MAX_PLOT + 0.5)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('SER')
    ax.xaxis.set_major_locator(FixedLocator(range(0, SNR_MAX_PLOT + 1, 1 if SNR_MAX_PLOT <= 15 else 2)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) if v % 2 == 0 else '' for v in range(0, SNR_MAX_PLOT + 1)]))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))


# Fig. letter-1 / journal: SER vs SNR, fair baseline
def fig_ser_snr(name, extra=False):
    fig, ax = plt.subplots(figsize=(COL, 2.55 if not extra else 2.9))
    if extra:
        for pct, c in ((25, '#9ecae1'), (50, '#6baed6'), (100, '#2171b5')):
            plot_curve(ax, f'ClassicViterbi_csi{pct}', f'Viterbi, {pct}% CSI error', c, 'd', ':', lw=0.9)
        plot_curve(ax, 'Transformer', 'Transformer + Viterbi (6,960)', PURPLE, 'x', lw=0.9)
    plot_curve(ax, 'ClassicViterbi', 'Viterbi, perfect CSI', K, 'o', '--', zorder=5)
    plot_curve(ax, 'ClassicViterbi_LS', 'LS-Viterbi, no CSI', VERM, 's', zorder=6)
    plot_curve(ax, 'ViterbiNet', 'ViterbiNet, 200 steps', ORANGE, '^')
    plot_curve(ax, 'ViterbiNet_on5', 'ViterbiNet, 5 steps', BLUE, 'o')
    plot_curve(ax, 'VNet_affine', 'VNet-affine (32 par.), 200 steps', GREEN, 'D')
    ax.plot([], [], 'v', color='grey', mfc='none', label='zero errors (95% bound)')
    snr_axes(ax, (1e-8, 0.4) if not extra else (1e-8, 0.5))
    ax.legend(loc='lower left', ncol=1, handlelength=2.2)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


fig_ser_snr('ser_snr_letter.pdf')
fig_ser_snr('ser_snr_journal.pdf', extra=True)

# ----------------------------------------------------------- topology study ---
topo = [dict(r, params=int(r['params']), mb=int(r['train_minibatches']), on=int(r['online_iters']),
             ser=float(r['ser_mean']) * DATA_SCALE, ci=float(r['ser_ci95']) * DATA_SCALE)
        for r in read('vnet_topology_study_snr7.csv')]
ref7 = {m: v for m, pts in ((m, bpsk_curve(m)) for m in ('ClassicViterbi', 'ClassicViterbi_LS'))
        for s, v, b in pts if s == 7}


def refs(ax, xtext):
    ax.axhline(ref7['ClassicViterbi'], color=K, ls='--', lw=0.8)
    ax.axhline(ref7['ClassicViterbi_LS'], color=VERM, ls=':', lw=1.0)
    ax.text(xtext, ref7['ClassicViterbi'] * 0.99, 'perfect CSI', va='top', ha='right', fontsize=6.5)
    ax.text(xtext, ref7['ClassicViterbi_LS'] * 1.008, 'LS-Viterbi (no CSI)', va='bottom', ha='right', fontsize=6.5, color=VERM)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, ref7['ClassicViterbi'] * 0.94), hi)


def online_panel(ax):
    for t, c, lab in (('affine', GREEN, 'VNet-affine (32)'), ('100-58', BLUE, 'ViterbiNet (7,002)')):
        p = sorted((r['on'], r['ser'], r['ci']) for r in topo if r['topology'] == t and r['mb'] == 25)
        xs = [max(x, 0.5) for x, _, _ in p]
        ax.errorbar(xs, [q[1] for q in p], yerr=[q[2] for q in p], color=c, marker='o', capsize=1.5,
                    elinewidth=0.6, label=lab)
    ticks = [0.5, 5, 10, 25, 50, 100, 200]
    ax.set_xscale('log'); ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(['0'] + [str(t) for t in ticks[1:]]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(0.38, 280)
    refs(ax, 260)
    ax.set_xlabel('Online adaptation steps per accepted word'); ax.set_ylabel('SER at 7 dB')
    ax.legend(loc='upper right')


fig, ax = plt.subplots(figsize=(COL, 2.2))
online_panel(ax)
fig.savefig(os.path.join(OUT, 'online_steps.pdf')); plt.close(fig)

# journal: size / offline budget / latency (3 panels)
lat = sorted(({'params': int(r['params']), 'nn': float(r['nn_forward_us_per_word']),
               'det': float(r['detect_us_per_word']), 'on': float(r['online_iter_us_per_word'])}
              for r in read('vnet_topology_latency.csv')), key=lambda r: r['params'])
fig, (a, b, c) = plt.subplots(1, 3, figsize=(DCOL, 2.2), gridspec_kw={'wspace': 0.42})
pts = sorted((r['params'], r['ser'], r['ci'], r['topology']) for r in topo if r['mb'] == 25 and r['on'] == 200)
for grp, col, lab in (([p for p in pts if '-' not in p[3]], BLUE, '1 hidden layer / affine'),
                      ([p for p in pts if '-' in p[3]], ORANGE, '2 hidden layers')):
    a.errorbar([p[0] for p in grp], [p[1] for p in grp], yerr=[p[2] for p in grp], color=col, lw=0, marker='o',
               capsize=1.5, elinewidth=0.6, label=lab)
for r in (r for r in topo if r['mb'] == 25 and r['on'] == 5):
    a.errorbar([r['params']], [r['ser']], yerr=[r['ci']], color=GREEN, marker='D', lw=0, capsize=1.5, elinewidth=0.6,
               label='5 steps/word' if r['topology'] == 'affine' else None)
a.set_xscale('log'); a.set_xlim(20, 12000); refs(a, 11000)
a.set_xlabel('Trainable parameters'); a.set_ylabel('SER at 7 dB'); a.legend(loc='upper center', bbox_to_anchor=(0.6, 1.0))
a.set_title('(a) size, 200 online steps/word', fontsize=8)
for t, col, lab in (('affine', GREEN, 'VNet-affine (32)'), ('100-58', BLUE, 'ViterbiNet (7,002)')):
    p = sorted((r['mb'], r['ser'], r['ci']) for r in topo if r['topology'] == t and r['on'] == 200)
    b.errorbar([q[0] for q in p], [q[1] for q in p], yerr=[q[2] for q in p], color=col, marker='o', capsize=1.5,
               elinewidth=0.6, label=lab)
b.set_xscale('log'); b.xaxis.set_major_locator(FixedLocator([2, 5, 10, 25]))
b.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '25'])); b.xaxis.set_minor_locator(NullLocator())
refs(b, 30); b.set_xlim(1.6, 32)
b.set_xlabel('Offline training minibatches'); b.set_ylabel('SER at 7 dB'); b.legend(loc='upper right')
b.set_title('(b) offline budget, 200 online steps', fontsize=8)
c.plot([r['params'] for r in lat], [r['nn'] for r in lat], color=BLUE, marker='o', label='NN forward')
c.plot([r['params'] for r in lat], [r['on'] for r in lat], color=ORANGE, marker='s', label='one online step')
trel = sum(r['det'] - r['nn'] for r in lat) / len(lat)
c.axhline(trel, color=K, ls='--', lw=0.8)
c.text(25, trel * 1.15, f'Viterbi trellis ({trel / 1000:.1f} ms)', fontsize=6.5)
c.set_xscale('log'); c.set_yscale('log'); c.set_ylim(10, 3e4)
c.set_xlabel('Trainable parameters'); c.set_ylabel('$\\mu$s per 136-sample word'); c.legend(loc='center', bbox_to_anchor=(0.6, 0.36))
c.set_title('(c) latency, one CPU thread', fontsize=8)
fig.savefig(os.path.join(OUT, 'topology.pdf')); plt.close(fig)

# ------------------------------------------------- architecture / budget search ---
mbs = read('minibatch_search_snr7.csv')
curves = {}
for r in mbs:
    curves.setdefault(r['model'], {})[int(r['train_minibatches'])] = (float(r['ser_mean']) * DATA_SCALE,
                                                                      float(r['ser_ci95']) * DATA_SCALE)
# fixed-budget runs recorded in Results/figures/plot_snr7_trials.py (same 20-rep protocol)
for m, mb, s, ci in (('ViterbiNet', 250, 0.03008, 0.00104), ('ViterbiNet', 1000, 0.02982, 0.00099),
                     ('TransformerV2', 25, 0.03910, 0.00100), ('TransformerV2', 250, 0.03656, 0.00065),
                     ('TransformerV2', 1000, 0.03631, 0.00105), ('ViT_overlap', 250, 0.03211, 0.00078)):
    curves.setdefault(m, {})[mb] = (s * DATA_SCALE, ci * DATA_SCALE)
fig, ax = plt.subplots(figsize=(COL, 2.4))
for m, lab, col, mk in (('ViterbiNet', 'ViterbiNet MLP (7,002)', BLUE, 'o'), ('TransformerV2', 'Transformer V2 (6,960)', PURPLE, 'x'),
                        ('ViterbiTransformerV3', 'Transformer V3, MLP embed. (7,248)', ORANGE, '^'),
                        ('ViterbiTransformerV4', 'Transformer V4, MLP-attn-MLP (6,880)', GREEN, 's'),
                        ('ViT_overlap', 'ViT, overlapping patches (6,976)', SKY, 'D')):
    p = sorted(curves[m].items())
    ax.errorbar([x for x, _ in p], [v[0] for _, v in p], yerr=[v[1] for _, v in p], color=col, marker=mk, capsize=1.2,
                elinewidth=0.5, label=lab)
refs(ax, 1100)
ax.set_xscale('log'); ax.set_ylim(0.021, 0.047)
ax.set_xlabel('Offline training minibatches'); ax.set_ylabel('SER at 7 dB (200 online steps/word)')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=6)
fig.savefig(os.path.join(OUT, 'architectures.pdf')); plt.close(fig)

# ------------------------------------------------------------------ latency ---
dl = [r for r in read('detector_latency_symbol_budget.csv') if r['model'] != 'Mamba2']
NAMES = {'VNet_affine': 'VNet-affine (32)', 'VNet_4': 'VNet, 4 hidden (88)', 'ViterbiNet_100-58': 'ViterbiNet (7,002)',
         'TransformerV2': 'Transformer V2 (6,960)', 'ViterbiTransformerV4': 'Transformer V4 (6,880)',
         'ViT_overlap': 'ViT overlap (6,976)'}
dl.sort(key=lambda r: float(r['online_iter_ms_per_word']))
fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True, gridspec_kw={'wspace': 0.08, 'width_ratios': [1.3, 1]})
y = range(len(dl))
a.barh(y, [float(r['trellis_ms_per_word']) for r in dl], color='#bbbbbb', label='trellis')
a.barh(y, [float(r['nn_forward_ms_per_word']) for r in dl], left=[float(r['trellis_ms_per_word']) for r in dl],
       color=BLUE, label='NN forward')
a.set_yticks(list(y)); a.set_yticklabels([NAMES[r['model']] for r in dl], fontsize=6.5)
a.set_xlabel('Detection (ms/word)'); a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6); a.set_xlim(0, 16)
b.barh(y, [float(r['online_iter_ms_per_word']) for r in dl], color=ORANGE)
b.set_xscale('log'); b.set_xlabel('One online step (ms)'); b.set_xlim(0.3, 15)
for ax_ in (a, b):
    ax_.grid(axis='y', visible=False)
fig.savefig(os.path.join(OUT, 'latency.pdf')); plt.close(fig)

# --------------------------------------------------------------------- QPSK ---
q = [r for r in read('qpsk_sweep.csv') if r['receiver'] != 'receiver']


def qcurve(key):
    p = sorted((int(r['es_n0_db']), int(r['symbol_errors']), int(r['symbols'])) for r in q
               if r['receiver'] == key and int(r['es_n0_db']) <= SNR_MAX_PLOT)
    return [(s, e / n) for s, e, n in p if e > 0], [(s, 3 / n) for s, e, n in p if e == 0]


fig, ax = plt.subplots(figsize=(COL, 2.3))
for key, lab, col, mk, ls in (('classic_csi', 'Viterbi, perfect CSI', K, 'o', '--'),
                              ('classic_ls', 'LS-Viterbi, no CSI', VERM, 's', '-'),
                              ('tied@5', 'Tied-tap affine + trellis (7)', GREEN, 'D', '-'),
                              ('tied_mlp@5', 'Tied + residual MLP + trellis (88)', SKY, 'P', '-'),
                              ('mlp@10', 'Free per-branch MLP, ViterbiNet-style (9,934)', ORANGE, '^', '-'),
                              ('affine@10', 'Free per-branch affine (192)', YELLOW, 'v', '-'),
                              ('le_oracle', 'Linear MMSE equalizer, true taps (9 taps)', '#999999', '.', ':'),
                              ('sym_eq@30', '4-class classifier, equalizer struct. (21)', PURPLE, 'x', '-'),
                              ('sym_affine@10', '4-class linear classifier, CE tracking (44)', BLUE, '+', '-.')):
    m, c_ = qcurve(key)
    if m:
        ax.plot(*zip(*m), ls, color=col, marker=mk, label=lab)
    if c_:
        ax.plot(*zip(*c_), 'v', color=col, mfc='none', ms=4, mew=0.8)
ax.set_yscale('log'); ax.set_ylim(3e-3, 1); ax.set_xlim(-0.5, SNR_MAX_PLOT + 0.5)
ax.xaxis.set_major_locator(FixedLocator(range(0, SNR_MAX_PLOT + 1, 2)))
ax.set_xlabel('$E_s/N_0$ (dB)'); ax.set_ylabel('SER')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.18), fontsize=5.8, ncol=2)
fig.savefig(os.path.join(OUT, 'qpsk.pdf')); plt.close(fig)

# -------------------------------------------------------------- fast fading ---
ff = read('fast_fading.csv')
fig, ax = plt.subplots(figsize=(COL, 2.3))
for key, lab, col, mk, ls in (('ClassicViterbi_genie', 'Viterbi, true per-sample taps (genie)', K, 'o', '--'),
                              ('ClassicViterbi_LS', 'LS-Viterbi, last accepted word', VERM, 's', '-'),
                              ('ClassicViterbi_PSP', 'PSP-LMS Viterbi (per survivor)', YELLOW, 'v', '-'),
                              ('ClassicViterbi_RLS', 'LS-Viterbi, forgetting 0.8', PURPLE, 'x', '-'),
                              ('ViterbiNet_on5', 'ViterbiNet (7,002), 5 steps', BLUE, 'o', '-'),
                              ('VNet_affine', 'VNet-affine (32), 200 steps', GREEN, 'D', '-')):
    p = sorted((float(r['doppler']), float(r['ser_mean']) * DATA_SCALE, float(r['ser_ci95']) * DATA_SCALE)
               for r in ff if r['model'] == key)
    ax.errorbar([x[0] for x in p], [x[1] for x in p], yerr=[x[2] for x in p], color=col, marker=mk, ls=ls,
                capsize=1.2, elinewidth=0.5, label=lab)
ax.set_xscale('log'); ax.set_yscale('log')
ax.xaxis.set_major_locator(FixedLocator([0.001, 0.005, 0.01, 0.03]))
ax.xaxis.set_major_formatter(FixedFormatter(['0.001', '0.005', '0.01', '0.03'])); ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([0.06, 0.08, 0.1, 0.12, 0.15, 0.2]))
ax.yaxis.set_major_formatter(FixedFormatter(['0.06', '0.08', '0.10', '0.12', '0.15', '0.20'])); ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(0.058, 0.21); ax.set_xlabel('Normalized Doppler $f_D T_s$'); ax.set_ylabel('SER at 7 dB')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=6)
fig.savefig(os.path.join(OUT, 'fast_fading_bpsk.pdf')); plt.close(fig)

FIELDS = ['receiver', 'params', 'doppler', 'es_n0_db', 'ser', 'symbol_errors', 'symbols', 'frames',
          'words', 'pilot_words', 'online_iters', 'dd_lr', 'sym_window']
qf = [r for r in csv.DictReader(open(os.path.join(MET, 'qpsk_fast_fading.csv')), fieldnames=FIELDS)
      if r['receiver'] != 'receiver']
fig, axs = plt.subplots(1, 2, figsize=(COL, 2.0), sharey=True, gridspec_kw={'wspace': 0.08})
for ax, snr in zip(axs, (10, 14)):
    for key, lab, col, mk, ls in (('classic_csi', 'genie (true per-sample taps)', K, 'o', '--'),
                                  ('classic_ls', 'LS-Viterbi', VERM, 's', '-'),
                                  ('tied@5', 'tied affine + trellis', GREEN, 'D', '-'),
                                  ('tied_mlp@5', 'tied + MLP + trellis', SKY, 'P', '-'),
                                  ('sym_eq@30', '4-class classifier', PURPLE, 'x', '-')):
        p = sorted((float(r['doppler']), float(r['ser'])) for r in qf if r['receiver'] == key and int(r['es_n0_db']) == snr)
        ax.plot(*(zip(*p) if p else ([], [])), ls, color=col, marker=mk, label=lab)
    ax.axhline(0.75, color='grey', lw=0.6, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-3, 1.2); ax.set_xlim(1.4e-4, 1.4e-2)
    ax.set_title(f'$E_s/N_0$ = {snr} dB', fontsize=8); ax.set_xlabel('$f_D T_s$')
axs[0].set_ylabel('SER'); axs[0].text(2.1e-4, 0.8, 'chance (0.75)', fontsize=6, color='grey')
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=6)
fig.savefig(os.path.join(OUT, 'fast_fading_qpsk.pdf')); plt.close(fig)

# ------------------------------------------------ noisy-CSI baseline (journal) ---
fig, ax = plt.subplots(figsize=(COL, 2.5))
for pct, c in ((25, '#c6dbef'), (50, '#9ecae1'), (75, '#6baed6'), (100, '#2171b5')):
    plot_curve(ax, f'ClassicViterbi_csi{pct}', f'Viterbi, {pct}% tap error', c, 'd', ':', lw=0.9)
plot_curve(ax, 'ClassicViterbi', 'Viterbi, perfect CSI', K, 'o', '--', zorder=5)
plot_curve(ax, 'ClassicViterbi_LS', 'LS-Viterbi (estimates taps)', VERM, 's', zorder=6)
plot_curve(ax, 'ViterbiNet', 'ViterbiNet, 200 steps', ORANGE, '^')
snr_axes(ax, (1e-7, 0.5))
ax.legend(loc='lower left', fontsize=6)
fig.savefig(os.path.join(OUT, 'csi_uncertainty.pdf')); plt.close(fig)

# ---------------------------------------- SER relative to the perfect-CSI bound ---
fig, ax = plt.subplots(figsize=(COL, 2.3))
csi = {s: v for s, v, b in bpsk_curve('ClassicViterbi', 14) if not b}
for model, lab, col, mk in (('ClassicViterbi_LS', 'LS-Viterbi, no CSI', VERM, 's'),
                            ('ViterbiNet_on5', 'ViterbiNet, 5 steps', BLUE, 'o'),
                            ('VNet_affine', 'VNet-affine, 200 steps', GREEN, 'D'),
                            ('ViterbiNet', 'ViterbiNet, 200 steps', ORANGE, '^'),
                            ('Transformer', 'Transformer, 200 steps', PURPLE, 'x'),
                            ('ClassicViterbi_csi25', 'Viterbi, 25% tap error', '#6baed6', 'd')):
    p = [(s, v / csi[s]) for s, v, b in bpsk_curve(model, 14) if not b and s in csi]
    ax.plot(*zip(*p), color=col, marker=mk, label=lab, ls=':' if 'csi' in model else '-')
# LS estimation loss (Section V): an LS tap estimate from T samples costs
# 10 log10(1 + L/T) dB of SNR; shift the perfect-CSI curve by that much.
import math
LS_LOSS_DB = 10 * math.log10(1 + 4 / 136)
cs = sorted(csi.items())
def csi_at(snr):
    for (s0, v0), (s1, v1) in zip(cs, cs[1:]):
        if s0 <= snr <= s1:
            return 10 ** (math.log10(v0) + (snr - s0) / (s1 - s0) * (math.log10(v1) - math.log10(v0)))
pred = [(s, csi_at(s - LS_LOSS_DB) / v) for s, v in cs if s >= 1]
ax.plot(*zip(*pred), color=VERM, ls='--', lw=0.8, label=f'LS, predicted ({LS_LOSS_DB:.3f} dB loss)')
ax.axhline(1, color=K, lw=0.8, ls='--')
ax.set_yscale('log'); ax.set_xlim(-0.5, 14.5); ax.xaxis.set_major_locator(FixedLocator(range(0, 15, 2)))
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('SER / SER(perfect CSI)')
ax.legend(loc='upper left', fontsize=6)
fig.savefig(os.path.join(OUT, 'gap_ratio.pdf')); plt.close(fig)

# ------------------------------------------- real-time budget per 136-symbol word ---
LS_SOLVE_MS = 0.02   # median numpy lstsq on a 136x4 system, one thread
dlm = {r['model']: r for r in read('detector_latency_symbol_budget.csv')}
rt = [('LS-Viterbi', float(dlm['ViterbiNet_100-58']['trellis_ms_per_word']), LS_SOLVE_MS),
      ('ViterbiNet, K=5', float(dlm['ViterbiNet_100-58']['detect_ms_per_word']), 5 * float(dlm['ViterbiNet_100-58']['online_iter_ms_per_word'])),
      ('VNet-affine, K=200', float(dlm['VNet_affine']['detect_ms_per_word']), 200 * float(dlm['VNet_affine']['online_iter_ms_per_word'])),
      ('ViterbiNet, K=200', float(dlm['ViterbiNet_100-58']['detect_ms_per_word']), 200 * float(dlm['ViterbiNet_100-58']['online_iter_ms_per_word'])),
      ('Transformer V4, K=200', float(dlm['ViterbiTransformerV4']['detect_ms_per_word']), 200 * float(dlm['ViterbiTransformerV4']['online_iter_ms_per_word'])),
      ('Transformer V2, K=200', float(dlm['TransformerV2']['detect_ms_per_word']), 200 * float(dlm['TransformerV2']['online_iter_ms_per_word']))]
fig, ax = plt.subplots(figsize=(COL, 1.9))
y = range(len(rt))
ax.barh(y, [d for _, d, _ in rt], color='#bbbbbb', label='detection')
ax.barh(y, [a for _, _, a in rt], left=[d for _, d, _ in rt], color=ORANGE, label='adaptation (per accepted word)')
word_ms = 136 * 0.125
ax.axvline(word_ms, color=VERM, ls='--', lw=0.9, label=f'word duration, 0.125 ms/symbol ({word_ms:.0f} ms)')
for i, (_, d, a) in enumerate(rt):
    ax.text((d + a) * 1.12, i, f'{d + a:.1f} ms' if d + a < 100 else f'{d + a:.0f} ms', va='center', fontsize=6)
ax.set_xscale('log'); ax.set_xlim(5, 3e3)
ax.set_yticks(list(y)); ax.set_yticklabels([n for n, _, _ in rt], fontsize=6.5); ax.grid(axis='y', visible=False)
ax.set_xlabel('Time per 136-symbol word (ms, one CPU thread)'); ax.set_ylim(-0.8, len(rt) - 0.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.3), ncol=2, fontsize=6)
fig.savefig(os.path.join(OUT, 'realtime.pdf')); plt.close(fig)

# ------------------------------------ accuracy vs adaptation cost at 7 dB (Pareto) ---
tl = {int(r['params']): float(r['online_iter_us_per_word']) / 1000 for r in read('vnet_topology_latency.csv')}
fig, ax = plt.subplots(figsize=(COL, 2.4))
pts = [(r['on'] * tl[r['params']], r['ser'], r['topology'], r['params'], r['on']) for r in topo
       if r['mb'] == 25 and r['on'] in (5, 25, 200) and r['params'] in tl]
for on, col, mk in ((200, ORANGE, 'o'), (25, SKY, 's'), (5, BLUE, 'D')):
    q_ = [p for p in pts if p[4] == on]
    ax.scatter([p[0] for p in q_], [p[1] for p in q_], color=col, marker=mk, s=12, label=f'ViterbiNet topologies, K={on}', zorder=3)
    for p in q_:
        if p[2] in ('affine', '100-58'):
            ax.annotate('affine' if p[2] == 'affine' else '7,002', (p[0], p[1]), xytext=(3, 2), textcoords='offset points', fontsize=5.5)
for m, lab, col in (('TransformerV2', 'Transformer V2', PURPLE), ('ViterbiTransformerV4', 'Transformer V4', GREEN)):
    ax.scatter([200 * float(dlm[m]['online_iter_ms_per_word'])], [min(v[0] for v in curves[m].values())], color=col, marker='x',
               s=18, label=f'{lab}, K=200 (best budget)', zorder=3)
ax.scatter([LS_SOLVE_MS], [ref7['ClassicViterbi_LS']], color=VERM, marker='*', s=40, label='LS-Viterbi', zorder=4)
ax.axhline(ref7['ClassicViterbi'], color=K, ls='--', lw=0.8)
ax.text(0.012, ref7['ClassicViterbi'] * 0.99, 'perfect CSI', va='top', fontsize=6)
ax.set_xscale('log'); ax.set_xlim(0.008, 3e3)
ax.set_xlabel('Adaptation compute per accepted word (ms)'); ax.set_ylabel('SER at 7 dB')
ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.22), fontsize=5.8, ncol=2)
fig.savefig(os.path.join(OUT, 'pareto.pdf')); plt.close(fig)

# ------------------------------------------------- QPSK diagnostics (if present) ---
def maybe(name):
    path = os.path.join(MET, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


wt = maybe('qpsk_diag_word_trace.csv')
if wt:
    fig, axs = plt.subplots(1, 2, figsize=(COL, 2.0), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, snr in zip(axs, (12, 14)):
        for key, lab, col, mk in (('classic_csi', 'perfect CSI', K, 'o'), ('classic_ls', 'LS-Viterbi', VERM, 's'),
                                  ('tied@5', 'tied affine + trellis', GREEN, 'D'),
                                  ('sym_affine@10', '4-class linear, CE tracking', BLUE, '+'),
                                  ('sym_eq@30', '4-class, equalizer + LMS', PURPLE, 'x'),
                                  ('le_ls', 'LS linear equalizer', '#999999', '.')):
            p = sorted((int(r['word']), float(r['ser'])) for r in wt if r['receiver'] == key and int(r['es_n0_db']) == snr)
            p = [(w, v) for w, v in p if v > 0]   # words with no errors are not drawn
            ax.plot(*(zip(*p) if p else ([], [])), color=col, marker=mk, ms=2.5, lw=0.9, label=lab)
        ax.set_yscale('log'); ax.set_ylim(1e-5, 1); ax.set_title(f'$E_s/N_0$ = {snr} dB', fontsize=8)
        ax.set_xlabel('Data word in frame')
    axs[0].set_ylabel('SER per word')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=6)
    fig.savefig(os.path.join(OUT, 'qpsk_trace.pdf')); plt.close(fig)

pl = maybe('qpsk_diag_pilot.csv')
if pl:
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for key, lab, col, mk in (('classic_ls', 'LS-Viterbi', VERM, 's'), ('tied@5', 'tied affine (7)', GREEN, 'D'),
                              ('affine@10', 'free per-branch affine (192)', YELLOW, 'v'),
                              ('mlp@10', 'free per-branch MLP (9,934)', ORANGE, '^')):
        for iters, ls in ((200, '-'), (2000, ':')):
            p = sorted((int(r['pilot_words']), max(float(r['ser']), 3 / int(r['symbols']))) for r in pl
                       if r['receiver'] == key and int(r['pilot_iters']) == iters)
            ax.plot(*zip(*p), color=col, marker=mk, ls=ls, label=f'{lab}' if iters == 200 else None)
    ax.plot([], [], 'k-', label='200 pilot steps'); ax.plot([], [], 'k:', label='2000 pilot steps')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FixedFormatter(['1', '2', '5', '10'])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Pilot words (120 symbols each)'); ax.set_ylabel('SER, static channel, 14 dB')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2), fontsize=5.8, ncol=2)
    fig.savefig(os.path.join(OUT, 'qpsk_pilot.pdf')); plt.close(fig)

wd = maybe('qpsk_diag_window.csv')
if wd:
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for key, lab, ls in (('le_oracle', 'MMSE, true taps', '--'), ('le_ls', 'LS, own decisions', '-')):
        for snr, col in ((12, BLUE), (14, VERM)):
            p = sorted((int(r['window']), float(r['ser'])) for r in wd if r['receiver'] == key and int(r['es_n0_db']) == snr)
            ax.plot(*zip(*p), color=col, ls=ls, marker='o', label=f'{lab}, {snr} dB')
    ax.set_yscale('log'); ax.set_xlabel('Equalizer window (samples)'); ax.set_ylabel('SER')
    ax.xaxis.set_major_locator(FixedLocator([3, 5, 7, 9, 11, 15]))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6)
    fig.savefig(os.path.join(OUT, 'qpsk_window.pdf')); plt.close(fig)

CKPT = os.path.join(MET, '.mc_sweep_checkpoints')


def ff_reps(name, fd):
    path = os.path.join(CKPT, f'FF_{name}_fd{fd:g}_K3_snr7.json')
    return json.load(open(path))['per_rep_means'] if os.path.exists(path) else None


def paired(name, ref, fd):
    """Mean and 95% CI of SER(name) - SER(ref) over the repetitions both ran (same draws)."""
    a, b = ff_reps(name, fd), ff_reps(ref, fd)
    if not a or not b:
        return None
    d = [(x - y) * DATA_SCALE for x, y in zip(a, b)]
    m = sum(d) / len(d)
    sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    return m, 1.96 * sd / len(d) ** 0.5


import json
if any(r['model'].startswith('tune_') for r in ff):
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 2.1), sharey=True, gridspec_kw={'wspace': 0.08})
    for k, (fd, col, mk) in enumerate(((0.001, BLUE, 'o'), (0.01, ORANGE, 's'), (0.03, VERM, 'D'))):
        off = 1 + (k - 1) * 0.06
        pp = [(mu * off,) + paired(f'tune_PSP_mu{mu:g}', 'VNet_affine', fd) for mu in (0.003, 0.01, 0.02, 0.05)
              if paired(f'tune_PSP_mu{mu:g}', 'VNet_affine', fd)]
        if pp:
            a.errorbar([p[0] for p in pp], [p[1] for p in pp], yerr=[p[2] for p in pp], color=col, marker=mk,
                       capsize=1.5, elinewidth=0.6, label=f'$f_DT_s$={fd:g}')
        xs = [(0.0, 'tune_LS_last')] + [(lam, f'tune_RLS_lam{lam:g}') for lam in (0.5, 0.8, 0.9, 0.95, 0.99)]
        rl = [(off / (1 - x),) + paired(n, 'VNet_affine', fd) for x, n in xs if paired(n, 'VNet_affine', fd)]
        if rl:
            b.errorbar([p[0] for p in rl], [p[1] for p in rl], yerr=[p[2] for p in rl], color=col, marker=mk,
                       capsize=1.5, elinewidth=0.6)
    for ax_ in (a, b):
        ax_.axhline(0, color=K, lw=0.8, ls='--')
    a.set_xscale('log'); a.xaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.02, 0.05]))
    a.xaxis.set_major_formatter(FixedFormatter(['0.003', '0.01', '0.02', '0.05'])); a.xaxis.set_minor_locator(NullLocator())
    a.set_xlabel('PSP-LMS step size $\\mu$'); a.set_ylabel('SER $-$ SER(VNet-affine), 7 dB')
    b.set_xscale('log'); b.set_xlabel('LS memory $1/(1-\\lambda)$ (words)')
    b.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 100]))
    b.xaxis.set_major_formatter(FixedFormatter(['1', '2', '5', '10', '20', '100'])); b.xaxis.set_minor_locator(NullLocator())
    a.set_ylim(-0.012, 0.06)
    a.text(0.0125, 0.058, '$\\mu$=0.05:\n+0.09 to +0.19 (off scale)', fontsize=5.5, ha='center', va='top')
    a.legend(loc='upper left', fontsize=6)
    fig.savefig(os.path.join(OUT, 'tracker_tuning.pdf')); plt.close(fig)

# ------------------------------------------- channel illustration (journal) ---
import sys, numpy as np
sys.path.insert(0, ROOT)
from Code.channel.channel_estimation import _load_cost2100_taps
from Code.channel.channel_dataset import jakes_process
fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw={'wspace': 0.42, 'width_ratios': [1.2, 1]})
taps = _load_cost2100_taps(4)
for i, col in enumerate((BLUE, ORANGE, GREEN, VERM)):
    a.plot(np.arange(1, taps.shape[0] + 1), taps[:, i], color=col, lw=0.9, label=f'$h_{i}$')
a.axvspan(1, 125, color='#eeeeee', zorder=0, label='one frame')
a.set_xlabel('Word index $w$'); a.set_ylabel('Tap magnitude'); a.set_xlim(1, taps.shape[0])
a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.86), fontsize=5.5, ncol=2, handlelength=1.2)
T_WORD, kappa = 136, 3
for fd, col in ((0.001, BLUE), (0.01, ORANGE), (0.03, VERM)):
    g = jakes_process(3 * T_WORD, 1, fd, np.random.RandomState(7))[:, 0]
    b.plot(np.arange(3 * T_WORD) / T_WORD, np.sqrt(kappa / (kappa + 1)) + np.sqrt(1 / (kappa + 1)) * g,
           color=col, lw=0.9, label=f'$f_DT_s$={fd:g}')
for w in (1, 2):
    b.axvline(w, color='#999999', lw=0.6, ls=':')
b.set_xlabel('Time (words of 136 symbols)'); b.set_ylabel('Tap gain $h_k(t)/\\bar h_k^{(w)}$', labelpad=1); b.set_xlim(0, 3)
b.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.3, handlelength=1.0, columnspacing=0.6)
fig.savefig(os.path.join(OUT, 'channel.pdf')); plt.close(fig)

# --------------------------------------- operation counts vs channel memory ---
Ls = np.arange(2, 9); T = 136
def mlp_params(L):   # ViterbiNet 1->100->58->2^L
    return 100 + 100 + 100 * 58 + 58 + 58 * 2 ** L + 2 ** L
ops = {'trellis (add-compare-select)': (T * 2 ** (Ls + 1), K, '-', 'o'),
       'LS re-estimation': (T * Ls ** 2 + Ls ** 3, VERM, '-', 's'),
       'ViterbiNet forward': (T * np.array([mlp_params(L) for L in Ls]), BLUE, '-', 'D'),
       'ViterbiNet, K=5 steps': (5 * 3 * T * np.array([mlp_params(L) for L in Ls]), SKY, '--', 'D'),
       'ViterbiNet, K=200 steps': (200 * 3 * T * np.array([mlp_params(L) for L in Ls]), ORANGE, '--', '^'),
       'VNet-affine, K=200 steps': (200 * 3 * T * 2 * 2 ** Ls, GREEN, '--', 'v')}
fig, ax = plt.subplots(figsize=(COL, 2.3))
for lab, (y, col, ls, mk) in ops.items():
    ax.plot(Ls, y, color=col, ls=ls, marker=mk, label=lab)
ax.axvline(4, color='#999999', lw=0.6, ls=':')
ax.set_yscale('log'); ax.set_xlabel('Channel memory $L$ (BPSK, $2^L$ states)')
ax.set_ylabel('Operations per word ($T$=136)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=5.8)
fig.savefig(os.path.join(OUT, 'complexity.pdf')); plt.close(fig)

# ------------------------------------------------------------- numbers.tex ---
def v(model, snr):
    for s, val, b in bpsk_curve(model):
        if s == snr:
            return val, b
    return None, None


def fmt(x, bound=False):
    if x is None:
        return '--'
    m, e = f'{x:.2e}'.split('e')
    e = int(e)
    s = f'{float(m):.2f}' if e >= -1 else f'{float(m):.1f}\\mathrm{{e}}{{{e}}}'
    if e >= -1:
        s = f'{x:.3f}' if x >= 0.01 else f'{x:.4f}'
    return ('$<' + s + '$') if bound else ('$' + s + '$')


lines = ['% generated by paper/make_figures.py -- do not edit']
for key, model in (('csi', 'ClassicViterbi'), ('ls', 'ClassicViterbi_LS'), ('vnet', 'ViterbiNet'),
                   ('vnetfive', 'ViterbiNet_on5'), ('affine', 'VNet_affine'), ('csitwentyfive', 'ClassicViterbi_csi25'),
                   ('transformer', 'Transformer')):
    for snr, w in ((0, 'zero'), (4, 'four'), (7, 'seven'), (10, 'ten'), (12, 'twelve'), (13, 'thirteen'),
                   (14, 'fourteen'), (15, 'fifteen'), (17, 'seventeen')):
        val, b = v(model, snr)
        lines.append(f'\\newcommand{{\\ser{key}{w}}}{{{fmt(val, b)}}}')
open(os.path.join(OUT, 'numbers.tex'), 'w').write('\n'.join(lines) + '\n')
print('figures + numbers.tex written to', OUT)
for model in ('ClassicViterbi', 'ClassicViterbi_LS', 'ViterbiNet', 'ViterbiNet_on5', 'VNet_affine', 'ClassicViterbi_csi25', 'Transformer'):
    print(f'{model:22s}', ' '.join(f'{s}:{val:.3g}{"<" if b else ""}' for s, val, b in bpsk_curve(model)))
