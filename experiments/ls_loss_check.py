"""Check of the LS estimation-loss argument (letter Sec. III-A, journal Sec. V): for RS-coded BPSK
words, average s(x)^T (X^T X)^{-1} s(x) over random words and the 2^L states x, compared with the
near-orthogonal approximation L/T, and the condition number of X^T X / T.

Words are built exactly as Code/channel/channel_dataset.py does: 120 random information bits,
RS-encoded with 2 parity symbols (136 bits), L zero bits appended, BPSK-mapped (0 -> +1), and
the T = 136 regressor rows are the L-sample windows the channel convolves.

    python3 experiments/ls_loss_check.py [words] [seed]   -> Results/metrics/ls_loss_check.json
"""
import itertools, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Code.ecc.rs_main import encode

WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
L, K_INFO, NSYM = 4, 120, 2
rng = np.random.RandomState(SEED)
states = np.array(list(itertools.product([-1.0, 1.0], repeat=L)))      # all 2^L state vectors

quad, cond = [], []
for _ in range(WORDS):
    c = np.asarray(encode(rng.randint(0, 2, size=K_INFO), NSYM)).reshape(-1)
    s = 1.0 - 2.0 * np.concatenate([c, np.zeros(L)])                     # BPSK, zero padding -> +1
    T = len(c)
    X = np.stack([s[i:i + T] for i in range(L)], axis=1)                 # T x L regressor
    G = X.T @ X
    Ginv = np.linalg.inv(G)
    quad.append(float(np.mean(np.einsum('ij,jk,ik->i', states, Ginv, states))))
    cond.append(float(np.linalg.cond(G / T)))

T = 136
out = {'words': WORDS, 'seed': SEED, 'L': L, 'T': T,
       'mean_quadratic_form': float(np.mean(quad)), 'L_over_T': L / T,
       'loss_db_exact': float(10 * np.log10(1 + np.mean(quad))), 'loss_db_approx': float(10 * np.log10(1 + L / T)),
       'median_condition_number': float(np.median(cond)), 'p95_condition_number': float(np.percentile(cond, 95))}
json.dump(out, open(os.path.join(ROOT, 'Results', 'metrics', 'ls_loss_check.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
