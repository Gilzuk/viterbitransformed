"""Paired LS-Viterbi vs learned-receiver differences on shared Monte-Carlo draws.

Receivers evaluated in this container read repetition r's (bits, noise) from the
same on-disk cache entry (Code/channel/data_cache.py, reps < 200), so their
per-repetition SERs (saved in Results/metrics/.mc_sweep_checkpoints) are paired
over the first min(n_a, n_b, 200) repetitions. Intervals are pointwise paired
Student-t 95% intervals (one per SNR, no multiplicity correction). Rows run on other machines (ViterbiNet
K=200, Transformer: Colab) used independent draws and are not paired here.
Applies the 125/120 pilot-word correction used by paper/make_figures.py.

    python3 experiments/paired_stats.py   -> Results/metrics/paired_ls_vs_learned.csv
"""
import csv, json, os
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, 'Results', 'metrics')
CKPT = os.path.join(MET, '.mc_sweep_checkpoints')
n_csv = {(r['model'], int(r['snr'])): int(r['n_reps']) for r in csv.DictReader(open(os.path.join(MET, 'mc_sweep_validation.csv')))}


def reps(model, snr):
    """Per-repetition SERs, only if the checkpoint holds exactly the reported repetitions."""
    path = os.path.join(CKPT, f'{model}_snr{snr}.json')
    if not os.path.exists(path):
        return None
    r = json.load(open(path))['per_rep_means']
    return np.array(r) * 125 / 120 if len(r) == n_csv.get((model, snr)) else None


rows = []
for snr in range(0, 15):
    a = reps('ClassicViterbi_LS', snr)
    for other in ('ViterbiNet_on5', 'VNet_affine'):
        b = reps(other, snr)
        if a is None or b is None:
            continue
        n = min(len(a), len(b), 200)   # the shared cache covers repetitions r < 200 only
        d = a[:n] - b[:n]
        ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)   # pointwise paired Student-t 95% interval
        rows.append(['ClassicViterbi_LS', other, snr, n, d.mean(), ci, a[:n].mean(), b[:n].mean(),
                     int((d < 0).sum()), int((d > 0).sum()), np.corrcoef(a[:n], b[:n])[0, 1],
                     'LS better' if d.mean() + ci < 0 else ('LS worse' if d.mean() - ci > 0 else 'not resolved')])
with open(os.path.join(MET, 'paired_ls_vs_learned.csv'), 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['receiver_a', 'receiver_b', 'snr', 'n_shared_reps', 'mean_diff', 'ci95', 'ser_a_shared',
                'ser_b_shared', 'reps_a_better', 'reps_b_better', 'corr', 'verdict'])
    w.writerows(rows)
for r in rows:
    print(f'{r[2]:3d} {r[1]:15s} n={r[3]:3d} diff={r[4]:+.2e} +-{r[5]:.1e} corr={r[10]:+.2f} {r[11]}')

# ---- implementable (RS) vs oracle gate, paired over the same repetitions (experiments/gate_check.py) ----
# gate_check.py averages SER over data words only, so no 125/120 correction is needed here.
gpath = os.path.join(MET, 'gate_check_reps.json')
if os.path.exists(gpath):
    g = json.load(open(gpath))
    grows = []
    for key in sorted(k for k in g if '|oracle|' in k):
        rec, _, snr = key.split('|')
        o, q = np.array(g[key]), np.array(g.get(f'{rec}|rs|{snr}', []))
        n = min(len(o), len(q))
        if n < 2:
            continue
        d = q[:n] - o[:n]
        ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        grows.append([rec, int(snr), n, d.mean(), ci, o[:n].mean(), q[:n].mean(),
                      'RS higher' if d.mean() - ci > 0 else ('RS lower' if d.mean() + ci < 0 else 'not resolved')])
    grows.sort(key=lambda r: (r[1], r[0]))
    with open(os.path.join(MET, 'gate_paired_diff.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['receiver', 'snr', 'n_reps', 'mean_diff_rs_minus_oracle', 'ci95', 'ser_oracle', 'ser_rs', 'verdict'])
        w.writerows(grows)
    for r in grows:
        print(f'gate {r[1]:3d} {r[0]:18s} n={r[2]:3d} RS-oracle={r[3]:+.2e} +-{r[4]:.1e} {r[7]}')
