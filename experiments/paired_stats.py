"""Paired LS-Viterbi vs learned-receiver differences on shared Monte-Carlo draws.

Receivers evaluated in this container read repetition r's (bits, noise) from the
same on-disk cache entry (Code/channel/data_cache.py, reps < 200), so their
per-repetition SERs (saved in Results/metrics/.mc_sweep_checkpoints) are paired
over the first min(n_a, n_b) repetitions. Rows run on other machines (ViterbiNet
K=200, Transformer: Colab) used independent draws and are not paired here.
Applies the 125/120 pilot-word correction used by paper/make_figures.py.

    python3 experiments/paired_stats.py   -> Results/metrics/paired_ls_vs_learned.csv
"""
import csv, json, os
import numpy as np
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
        n = min(len(a), len(b)); d = a[:n] - b[:n]
        ci = 1.96 * d.std(ddof=1) / np.sqrt(n)
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
