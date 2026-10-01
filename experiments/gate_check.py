"""Oracle vs implementable adaptation gate, on identical Monte-Carlo draws.

The sweeps (run_mc_sweep.py) gate online adaptation with the reference rule
'oracle': decoded-word SER against the transmitted bits <= 0.02. A receiver
cannot compute that. 'rs' is the implementable rule (Code/trainer.py,
gate_mode): accept a data word when the re-encoded decoded word differs from
the detector's hard decisions in at most one RS symbol (bounded-distance
decoding for 2 parity symbols) and adapt on the re-encoded word.

Each configuration starts from the offline weights of its SNR sweep (the weights label in
CONFIGS), except 'ViterbiNet K=200', which starts from the ViterbiNet_on5 (K=5) sweep's weights
rather than those of the separately trained K=200 sweep. Every configuration
evaluates repetitions 0..N-1, i.e. the same cached (bits, noise) draws the
sweeps used, so differences between rows are paired.

    python3 experiments/gate_check.py <snr> [reps]
Appends to Results/metrics/gate_check.csv (per-rep SER in gate_check_reps.json).
GATE_CHECK_OUT overrides the CSV path; GATE_CHECK_ONLY=0,1 runs a subset of CONFIGS.
"""
import csv, fcntl, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch
from Code.trainer import Trainer
from run_mc_sweep import weights_dir_for

SNR = int(sys.argv[1]); REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = os.environ.get('GATE_CHECK_OUT', 'Results/metrics/gate_check.csv')
REPS_JSON = OUT.replace('.csv', '_reps.json')
# label, trainer model, method, online iters, weights taken from this sweep label
CONFIGS = [('LS-Viterbi', 'ClassicViterbi_LS', 'Statistical', None, None),
           ('ViterbiNet K=5', 'ViterbiNet', 'ModelBased', 5, 'ViterbiNet_on5'),
           ('VNet-affine K=200', 'VNet_affine', 'ModelBased', None, 'VNet_affine'),
           ('ViterbiNet K=200', 'ViterbiNet', 'ModelBased', 200, 'ViterbiNet_on5')]


def run(label, model, method, iters, wlabel, gate):
    kw = {} if iters is None else {'self_supervised_iterations': iters}
    tr = Trainer(model_name=model, detector_method=method, curr_SNR=SNR, val_block_length=120,
                 train_block_length=120, pilots_num=25, gate_mode=gate,
                 weights_dir=weights_dir_for(wlabel or model, method), **kw)
    if wlabel:
        w = torch.load(os.path.join(weights_dir_for(wlabel, method), f'snr_{SNR}_gamma_{tr.gamma}.pt'))
        tr.detector.model.load_state_dict(w['model_state_dict'])
    tr._eval_rep_counter = 0
    per_rep = []
    for _ in range(REPS):
        ser = np.asarray(tr.online_evaluation(num_of_rep=1)).reshape(-1)
        per_rep.append(float(ser[tr.data_indices].mean()))
    return per_rep, getattr(tr, 'gate_stats', {})


ONLY = os.environ.get('GATE_CHECK_ONLY')   # e.g. '0,1' to run a subset of CONFIGS
rows, reps = [], {}
for label, model, method, iters, wlabel in [c for i, c in enumerate(CONFIGS) if ONLY is None or str(i) in ONLY.split(',')]:
    for gate in ('oracle', 'rs'):
        t0 = time.time()
        per_rep, st = run(label, model, method, iters, wlabel, gate)
        m = float(np.mean(per_rep)); ci = 1.96 * float(np.std(per_rep, ddof=1)) / len(per_rep) ** 0.5
        acc = st.get('accepted', 0) / max(1, st.get('data_words', 1))
        wrong = st.get('wrong_labels', 0) / max(1, st.get('accepted', 1))
        rows.append([label, gate, SNR, m, ci, len(per_rep), round(m * len(per_rep) * 14400), acc, wrong,
                     round(time.time() - t0)])
        reps[f'{label}|{gate}|{SNR}'] = per_rep
        print(f'snr={SNR} {label:18s} {gate:6s} SER={m:.3e}+-{ci:.1e} accepted={acc:.3f} '
              f'wrong_labels={wrong:.4f} ({time.time() - t0:.0f}s)', flush=True)

with open(OUT, 'a', newline='') as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    w = csv.writer(f, lineterminator='\n')
    if f.tell() == 0:
        w.writerow(['receiver', 'gate', 'snr', 'ser', 'ci95', 'n_reps', 'bit_errors', 'accept_rate',
                    'wrong_label_rate', 'run_time_sec'])
    w.writerows(rows)
    allreps = json.load(open(REPS_JSON)) if os.path.exists(REPS_JSON) else {}
    allreps.update(reps)
    json.dump(allreps, open(REPS_JSON, 'w'), indent=0)
    fcntl.flock(f, fcntl.LOCK_UN)
print('gate check complete', flush=True)
