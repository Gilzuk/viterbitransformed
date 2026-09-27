"""Adds the tracking 4-class detector and the linear-equalizer references to
Results/metrics/qpsk_sweep.csv (same channel, frames and SNR grid as
run_qpsk_sweep.py):
    sym_eq@30   4-class classifier with equalizer structure, 9-sample window,
                pilot: cross-entropy; tracking: decision-directed LMS, 30 steps/word
    le_ls       classical LS linear equalizer (pilot, then re-solved on own decisions)
    le_oracle   MMSE linear equalizer with the true taps (linear bound)
The original sym_affine/sym_mlp rows stay: they show a 4-class classifier that
does not track (cross-entropy on its own decisions has ~zero gradient)."""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Code.qpsk_sim import Config, run_snr

OUT = 'Results/metrics/qpsk_sweep.csv'
SNRS = list(range(0, 26, 2))
FRAMES = 64
RECEIVERS = ['le_oracle', 'le_ls', 'sym_eq@30']
ITERS = {'sym_eq@30': 30}
cfg = Config(dd_lr=0.03, sym_window=9)
done = {(r['receiver'], int(r['es_n0_db'])) for r in csv.DictReader(open(OUT))}
with open(OUT, 'a', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    for snr in SNRS:
        if all((r, snr) in done for r in RECEIVERS):
            continue
        t0 = time.time()
        stats, npar = run_snr(cfg, snr, FRAMES, RECEIVERS, ITERS, seed=snr)
        for r, (e, n) in stats.items():
            w.writerow([r, npar.get(r, 0 if r == 'le_oracle' else 2 * (cfg.W + 1)), snr, e / n, e, n, cfg.M, cfg.L,
                        cfg.rho, cfg.pilot_words, cfg.pilot_iters, ITERS.get(r, ''), cfg.adapt, FRAMES,
                        cfg.online_lr, cfg.dd_lr])
        f.flush()
        print(f'[{time.strftime("%H:%M:%S")}] Es/N0={snr:2d} dB: ' +
              '  '.join(f'{r}={e / n:.2e}' for r, (e, n) in stats.items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
print('qpsk sym sweep complete', flush=True)
