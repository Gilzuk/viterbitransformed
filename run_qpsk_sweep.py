"""SER vs Es/N0 for the complex QPSK few-tap fading channel (Code/qpsk_sim.py).
Writes Results/metrics/qpsk_sweep.csv."""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Code.qpsk_sim import Config, run_snr

OUT = 'Results/metrics/qpsk_sweep.csv'
SNRS = list(range(0, 26, 2))
FRAMES = 64
RECEIVERS = ['classic_csi', 'classic_ls', 'tied@5', 'tied@10', 'affine@10', 'mlp@10']
ITERS = {'tied@5': 5, 'tied@10': 10, 'affine@10': 10, 'mlp@10': 10}

cfg = Config()
new = not os.path.exists(OUT)
with open(OUT, 'a', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    if new:
        w.writerow(['receiver', 'params', 'es_n0_db', 'ser', 'symbol_errors', 'symbols', 'M', 'L', 'rho',
                    'pilot_words', 'pilot_iters', 'online_iters', 'adapt', 'frames'])
    for snr in SNRS:
        t0 = time.time()
        stats, npar = run_snr(cfg, snr, FRAMES, RECEIVERS, ITERS, seed=snr)
        for r, (e, n) in stats.items():
            w.writerow([r, npar.get(r, {'classic_csi': 0, 'classic_ls': 2 * cfg.L + 1}[r] if r.startswith('classic') else ''),
                        snr, e / n, e, n, cfg.M, cfg.L, cfg.rho, cfg.pilot_words, cfg.pilot_iters,
                        ITERS.get(r, ''), cfg.adapt, FRAMES])
        f.flush()
        print(f'[{time.strftime("%H:%M:%S")}] Es/N0={snr:2d} dB: ' +
              '  '.join(f'{r}={e / n:.2e}' for r, (e, n) in stats.items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
print('qpsk sweep complete', flush=True)
