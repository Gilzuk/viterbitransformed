"""QPSK receivers on a fast-fading channel (Code/qpsk_sim.py with doppler > 0):
every tap is a complex Rayleigh (Jakes) process that varies sample by sample,
continuously across words, instead of the word-to-word AR(1) drift of
run_qpsk_sweep.py. One Es/N0, a grid of normalized Doppler f_D*T_symbol.
All receivers see the same channel and data (run_snr reseeds after offline
training when doppler > 0). Writes Results/metrics/qpsk_fast_fading.csv.

    python run_qpsk_fast_fading.py [es_n0_db]
"""
import csv, fcntl, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Code.qpsk_sim import Config, run_snr

OUT = 'Results/metrics/qpsk_fast_fading.csv'
SNR = int(sys.argv[1]) if len(sys.argv) > 1 else 16
DOPPLERS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
FRAMES = 64
GROUPS = [  # (config overrides, receivers, online iters) -- same settings as the static sweeps
    (dict(dd_lr=0.01), ['classic_csi', 'classic_ls', 'tied@5', 'tied_mlp@5'], {'tied@5': 5, 'tied_mlp@5': 5}),
    (dict(dd_lr=0.03, sym_window=9), ['le_ls', 'sym_eq@30'], {'sym_eq@30': 30}),
]
done = set()
if os.path.exists(OUT):
    done = {(r['receiver'], float(r['doppler']), int(r['es_n0_db'])) for r in csv.DictReader(open(OUT))}
with open(OUT, 'a', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    # several SNRs may run in parallel: write the header under a lock, and only
    # into an empty file (an unlocked exists() check put it on line 7 once)
    fcntl.flock(f, fcntl.LOCK_EX)
    if f.tell() == 0 and os.path.getsize(OUT) == 0:
        w.writerow(['receiver', 'params', 'doppler', 'es_n0_db', 'ser', 'symbol_errors', 'symbols', 'frames',
                    'words', 'pilot_words', 'online_iters', 'dd_lr', 'sym_window'])
        f.flush()
    fcntl.flock(f, fcntl.LOCK_UN)
    for fd in DOPPLERS:
        for kw, recv, iters in GROUPS:
            if all((r, fd, SNR) in done for r in recv):
                continue
            cfg = Config(doppler=fd, **kw)
            t0 = time.time()
            stats, npar = run_snr(cfg, SNR, FRAMES, recv, iters, seed=SNR)
            for r, (e, n) in stats.items():
                w.writerow([r, npar.get(r, 0), fd, SNR, e / n, e, n, FRAMES, cfg.words, cfg.pilot_words,
                            iters.get(r, ''), cfg.dd_lr, cfg.W])
            f.flush()
            print(f'[{time.strftime("%H:%M:%S")}] fd={fd:<7g} Es/N0={SNR} dB: ' +
                  '  '.join(f'{r}={e / n:.2e}' for r, (e, n) in stats.items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
print('qpsk fast-fading run complete', flush=True)
