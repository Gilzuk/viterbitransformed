"""QPSK diagnostics behind the journal paper's Section on higher-order modulation
(Code/qpsk_sim.py). Three studies, each writing one CSV in Results/metrics/:

  trace   qpsk_diag_word_trace.csv -- SER by word position within the frame
          (drifting channel, one pilot): shows which receivers track the drift.
  pilot   qpsk_diag_pilot.csv      -- static channel: free per-branch vs tied
          metrics as a function of pilot words and pilot training steps.
  window  qpsk_diag_window.csv     -- linear equalizers vs window length
          (true-tap MMSE and LS re-solved on own decisions).

    python3 experiments/qpsk_diagnostics.py {trace|pilot|window|all}
"""
import csv, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Code.qpsk_sim import Config, run_snr

MET = os.path.join(ROOT, 'Results', 'metrics')


def write(name, header, rows):
    with open(os.path.join(MET, name), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(header); w.writerows(rows)
    print('wrote', name, len(rows), 'rows', flush=True)


def trace_study(frames=64):
    rows = []
    for snr in (12, 14):
        groups = [(Config(dd_lr=0.01), ['classic_csi', 'classic_ls', 'tied@5'], {'tied@5': 5}, 5),
                  (Config(dd_lr=0.01), ['sym_affine@10'], {'sym_affine@10': 10}, 5),
                  (Config(dd_lr=0.03, sym_window=9), ['sym_eq@30', 'le_ls'], {'sym_eq@30': 30}, 9)]
        for cfg, recv, iters, W in groups:
            tr = {}
            t0 = time.time()
            run_snr(cfg, snr, frames, recv, iters, seed=snr, trace=tr)
            for r, per_word in tr.items():
                for w, (e, n) in sorted(per_word.items()):
                    rows.append([r, snr, w, W, e / n, e, n, frames])
            print(f'trace snr={snr} {recv} ({time.time() - t0:.0f}s)', flush=True)
    write('qpsk_diag_word_trace.csv', ['receiver', 'es_n0_db', 'word', 'window', 'ser', 'symbol_errors', 'symbols', 'frames'], rows)


def pilot_study(frames=32, snr=14):
    rows = []
    recv, iters = ['classic_ls', 'tied@5', 'affine@10', 'mlp@10'], {'tied@5': 5, 'affine@10': 10, 'mlp@10': 10}
    for pw in (1, 2, 5, 10):
        for pi in (200, 2000):
            cfg = Config(dd_lr=0.01, rho=1.0, pilot_words=pw, pilot_iters=pi)
            t0 = time.time()
            st, npar = run_snr(cfg, snr, frames, recv, iters, seed=snr)
            for r, (e, n) in st.items():
                rows.append([r, npar.get(r, 7 if r == 'classic_ls' else 0), snr, pw, pi, e / n, e, n, frames])
            print(f'pilot pw={pw} iters={pi}: ' + ' '.join(f'{r}={e / n:.3g}' for r, (e, n) in st.items())
                  + f' ({time.time() - t0:.0f}s)', flush=True)
    write('qpsk_diag_pilot.csv', ['receiver', 'params', 'es_n0_db', 'pilot_words', 'pilot_iters', 'ser', 'symbol_errors',
                                  'symbols', 'frames'], rows)


def window_study(frames=64):
    rows = []
    for snr in (12, 14):
        for W in (3, 5, 7, 9, 11, 15):
            st, _ = run_snr(Config(dd_lr=0.03, sym_window=W), snr, frames, ['le_oracle', 'le_ls'], {}, seed=snr)
            for r, (e, n) in st.items():
                rows.append([r, snr, W, e / n, e, n, frames])
            print(f'window snr={snr} W={W}: ' + ' '.join(f'{r}={e / n:.3g}' for r, (e, n) in st.items()), flush=True)
    write('qpsk_diag_window.csv', ['receiver', 'es_n0_db', 'window', 'ser', 'symbol_errors', 'symbols', 'frames'], rows)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, fn in (('window', window_study), ('trace', trace_study), ('pilot', pilot_study)):
        if which in (name, 'all'):
            fn()
