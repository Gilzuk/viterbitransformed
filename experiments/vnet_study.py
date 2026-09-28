"""ViterbiNet MLP topology study at snr=7 (20 reps / 40k bits per point).

Stage 1  topology sweep: every hidden-size spec at the default budget
         (25 offline minibatches, 200 online iterations per word).
Stage 2  for the smallest topology within 5% of the ViterbiNet (100-58)
         baseline, plus the baseline itself:
           offline minibatches in {2, 5, 10} (online iters 200)
           online iterations  in {0, 25, 50, 100} (25 offline minibatches)
Stage 3  latency benchmark (separate script, vnet_latency.py).
Two runs at a time. Results -> Results/metrics/vnet_topology_study_snr7.csv
"""
import ast, csv, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, 'experiments', 'run_mb_test.py')
LOG_DIR = os.environ.get('EXP_LOG_DIR', '/tmp')
CSV_OUT = os.path.join(REPO, 'Results/metrics/vnet_topology_study_snr7.csv')
SNR = 7
TOPOLOGIES = ['affine', '4', '8', '16', '32', '64', '8-8', '16-16', '32-16', '100-58']
PARAMS = {'affine': 32, '4': 88, '8': 160, '16': 304, '32': 592, '64': 1168,
          '8-8': 232, '16-16': 576, '32-16': 864, '100-58': 7002}
PARALLEL = 2
results = {}  # (topo, mb, online_iters) -> row


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def write_row(topo, mb, on, row):
    new = not os.path.exists(CSV_OUT)
    with open(CSV_OUT, 'a', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        if new:
            w.writerow(['topology', 'params', 'train_minibatches', 'online_iters', 'snr', 'ser_mean',
                        'ser_ci95', 'n_reps', 'bits_run', 'run_time_sec'])
        w.writerow([topo, PARAMS[topo], mb, on, SNR, row['ser_mean'], row['ser_ci95'], row['n_reps'],
                    row['bits_run'], round(row['run_time_sec'], 1)])


def run_jobs(jobs):
    jobs = [j for j in jobs if j not in results]
    while jobs:
        batch, jobs = jobs[:PARALLEL], jobs[PARALLEL:]
        procs = []
        for topo, mb, on in batch:
            env = dict(os.environ, ONLINE_ITERS=str(on))
            path = os.path.join(LOG_DIR, f'vnet_{topo}_mb{mb}_on{on}.log')
            f = open(path, 'w')
            p = subprocess.Popen(['nice', '-n', '10', 'python3', '-u', RUNNER, f'VNet_{topo}', str(mb), str(SNR)],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env)
            procs.append(((topo, mb, on), p, path))
            log(f'launched VNet_{topo} mb={mb} online_iters={on}')
        for key, p, path in procs:
            p.wait()
            text = open(path).read()
            if '=== RESULT ===' not in text:
                log(f'FAILED {key}; see {path}')
                continue
            row = ast.literal_eval(text.split('=== RESULT ===')[1].strip().splitlines()[0])
            results[key] = row
            write_row(*key, row)
            log(f'VNet_{key[0]} ({PARAMS[key[0]]} params) mb={key[1]} on={key[2]}: '
                f'SER={row["ser_mean"]:.5f} +/- {row["ser_ci95"]:.5f}  ({row["run_time_sec"]:.0f}s)')


def ser(key):
    return results[key]['ser_mean']


if __name__ == '__main__':
    log('===== stage 1: topology sweep (mb=25, online 200) =====')
    run_jobs([(t, 25, 200) for t in TOPOLOGIES])
    base = ser(('100-58', 25, 200))
    good = [t for t in TOPOLOGIES if (t, 25, 200) in results and ser((t, 25, 200)) <= 1.05 * base]
    smallest = min(good, key=lambda t: PARAMS[t])
    log(f'baseline 100-58 SER={base:.5f}; within 5%: {good}; smallest: {smallest} ({PARAMS[smallest]} params)')

    log('===== stage 2: training budget (offline minibatches, online iterations) =====')
    for topo in dict.fromkeys([smallest, '100-58']):
        run_jobs([(topo, mb, 200) for mb in (2, 5, 10)] + [(topo, 25, on) for on in (0, 25, 50, 100)])

    log('===== summary =====')
    for key in sorted(results, key=lambda k: (PARAMS[k[0]], k[1], k[2])):
        log(f'   VNet_{key[0]:7s} {PARAMS[key[0]]:5d} params  mb={key[1]:3d}  online={key[2]:3d}  '
            f'SER={ser(key):.5f} +/- {results[key]["ser_ci95"]:.5f}')
    log('study complete')
