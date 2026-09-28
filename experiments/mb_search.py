"""Training-budget search at snr=7 (20 reps / 40k bits per point, same protocol
as the earlier minibatch tests).

Per model:
  1. Ramp: 25, 50, 75, ... (+25) while each step lowers SER. Two budgets are
     run in parallel per round (n+25 and n+50) to halve wall-clock time.
  2. At the first step that does not improve, refine around the best budget b
     by binary search: try b-h and b+h for h = 12, 6, 3, recentring on the
     best point each round.
Results are appended to Results/metrics/minibatch_search_snr7.csv.
"""
import ast, csv, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, 'experiments', 'run_mb_test.py')
LOG_DIR = os.environ.get('EXP_LOG_DIR', '/tmp')
CSV_OUT = os.path.join(REPO, 'Results/metrics/minibatch_search_snr7.csv')
SNR = 7
STEP = 25
MAX_MB = 1000
REFINE_STEPS = (12, 6, 3)

results = {}  # (model, mb) -> row dict


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run_many(model, mbs):
    todo = sorted({mb for mb in mbs if 1 <= mb <= MAX_MB and (model, mb) not in results})
    procs = {}
    for mb in todo:
        path = os.path.join(LOG_DIR, f'search_{model}_mb{mb}.log')
        f = open(path, 'w')
        procs[mb] = (subprocess.Popen(['nice', '-n', '10', 'python3', '-u', RUNNER, model, str(mb), str(SNR)],
                                      stdout=f, stderr=subprocess.STDOUT, cwd=REPO), path)
        log(f'launched {model} mb={mb}')
    for mb, (p, path) in procs.items():
        p.wait()
        text = open(path).read()
        if '=== RESULT ===' not in text:
            raise RuntimeError(f'{model} mb={mb} failed; see {path}')
        row = ast.literal_eval(text.split('=== RESULT ===')[1].strip().splitlines()[0])
        results[(model, mb)] = row
        write_row(model, mb, row)
        log(f'{model} mb={mb}: SER={row["ser_mean"]:.5f} +/- {row["ser_ci95"]:.5f}')


def write_row(model, mb, row):
    new = not os.path.exists(CSV_OUT)
    with open(CSV_OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['model', 'train_minibatches', 'snr', 'ser_mean', 'ser_ci95', 'n_reps',
                        'bits_run', 'model_size', 'run_time_sec'])
        w.writerow([model, mb, SNR, row['ser_mean'], row['ser_ci95'], row['n_reps'],
                    row['bits_run'], row['model_size'], round(row['run_time_sec'], 1)])


def ser(model, mb):
    return results[(model, mb)]['ser_mean']


def search(model):
    log(f'===== {model}: ramp =====')
    run_many(model, [25, 50])
    best = 25
    if ser(model, 50) < ser(model, 25):
        best = 50
        while best + STEP <= MAX_MB:
            run_many(model, [best + STEP, best + 2 * STEP])
            nxt = best + STEP
            if ser(model, nxt) < ser(model, best):
                best = nxt
                nxt2 = best + STEP
                if (model, nxt2) in results and ser(model, nxt2) < ser(model, best):
                    best = nxt2
                    continue
            break
    log(f'{model}: ramp stopped, best so far mb={best} SER={ser(model, best):.5f}')

    log(f'===== {model}: binary-search refinement around mb={best} =====')
    for h in REFINE_STEPS:
        cands = [best - h, best + h]
        run_many(model, cands)
        for c in cands:
            if (model, c) in results and ser(model, c) < ser(model, best):
                best = c
        log(f'{model}: h={h} -> best mb={best} SER={ser(model, best):.5f}')

    log(f'***** {model}: sweet spot mb={best}, SER={ser(model, best):.5f} '
        f'+/- {results[(model, best)]["ser_ci95"]:.5f} *****')
    ranked = sorted((ser(m, mb), mb) for (m, mb) in results if m == model)
    for s, mb in sorted(ranked, key=lambda x: x[1]):
        log(f'   {model} mb={mb:4d}  SER={s:.5f}')
    return best


if __name__ == '__main__':
    for model in sys.argv[1:]:
        search(model)
    log('search complete')
