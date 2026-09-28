"""Fast-fading check at one SNR: which detectors survive taps that change
WITHIN a word?

Channel: the COST2100 taps times a real Rician factor (K = rician_k) whose
diffuse part is a Jakes process with normalized Doppler fd = f_D * T_symbol
(see Code/channel/channel_dataset.py). Every detector sees the same draws
(evaluation data is seeded per repetition), so differences are paired.

    python run_fast_fading.py <snr> <fd> <label=trainer_model:method[:online_iters][:key=value]> ...

e.g.
    python run_fast_fading.py 7 0.01 ClassicViterbi_genie=ClassicViterbi_genie:Statistical \
        ViterbiNet_on5=ViterbiNet:ModelBased:5

Rows go to Results/metrics/fast_fading.csv. Git is left to the caller.
"""
import csv, fcntl, os, sys

os.environ['MC_SWEEP_NO_GIT'] = '1'
import run_mc_sweep as sweep

CSV_PATH = os.path.join(sweep.RESULTS_DIR, 'metrics', 'fast_fading.csv')
FIELDS = ['model', 'doppler', 'rician_k', 'psp_step'] + sweep.FIELDNAMES[1:]
REPS = int(os.environ.get('FF_REPS', '50'))
RICIAN_K = float(os.environ.get('FF_RICIAN_K', '3'))
PSP_STEP = float(os.environ.get('FF_PSP_STEP', '0.01'))


def done_keys():
    if not os.path.isfile(CSV_PATH):
        return set()
    return {(r['model'], float(r['doppler']), float(r['rician_k']), int(r['snr']))
            for r in csv.DictReader(open(CSV_PATH))}


def main(snr, fd, specs):
    lock = open(CSV_PATH + '.lock', 'w')
    for spec in specs:
        label, rest = spec.split('=', 1)
        parts = rest.split(':')
        trainer_model, method = parts[0], parts[1]
        kwargs = {'doppler': fd, 'rician_k': RICIAN_K, 'psp_step': PSP_STEP}
        for extra in parts[2:]:
            if '=' in extra:  # any Trainer attribute, e.g. ls_forget=0.8
                k, v = extra.split('=')
                kwargs[k] = float(v)
            else:
                kwargs['self_supervised_iterations'] = int(extra)
        if (label, fd, RICIAN_K, snr) in done_keys():
            print(f'[skip] {label} fd={fd} K={RICIAN_K} snr={snr}', flush=True)
            continue
        # distinct name per (fd, K) for checkpoints and weights
        point = f'FF_{label}_fd{fd}_K{RICIAN_K:g}'
        print(f'\n[run] {point} snr={snr}', flush=True)
        row = sweep.run_point(point, method, snr, REPS, REPS * 2000, 1,
                              trainer_model_name=trainer_model, trainer_kwargs=kwargs)
        row.update(model=label, doppler=fd, rician_k=RICIAN_K,
                   psp_step=kwargs['psp_step'] if trainer_model == 'ClassicViterbi_PSP' else '')
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = not os.path.isfile(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerow(row)
        fcntl.flock(lock, fcntl.LOCK_UN)
        print(f'[done] {row}', flush=True)
    print('Worker complete.', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3:])
