"""SNR sweep for the 32-parameter affine ViterbiNet variant (VNet_affine).

Same Monte-Carlo protocol as ViterbiNet's row in run_mc_sweep.MODELS
(ModelBased, min 20 reps, max 100k bits, checkpoint every rep), so its rows in
Results/metrics/mc_sweep_validation.csv compare directly with ViterbiNet's and
ClassicViterbi's. Several workers can run at once, each on its own SNR list:

    python run_affine_sweep.py 0 2 4 6 ...    # worker A
    python run_affine_sweep.py 1 3 5 7 ...    # worker B

Git is left to the caller (MC_SWEEP_NO_GIT is forced on); resume works from the
same checkpoints as run_mc_sweep.
"""
import fcntl, os, sys

os.environ['MC_SWEEP_NO_GIT'] = '1'
import run_mc_sweep as sweep

MODEL = 'VNet_affine'
LOCK = sweep.CSV_PATH + '.lock'


def main(snrs):
    with open(LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        sweep.ensure_header()
        done = sweep.already_done()
        fcntl.flock(lk, fcntl.LOCK_UN)
    for snr in snrs:
        if (MODEL, snr) in done:
            print(f'[skip] {MODEL} snr={snr} already in CSV', flush=True)
            continue
        print(f'\n[run] {MODEL} snr={snr}', flush=True)
        row = sweep.run_point(MODEL, 'ModelBased', snr, 20, 100_000, 1)
        with open(LOCK, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            sweep.drop_existing_row(MODEL, snr)
            sweep.append_row(row)
            fcntl.flock(lk, fcntl.LOCK_UN)
        print(f'[done] {row}', flush=True)
    print('Worker complete.', flush=True)


if __name__ == '__main__':
    main([int(s) for s in sys.argv[1:]])
