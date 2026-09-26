"""SNR sweep for a trainer variant under its own CSV label.

Same Monte-Carlo protocol as ViterbiNet's row in run_mc_sweep.MODELS
(ModelBased, min 20 reps, max 100k bits, checkpoint every rep), so rows in
Results/metrics/mc_sweep_validation.csv compare directly with the others.

    python run_variant_sweep.py <label> <trainer_model> <online_iters|-> <snr> [<snr> ...]

e.g. ViterbiNet with 5 online adaptation iterations per word:
    python run_variant_sweep.py ViterbiNet_on5 ViterbiNet 5 7 0 2 4 ...
or classical Viterbi with LS-estimated taps (pilot + decision-directed):
    VARIANT_METHOD=Statistical VARIANT_MIN_REPS=100 VARIANT_MAX_BITS=2000000 \
    python run_variant_sweep.py ClassicViterbi_LS ClassicViterbi_LS - 7 0 2 ...
Several workers can run at once on disjoint SNR lists. Git is left to the
caller (MC_SWEEP_NO_GIT is forced on); resume uses run_mc_sweep's checkpoints.
"""
import fcntl, os, sys

os.environ['MC_SWEEP_NO_GIT'] = '1'
import run_mc_sweep as sweep

LOCK = sweep.CSV_PATH + '.lock'
# Protocol overrides (defaults = ViterbiNet's row). A classical/statistical variant uses e.g.
# VARIANT_METHOD=Statistical VARIANT_MIN_REPS=100 VARIANT_MAX_BITS=2000000.
METHOD = os.environ.get('VARIANT_METHOD', 'ModelBased')
MIN_REPS = int(os.environ.get('VARIANT_MIN_REPS', '20'))
MAX_BITS = int(os.environ.get('VARIANT_MAX_BITS', '100000'))


def main(label, trainer_model, online_iters, snrs):
    kwargs = {} if online_iters is None else {'self_supervised_iterations': online_iters}
    with open(LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        sweep.ensure_header()
        done = sweep.already_done()
        fcntl.flock(lk, fcntl.LOCK_UN)
    for snr in snrs:
        if (label, snr) in done:
            print(f'[skip] {label} snr={snr} already in CSV', flush=True)
            continue
        print(f'\n[run] {label} snr={snr}', flush=True)
        row = sweep.run_point(label, METHOD, snr, MIN_REPS, MAX_BITS, 1,
                              trainer_model_name=trainer_model, trainer_kwargs=kwargs)
        with open(LOCK, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            sweep.drop_existing_row(label, snr)
            sweep.append_row(row)
            fcntl.flock(lk, fcntl.LOCK_UN)
        print(f'[done] {row}', flush=True)
    print('Worker complete.', flush=True)


if __name__ == '__main__':
    label, trainer_model, iters = sys.argv[1], sys.argv[2], sys.argv[3]
    main(label, trainer_model, None if iters == '-' else int(iters), [int(s) for s in sys.argv[4:]])
