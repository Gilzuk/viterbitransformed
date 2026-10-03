"""
Add more Monte Carlo evaluation repetitions to an ALREADY-FINISHED (model, snr)
point, reusing its trained weights -- no retraining. Useful for a point whose
CSV row was sized off very few observed errors (a wide confidence interval),
where more reps on the same model tighten the estimate.

Requires that point's eval checkpoint to still be on disk: the raw per-rep SER
values the finished row's stats were computed from. Any point finished after
the checkpoint-retention change keeps this automatically; a point finished
before that change has nothing to extend from and must be re-run in full with
run_single_point.py instead.

Writes/commits the result exactly as a normal sweep point would: same
run_point() call (with forced_reps overriding the usual predicted-SER sizing
formula), same CSV row replacement, same weights/checkpoint commit -- so it's
safe to interrupt/restart just like the normal sweep.

Usage: python extend_point.py <ModelName> <snr> <extra_reps>
"""
import sys

from run_mc_sweep import (
    MODELS, ensure_header, drop_existing_row, append_row, commit_and_push,
    run_point, load_checkpoint,
)


def main():
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python extend_point.py <ModelName> <snr> <extra_reps>')
    model_name, snr, extra_reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    match = [m for m in MODELS if m[0] == model_name]
    if not match:
        valid = ', '.join(m[0] for m in MODELS)
        raise SystemExit(f'Unknown model {model_name!r} -- choose one of: {valid}')
    _, detector_method, min_reps, max_bits, step = match[0]

    ensure_header()
    checkpoint = load_checkpoint(model_name, snr)
    if not checkpoint:
        raise SystemExit(
            f'No eval checkpoint on disk for {model_name} snr={snr} -- nothing to extend '
            f'from. (A point finished before the checkpoint-retention change has none; '
            f're-run it in full with run_single_point.py instead.)')
    current_reps = len(checkpoint['per_rep_means'])
    target_reps = current_reps + extra_reps
    print(f'[extend] {model_name} snr={snr}: {current_reps} reps on disk, '
          f'adding {extra_reps} more (target {target_reps})', flush=True)

    row = run_point(model_name, detector_method, snr, min_reps, max_bits, step,
                     forced_reps=target_reps)
    drop_existing_row(model_name, snr)
    append_row(row)
    print(f'[done] {row}', flush=True)
    commit_and_push(model_name, detector_method, snr)


if __name__ == '__main__':
    main()
