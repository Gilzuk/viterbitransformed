"""
Run exactly one (model, snr) point outside the normal per-SNR sweep order,
with an explicit max_bits override -- for a point that needs a different
bit budget than the rest of that model's MODELS row (e.g. a much higher
SNR, where the model-wide budget would cap it into a thin/censored result
before it observes enough errors).

Writes/commits the result exactly as main() in run_mc_sweep.py would for
that point -- same run_point() call, same CSV row, same weights commit --
so it participates in the sweep exactly like any other completed point,
and is safe to interrupt/restart just like the normal sweep (warm-start
and mid-training commits both still apply, since it's the same run_point).

Usage: python run_single_point.py <ModelName> <snr> <max_bits>
"""
import sys

from run_mc_sweep import (
    MODELS, ensure_header, drop_existing_row, append_row, commit_and_push,
    run_point,
)


def main():
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python run_single_point.py <ModelName> <snr> <max_bits>')
    model_name, snr, max_bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    match = [m for m in MODELS if m[0] == model_name]
    if not match:
        valid = ', '.join(m[0] for m in MODELS)
        raise SystemExit(f'Unknown model {model_name!r} -- choose one of: {valid}')
    _, detector_method, min_reps, _, step = match[0]

    ensure_header()
    print(f'[run] {model_name} snr={snr} (single point, max_bits={max_bits})', flush=True)
    row = run_point(model_name, detector_method, snr, min_reps, max_bits, step)
    drop_existing_row(model_name, snr)
    append_row(row)
    print(f'[done] {row}', flush=True)
    commit_and_push(model_name, detector_method, snr)  # also clears resume state


if __name__ == '__main__':
    main()
