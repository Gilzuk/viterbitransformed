"""
Sweep ClassicViterbi across SNR with a fixed CSI uncertainty level: the
decoder's own channel estimate is perturbed by Gaussian noise whose std is
`uncertainty_pct` percent of that word's own channel energy (see
ClassicViterbi.compute_likelihood_priors in Code/models.py), while the
channel that actually transmits the word stays exact. This models a
decoder mismatched against a perfect channel, not a noisier physical one.

Results land under a distinct model label ('ClassicViterbi_csi<pct>'), via
run_point()'s trainer_model_name/trainer_kwargs hooks -- so this never
collides with the perfect-CSI ClassicViterbi row for the same SNR, in the
CSV or in checkpoint/weights files. Same CSV, same commit/push/resume
machinery as the main sweep, so it is safe to interrupt/restart and can run
alongside the main sweep in a sibling worktree without racing it.

Usage: python run_csi_sweep.py <uncertainty_pct> [start_snr] [end_snr]
  uncertainty_pct: 25, 50, 75, or 100 (percent)
  start_snr/end_snr: inclusive SNR range, default 0..17 (matches the
    perfect-CSI ClassicViterbi row in MODELS)
"""
import sys

from run_mc_sweep import (
    ensure_header, already_done, drop_existing_row, append_row, commit_and_push,
    run_point,
)

MIN_REPS, MAX_BITS, STEP = 100, 20_000_000, 1  # matches ClassicViterbi's own row in MODELS


def main():
    if len(sys.argv) not in (2, 4):
        raise SystemExit('Usage: python run_csi_sweep.py <uncertainty_pct> [start_snr] [end_snr]')
    pct = int(sys.argv[1])
    if pct not in (25, 50, 75, 100):
        raise SystemExit(f'uncertainty_pct must be 25, 50, 75, or 100, got {pct}')
    start_snr = int(sys.argv[2]) if len(sys.argv) == 4 else 0
    end_snr = int(sys.argv[3]) if len(sys.argv) == 4 else 17

    model_label = f'ClassicViterbi_csi{pct}'
    csi_uncertainty = pct / 100.0

    ensure_header()
    done = already_done()

    for snr in range(start_snr, end_snr + 1):
        if (model_label, snr) in done:
            print(f'[skip] {model_label} snr={snr} already in CSV', flush=True)
            continue
        print(f'[run] {model_label} snr={snr} (csi_uncertainty={csi_uncertainty})', flush=True)
        row = run_point(model_label, 'Statistical', snr, MIN_REPS, MAX_BITS, STEP,
                        trainer_model_name='ClassicViterbi',
                        trainer_kwargs={'csi_uncertainty': csi_uncertainty})
        drop_existing_row(model_label, snr)
        append_row(row)
        print(f'[done] {row}', flush=True)
        commit_and_push(model_label, 'Statistical', snr)


if __name__ == '__main__':
    main()
