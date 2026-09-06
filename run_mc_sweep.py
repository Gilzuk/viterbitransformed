"""
Higher-MC validation sweep (Colab/GPU configuration): ClassicViterbi and
Viterbi-Transformer over SNR 0-17, evaluating each with an SNR-adaptive
number of repetitions so every point actually observes a meaningful number
of errors, rather than a fixed rep count that silently floors to a
meaningless "0" once the true SER drops below what that many bits can
resolve.

Sizing method (per point):
  1. Predict the SER at this SNR with Q(sqrt(2*snr_eff_linear)), where
     snr_eff = snr - ISI_PENALTY_DB. The ideal single-tap AWGN expression
     (penalty 0) is badly miscalibrated for this 4-tap ISI channel --
     under-predicting by 2.2x at 0 dB and 258x at 10 dB against our own
     measurements -- but the gap is a near-constant effective-SNR shift,
     so applying that shift makes it a usable prior. See ISI_PENALTY_DB.
  2. Target ~100 expected errors (the standard rule of thumb for a stable
     Monte-Carlo BER/SER estimate) purely to size the OPENING run:
     required_bits = 100 / predicted_ser. This is only a prior -- the
     observed error count, not the prediction, decides what happens next.
  3. Convert to repetitions via the trainer's actual words/rep and bits/word,
     clipped to a per-model [MIN_REPS, MAX_REPS].
  4. If the opening run sees zero errors, double the bits and look again
     ("run until the first error") -- there is nothing to estimate a rate
     from yet.
  5. Once the first error is observed, at bits_to_first_error, run to a
     FIXED cap of FIRST_ERROR_BITS_MULTIPLIER (100) times that bit count and
     stop -- e.g. first error at 1e6 bits means run to 1e8 bits, then stop,
     however many errors that ends up with. This is deliberately NOT
     re-estimated from the running SER as more errors come in: re-targeting
     off a noisy observed rate can keep chasing a moving goalpost and never
     converge, whereas the first-error bit count is measured once and the
     cap it sets is fixed. Bounded by the per-model max_bits so a point
     cannot run away.
  6. If max_bits is spent with still zero errors, the point is recorded as
     CENSORED: ser_mean is 0.0, but `censored=1` and `bits_run` are recorded
     so it reads as an honest upper bound rather than a converged zero. (For
     zero errors in N bits the correct 95% bound is the rule of three, 3/N.)
     If the 100x-first-error cap is reached with very few errors (bursty
     luck), the point is kept and flagged `[thin]` in the log -- its CI is
     real but wide.

On CPU, ClassicViterbi was ~6.5s/rep before the COST2100 tap-load cache fix
(now ~0.02s/rep); the Transformer's self-supervised online training fires on
almost every word during each eval rep, making each Transformer rep on the
order of minutes there -- see COLAB_MC_SWEEP.md for why this copy of the
script targets a GPU runtime instead.

RESILIENCE (this matters a lot more here than on a persistent machine --
a Colab runtime is ephemeral: on disconnect/restart, everything not already
pushed to GitHub is gone, full stop):
  - Every completed (model, snr) point is committed AND pushed to the
    mc-sweep-colab-gpu branch before moving to the next point.
  - If push fails, it is retried with capped backoff for several minutes;
    if it still fails, the script stops immediately (raises) rather than
    silently continuing and leaving that point committed-but-unpushed
    (which a Colab restart would then drop entirely).
  - On (re)start, already_done() reads the CSV already on disk (i.e. after
    re-cloning the branch, which carries every point pushed so far) and
    skips every point already present, so a rerun after a disconnect
    resumes from the last pushed point instead of starting over from SNR=0.
    Re-running this script (e.g. re-running the notebook cell after a
    reconnect) is therefore always safe and always cheap for already-done
    points.
  - Within a point, an extend round can be thousands of reps in a single
    trainer call that does not return for hours. run_point() also
    checkpoints per-rep SER means to disk after every `step`-sized chunk
    and resumes from that checkpoint if this process restarts mid-point,
    so a restart loses at most one chunk instead of the whole point. This
    checkpoint is local-only (not committed to git, see .gitignore) --
    it does not survive a full Colab disconnect, only a restart of this
    process on the same runtime.

Run standalone: python run_mc_sweep.py
"""
import csv
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from Code.dir_definitions import RESULTS_DIR, WEIGHTS_DIR
from Code.trainer import Trainer

CSV_PATH = os.path.join(RESULTS_DIR, 'metrics', 'mc_sweep_validation_colab.csv')
FIELDNAMES = ['model', 'snr', 'ser_mean', 'ser_std', 'ser_ci95', 'n_reps',
              'words_run', 'bits_run', 'errors_observed', 'censored',
              'model_size', 'run_time_sec']
SNR_VALUES = list(range(0, 18))
# Used only to size the opening run (see docstring step 2) -- NOT the
# stopping condition. The actual stopping rule is FIRST_ERROR_BITS_MULTIPLIER.
INITIAL_SIZING_TARGET_ERRORS = 100
# Once the first error is observed at N bits, run to a fixed cap of this many
# times N and stop, rather than re-targeting an error count from the
# (noisy) observed rate as more errors accumulate. See docstring step 5.
FIRST_ERROR_BITS_MULTIPLIER = 100
# Below this many observed errors at the cap, flag the point [thin] -- the
# result is still a valid unbiased estimate, just with a wide CI.
THIN_ERROR_THRESHOLD = 10

# (model_name, detector_method, min_reps, max_reps, max_bits, step)
#   min_reps   -- floor
#   max_reps   -- cap on the predictor-sized opening run
#   max_bits   -- total bit budget for the point. The run keeps going past
#                 max_reps until the FIRST_ERROR_BITS_MULTIPLIER cap is
#                 reached (see docstring); this is what stops it running
#                 away when the first error never comes.
#   step       -- minimum rep increment while extending
#
# ClassicViterbi's max_bits (20M) is copied from the CPU branch's
# measured-throughput calibration: the COST2100 tap-load cache fix is CPU
# logic, not GPU-dependent, so the same throughput should transfer here.
#
# ViterbiNet and Transformer max_bits (2M) are an UNCALIBRATED placeholder --
# no run has completed on this branch yet, so GPU throughput for their
# per-word online-training backprop is unknown. Check the first [done] log
# lines' run_time_sec once this actually runs, recompute bits/sec, and raise
# or lower max_bits to target a similar few-hours-per-point budget as
# ClassicViterbi -- do not leave this unexamined after the first real timing
# comes back.
#
# ViterbiNet is included because the cache bug invalidated its old n=84
# baseline (Results/metrics/model_performance_final_mc_83.csv) too -- the
# paper's three-way comparison needs all three detectors measured under the fix.
MODELS = [
    ('ClassicViterbi', 'Statistical', 100, 500, 20_000_000, 100),
    ('ViterbiNet', 'ModelBased', 20, 30, 2_000_000, 5),
    ('Transformer', 'ModelBased', 20, 30, 2_000_000, 5),
]
BRANCH = 'mc-sweep-colab-gpu'


# Effective-SNR penalty of the ISI channel relative to ideal single-tap AWGN.
#
# The plain AWGN expression Q(sqrt(2*snr_linear)) is badly miscalibrated here
# because the channel has 4-tap ISI (mean COST2100 taps [0.944, 0.430, 0.172,
# 0.079]) and the measured quantity is post-FEC SER, not raw channel BER:
# against our own ClassicViterbi (perfect-CSI) measurements it under-predicts
# by 2.2x at 0 dB, growing to 258x at 10 dB.
#
# What it under-predicts by is, however, an almost constant shift in effective
# SNR. Solving Q(sqrt(2*snr_eff)) = measured for each of SNR 0..10 dB gives an
# implied penalty of +3.66, +3.83, +3.96, +4.06, +4.02, +3.90, +3.77, +3.67,
# +3.55, +3.30, +3.21 dB -- i.e. the ISI channel costs a stable ~3.2-4.1 dB.
# (Note the pure MLSE minimum-distance bound does NOT explain this: for these
# taps d_min^2 = 4*||h||^2 = 4.61 > 4, so minimum-distance theory predicts
# slightly *better* than AWGN, while the real detector does worse. The loss is
# dominated by fading across the trace and by post-FEC error behaviour, which
# is why this is calibrated empirically rather than derived.)
#
# 3.0 dB is deliberately at the conservative end of the measured range: a
# smaller penalty predicts a lower SER, which asks for MORE bits, which is the
# safe direction for a sizing heuristic (over-running wastes time; under-running
# silently produces an under-powered estimate).
ISI_PENALTY_DB = 3.0


def expected_ser_isi(snr_db):
    """Predicted SER for this ISI channel: Q(sqrt(2 * snr_eff_linear)).

    snr_eff = snr - ISI_PENALTY_DB, i.e. the ideal single-tap AWGN expression
    evaluated at the effective SNR the ISI channel actually delivers. A sizing
    heuristic only -- used solely to decide how many bits a point should run.
    """
    snr_eff_db = snr_db - ISI_PENALTY_DB
    snr_linear = 10 ** (snr_eff_db / 10)
    x = math.sqrt(2 * snr_linear)
    return 0.5 * math.erfc(x / math.sqrt(2))


def already_done():
    done = set()
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                # censored=1 means this point ran all the way to its extend
                # cap and honestly observed zero errors -- a legitimate upper
                # bound, already as good as it is going to get. It must count
                # as DONE, or every restart re-runs it forever and the sweep
                # can never advance past the first censored point.
                if row.get('censored') == '1':
                    done.add((row['model'], int(row['snr'])))
                    continue

                # An un-censored zero, by contrast, can only come from the old
                # fixed-rep methodology (post-fix, zero errors always sets
                # censored=1) -- that is the floor artifact this rewrite
                # replaces, so re-run it rather than trusting it.
                try:
                    is_zero = float(row['ser_mean']) == 0.0
                except (KeyError, ValueError):
                    is_zero = True
                if is_zero:
                    continue
                done.add((row['model'], int(row['snr'])))
    return done


def ensure_header():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def drop_existing_row(model, snr):
    """Remove any prior row for (model, snr) before appending its replacement
    (used for the zero-SER rows this scheme re-runs)."""
    if not os.path.isfile(CSV_PATH):
        return
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader
                if not (r['model'] == model and int(r['snr']) == snr)]
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def append_row(row):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


# Mid-point checkpoint: a restart during a multi-thousand-rep extend round
# (which can be a single trainer.online_evaluation() call that does not
# return until it finishes) previously lost all of it -- on the CPU branch
# this happened twice on SNR=13, each costing several hours of real compute,
# and it matters even more here since a Colab runtime disconnect wipes local
# disk entirely. Checkpointing only the per-rep SER means (not the raw
# per-word arrays) is enough to reconstruct ser_mean/ser_std/ci95/
# errors_observed exactly on resume, and keeps each checkpoint write small.
# This directory is NOT pushed to git (see .gitignore) -- it is local
# resilience against a mid-point restart, superseded by the committed CSV
# row once a point finishes; a full Colab disconnect (not just a restart of
# this process) still loses whatever chunk was in flight, same as before.
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'metrics', '.mc_sweep_checkpoints')


def checkpoint_path(model, snr):
    return os.path.join(CHECKPOINT_DIR, f'{model}_snr{snr}.json')


def load_checkpoint(model, snr):
    path = checkpoint_path(model, snr)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_checkpoint(model, snr, per_rep_means):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(model, snr)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'per_rep_means': per_rep_means}, f)
    os.replace(tmp, path)  # atomic: a restart mid-write never leaves a corrupt checkpoint


def clear_checkpoint(model, snr):
    path = checkpoint_path(model, snr)
    if os.path.isfile(path):
        os.remove(path)


def repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def commit_and_push(model, snr):
    subprocess.run(['git', 'add', 'Results/metrics/mc_sweep_validation_colab.csv'],
                    check=True, cwd=repo_dir())
    commit = subprocess.run(
        ['git', 'commit', '-q', '-m', f'Add MC-sweep validation point: {model} snr={snr}'],
        cwd=repo_dir(), capture_output=True, text=True)
    if commit.returncode != 0:
        # The only expected/benign failure is "nothing to commit" (append_row
        # already wrote a fresh row before this is called, so that should
        # never actually happen -- but check for it specifically rather than
        # swallowing every commit failure, since a real failure here (e.g.
        # git identity not configured: "Please tell me who you are") would
        # otherwise be silently mislabeled as "nothing to commit" and the
        # point would never reach the remote.
        combined = (commit.stdout or '') + (commit.stderr or '')
        if 'nothing to commit' in combined.lower():
            print(f'[git] nothing to commit for {model} snr={snr}', flush=True)
            return
        raise RuntimeError(
            f'git commit failed for {model} snr={snr} (not a "nothing to commit" '
            f'case): {combined.strip()}')

    # On Colab specifically, a point that is committed locally but never
    # reaches the remote WILL be lost -- the runtime is ephemeral and a
    # disconnect wipes local disk entirely. Retry with capped backoff for
    # several minutes; if it still can't push, stop the whole sweep loudly
    # instead of silently moving on and stranding this point local-only.
    delays = (0, 2, 4, 8, 16, 30, 60, 60, 60, 60)
    for attempt, delay in enumerate(delays, 1):
        if delay:
            time.sleep(delay)
        r = subprocess.run(['git', 'push', '-q', 'origin', BRANCH], cwd=repo_dir())
        if r.returncode == 0:
            return
        print(f'[git] push attempt {attempt}/{len(delays)} failed for {model} snr={snr}',
              flush=True)

    raise RuntimeError(
        f'git push failed after {len(delays)} attempts for {model} snr={snr} -- '
        f'stopping the sweep so this point is not silently lost (committed '
        f'locally but not pushed -- a Colab disconnect would drop it entirely).')


def run_point(model_name, detector_method, snr, min_reps, max_reps, max_bits, step):
    method_name = f'{model_name}_{detector_method}'
    weights_dir = os.path.join(
        WEIGHTS_DIR,
        f'{method_name}_training_120_2_channel1_cost2100_mcsweep_colab')

    t0 = time.time()
    trainer = Trainer(
        model_name=model_name,
        detector_method=detector_method,
        curr_SNR=snr,
        val_block_length=120,
        train_block_length=120,
        pilots_num=25,
        weights_dir=weights_dir,
    )

    model_size = 0
    if hasattr(trainer.detector, 'model'):
        params = filter(lambda p: p.requires_grad, trainer.detector.model.parameters())
        model_size = sum(torch.numel(p) for p in params)

    # Train once (no-op for ClassicViterbi/Statistical).
    trainer.load_train_weights(run_over=2)

    # Total words drawn per online_evaluation repetition (matches
    # transmitted_words.shape[0] inside trainer.online_evaluation, i.e. all
    # words including pilots -- NOT trainer.data_indices, which is only the
    # non-pilot subset and would under-count the array size ser_by_word
    # actually comes back in).
    words_per_rep = trainer.val_frames * trainer.subframes_in_frame
    bits_per_word = trainer.n_symbols * 8
    bits_per_rep = words_per_rep * bits_per_word

    # Resume from a checkpoint left by a run that was interrupted mid-point
    # (a restart, not a clean finish -- a finished point is a committed CSV
    # row and has no checkpoint). per_rep_means is the only state needed to
    # reconstruct every final statistic exactly; see save_checkpoint().
    checkpoint = load_checkpoint(model_name, snr)
    per_rep_means = list(checkpoint['per_rep_means']) if checkpoint else []
    reps_done = len(per_rep_means)
    if reps_done:
        print(f'[resume] {model_name} snr={snr}: found checkpoint with {reps_done} reps '
              f'already done, continuing from there', flush=True)
    # The rep-index cache-key fix (see Code/trainer.py online_evaluation) keys
    # each draw by a per-trainer counter starting at 0. A fresh trainer after
    # a restart would replay reps_done..0's cache entries from the START,
    # which are exactly the ones the abandoned run already consumed --
    # pre-seed the counter so resumed reps get genuinely new indices instead.
    trainer._eval_rep_counter = reps_done

    def total_errors():
        return sum(m * bits_per_rep for m in per_rep_means)

    def bits_at_first_error():
        """Total bits run through the first rep at which the cumulative
        error count first became nonzero, or None if no error has been
        observed yet. Reconstructed from per_rep_means so it is correct
        across a checkpoint resume, not just within one process's run."""
        cum = 0.0
        for i, m in enumerate(per_rep_means):
            cum += m * bits_per_rep
            if cum > 0:
                return (i + 1) * bits_per_rep
        return None

    def run_batch(n):
        nonlocal reps_done
        # Chunk into at most `step` reps per trainer call and checkpoint
        # after every chunk. Without this, a single extend round can be
        # thousands of reps in one online_evaluation() call that does not
        # return for hours -- a restart mid-call loses all of it, and on a
        # Colab runtime that is even more likely than on a persistent box.
        remaining = n
        while remaining > 0:
            chunk = min(step, remaining)
            ser = np.asarray(trainer.online_evaluation(num_of_rep=chunk)).reshape(-1)
            chunk_means = ser.reshape(chunk, words_per_rep).mean(axis=1)
            per_rep_means.extend(float(m) for m in chunk_means)
            reps_done += chunk
            remaining -= chunk
            save_checkpoint(model_name, snr, per_rep_means)

    predicted_ser = max(expected_ser_isi(snr), 1e-300)
    required_bits = INITIAL_SIZING_TARGET_ERRORS / predicted_ser
    required_reps = math.ceil(required_bits / bits_per_rep)
    planned_reps = int(min(max(required_reps, min_reps), max_reps))

    print(f'[plan] {model_name} snr={snr}: predicted_ser={predicted_ser:.3e}, '
          f'planned_reps={planned_reps} (min={min_reps}, max={max_reps})', flush=True)

    if reps_done < planned_reps:
        run_batch(planned_reps - reps_done)

    # Phase 1: run until the first error is actually observed. The predictor
    # only sizes the opening move -- if it undershoots, there is nothing to
    # estimate a rate from yet, so double the bits and look again.
    while total_errors() == 0 and reps_done * bits_per_rep < max_bits:
        bits_so_far = reps_done * bits_per_rep
        target_reps = min(reps_done * 2, max_bits // bits_per_rep)
        batch = max(step, target_reps - reps_done)
        if batch <= 0:
            break
        print(f'[extend] {model_name} snr={snr}: 0 errors in {bits_so_far:,} bits -- '
              f'doubling toward first error; running {batch} more', flush=True)
        run_batch(batch)

    # Phase 2: once the first error is observed at N bits, run to a FIXED cap
    # of FIRST_ERROR_BITS_MULTIPLIER x N bits and stop -- not re-targeted off
    # the observed rate as more errors come in, since that can chase a moving
    # goalpost and never converge. Bounded by max_bits as always.
    first_bits = bits_at_first_error()
    if first_bits is not None:
        cap_bits = min(FIRST_ERROR_BITS_MULTIPLIER * first_bits, max_bits)
        cap_reps = int(cap_bits // bits_per_rep)
        if reps_done < cap_reps:
            print(f'[extend] {model_name} snr={snr}: first error at {first_bits:,} bits -- '
                  f'running to {FIRST_ERROR_BITS_MULTIPLIER}x cap = {cap_bits:,.0f} bits '
                  f'({cap_reps} reps)', flush=True)
            run_batch(cap_reps - reps_done)

    censored = total_errors() == 0
    if censored:
        print(f'[censored] {model_name} snr={snr}: 0 errors in {reps_done} reps '
              f'({reps_done * bits_per_rep:,} bits) -- reporting as upper bound',
              flush=True)
    elif total_errors() < THIN_ERROR_THRESHOLD:
        print(f'[thin] {model_name} snr={snr}: only {total_errors():.0f} errors in '
              f'{reps_done * bits_per_rep:,} bits (100x-first-error cap reached) -- '
              f'CI will be wide', flush=True)

    run_time = time.time() - t0

    per_rep_means_arr = np.array(per_rep_means)
    ser_mean = float(per_rep_means_arr.mean())
    if reps_done > 1:
        ser_std = float(per_rep_means_arr.std(ddof=1))
        ser_ci95 = 1.96 * ser_std / math.sqrt(reps_done)
    else:
        ser_std = 0.0
        ser_ci95 = 0.0

    words_run = reps_done * words_per_rep
    bits_run = words_run * bits_per_word
    errors_observed = total_errors()

    return {
        'model': model_name,
        'snr': snr,
        'ser_mean': ser_mean,
        'ser_std': ser_std,
        'ser_ci95': ser_ci95,
        'n_reps': reps_done,
        'words_run': words_run,
        'bits_run': bits_run,
        'errors_observed': errors_observed,
        'censored': int(censored),
        'model_size': int(model_size),
        'run_time_sec': run_time,
    }


def main():
    ensure_header()
    done = already_done()
    print(f'Already completed points: {sorted(done)}', flush=True)

    for model_name, detector_method, min_reps, max_reps, max_bits, step in MODELS:
        for snr in SNR_VALUES:
            key = (model_name, snr)
            if key in done:
                print(f'[skip] {model_name} snr={snr} already in CSV', flush=True)
                continue

            print(f'\n{"="*70}\n[run] {model_name} snr={snr}\n{"="*70}', flush=True)
            try:
                row = run_point(model_name, detector_method, snr,
                                 min_reps, max_reps, max_bits, step)
            except Exception as e:
                print(f'[ERROR] {model_name} snr={snr} failed: {e}', flush=True)
                import traceback
                traceback.print_exc()
                continue

            drop_existing_row(model_name, snr)
            append_row(row)
            print(f'[done] {row}', flush=True)
            commit_and_push(model_name, snr)
            clear_checkpoint(model_name, snr)
            done.add(key)

    print('\nAll points complete.', flush=True)


if __name__ == '__main__':
    main()
