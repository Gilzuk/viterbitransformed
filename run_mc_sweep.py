"""
Higher-MC validation sweep: ClassicViterbi and Viterbi-Transformer over SNR
0-17, evaluating each with an SNR-adaptive number of repetitions so every
point actually observes a meaningful number of errors, rather than a fixed
rep count that silently floors to a meaningless "0" once the true SER drops
below what that many bits can resolve.

Sizing method (per point):
  1. Predict the SER at this SNR with Q(sqrt(2*snr_eff_linear)), where
     snr_eff = snr - ISI_PENALTY_DB. The ideal single-tap AWGN expression
     (penalty 0) is badly miscalibrated for this 4-tap ISI channel --
     under-predicting by 2.2x at 0 dB and 258x at 10 dB against our own
     measurements -- but the gap is a near-constant effective-SNR shift,
     so applying that shift makes it a usable prior. See ISI_PENALTY_DB.
  2. Target ~100 expected errors (the standard rule of thumb for a stable
     Monte-Carlo BER/SER estimate): required_bits = 100 / predicted_ser.
  3. Convert to repetitions via the trainer's actual words/rep and bits/word,
     clipped to a per-model [MIN_REPS, MAX_REPS]. This only sizes the opening
     move -- the observed error count, not the prediction, decides when to
     stop.
  4. Keep running until TARGET_ERRORS errors have actually been OBSERVED:
       - zero errors so far: nothing to estimate a rate from, so double the
         bits and look again ("run until the first error").
       - errors seen at N bits: SER ~ errors/N, so ~TARGET_ERRORS errors
         needs TARGET_ERRORS/SER bits. At exactly one error that is the
         100x-the-bits-to-first-error rule; more errors sharpen the estimate.
     Bounded by the per-model max_bits so a point cannot run away.
  5. If max_bits is spent with still zero errors, the point is recorded as
     CENSORED: ser_mean is 0.0, but `censored=1` and `bits_run` are recorded
     so it reads as an honest upper bound rather than a converged zero. (For
     zero errors in N bits the correct 95% bound is the rule of three, 3/N.)
     If it is spent with some errors but fewer than TARGET_ERRORS, the point
     is kept and flagged `[thin]` in the log -- its CI is real but wide.

Commits and pushes Results/metrics/mc_sweep_validation.csv after every
completed (model, snr) point, so a container reset loses at most one point.
Any existing row with ser_mean==0.0 is treated as not-done and re-run under
this adaptive scheme (that is exactly the floor artifact this rewrite
fixes) -- everything else already in the CSV is left alone.

Run standalone: python run_mc_sweep.py
"""
import csv
import math
import os
import subprocess
import time

import numpy as np
import torch

from Code.dir_definitions import RESULTS_DIR, WEIGHTS_DIR
from Code.trainer import Trainer

CSV_PATH = os.path.join(RESULTS_DIR, 'metrics', 'mc_sweep_validation.csv')
FIELDNAMES = ['model', 'snr', 'ser_mean', 'ser_std', 'ser_ci95', 'n_reps',
              'words_run', 'bits_run', 'errors_observed', 'censored',
              'model_size', 'run_time_sec']
SNR_VALUES = list(range(0, 18))
TARGET_ERRORS = 100

# (model_name, detector_method, min_reps, max_reps, max_bits, step)
#   min_reps   -- floor
#   max_reps   -- cap on the predictor-sized opening run
#   max_bits   -- total bit budget for the point. The run keeps going past
#                 max_reps until TARGET_ERRORS errors are actually observed;
#                 this is what stops it running away when they never are.
#   step       -- minimum rep increment while extending
#
# Sizing max_bits, from MEASURED throughput on this box (2000 bits/rep):
#   ClassicViterbi  1.15 s/rep = ~1760 bits/s  -> 2e7 bits = 3.2 h/point
#   Transformer     ~210 s/rep = ~9.5 bits/s   -> 1e5 bits = 2.9 h/point
# (ClassicViterbi was 6.5 s/rep before the COST2100 tap-load cache in
# 0ad20cd; that is a 5.7x end-to-end speedup, not the 224x that applies to
# estimate_channel alone.)
#
# What this does and does not buy: ~100 errors needs ~100/SER bits, so
# SNR<=13 (SER >= 5.5e-6) now reaches a full 100 errors. The error floor at
# SNR>=14 (SER < 5e-7) would need ~2e8 bits = 31.5 h for ONE point, so those
# stay censored -- but at 2e7 bits their upper bound tightens 10x, to
# ~1.5e-7. Brute force cannot reach the floor here; that needs importance
# sampling, or a much faster detector implementation.
MODELS = [
    ('ClassicViterbi', 'Statistical', 100, 500, 20_000_000, 100),
    ('Transformer', 'ModelBased', 20, 30, 100_000, 5),
]
BRANCH = 'claude/transformer-sionna-mlp-comparison-wc67zp'


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
    (used for the zero-SER rows this rewrite is re-running)."""
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


def repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def commit_and_push(model, snr):
    subprocess.run(['git', 'add', 'Results/metrics/mc_sweep_validation.csv'],
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

    # A point that is committed locally but never reaches the remote is a
    # point that can still be lost (container reclaim, disconnect, etc).
    # Retry with capped backoff for several minutes; if it still can't push,
    # stop the whole sweep loudly instead of silently moving on to the next
    # point and leaving this one stranded local-only.
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
        f'stopping the sweep so this point is not silently lost (it is committed '
        f'locally but not on the remote yet).')


def run_point(model_name, detector_method, snr, min_reps, max_reps, max_bits, step):
    method_name = f'{model_name}_{detector_method}'
    weights_dir = os.path.join(
        WEIGHTS_DIR,
        f'{method_name}_training_120_2_channel1_cost2100_mcsweep')

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

    predicted_ser = max(expected_ser_isi(snr), 1e-300)
    required_bits = TARGET_ERRORS / predicted_ser
    required_reps = math.ceil(required_bits / (words_per_rep * bits_per_word))
    planned_reps = int(min(max(required_reps, min_reps), max_reps))

    print(f'[plan] {model_name} snr={snr}: predicted_ser={predicted_ser:.3e}, '
          f'planned_reps={planned_reps} (min={min_reps}, max={max_reps})', flush=True)

    bits_per_rep = words_per_rep * bits_per_word

    ser_batches = []
    reps_done = 0

    def total_errors(batches):
        return sum(float(b.sum()) * bits_per_word for b in batches)

    def run_batch(n):
        nonlocal reps_done
        ser_batches.append(np.asarray(trainer.online_evaluation(num_of_rep=n)).reshape(-1))
        reps_done += n

    while reps_done < planned_reps:
        run_batch(min(step, planned_reps - reps_done))

    # Keep going until TARGET_ERRORS errors have actually been observed, not
    # merely until the predictor's bit budget is spent. The predictor only
    # sizes the opening move; the observed error rate is the real authority.
    #
    #   - zero errors so far: nothing to estimate a rate from, so double the
    #     bits and look again (this is the "run until the first error" phase).
    #   - at least one error at N bits: the rate implies SER ~ errors/N, so
    #     ~TARGET_ERRORS errors needs TARGET_ERRORS/SER bits. With a single
    #     error that is exactly the 100x-the-bits-to-first-error rule; with
    #     more errors the estimate simply gets better.
    #
    # Bounded by max_bits so a point cannot run away; a point that exhausts it
    # without any error is recorded as censored (an honest upper bound).
    while total_errors(ser_batches) < TARGET_ERRORS and reps_done * bits_per_rep < max_bits:
        errs = total_errors(ser_batches)
        bits_so_far = reps_done * bits_per_rep
        if errs == 0:
            target_reps = reps_done * 2
            why = f'0 errors in {bits_so_far:,} bits -- doubling'
        else:
            observed_ser = errs / bits_so_far
            target_reps = math.ceil((TARGET_ERRORS / observed_ser) / bits_per_rep)
            why = (f'{errs:.0f} errors in {bits_so_far:,} bits '
                   f'(ser~{observed_ser:.2e}) -- need ~{target_reps} reps for {TARGET_ERRORS}')
        # cap the target by the bit budget, and always make forward progress
        target_reps = min(target_reps, max_bits // bits_per_rep)
        batch = max(step, min(target_reps - reps_done, reps_done))  # grow at most 2x per round
        if batch <= 0:
            break
        print(f'[extend] {model_name} snr={snr}: {why}; running {batch} more', flush=True)
        run_batch(batch)

    censored = total_errors(ser_batches) == 0
    if censored:
        print(f'[censored] {model_name} snr={snr}: 0 errors in {reps_done} reps '
              f'({reps_done * bits_per_rep:,} bits) -- reporting as upper bound',
              flush=True)
    elif total_errors(ser_batches) < TARGET_ERRORS:
        print(f'[thin] {model_name} snr={snr}: only {total_errors(ser_batches):.0f} errors in '
              f'{reps_done * bits_per_rep:,} bits (max_bits budget spent) -- CI will be wide',
              flush=True)

    run_time = time.time() - t0

    ser_all = np.concatenate(ser_batches)
    per_rep_means = ser_all.reshape(reps_done, words_per_rep).mean(axis=1)

    ser_mean = float(per_rep_means.mean())
    if reps_done > 1:
        ser_std = float(per_rep_means.std(ddof=1))
        ser_ci95 = 1.96 * ser_std / math.sqrt(reps_done)
    else:
        ser_std = 0.0
        ser_ci95 = 0.0

    words_run = reps_done * words_per_rep
    bits_run = words_run * bits_per_word
    errors_observed = total_errors(ser_batches)

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
            done.add(key)

    print('\nAll points complete.', flush=True)


if __name__ == '__main__':
    main()
