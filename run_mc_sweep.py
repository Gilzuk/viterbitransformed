"""
Higher-MC validation sweep (Colab/GPU configuration): ClassicViterbi and
Viterbi-Transformer over SNR 0-17, evaluating each with an SNR-adaptive
number of repetitions so every point actually observes a meaningful number
of errors, rather than a fixed rep count that silently floors to a
meaningless "0" once the true SER drops below what that many bits can
resolve.

Sizing method (per point):
  1. Predict the SER at this SNR using the closed-form BPSK/AWGN error
     probability Q(sqrt(2*snr_linear)). This is only a heuristic -- the
     real channel has ISI and fading on top of AWGN -- but it is a
     reasonable prior for how many bits are needed.
  2. Target ~100 expected errors (the standard rule of thumb for a stable
     Monte-Carlo BER/SER estimate): required_bits = 100 / predicted_ser.
  3. Convert to repetitions via the trainer's actual words/rep and bits/word,
     clipped to a per-model [MIN_REPS, MAX_REPS] so runtime stays bounded
     even where the AWGN prediction badly underestimates the requirement.
  4. Run that many reps. If literally zero errors were observed, keep
     extending in fixed increments -- "run until first error" -- up to a
     separate, higher EXTEND_MAX_REPS safety cap.
  5. If the safety cap is hit with still zero errors, the point is recorded
     as CENSORED: ser_mean is reported as 0.0, but `censored=1` and
     `bits_run` are also recorded, so downstream analysis can report it
     honestly as an upper bound (< 1/bits_run) instead of a converged zero.

On CPU, ClassicViterbi is ~6s/rep (no training, no online adaptation), while
the Transformer's self-supervised online training fires on almost every word
during each eval rep, making each Transformer rep ~3.5 min there -- see
COLAB_MC_SWEEP.md for why this copy of the script targets a GPU runtime
instead: both models use higher rep caps here than the CPU run, since GPU
should make them tractable.

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

CSV_PATH = os.path.join(RESULTS_DIR, 'metrics', 'mc_sweep_validation_colab.csv')
FIELDNAMES = ['model', 'snr', 'ser_mean', 'ser_std', 'ser_ci95', 'n_reps',
              'words_run', 'bits_run', 'errors_observed', 'censored',
              'model_size', 'run_time_sec']
SNR_VALUES = list(range(0, 18))
TARGET_ERRORS = 100

# (model_name, detector_method, min_reps, max_reps, extend_max_reps, step)
#   min_reps        -- floor
#   max_reps         -- cap on the AWGN-formula-driven primary run
#   extend_max_reps  -- higher safety cap for the "run until first error"
#                       fallback when the primary run sees zero errors
#   step             -- rep increment used while extending
# GPU run: higher caps than the CPU run (see COLAB_MC_SWEEP.md), since a
# real GPU should make the Transformer's per-word online-training updates
# cheap enough to afford them.
MODELS = [
    ('ClassicViterbi', 'Statistical', 100, 1000, 3000, 200),
    ('Transformer', 'ModelBased', 20, 100, 150, 20),
]
BRANCH = 'mc-sweep-colab-gpu'


def expected_ser_awgn(snr_db):
    """Closed-form BPSK/AWGN bit-error probability Q(sqrt(2*snr_linear)).

    A sizing heuristic only -- this channel has ISI/fading on top of AWGN,
    so actual SER differs. Used solely to decide how many bits to run.
    """
    snr_linear = 10 ** (snr_db / 10)
    x = math.sqrt(2 * snr_linear)
    return 0.5 * math.erfc(x / math.sqrt(2))


def already_done():
    done = set()
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                # A row measuring exactly zero errors is the floor artifact
                # this adaptive scheme fixes -- treat it as not-done so it
                # gets re-run, instead of trusting it as converged. (Also
                # covers rows written by an older, non-adaptive version of
                # this script, which lack the newer columns entirely.)
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


def run_point(model_name, detector_method, snr, min_reps, max_reps, extend_max_reps, step):
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

    predicted_ser = max(expected_ser_awgn(snr), 1e-300)
    required_bits = TARGET_ERRORS / predicted_ser
    required_reps = math.ceil(required_bits / (words_per_rep * bits_per_word))
    planned_reps = int(min(max(required_reps, min_reps), max_reps))

    print(f'[plan] {model_name} snr={snr}: predicted_ser={predicted_ser:.3e}, '
          f'planned_reps={planned_reps} (min={min_reps}, max={max_reps})', flush=True)

    ser_batches = []
    reps_done = 0
    while reps_done < planned_reps:
        batch = min(step, planned_reps - reps_done)
        ser_batches.append(np.asarray(trainer.online_evaluation(num_of_rep=batch)).reshape(-1))
        reps_done += batch

    def total_errors(batches):
        return sum(float(b.sum()) * bits_per_word for b in batches)

    censored = False
    # Safety net: if the primary (formula-sized) run saw literally zero
    # errors, keep extending until the first error appears or we hit the
    # higher extend cap.
    while total_errors(ser_batches) == 0 and reps_done < extend_max_reps:
        batch = min(step, extend_max_reps - reps_done)
        print(f'[extend] {model_name} snr={snr}: 0 errors after {reps_done} reps, '
              f'running {batch} more (cap {extend_max_reps})', flush=True)
        ser_batches.append(np.asarray(trainer.online_evaluation(num_of_rep=batch)).reshape(-1))
        reps_done += batch

    if total_errors(ser_batches) == 0:
        censored = True
        print(f'[censored] {model_name} snr={snr}: 0 errors in {reps_done} reps '
              f'({reps_done * words_per_rep * bits_per_word} bits) -- reporting as upper bound',
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

    for model_name, detector_method, min_reps, max_reps, extend_max_reps, step in MODELS:
        for snr in SNR_VALUES:
            key = (model_name, snr)
            if key in done:
                print(f'[skip] {model_name} snr={snr} already in CSV', flush=True)
                continue

            print(f'\n{"="*70}\n[run] {model_name} snr={snr}\n{"="*70}', flush=True)
            try:
                row = run_point(model_name, detector_method, snr,
                                 min_reps, max_reps, extend_max_reps, step)
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
