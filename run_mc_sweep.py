"""
Higher-MC validation sweep: ClassicViterbi, ViterbiNet, and the
Viterbi-Transformer over SNR 0-17, evaluating each with an SNR-adaptive
number of repetitions so every
point actually observes a meaningful number of errors, rather than a fixed
rep count that silently floors to a meaningless "0" once the true SER drops
below what that many bits can resolve.

Sizing method (per point) -- calculated once, run once, no iteration:
  1. Predict the SER at this SNR with Q(sqrt(2*snr_eff_linear)), where
     snr_eff = snr - ISI_PENALTY_DB. The ideal single-tap AWGN expression
     (penalty 0) is badly miscalibrated for this 4-tap ISI channel --
     under-predicting by 2.2x at 0 dB and 258x at 10 dB against our own
     measurements -- but the gap is a near-constant effective-SNR shift,
     so applying that shift makes it a usable prior. See ISI_PENALTY_DB.
  2. Target ~100 expected errors (the standard rule of thumb for a stable
     Monte-Carlo BER/SER estimate): required_bits = TARGET_ERRORS /
     predicted_ser, capped at the per-model max_bits so a point cannot run
     away when the prediction is optimistic.
  3. Convert to repetitions via the trainer's actual words/rep and bits/word,
     floored at a per-model MIN_REPS. Run exactly that many reps, in
     checkpointed `step`-sized chunks for restart safety, and stop --
     whatever error count that ends up with is the result. There is
     deliberately no adaptive re-estimation or extending after this: a
     previous version doubled the bit budget over and over hunting for a
     first error at high SNR, which could run for days without ever
     stopping when the true SER was far below what the predictor assumed.
     Calculating once and running exactly that is fast and bounded; if the
     prediction was too optimistic, the point comes back censored or thin
     (see below) instead of running forever.
  4. If the run sees zero errors, the point is recorded as CENSORED:
     ser_mean is 0.0, but `censored=1` and `bits_run` are recorded so it
     reads as an honest upper bound rather than a converged zero. (For zero
     errors in N bits the correct 95% bound is the rule of three, 3/N.) If
     it finishes with some errors but fewer than THIN_ERROR_THRESHOLD, the
     point is kept and flagged `[thin]` in the log -- its CI is real but
     wide.

Commits and pushes Results/metrics/mc_sweep_validation.csv after every
completed (model, snr) point, so a container reset loses at most one point.
Any existing row with ser_mean==0.0 is treated as not-done and re-run under
this adaptive scheme (that is exactly the floor artifact this rewrite
fixes) -- everything else already in the CSV is left alone.

Run standalone: python run_mc_sweep.py
"""
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from Code.dir_definitions import RESULTS_DIR, WEIGHTS_DIR
from Code.trainer import Trainer

CSV_PATH = os.path.join(RESULTS_DIR, 'metrics', 'mc_sweep_validation.csv')
FIELDNAMES = ['model', 'snr', 'ser_mean', 'ser_std', 'ser_ci95', 'n_reps',
              'words_run', 'bits_run', 'errors_observed', 'censored',
              'model_size', 'run_time_sec', 'source']

# Label recorded on rows written before provenance was tracked. Not a guess at
# which machine produced them -- just an honest "not recorded".
PRE_TRACKING_SOURCE = 'unknown'
SNR_VALUES = list(range(0, 18))
# Target expected error count the sizing calculation aims for (see docstring
# step 2). This is the ONLY thing that decides how many bits a point runs --
# calculated once up front, not adjusted afterward based on what's observed.
TARGET_ERRORS = 100
# Below this many observed errors at the end of the run, flag the point
# [thin] -- the result is still a valid unbiased estimate, just wide CI.
THIN_ERROR_THRESHOLD = 10

# (model_name, detector_method, min_reps, max_bits, step)
#   min_reps   -- floor on the calculated rep count
#   step       -- repetitions per online_evaluation() call, i.e. how much work
#                 is in flight and unsaved at any moment. Nothing is written
#                 until the call returns, so this is a DATA-LOSS window, and it
#                 is counted in reps while the risk is measured in hours: at
#                 step=100 ClassicViterbi went ~10h between saves, and at
#                 step=5 Transformer ~4h, so a container reclaim could discard
#                 most of a day. It is 1 everywhere now -- one repetition is
#                 the smallest unit online_evaluation can return, so a restart
#                 loses at most the rep in progress. Chunking bought nothing
#                 anyway: each repetition already reloads its own data inside
#                 the loop, so the per-call overhead it avoided is negligible.
#   max_bits   -- ceiling on the calculated bit budget, so an optimistic
#                 prediction (or a genuinely very low SER) cannot make a
#                 point run away; the point comes back censored/thin instead
#   step       -- checkpoint every this many reps, for restart safety
#
# Sizing max_bits, from MEASURED throughput on this box (2000 bits/rep):
#   ClassicViterbi  1.15 s/rep = ~1760 bits/s  -> 2e7 bits = 3.2 h/point
#   Transformer     ~210 s/rep = ~9.5 bits/s   -> 1e5 bits = 2.9 h/point
# (ClassicViterbi was 6.5 s/rep before the COST2100 tap-load cache in
# 0ad20cd; that is a 5.7x end-to-end speedup, not the 224x that applies to
# estimate_channel alone.)
#
# ViterbiNet's max_bits (100_000, matching Transformer) is an UNCALIBRATED
# placeholder -- no run has completed on this branch yet. It is included
# because the data-cache bug invalidated its old n=84 baseline
# (Results/metrics/model_performance_final_mc_83.csv) too -- the paper's
# three-way comparison needs all three detectors measured under the fix, and
# it is already in the Colab branch's MODELS list for the same reason. Check
# the first [done] log line's run_time_sec once this actually runs, recompute
# bits/sec, and raise or lower max_bits to target a similar per-point budget
# as the other two -- do not leave this unexamined after the first real
# timing comes back.
#
# What this does and does not buy: ~100 errors needs ~100/SER bits, so
# SNR<=13 (SER >= 5.5e-6) now reaches a full 100 errors. The error floor at
# SNR>=14 (SER < 5e-7) would need ~2e8 bits = 31.5 h for ONE point, so those
# stay censored -- but at 2e7 bits their upper bound tightens 10x, to
# ~1.5e-7. Brute force cannot reach the floor here; that needs importance
# sampling, or a much faster detector implementation.
MODELS = [
    ('Transformer', 'ModelBased', 20, 100_000, 1),
    ('ViterbiNet', 'ModelBased', 20, 100_000, 1),
    ('ClassicViterbi', 'Statistical', 100, 20_000_000, 1),
]
# Push target. Override with MC_SWEEP_BRANCH when running this on a second
# machine so it does not push into the same branch another runner is already
# advancing -- see "Resuming on another machine" in README.md.
BRANCH = os.environ.get('MC_SWEEP_BRANCH') or 'claude/transformer-sionna-mlp-comparison-wc67zp'

# Set MC_SWEEP_NO_GIT=1 to keep results local: no commits, no pushes. Resume is
# unaffected -- a restart reads the same CSV row / weights / checkpoint files
# from disk either way; they just are not mirrored to a remote. Needed on a
# machine without push credentials, where the default behaviour would otherwise
# abort the sweep after push_with_retry exhausts its attempts.
NO_GIT = (os.environ.get('MC_SWEEP_NO_GIT') or '').lower() not in ('', '0', 'false', 'no')


def default_source():
    """Which machine produced a result. Rows from several machines end up in
    one CSV once a second runner's branch is merged back, and the hardware is
    part of how a row should be read -- run_time_sec in particular is not
    comparable between a CPU container and a local GPU. Override with
    MC_SWEEP_SOURCE to label a run explicitly."""
    host = socket.gethostname()
    try:
        device = (torch.cuda.get_device_name(0).replace(' ', '_')
                  if torch.cuda.is_available() else 'cpu')
    except Exception:
        device = 'cpu'
    return f'{host}:{device}'


SOURCE = os.environ.get('MC_SWEEP_SOURCE') or default_source()


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
        return

    # Upgrade a CSV written before a column existed. Without this, appending a
    # row with the new field to a file still carrying the old header writes
    # more values than there are columns, and every later read misaligns.
    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        if (reader.fieldnames or []) == FIELDNAMES:
            return
        rows = list(reader)
    for row in rows:
        if not row.get('source'):
            row['source'] = PRE_TRACKING_SOURCE
    tmp = CSV_PATH + '.tmp'
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval='',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, CSV_PATH)
    print(f'[csv] upgraded {len(rows)} existing rows to the current columns '
          f'(source={PRE_TRACKING_SOURCE!r} for rows predating provenance)', flush=True)


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
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval='',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def append_row(row):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES, restval='',
                       extrasaction='ignore').writerow(row)


# In-progress-point resume state. A point can take hours (an extend batch is a
# single trainer.online_evaluation() call that does not return until it
# finishes), and until now a mid-point restart lost all of it -- observed
# twice on SNR=13, each costing several hours of real compute. Checkpointing
# only the per-rep SER means (not the raw per-word arrays) is enough to
# reconstruct ser_mean/ser_std/ci95/errors_observed exactly on resume, and
# keeps each checkpoint write small.
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'metrics', '.mc_sweep_checkpoints')


def checkpoint_path(model, snr):
    return os.path.join(CHECKPOINT_DIR, f'{model}_snr{snr}.json')


def load_checkpoint(model, snr):
    path = checkpoint_path(model, snr)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


_UNTAGGED = object()  # sentinel: checkpoint predates weights tagging


def weights_fingerprint(weights_dir, snr, gamma):
    """Identity of the exact weights a point's eval reps were measured
    against, or None for a method with no weights (Statistical). Checkpoints
    are committed and therefore travel between worktrees via git, so reps
    must be matched to their model rather than trusted by filename alone --
    the on-disk weights file does not change during evaluation (save_weights
    is only called from train()), so this is stable across the whole run."""
    if weights_dir is None:
        return None
    path = os.path.join(weights_dir, f'snr_{snr}_gamma_{gamma}.pt')
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_checkpoint(model, snr, per_rep_means, weights_tag=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(model, snr)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'per_rep_means': per_rep_means, 'weights_tag': weights_tag}, f)
    os.replace(tmp, path)  # atomic: a restart mid-write never leaves a corrupt checkpoint


def clear_eval_checkpoint(model, snr):
    """Drop only the banked eval reps, keeping training progress."""
    path = checkpoint_path(model, snr)
    if os.path.isfile(path):
        os.remove(path)


def clear_checkpoint(model, snr):
    """Clear all per-point resume state -- called once the point is finished
    and committed, so the next run starts it clean if it ever re-runs."""
    clear_eval_checkpoint(model, snr)
    clear_training_state(model, snr)


def training_state_path(model, snr):
    return os.path.join(CHECKPOINT_DIR, f'{model}_snr{snr}_training.json')


def load_training_state(model, snr):
    """Where training got to for this point, across restarts: how many
    minibatches have run, the best validation SER seen so far (so a resumed
    run does not overwrite good weights with a worse first evaluation), and
    whether the full training budget has been spent. Without this, every
    restart replayed the whole minibatch loop from 1 -- and a point whose
    training takes longer than the container lives (e.g. ~6 min/minibatch x
    25 at high SNR, vs a ~1h container) could never finish training at all,
    so it never reached the eval phase and never produced a CSV row."""
    path = training_state_path(model, snr)
    if not os.path.isfile(path):
        return {'minibatches_done': 0, 'best_ser': math.inf, 'complete': False}
    with open(path) as f:
        state = json.load(f)
    # json has no inf: it round-trips as None.
    if state.get('best_ser') is None:
        state['best_ser'] = math.inf
    return state


def save_training_state(model, snr, minibatches_done, best_ser, complete):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = training_state_path(model, snr)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'minibatches_done': minibatches_done,
                   'best_ser': None if math.isinf(best_ser) else best_ser,
                   'complete': complete}, f)
    os.replace(tmp, path)


def clear_training_state(model, snr):
    path = training_state_path(model, snr)
    if os.path.isfile(path):
        os.remove(path)


def repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def weights_dir_for(model_name, detector_method):
    method_name = f'{model_name}_{detector_method}'
    return os.path.join(
        WEIGHTS_DIR,
        f'{method_name}_training_120_2_channel1_cost2100_mcsweep')


def push_with_retry(context):
    # A commit that never reaches the remote is a commit that can still be
    # lost (container reclaim, disconnect, etc). Retry with capped backoff
    # for several minutes; if it still can't push, stop the whole sweep
    # loudly instead of silently moving on and leaving this stranded local-only.
    #
    # Push HEAD explicitly (not the local branch name) so this works the same
    # whether this process is the main worktree (local branch == BRANCH) or a
    # sibling worktree running a different model concurrently, which is on
    # its own local branch pointed at the same remote BRANCH -- see
    # PARALLEL_MC_SWEEP notes.
    delays = (0, 2, 4, 8, 16, 30, 60, 60, 60, 60)
    for attempt, delay in enumerate(delays, 1):
        if delay:
            time.sleep(delay)
        r = subprocess.run(['git', 'push', '-q', 'origin', f'HEAD:{BRANCH}'], cwd=repo_dir())
        if r.returncode == 0:
            return
        print(f'[git] push attempt {attempt}/{len(delays)} failed for {context}', flush=True)

        # The expected cause when multiple sweep processes push to the same
        # branch concurrently: a sibling committed and pushed its own point
        # first, making this push non-fast-forward. Every commit here is a
        # pure append (one new CSV row, one model's own weights file), so
        # rebasing onto the new tip essentially never conflicts -- do that
        # and let the next loop iteration retry the push.
        subprocess.run(['git', 'fetch', '-q', 'origin', BRANCH], cwd=repo_dir())
        rebase = subprocess.run(['git', 'rebase', f'origin/{BRANCH}'],
                                 cwd=repo_dir(), capture_output=True, text=True)
        if rebase.returncode != 0:
            subprocess.run(['git', 'rebase', '--abort'], cwd=repo_dir())
            raise RuntimeError(
                f'git rebase onto origin/{BRANCH} failed for {context} (not a '
                f'plain non-fast-forward push failure) -- aborted the rebase '
                f'rather than risk a broken tree; needs manual resolution. '
                f'{rebase.stdout.strip()} {rebase.stderr.strip()}')

    raise RuntimeError(
        f'git push failed after {len(delays)} attempts for {context} -- '
        f'stopping the sweep so this is not silently lost (it is committed '
        f'locally but not on the remote yet).')


def commit_weights_snapshot(model_name, detector_method, snr, in_progress=False):
    """Commit+push a snapshot of a point's weights. Called once right after
    training finishes, before the (often much longer) eval-rep phase runs --
    and, with in_progress=True, also periodically *during* training (see
    make_training_committer) so that even a full disk loss mid-training
    (not just a process restart, which local disk alone already survives)
    cannot erase more than a bounded amount of training progress."""
    if NO_GIT:
        return
    weights_dir = weights_dir_for(model_name, detector_method)
    add_paths = []
    if os.path.isdir(weights_dir):
        add_paths.append(os.path.relpath(weights_dir, repo_dir()))
    # Stage the resume state in the SAME commit as the weights that produced
    # it, so the two can never drift apart on the remote: a restart that
    # pulls this commit gets banked reps and the model they were measured
    # against together, or neither.
    #
    # Stage only THIS point's files, never the whole directory. The directory
    # is shared: sibling worktrees running other models pull each other's
    # checkpoints in, so staging all of it would commit this worktree's
    # possibly-stale copy of another model's file and revert that model's
    # progress on the remote.
    for path in (checkpoint_path(model_name, snr), training_state_path(model_name, snr)):
        if os.path.isfile(path):
            add_paths.append(os.path.relpath(path, repo_dir()))
    if not add_paths:
        return
    subprocess.run(['git', 'add'] + add_paths, check=True, cwd=repo_dir())
    suffix = ' (training in progress)' if in_progress else ''
    commit = subprocess.run(
        ['git', 'commit', '-q', '-m', f'Train MC-sweep weights: {model_name} snr={snr}{suffix}'],
        cwd=repo_dir(), capture_output=True, text=True)
    if commit.returncode != 0:
        combined = (commit.stdout or '') + (commit.stderr or '')
        if 'nothing to commit' in combined.lower():
            return
        raise RuntimeError(
            f'git commit failed for {model_name} snr={snr} weights snapshot '
            f'(not a "nothing to commit" case): {combined.strip()}')
    push_with_retry(f'{model_name} snr={snr} weights snapshot')


def make_training_committer(model_name, detector_method, snr):
    """Callback for Trainer.train()'s on_checkpoint hook: commits+pushes the
    weights file every time training saves improved weights, rate-limited so
    it does not push on every single improving minibatch. A failure here is
    logged and swallowed rather than raised -- it runs deep inside the
    training loop, and a transient git/network hiccup mid-training should
    not abort the run; the next improvement (or the unconditional snapshot
    once training finishes) will pick it up."""
    last_commit = [0.0]
    min_interval_sec = 120
    def on_checkpoint():
        now = time.time()
        if now - last_commit[0] < min_interval_sec:
            return
        last_commit[0] = now
        try:
            commit_weights_snapshot(model_name, detector_method, snr, in_progress=True)
        except Exception as e:
            print(f'[git] mid-training commit failed for {model_name} snr={snr}: {e} '
                  f'-- continuing training, will retry at the next improvement', flush=True)
    return on_checkpoint


def stageable_paths(*paths):
    """Of the given paths, the ones `git add` will accept: present on disk,
    or absent but tracked (so the deletion stages). Passing a path that is
    neither makes git fail the whole invocation with "pathspec did not
    match", staging nothing at all -- which would silently drop the CSV row
    staged alongside it."""
    out = []
    for path in paths:
        rel = os.path.relpath(path, repo_dir())
        if os.path.exists(path):
            out.append(rel)
            continue
        tracked = subprocess.run(['git', 'ls-files', '--', rel],
                                  cwd=repo_dir(), capture_output=True, text=True)
        if tracked.stdout.strip():
            out.append(rel)
    return out


def commit_and_push(model, detector_method, snr):
    # Other models' weight checkpoints are already tracked in this repo (see
    # Results/weights/*), so this sweep's are too -- add them alongside the
    # CSV row so each point's commit is atomic and a training run this sweep
    # produced isn't left as an untracked, unpushed pile on disk. (Usually a
    # no-op here: commit_weights_snapshot already committed them right after
    # training, before the eval-rep phase ran.)
    weights_dir = weights_dir_for(model, detector_method)
    add_paths = ['Results/metrics/mc_sweep_validation.csv']
    if os.path.isdir(weights_dir):
        add_paths.append(os.path.relpath(weights_dir, repo_dir()))
    # The finished point's resume state is cleared here rather than by the
    # caller, so its removal lands in the same commit as the CSV row that
    # supersedes it -- otherwise the tracked checkpoint files would stay in
    # git forever and show as an unstaged deletion after every point.
    clear_checkpoint(model, snr)
    if NO_GIT:
        # The CSV row on disk is the result; clearing the resume state above
        # still matters so a finished point leaves nothing stale behind.
        return
    add_paths += stageable_paths(checkpoint_path(model, snr),
                                 training_state_path(model, snr))
    # -A so the deletions above are staged, not just modifications.
    subprocess.run(['git', 'add', '-A', '--'] + add_paths, check=True, cwd=repo_dir())
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

    push_with_retry(f'{model} snr={snr}')


def run_point(model_name, detector_method, snr, min_reps, max_bits, step):
    weights_dir = weights_dir_for(model_name, detector_method)

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

    # Train once (no-op for ClassicViterbi/Statistical). trainer.train() always
    # starts from the model's current in-memory weights -- Trainer.__init__
    # leaves those at a fresh random init, so a restart mid-training would
    # normally throw away whatever progress the interrupted attempt made and
    # start over. If this exact (model, snr) already has a weights file on
    # disk -- left there by a previous run_point() call for this same point
    # that got interrupted before its point was fully done (a finished point
    # is never re-entered; see the `done` skip in main()) -- load it first so
    # training continues from there instead of from scratch. Local disk alone
    # already survives a plain process restart, but not necessarily a full
    # container/disk loss -- on_checkpoint commits+pushes each improvement
    # (rate-limited) during training too, so that case is covered as well.
    #
    # The minibatch loop itself also resumes: load_training_state() says how
    # much of the training budget this point has already spent, so a point
    # whose training takes longer than the container lives finishes it over
    # several restarts instead of restarting the loop every time and never
    # reaching the eval phase at all.
    training_complete_on_entry = False
    if detector_method != 'Statistical':
        # load_train_weights() used to do this before calling train(); calling
        # train() directly means save_weights() would otherwise fail on a
        # model whose weights directory does not exist yet.
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f'snr_{snr}_gamma_{trainer.gamma}.pt')
        state = load_training_state(model_name, snr)
        training_complete_on_entry = state['complete'] and os.path.isfile(weights_path)
        if os.path.isfile(weights_path):
            prior = torch.load(weights_path)
            trainer.detector.model.load_state_dict(prior['model_state_dict'])
            print(f'[resume] {model_name} snr={snr}: warm-starting training from '
                  f'existing weights on disk (loss={prior["loss"]:.4f})', flush=True)

        if training_complete_on_entry:
            # Training already spent its full budget in an earlier run, and
            # those exact weights are what we just loaded. Re-running it would
            # produce a *different* model and invalidate any eval reps already
            # banked against this one, so leave the model frozen here and go
            # straight to eval.
            print(f'[resume] {model_name} snr={snr}: training already complete '
                  f'({state["minibatches_done"]}/{trainer.train_minibatch_num} minibatches, '
                  f'best_ser={state["best_ser"]:.6f}) -- skipping to evaluation', flush=True)
        else:
            commit_training = make_training_committer(model_name, detector_method, snr)

            def record_progress(minibatch, best_ser):
                save_training_state(model_name, snr, minibatch, best_ser, complete=False)
                # Push after every minibatch, not only the ones that improve
                # the loss. Training can run a long stretch without improving,
                # and that stretch is still real progress: the minibatch index
                # is what lets a restart skip it. It is rate-limited inside, so
                # this is cheap when minibatches are fast.
                commit_training()

            done_so_far = state['minibatches_done']
            if done_so_far:
                print(f'[resume] {model_name} snr={snr}: continuing training at minibatch '
                      f'{done_so_far + 1}/{trainer.train_minibatch_num} '
                      f'(best_ser={state["best_ser"]:.6f})', flush=True)
            trainer.fading_taps_type = 1
            trainer.train(
                on_checkpoint=commit_training,
                start_minibatch=done_so_far + 1,
                best_ser=state['best_ser'],
                on_minibatch=record_progress)
            trainer.fading_taps_type = 2
            save_training_state(model_name, snr, trainer.train_minibatch_num,
                                load_training_state(model_name, snr)['best_ser'],
                                complete=True)
            final = torch.load(weights_path)
            trainer.detector.model.load_state_dict(final['model_state_dict'])
    else:
        trainer.load_train_weights(run_over=2)
    commit_weights_snapshot(model_name, detector_method, snr)

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
    #
    # For a model-based method this is only valid once the model is FROZEN.
    # If training ran again in this process (training_complete_on_entry is
    # False), it produced a genuinely different model than the one an
    # existing checkpoint's reps were measured against, and mixing the two
    # would blend SER measurements from two different models into one
    # point's statistics -- an invalid Monte-Carlo estimate. Discard those
    # reps and start eval fresh in that case. When training was already
    # complete on entry, though, the weights loaded above are byte-identical
    # to the ones that produced those reps, so they accumulate legitimately
    # and the point can finish its eval budget across several restarts.
    weights_tag = weights_fingerprint(
        weights_dir if detector_method != 'Statistical' else None, snr, trainer.gamma)

    checkpoint = load_checkpoint(model_name, snr)
    if checkpoint:
        banked = len(checkpoint['per_rep_means'])
        stored_tag = checkpoint.get('weights_tag', _UNTAGGED)
        discard_reason = None

        if detector_method != 'Statistical' and not training_complete_on_entry:
            discard_reason = ('training ran again this pass, so those reps were '
                              'computed against a different trained model')
        elif stored_tag is _UNTAGGED:
            # Written before checkpoints carried a weights tag. Training is
            # complete and the weights have not changed since (save_weights
            # only runs during training), so these reps do belong to the model
            # loaded here: adopt them and let the next save stamp the tag,
            # rather than discarding valid evaluation over a format change.
            print(f'[resume] {model_name} snr={snr}: adopting untagged eval checkpoint '
                  f'with {banked} reps (predates weight tagging); it will be tagged to '
                  f'the current weights on the next save', flush=True)
        elif stored_tag != weights_tag:
            # Checkpoints are committed, so one can arrive from another
            # worktree or an older run carrying reps measured against
            # different weights. The complete-flag check cannot see that.
            discard_reason = ('it is tagged to different weights than the ones '
                              'loaded here, so its reps belong to another model')

        if discard_reason:
            print(f'[resume] {model_name} snr={snr}: discarding eval checkpoint with '
                  f'{banked} reps -- {discard_reason}', flush=True)
            clear_eval_checkpoint(model_name, snr)
            checkpoint = None
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

    last_eval_commit = [0.0]
    eval_commit_interval_sec = 120

    def commit_eval_progress():
        now = time.time()
        if now - last_eval_commit[0] < eval_commit_interval_sec:
            return
        last_eval_commit[0] = now
        try:
            commit_weights_snapshot(model_name, detector_method, snr, in_progress=True)
        except Exception as e:
            print(f'[git] mid-eval commit failed for {model_name} snr={snr}: {e} '
                  f'-- continuing, will retry after the next chunk', flush=True)

    def run_batch(n):
        nonlocal reps_done
        # Chunk into at most `step` reps per trainer call and checkpoint
        # after every chunk. Without this, a single extend round can be
        # thousands of reps in one online_evaluation() call that does not
        # return for hours -- a restart mid-call loses all of it, which is
        # exactly what happened twice on SNR=13 before this existed.
        remaining = n
        while remaining > 0:
            chunk = min(step, remaining)
            ser = np.asarray(trainer.online_evaluation(num_of_rep=chunk)).reshape(-1)
            chunk_means = ser.reshape(chunk, words_per_rep).mean(axis=1)
            per_rep_means.extend(float(m) for m in chunk_means)
            reps_done += chunk
            remaining -= chunk
            save_checkpoint(model_name, snr, per_rep_means, weights_tag)
            # Push each banked chunk. Saving it locally only protected against
            # a process restart; a point can now represent tens of hours of
            # evaluation, so get it onto the remote where a container/disk
            # loss cannot take it. Rate-limited, and failures are logged
            # rather than raised so a git hiccup never aborts the run.
            commit_eval_progress()

    # Calculate the required bit budget once, up front, and run exactly that
    # many reps -- no doubling, no re-targeting off what gets observed. See
    # the "Sizing method" note at the top of this file.
    predicted_ser = max(expected_ser_isi(snr), 1e-300)
    required_bits = TARGET_ERRORS / predicted_ser
    required_reps = math.ceil(required_bits / bits_per_rep)
    capped_reps = max_bits // bits_per_rep
    planned_reps = int(min(max(required_reps, min_reps), capped_reps))

    print(f'[plan] {model_name} snr={snr}: predicted_ser={predicted_ser:.3e}, '
          f'planned_reps={planned_reps} (min={min_reps}, cap={capped_reps})', flush=True)

    if reps_done < planned_reps:
        run_batch(planned_reps - reps_done)

    censored = total_errors() == 0
    if censored:
        print(f'[censored] {model_name} snr={snr}: 0 errors in {reps_done} reps '
              f'({reps_done * bits_per_rep:,} bits) -- reporting as upper bound',
              flush=True)
    elif total_errors() < THIN_ERROR_THRESHOLD:
        print(f'[thin] {model_name} snr={snr}: only {total_errors():.0f} errors in '
              f'{reps_done * bits_per_rep:,} bits (one-shot budget reached) -- '
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
        'source': SOURCE,
    }


def main():
    # Optional: python run_mc_sweep.py <ModelName> restricts this process to
    # one model, so it can run concurrently with sibling processes (each in
    # its own worktree -- see PARALLEL_MC_SWEEP notes) covering the other
    # models, instead of this one process working through all of them in
    # sequence. Point-level commit/push and checkpointing are unaffected;
    # only which models this particular process considers is scoped.
    models = MODELS
    if len(sys.argv) > 1:
        wanted = sys.argv[1]
        models = [m for m in MODELS if m[0] == wanted]
        if not models:
            valid = ', '.join(m[0] for m in MODELS)
            raise SystemExit(f'Unknown model {wanted!r} -- choose one of: {valid}')
        print(f'[filter] restricting this process to model={wanted}', flush=True)

    ensure_header()
    done = already_done()
    print(f'Already completed points: {sorted(done)}', flush=True)

    # Per-SNR, per-model: at each SNR, try every model before moving to the
    # next SNR, rather than exhausting one model's whole 0-17 range first.
    # Keeps progress broad instead of narrow, so no single model's full
    # sweep has to finish before the other model's points at the same SNR
    # are even attempted.
    for snr in SNR_VALUES:
        for model_name, detector_method, min_reps, max_bits, step in models:
            key = (model_name, snr)
            if key in done:
                print(f'[skip] {model_name} snr={snr} already in CSV', flush=True)
                continue

            print(f'\n{"="*70}\n[run] {model_name} snr={snr}\n{"="*70}', flush=True)
            try:
                row = run_point(model_name, detector_method, snr,
                                 min_reps, max_bits, step)
            except Exception as e:
                print(f'[ERROR] {model_name} snr={snr} failed: {e}', flush=True)
                import traceback
                traceback.print_exc()
                continue

            drop_existing_row(model_name, snr)
            append_row(row)
            print(f'[done] {row}', flush=True)
            commit_and_push(model_name, detector_method, snr)  # also clears resume state
            done.add(key)

    print('\nAll points complete.', flush=True)


if __name__ == '__main__':
    main()
