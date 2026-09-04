"""
Higher-MC validation sweep (Colab/GPU configuration): ClassicViterbi and
Viterbi-Transformer over SNR 0-17, evaluating each with more Monte-Carlo
repetitions than the earlier n=3 sweep
(Results/metrics/transformer_mask_fix_validation.csv), to tighten the
confidence interval on the paper's central result.

Each (model, snr) point trains once (Transformer only; ClassicViterbi has no
training) and then evaluates num_of_rep times (train once, evaluate N times),
using Code/configuration.yaml defaults otherwise. On CPU, ClassicViterbi is
~6s/rep (no training, no online adaptation), while the Transformer's
self-supervised online training fires on almost every word during each eval
rep (avg SER is usually well under the 0.02 gating threshold), making each
Transformer rep ~3.5 min there -- see COLAB_MC_SWEEP.md for why this copy of
the script targets a GPU runtime instead: both models run at num_of_rep=100
here, which is only tractable with GPU-accelerated backprop for the
Transformer's per-word online updates.

Commits and pushes Results/metrics/mc_sweep_validation_colab.csv (a separate
file from the CPU run's mc_sweep_validation.csv) to the mc-sweep-colab-gpu
branch (separate from the CPU run's branch) after every completed
(model, snr) point, so a Colab disconnect loses at most one point. See
COLAB_MC_SWEEP.md for setup and how the two runs' results get merged back.

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
              'model_size', 'run_time_sec']
SNR_VALUES = list(range(0, 18))
# (model_name, detector_method, num_of_rep) -- ClassicViterbi first (cheap).
# GPU run: both at n=100 (CPU run elsewhere caps the Transformer at n=20
# because of per-rep online-training cost -- see COLAB_MC_SWEEP.md).
MODELS = [
    ('ClassicViterbi', 'Statistical', 100),
    ('Transformer', 'ModelBased', 100),
]
BRANCH = 'mc-sweep-colab-gpu'


def already_done():
    done = set()
    if os.path.isfile(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add((row['model'], int(row['snr'])))
    return done


def ensure_header():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_row(row):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def commit_and_push(model, snr):
    try:
        subprocess.run(['git', 'add', 'Results/metrics/mc_sweep_validation_colab.csv'],
                        check=True, cwd=repo_dir())
        subprocess.run(['git', 'commit', '-q', '-m',
                         f'Add MC-sweep validation point: {model} snr={snr}'],
                        check=True, cwd=repo_dir())
    except subprocess.CalledProcessError as e:
        print(f'[git] nothing to commit for {model} snr={snr}: {e}', flush=True)
        return

    for delay in (0, 2, 4, 8, 16):
        if delay:
            time.sleep(delay)
        r = subprocess.run(['git', 'push', '-q', 'origin', BRANCH], cwd=repo_dir())
        if r.returncode == 0:
            return
    print(f'[git] push FAILED after retries for {model} snr={snr}', flush=True)


def run_point(model_name, detector_method, snr, num_of_rep):
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

    ser = trainer.run(run_over=2, num_of_rep=num_of_rep)
    run_time = time.time() - t0

    ser = np.asarray(ser).reshape(-1)
    words_per_rep = ser.shape[0] // num_of_rep
    per_rep_means = ser.reshape(num_of_rep, words_per_rep).mean(axis=1)

    ser_mean = float(per_rep_means.mean())
    if num_of_rep > 1:
        ser_std = float(per_rep_means.std(ddof=1))
        ser_ci95 = 1.96 * ser_std / math.sqrt(num_of_rep)
    else:
        ser_std = 0.0
        ser_ci95 = 0.0

    return {
        'model': model_name,
        'snr': snr,
        'ser_mean': ser_mean,
        'ser_std': ser_std,
        'ser_ci95': ser_ci95,
        'n_reps': num_of_rep,
        'model_size': int(model_size),
        'run_time_sec': run_time,
    }


def main():
    ensure_header()
    done = already_done()
    print(f'Already completed points: {sorted(done)}', flush=True)

    for model_name, detector_method, num_of_rep in MODELS:
        for snr in SNR_VALUES:
            key = (model_name, snr)
            if key in done:
                print(f'[skip] {model_name} snr={snr} already in CSV', flush=True)
                continue

            print(f'\n{"="*70}\n[run] {model_name} snr={snr} (num_of_rep={num_of_rep})\n{"="*70}',
                  flush=True)
            try:
                row = run_point(model_name, detector_method, snr, num_of_rep)
            except Exception as e:
                print(f'[ERROR] {model_name} snr={snr} failed: {e}', flush=True)
                import traceback
                traceback.print_exc()
                continue

            append_row(row)
            print(f'[done] {row}', flush=True)
            commit_and_push(model_name, snr)
            done.add(key)

    print('\nAll points complete.', flush=True)


if __name__ == '__main__':
    main()
