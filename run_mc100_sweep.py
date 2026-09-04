"""
MC=100 validation sweep: Viterbi-Transformer and ClassicViterbi over SNR 0-17,
using num_of_rep=100 evaluation repetitions per (model, SNR) point (train once,
evaluate 100 times), matching Code/configuration.yaml defaults.

Commits and pushes Results/metrics/transformer_mc100_validation.csv after every
completed (model, snr) point, so a container reset loses at most one point.

Run standalone: python run_mc100_sweep.py
"""
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

from Code.dir_definitions import RESULTS_DIR, WEIGHTS_DIR
from Code.trainer import Trainer

CSV_PATH = os.path.join(RESULTS_DIR, 'metrics', 'transformer_mc100_validation.csv')
FIELDNAMES = ['model', 'snr', 'ser_mean', 'ser_std', 'ser_ci95', 'n_reps',
              'model_size', 'run_time_sec']
NUM_OF_REP = 100
SNR_VALUES = list(range(0, 18))
MODELS = [('Transformer', 'ModelBased'), ('ClassicViterbi', 'Statistical')]

WORDS_PER_REP = None  # inferred from returned ser array length


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


def run_git(*args):
    subprocess.run(['git'] + list(args), check=True, cwd=os.path.dirname(os.path.abspath(__file__)))


def commit_and_push(model, snr):
    try:
        run_git('add', 'Results/metrics/transformer_mc100_validation.csv')
        run_git('commit', '-q', '-m',
                f'Add MC=100 validation point: {model} snr={snr}')
        for attempt, delay in enumerate([0, 2, 4, 8, 16]):
            if delay:
                time.sleep(delay)
            r = subprocess.run(['git', 'push', '-q', 'origin',
                                 'claude/transformer-sionna-mlp-comparison-wc67zp'],
                                cwd=os.path.dirname(os.path.abspath(__file__)))
            if r.returncode == 0:
                break
    except subprocess.CalledProcessError as e:
        print(f'[git] nothing to commit or push failed for {model} snr={snr}: {e}')


def run_point(model_name, detector_method, snr):
    method_name = f'{model_name}_{detector_method}'
    weights_dir = os.path.join(
        WEIGHTS_DIR,
        f'{method_name}_training_120_2_channel1_cost2100_mc100')

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

    ser = trainer.run(run_over=2, num_of_rep=NUM_OF_REP)
    run_time = time.time() - t0

    ser = np.asarray(ser).reshape(-1)
    words_per_rep = ser.shape[0] // NUM_OF_REP
    per_rep_means = ser.reshape(NUM_OF_REP, words_per_rep).mean(axis=1)

    ser_mean = float(per_rep_means.mean())
    if NUM_OF_REP > 1:
        ser_std = float(per_rep_means.std(ddof=1))
        ser_ci95 = 1.96 * ser_std / math.sqrt(NUM_OF_REP)
    else:
        ser_std = 0.0
        ser_ci95 = 0.0

    return {
        'model': model_name,
        'snr': snr,
        'ser_mean': ser_mean,
        'ser_std': ser_std,
        'ser_ci95': ser_ci95,
        'n_reps': NUM_OF_REP,
        'model_size': int(model_size),
        'run_time_sec': run_time,
    }


def main():
    ensure_header()
    done = already_done()
    print(f'Already completed points: {sorted(done)}', flush=True)

    for model_name, detector_method in MODELS:
        for snr in SNR_VALUES:
            key = (model_name, snr)
            if key in done:
                print(f'[skip] {model_name} snr={snr} already in CSV', flush=True)
                continue

            print(f'\n{"="*70}\n[run] {model_name} snr={snr} (num_of_rep={NUM_OF_REP})\n{"="*70}', flush=True)
            try:
                row = run_point(model_name, detector_method, snr)
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
