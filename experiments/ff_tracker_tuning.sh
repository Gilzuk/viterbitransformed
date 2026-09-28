#!/bin/bash
# Classical tracker tuning under BPSK fast fading at 7 dB (20 paired repetitions):
# PSP-LMS step size and exponentially weighted LS forgetting factor, per Doppler.
# usage: experiments/ff_tracker_tuning.sh <fd>   (rows -> Results/metrics/fast_fading.csv)
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FF_REPS=20
specs=()
for mu in 0.003 0.01 0.02 0.05; do specs+=("tune_PSP_mu$mu=ClassicViterbi_PSP:Statistical:psp_step=$mu"); done
for lam in 0.5 0.8 0.9 0.95 0.99; do specs+=("tune_RLS_lam$lam=ClassicViterbi_LS:Statistical:ls_forget=$lam"); done
specs+=("tune_LS_last=ClassicViterbi_LS:Statistical")
nice -n 10 python3 -u run_fast_fading.py 7 "$1" "${specs[@]}"
