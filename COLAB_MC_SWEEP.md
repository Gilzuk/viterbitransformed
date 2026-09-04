# Running the MC-sweep validation on Colab (GPU)

This branch exists purely to run `run_mc_sweep.py` on a Colab GPU runtime,
in parallel with the CPU sweep already running elsewhere against
`claude/transformer-sionna-mlp-comparison-wc67zp`. Results land in a
separate CSV on a separate branch so the two runs cannot collide, and get
merged back afterward.

## Why bother with Colab

`run_mc_sweep.py` evaluates ClassicViterbi and the Viterbi-Transformer over
SNR 0-17, sizing the number of repetitions per SNR point adaptively (see
"How many reps run" below) instead of a fixed rep count, so high-SNR points
actually observe real errors instead of silently floors at a meaningless
"0". The Transformer side is slow on CPU because its self-supervised online
training does a backprop step on almost every word during evaluation --
measured at ~3.5 min/rep on a CPU-only sandbox. A real GPU should cut that
per-rep cost substantially, making the higher rep caps this branch uses
tractable in a few hours instead of the better part of a day.

## How many reps run per point

Fixed rep counts have a real problem: at high SNR the true SER can be far
below what a small, fixed number of bits can resolve, and a naive average
then reports an exact "0" that looks converged but is really just "we didn't
run enough bits to see an error." `run_mc_sweep.py` instead:

1. Predicts the SER at each SNR from the closed-form BPSK/AWGN error
   probability Q(sqrt(2*snr_linear)) -- a heuristic only (this channel has
   ISI/fading on top of AWGN), used just to size the run.
2. Targets ~100 expected errors (the standard rule of thumb for a stable
   Monte-Carlo estimate): `required_bits = 100 / predicted_ser`.
3. Converts that to repetitions, clipped to a per-model `[min_reps,
   max_reps]` (see `MODELS` near the top of the script) so runtime stays
   bounded even where the formula's prediction is unreasonably large.
4. If the point still saw zero errors after that, keeps extending in fixed
   steps -- "run until first error" -- up to a higher `extend_max_reps`
   safety cap.
5. If it's still zero at that cap, the row is recorded as `censored=1` with
   `bits_run` also recorded, so it can be read as an honest upper bound
   (`< 1/bits_run`) rather than a converged zero.

## Setup

**1. Clone this branch**
```python
import os
REPO_DIR = "viterbitransformed"
if not os.path.exists(REPO_DIR):
    !git clone --branch mc-sweep-colab-gpu \
        https://github.com/Gilzuk/viterbitransformed.git {REPO_DIR}
%cd {REPO_DIR}
```

**2. Install dependencies**
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q numpy scipy matplotlib pandas psutil tqdm
```

**3. Verify GPU** (Runtime > Change runtime type > GPU, if not already set)
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(0) if device.type == "cuda" else "NO GPU -- stop and enable one")
```

**4. Git identity + push auth**
```python
!git config user.email "gil.zukerman@gmail.com"
!git config user.name "Gil Zukerman"

# Fine-grained GitHub PAT with "Contents: read and write" on this repo:
from getpass import getpass
token = getpass("GitHub token: ")
!git remote set-url origin https://{token}@github.com/Gilzuk/viterbitransformed.git
```

## Run

```python
!python run_mc_sweep.py
```

The script as checked into this branch is configured to write results to
`Results/metrics/mc_sweep_validation_colab.csv` (not
`mc_sweep_validation.csv` -- that name is reserved for the CPU run) and push
to `mc-sweep-colab-gpu` (not the branch the CPU run is on). If you want to
change the rep-count bounds or which models run, edit the `MODELS` list near
the top of `run_mc_sweep.py` before starting.

### Resilience -- every point is pushed, and reruns resume, not restart

This matters more here than on a persistent machine: a Colab runtime is
ephemeral, so anything not already pushed to GitHub when it disconnects is
just gone.

- **Every completed point is committed and pushed before the next one
  starts.** If the push itself fails, the script retries with capped
  backoff for several minutes and then **stops the whole sweep** rather than
  silently moving on -- a point that's committed locally but never reached
  the remote would otherwise be dropped entirely on the next disconnect.
  If you see the sweep has stopped, check your network/token and just
  re-run the cell.
- **A rerun always resumes from the last pushed point, never from SNR=0.**
  `already_done()` reads whatever is already in
  `mc_sweep_validation_colab.csv` on disk -- which, right after a fresh
  clone, is exactly what was pushed before the disconnect -- and skips
  every `(model, snr)` pair already present. So after a disconnect, just
  re-run cells 1 (re-clone) through 5 (run); the re-clone pulls back
  everything already pushed, and the script picks up immediately after the
  last completed point instead of repeating work.
- The only work that can be lost is the *single point in progress* at the
  moment of a disconnect (it isn't written/pushed until it completes) --
  never anything already finished.

## Keeping the session alive

Colab kills idle runtimes. For a multi-hour unattended run, periodically
interact with the notebook (Colab Pro's background execution helps), or
just accept that a disconnect costs at most the one in-progress point and
re-run the cell -- per the resume behavior above, that's cheap.

## Merging back

Once this finishes, from the main repo (not this branch):
```bash
git fetch origin mc-sweep-colab-gpu
git show origin/mc-sweep-colab-gpu:Results/metrics/mc_sweep_validation_colab.csv \
    > Results/metrics/mc_sweep_validation_colab.csv
```
Say when it's done and I'll fold the two CSVs (the CPU run's
`mc_sweep_validation.csv` and this one) into the paper's data pipeline.
