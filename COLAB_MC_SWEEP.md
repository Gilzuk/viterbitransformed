# Running the MC-sweep validation on Colab (GPU)

This branch exists purely to run `run_mc_sweep.py` on a Colab GPU runtime,
in parallel with the CPU sweep already running elsewhere against
`claude/transformer-sionna-mlp-comparison-wc67zp`. Results land in a
separate CSV on a separate branch so the two runs cannot collide, and get
merged back afterward.

## Why bother with Colab

`run_mc_sweep.py` evaluates ClassicViterbi and the Viterbi-Transformer over
SNR 0-17 with more Monte-Carlo repetitions than the earlier n=3 sweep. The
Transformer side is slow on CPU because its self-supervised online training
does a backprop step on almost every word during evaluation (avg SER is
usually well under the 0.02 gating threshold) -- measured at ~3.5 min/rep on
a CPU-only sandbox, which is why the CPU run only attempts n=20 reps for the
Transformer (~21 hours) instead of the full n=100. A real GPU should cut
that per-rep cost substantially, making n=100 for the Transformer feasible
in a few hours instead of the better part of a day.

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

`run_mc_sweep.py` already targets this branch's own CSV/branch config (set
up below), commits after every completed `(model, snr)` point, and skips
points already present in the CSV -- so a Colab disconnect just means
re-running this cell; it resumes where it left off.

```python
!python run_mc_sweep.py
```

The script as checked into this branch is configured to:
- write results to `Results/metrics/mc_sweep_validation_colab.csv` (not
  `mc_sweep_validation.csv` -- that name is reserved for the CPU run)
- push to `mc-sweep-colab-gpu` (not the branch the CPU run is on)
- run ClassicViterbi and Transformer, both at `num_of_rep=100`, since GPU
  should make the full n=100 Transformer sweep tractable

If you want to change either the rep counts or which models run, edit the
`MODELS` list near the top of `run_mc_sweep.py` before starting.

## Keeping the session alive

Colab kills idle runtimes. For a multi-hour unattended run, periodically
interact with the notebook (Colab Pro's background execution helps), or
plan to just re-run the cell above if it disconnects -- it resumes cleanly.

## Merging back

Once this finishes, from the main repo (not this branch):
```bash
git fetch origin mc-sweep-colab-gpu
git show origin/mc-sweep-colab-gpu:Results/metrics/mc_sweep_validation_colab.csv \
    > Results/metrics/mc_sweep_validation_colab.csv
```
Say when it's done and I'll fold the two CSVs (the CPU run's
`mc_sweep_validation.csv` and this one) into the paper's data pipeline.
