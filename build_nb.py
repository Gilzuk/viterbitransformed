"""Generates run_mc_sweep_colab.ipynb. Kept in-tree so the notebook can be
regenerated rather than hand-edited as JSON."""
import json

def _lines(src):
    """ipynb `source` is a list of lines that each KEEP their trailing newline
    (all but the last). Splitting without keepends silently concatenates every
    line into one when the notebook is opened."""
    lines = src.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(src)}

cells = [
md("""
# MC-sweep validation on Colab (GPU)

Runs `run_mc_sweep.py` over SNR 0-17 for three detectors -- **ClassicViterbi**
(perfect-CSI reference), **ViterbiNet**, and the **Viterbi-Transformer** -- at
25 independent Monte-Carlo repetitions per point.

Results are appended to `Results/metrics/mc_sweep_validation_colab.csv` and
**committed + pushed after every completed `(model, snr)` point**, on the
`mc-sweep-colab-gpu` branch.

### If Colab disconnects
Nothing already pushed is lost, and a rerun does **not** start over. Just
re-run cells 1-5: the clone pulls back every point pushed so far, and the
script skips those and continues from the next one. At most the single
in-progress point is lost.

See `COLAB_MC_SWEEP.md` in this repo for the background (including the
data-cache bug that made every repetition identical, and the ISI-aware
sizing predictor).
"""),

md("## 1. Clone (or update) the branch"),
code("""
import os

REPO_DIR = "/content/viterbitransformed"
BRANCH   = "mc-sweep-colab-gpu"
REPO_URL = "https://github.com/Gilzuk/viterbitransformed.git"

if not os.path.exists(REPO_DIR):
    !git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}
else:
    # Already present (e.g. re-running after a reconnect): fast-forward so we
    # pick up every point pushed before the disconnect.
    !cd {REPO_DIR} && git fetch origin {BRANCH} && git checkout {BRANCH} && git pull --ff-only origin {BRANCH}

%cd {REPO_DIR}
!git log --oneline -1
"""),

md("## 2. Install dependencies"),
code("""
!pip install -q numpy scipy matplotlib pandas psutil tqdm
# torch/torchvision are preinstalled on Colab GPU runtimes; only install if missing.
import importlib
if importlib.util.find_spec("torch") is None:
    !pip install -q torch torchvision torchaudio
import torch
print("torch", torch.__version__)
"""),

md("""
## 3. Verify GPU

If this prints `NO GPU`, stop and set **Runtime > Change runtime type > GPU**,
then re-run from cell 1. Running this on a Colab CPU is slower than useless --
that is the whole reason for using Colab here.
"""),
code("""
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Memory: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
else:
    print("NO GPU -- set Runtime > Change runtime type > GPU and re-run from cell 1")
"""),

md("""
## 4. Git identity + push credentials

The sweep pushes after every point, so this has to work *before* the long run
starts -- the last cell here does a `--dry-run` push to prove it does, rather
than discovering a credentials problem hours in.

Use a fine-grained GitHub PAT with **Contents: read and write** on this repo.
Note the token gets written into `.git/config` on this (ephemeral) runtime.
"""),
code("""
from getpass import getpass

!git config user.email "gil.zukerman@gmail.com"
!git config user.name "Gil Zukerman"

token = getpass("GitHub token (fine-grained PAT, Contents: read+write): ")
!git remote set-url origin https://{token}@github.com/Gilzuk/viterbitransformed.git

# Prove push auth works now, before committing hours of compute to it.
print("\\n--- verifying push access (dry run) ---")
!git push --dry-run origin {BRANCH} && echo "PUSH OK -- credentials work" || echo "PUSH FAILED -- fix the token before running cell 5"
"""),

md("""
## 5. Run the sweep

Re-runnable and resumable: points already in the CSV are skipped.

The full log goes to `/content/mc_sweep.log`; only the meaningful progress
lines are shown here, since the raw output includes per-word training chatter
and would otherwise be megabytes of scrollback.
"""),
code("""
import re, subprocess, sys

LOG = "/content/mc_sweep.log"
# Progress lines worth surfacing, plus anything that indicates a stop -- a
# failed push raises, and its traceback must not be filtered out of view.
KEEP = re.compile(r"\\[run\\]|\\[plan\\]|\\[done\\]|\\[skip\\]|\\[extend\\]|\\[censored\\]"
                  r"|\\[git\\]|\\[ERROR\\]|All points complete"
                  r"|Traceback|Error|Exception")

with open(LOG, "w") as log:
    proc = subprocess.Popen(["python", "-u", "run_mc_sweep.py"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        log.write(line)
        # tqdm redraws with \\r; keep only the last redraw of a chunk so the
        # notebook does not accumulate megabytes of progress-bar scrollback.
        tail = line.rsplit("\\r", 1)[-1]
        if KEEP.search(tail):
            print(tail.rstrip())
            sys.stdout.flush()
    proc.wait()

print("\\n--- exit code %d ---" % proc.returncode)
print("full log: %s" % LOG)
"""),

md("## 6. Progress so far"),
code("""
import os
import pandas as pd

CSV = "Results/metrics/mc_sweep_validation_colab.csv"
if os.path.exists(CSV):
    df = pd.read_csv(CSV)
    print("%d point(s) complete" % len(df))
    if len(df):
        cols = ["model", "snr", "ser_mean", "ser_ci95", "n_reps",
                "bits_run", "errors_observed", "censored", "run_time_sec"]
        display(df[[c for c in cols if c in df.columns]])
        print("\\nRemaining: %d of %d" % (3 * 18 - len(df), 3 * 18))
else:
    print("No results yet -- run cell 5.")
"""),

md("""
## 7. Plot (once there is enough data)
"""),
code("""
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV = "Results/metrics/mc_sweep_validation_colab.csv"
if os.path.exists(CSV) and len(pd.read_csv(CSV)):
    df = pd.read_csv(CSV).sort_values("snr")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in df.groupby("model"):
        solid = g[g.censored == 0]
        ax.errorbar(solid.snr, solid.ser_mean, yerr=solid.ser_ci95,
                    marker="o", capsize=3, label=model)
        # censored points are upper bounds, not measurements -- mark them apart
        cens = g[g.censored == 1]
        if len(cens):
            ax.scatter(cens.snr, 1.0 / cens.bits_run, marker="v", s=60,
                       label="%s (upper bound, 0 errors)" % model)
    ax.set_yscale("log")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("Coded SER")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
else:
    print("No results yet -- run cell 5.")
"""),

md("""
## Troubleshooting

**The sweep stopped with a `git push failed` / `git commit failed` error.**
That is deliberate. A point that is committed locally but never pushed would
be lost outright when the runtime is reclaimed, so the script stops loudly
rather than continuing and stranding results. Fix the credentials (cell 4)
and re-run cell 5 -- it resumes.

**It is slower than expected.** Trim the SNR ladder rather than cutting reps:
edit `SNR_VALUES` near the top of `run_mc_sweep.py` (the 0-6 dB points sit well
below the interesting crossover region). Cutting reps below 25 starts to
compromise the confidence intervals.

**A point reports `censored=1`.** That means zero errors were observed even
after extending, so its SER is reported as an upper bound (`< 1/bits_run`),
not a converged zero. That is the honest result at high SNR, not a failure.
""")
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("run_mc_sweep_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote run_mc_sweep_colab.ipynb with %d cells" % len(cells))
