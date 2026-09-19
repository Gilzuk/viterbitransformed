"""
Generates run_mc_sweep_colab.ipynb.

The notebook is built here rather than hand-edited, because an .ipynb stores
each cell as a list of lines and every line except the last must keep its
trailing newline -- editing that JSON by hand has silently spliced lines
together before. nbformat.validate() at the end catches a malformed result
before it reaches Colab.

Run: python build_colab_nb.py
"""
import nbformat as nbf

REPO = 'https://github.com/Gilzuk/viterbitransformed'
BRANCH = 'claude/transformer-sionna-mlp-comparison-wc67zp'

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(f"""
# MC sweep on Colab (GPU)

Runs the Monte-Carlo validation sweep on Colab's GPU, **continuing** the run already in
progress rather than starting over: finished points, trained weights and mid-point resume
state all travel in the repo.

Runs alongside other machines. The rule is **one model per runner** -- two runners on the
same point duplicate the work and race on its CSV row. Pick a model no one else is running.

Branch: `{BRANCH}`
""")

md("## 1. Check the GPU\nIf this says no GPU: Runtime -> Change runtime type -> T4 GPU, then re-run.")

code("""
import torch
print('torch', torch.__version__)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('*** NO GPU -- Runtime > Change runtime type > T4 GPU, then re-run this cell ***')
""")

md("## 2. Which model to run\nSet this to a model nobody else is currently running.")

code("""
MODEL = 'Transformer'   # 'Transformer' | 'ViterbiNet' | 'ClassicViterbi'

# Push results to their own branch so two runners never fight over one branch.
# Leave PUSH = False to keep everything local to this Colab session (then use
# the Drive cell below, or the session's results are lost when it ends).
PUSH = False
PUSH_BRANCH = 'colab-gpu'
""")

md(f"""
## 3. Clone the repo
Brings the sweep's full state with it: `Results/metrics/mc_sweep_validation.csv` (finished
points), `Results/weights/` (trained weights) and `Results/metrics/.mc_sweep_checkpoints/`
(banked evaluation repetitions and training progress).
""")

code(f"""
import os, subprocess
REPO = '{REPO}'
BRANCH = '{BRANCH}'
if not os.path.isdir('/content/viterbitransformed'):
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO, '/content/viterbitransformed'], check=True)
os.chdir('/content/viterbitransformed')
subprocess.run(['git', 'pull', '--rebase', 'origin', BRANCH], check=False)
print(subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True).stdout)

import csv
rows = list(csv.DictReader(open('Results/metrics/mc_sweep_validation.csv')))
have = sorted(int(r['snr']) for r in rows if r['model'] == MODEL)
print(f'{{MODEL}}: {{len(have)}} points already done -> {{have}}')
print('missing ->', [s for s in range(18) if s not in have])
""")

md("""
## 4. Dependencies
Colab already ships torch, numpy, scipy, pandas, matplotlib, tqdm and pyyaml, so this is
usually a no-op. It installs only what is genuinely missing.
""")

code("""
import importlib, subprocess, sys
missing = [m for m in ['numpy','scipy','pandas','matplotlib','tqdm','yaml']
           if not importlib.util.find_spec(m)]
print('installing:', missing or 'nothing (all present)')
if missing:
    pkgs = ['pyyaml' if m == 'yaml' else m for m in missing]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)
""")

md("""
## 5. (Optional) Keep results when the session ends

A Colab session is reclaimed after a few hours and its disk goes with it. Pick **one**:

* **Push to GitHub** -- set `PUSH = True` above and add a `GITHUB_TOKEN` in the Colab
  Secrets panel (key icon, left sidebar) with `Contents: read and write` on this repo.
  Each repetition is committed and pushed, so a new session resumes from the last push.
* **Google Drive** -- run the cell below. It points `Results/` at a Drive folder, so
  progress survives the session. Merge it back into git yourself afterwards.

Skipping both is fine for a quick timing measurement, but anything the session computes
is lost when it ends.
""")

code("""
USE_DRIVE = False   # set True to persist Results/ to Google Drive

if USE_DRIVE:
    from google.colab import drive
    import shutil, os
    drive.mount('/content/drive')
    dest = '/content/drive/MyDrive/mc_sweep_results'
    os.makedirs(dest, exist_ok=True)
    # Seed Drive from the clone the first time, then symlink Results/ at it.
    if not os.path.exists(os.path.join(dest, 'metrics')):
        shutil.copytree('Results', dest, dirs_exist_ok=True)
    shutil.rmtree('Results', ignore_errors=True)
    os.symlink(dest, 'Results')
    print('Results/ ->', dest)
else:
    print('not using Drive')
""")

code("""
import os
os.environ['MC_SWEEP_SOURCE'] = 'colab:' + (
    torch.cuda.get_device_name(0).replace(' ', '_') if torch.cuda.is_available() else 'cpu')

if PUSH:
    from google.colab import userdata
    token = userdata.get('GITHUB_TOKEN')
    os.environ['MC_SWEEP_BRANCH'] = PUSH_BRANCH
    os.environ.pop('MC_SWEEP_NO_GIT', None)
    subprocess.run(['git', 'remote', 'set-url', 'origin',
                    f'https://{token}@github.com/Gilzuk/viterbitransformed'], check=True)
    subprocess.run(['git', 'config', 'user.email', 'colab@example.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'colab-runner'], check=True)
    # Fail here rather than hours into the run.
    probe = subprocess.run(['git', 'push', '--dry-run', 'origin', f'HEAD:{PUSH_BRANCH}'],
                           capture_output=True, text=True)
    print('push check:', 'OK' if probe.returncode == 0 else 'FAILED\\n' + probe.stderr)
else:
    os.environ['MC_SWEEP_NO_GIT'] = '1'
    print('MC_SWEEP_NO_GIT=1 (no commits or pushes)')
print('source label:', os.environ['MC_SWEEP_SOURCE'])
""")

md("""
## 6. Run

Confirm from the first lines that it **resumed**: `already in CSV` for finished points, and
`training already complete -- skipping to evaluation` or `found checkpoint with N reps` for
the one in progress. If it starts a finished point from scratch, stop -- the clone did not
get the state.

Time the first repetition and compare it against the CPU container (~48 min/rep for
Transformer at snr=7). That is the number that decides whether the GPU makes the remaining
points reachable.
""")

code("""
import subprocess, sys
# -u so progress streams into the cell instead of sitting in a buffer.
subprocess.run([sys.executable, '-u', 'run_mc_sweep.py', MODEL])
""")

md("""
## 7. Results so far
""")

code("""
import csv
rows = list(csv.DictReader(open('Results/metrics/mc_sweep_validation.csv')))
for m in ['ClassicViterbi', 'Transformer', 'ViterbiNet']:
    pts = sorted((int(r['snr']), float(r['ser_mean']), r.get('source', '')) for r in rows if r['model'] == m)
    print(f'\\n{m}:')
    for snr, ser, src in pts:
        print(f'   snr={snr:2d}  ser={ser:.3e}  source={src}')
""")

nb['cells'] = cells
nb.metadata.update({
    'accelerator': 'GPU',
    'colab': {'provenance': [], 'gpuType': 'T4'},
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
    'language_info': {'name': 'python'},
})

nbf.validate(nb)
with open('run_mc_sweep_colab.ipynb', 'w') as f:
    nbf.write(nb, f)
print(f'wrote run_mc_sweep_colab.ipynb ({len(cells)} cells), validated OK')
