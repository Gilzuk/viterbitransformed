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
  Read the token from Secrets only -- never type or print it in a cell, since cell output
  is saved into the notebook and a token has leaked that way before.
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

    # Keep the token OUT of the remote URL. git echoes the remote back in its
    # error messages ("could not read Password for 'https://<token>@github...'"),
    # which writes the secret straight into saved notebook output -- that is
    # how a token has already been leaked once here. A credentials file keeps
    # the remote clean, so git has nothing sensitive to print.
    os.makedirs('/root', exist_ok=True)
    with open('/root/.git-credentials', 'w') as f:
        f.write(f'https://x-access-token:{token}@github.com\\n')
    os.chmod('/root/.git-credentials', 0o600)
    subprocess.run(['git', 'config', 'credential.helper', 'store'], check=True)
    subprocess.run(['git', 'remote', 'set-url', 'origin',
                    'https://github.com/Gilzuk/viterbitransformed'], check=True)
    subprocess.run(['git', 'config', 'user.email', 'colab@example.com'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'colab-runner'], check=True)

    os.environ['MC_SWEEP_BRANCH'] = PUSH_BRANCH
    os.environ.pop('MC_SWEEP_NO_GIT', None)

    # Fail here rather than hours into the run. Scrub the token from whatever
    # git says, in case any path still surfaces it.
    probe = subprocess.run(['git', 'push', '--dry-run', 'origin', f'HEAD:{PUSH_BRANCH}'],
                           capture_output=True, text=True)
    msg = (probe.stderr or '').replace(token, '<redacted>')
    print('push check:', 'OK -> will push to ' + PUSH_BRANCH if probe.returncode == 0
          else 'FAILED\\n' + msg)
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
import os, subprocess, time

# Run detached, writing to a log file, instead of streaming into this cell.
#
# Streaming from a notebook cell is unreliable here: a subprocess inherits the
# KERNEL's stdout, which is not the cell, so a plain subprocess.run() prints
# into the kernel log and the cell just sits blank -- indistinguishable from a
# hung run. Piping it back has its own trap: iterating proc.stdout uses a
# read-ahead buffer, so lines arrive late or not at all.
#
# A log file sidesteps both, survives this cell being interrupted, and lets
# the next cell show progress on demand -- which suits a run measured in hours
# far better than one cell blocking the whole notebook.
LOG = '/content/sweep.log'

RESTART = False   # set True to kill a running sweep and start it fresh

running = subprocess.run(['pgrep', '-f', 'run_mc_sweep.py'],
                         capture_output=True, text=True).stdout.split()

# An earlier attempt may still be running WITHOUT writing to LOG (e.g. one
# started by a cell that let the child inherit the kernel's stdout). That
# looks identical to a healthy run whose log is merely slow, so say so rather
# than tailing an empty file forever.
if running and not os.path.exists(LOG):
    print(f'A sweep is running (pids {running}) but {LOG} does not exist, so it was not\\n'
          f'started by this cell and its output is not being captured anywhere you can\\n'
          f'see. Set RESTART = True above and re-run to replace it -- nothing is lost,\\n'
          f'it resumes from the banked repetitions.')
elif running and not RESTART:
    print('already running, pids:', running)
else:
    if running and RESTART:
        subprocess.run(['pkill', '-f', 'run_mc_sweep.py'])
        time.sleep(3)
        print('stopped', running)
    subprocess.Popen(f'nohup python -u run_mc_sweep.py {MODEL} > {LOG} 2>&1 &',
                     shell=True)
    print('launched', MODEL)

time.sleep(25)
try:
    tail = open(LOG).read()[-3000:]
except FileNotFoundError:
    tail = ''
print(tail or '(no log yet -- re-run this cell in a few seconds)')
""")

md("""
### Progress
Re-run this cell whenever you want an update -- the sweep keeps going regardless. A crash
shows up here as a traceback; a healthy run shows `[resume]`/`[plan]` lines and a climbing
repetition count.
""")

code("""
import subprocess
print(subprocess.run(['tail', '-25', '/content/sweep.log'],
                     capture_output=True, text=True).stdout)
print('--- banked so far ---')
import glob, json
for f in sorted(glob.glob('Results/metrics/.mc_sweep_checkpoints/*.json')):
    d = json.load(open(f))
    n = f.split('/')[-1]
    print(' ', n, '->', f"{len(d['per_rep_means'])} reps" if 'per_rep_means' in d else d)
alive = subprocess.run(['pgrep', '-f', 'run_mc_sweep.py'],
                       capture_output=True, text=True).stdout.split()
print('\\nprocess:', 'running ' + str(alive) if alive else 'NOT running (finished or died)')
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
