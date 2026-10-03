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
    # --global, not local to this one clone: section 6b below clones a SECOND
    # repo to run a model in parallel, and a bare `git config key value` (no
    # --global) writes only to THIS repo's .git/config. Verified on a clean
    # HOME with no pre-existing identity: the second clone's first commit hit
    # "fatal: unable to auto-detect email address" and its push would have no
    # credential helper at all -- global config is what makes both apply
    # identically to any repo on this VM, present or cloned later.
    subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
    subprocess.run(['git', 'remote', 'set-url', 'origin',
                    'https://github.com/Gilzuk/viterbitransformed'], check=True)
    subprocess.run(['git', 'config', '--global', 'user.email', 'colab@example.com'], check=True)
    subprocess.run(['git', 'config', '--global', 'user.name', 'colab-runner'], check=True)

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
import torch

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
PIDFILE = LOG + '.pid'   # tracks THIS launch specifically -- see section 6b,
                         # where a second model's process also matches
                         # "run_mc_sweep.py" in a plain pgrep and would
                         # otherwise be indistinguishable from this one.

RESTART = False   # set True to kill a running sweep and start it fresh

# Configure the environment HERE rather than relying on an earlier cell having
# been run. Skipping the setup cell leaves MC_SWEEP_NO_GIT unset, so the sweep
# commits every repetition and tries to push; with no credentials the push
# raises, the point is abandoned, main() walks through the remaining SNRs
# failing each one, and the run exits -- looking exactly like it "just stopped".
if 'PUSH' not in globals():
    PUSH = False
if not PUSH:
    os.environ['MC_SWEEP_NO_GIT'] = '1'
os.environ.setdefault('MC_SWEEP_SOURCE', 'colab:' + (
    torch.cuda.get_device_name(0).replace(' ', '_')
    if torch.cuda.is_available() else 'cpu'))
print('MC_SWEEP_NO_GIT =', os.environ.get('MC_SWEEP_NO_GIT'),
      '| MC_SWEEP_SOURCE =', os.environ['MC_SWEEP_SOURCE'])

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False

tracked = None
if os.path.exists(PIDFILE):
    try:
        tracked = int(open(PIDFILE).read().strip())
    except ValueError:
        pass

if tracked and pid_alive(tracked) and RESTART:
    subprocess.run(['kill', str(tracked)]); time.sleep(3)
    print('stopped pid', tracked)
    tracked = None

if tracked and pid_alive(tracked):
    print('already running, pid', tracked)
else:
    # A process from an untracked launch (no pidfile -- e.g. an older cell
    # version already run this session) looks identical to a hung run. Say so
    # rather than launching a duplicate or tailing a log nothing writes to.
    stray = subprocess.run(['pgrep', '-f', f'run_mc_sweep.py {MODEL}'],
                           capture_output=True, text=True).stdout.split()
    if stray and not os.path.exists(LOG):
        print(f'A {MODEL} sweep is running (pid {stray}) but was not started by this cell '
              f'-- set RESTART = True and re-run to replace it. Nothing is lost.')
    else:
        subprocess.Popen(
            f'nohup python -u run_mc_sweep.py {MODEL} > {LOG} 2>&1 & echo $! > {PIDFILE}',
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
try:
    pid = int(open('/content/sweep.log.pid').read().strip())
    os.kill(pid, 0)
    alive = True
except (FileNotFoundError, ValueError, ProcessLookupError, OSError):
    alive = False
print('\\nprocess:', f'running (pid {pid})' if alive else 'NOT running (finished or died)')
""")

md("""
## 6b. (Optional) Run a second model in parallel, e.g. ViterbiNet

Runs alongside `MODEL` above, in the **same** Colab session, sharing its one GPU. This does
**not** touch the process launched in section 6 -- it is a separate clone, a separate
process, a separate log.

A second, separate clone directory is required: two `run_mc_sweep.py` processes writing to
the same local `mc_sweep_validation.csv` / checkpoint files would corrupt them, for the same
reason the CPU-container runs use one git worktree per model rather than one shared
directory. If both push (`PUSH = True` above), `run_mc_sweep.py`'s own retry logic already
handles two runners appending to the same branch -- it fetches and rebases on a
non-fast-forward push, which is exactly how the CPU container's parallel worktrees stay
consistent with each other.

Sharing one GPU between two models means neither gets the full ~2.5 min/rep Transformer saw
running alone -- expect roughly half the throughput each.
""")

code("""
SECOND_MODEL = 'ViterbiNet'   # a model nobody else is running
SECOND_DIR = '/content/viterbitransformed_2'
SECOND_LOG = '/content/sweep_2.log'
SECOND_PIDFILE = SECOND_LOG + '.pid'
""")

code(f"""
import os, subprocess
if not os.path.isdir(SECOND_DIR):
    subprocess.run(['git', 'clone', '-b', BRANCH, REPO, SECOND_DIR], check=True)
subprocess.run(['git', '-C', SECOND_DIR, 'pull', '--rebase', 'origin', BRANCH], check=False)
print(subprocess.run(['git', '-C', SECOND_DIR, 'log', '--oneline', '-1'],
                     capture_output=True, text=True).stdout)

import csv
csv_path = os.path.join(SECOND_DIR, 'Results/metrics/mc_sweep_validation.csv')
rows = list(csv.DictReader(open(csv_path)))
have = sorted(int(r['snr']) for r in rows if r['model'] == SECOND_MODEL)
print(f'{{SECOND_MODEL}}: {{len(have)}} points already done -> {{have}}')
print('missing ->', [s for s in range(18) if s not in have])
""")

code("""
import os, subprocess, time

RESTART_SECOND = False   # set True to kill and restart just this second run

# git credentials (~/.git-credentials) are set up globally by section 5 in the
# FIRST clone's directory -- git itself does not care which clone reads them,
# so PUSH here reuses that same login rather than needing its own token cell.
env = os.environ.copy()
if 'PUSH' not in globals():
    PUSH = False
if not PUSH:
    env['MC_SWEEP_NO_GIT'] = '1'
else:
    env['MC_SWEEP_BRANCH'] = PUSH_BRANCH
    env.pop('MC_SWEEP_NO_GIT', None)
env['MC_SWEEP_SOURCE'] = 'colab:' + (
    torch.cuda.get_device_name(0).replace(' ', '_') if torch.cuda.is_available() else 'cpu')
print('MC_SWEEP_NO_GIT =', env.get('MC_SWEEP_NO_GIT'), '| MC_SWEEP_SOURCE =', env['MC_SWEEP_SOURCE'])

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False

tracked = None
if os.path.exists(SECOND_PIDFILE):
    try:
        tracked = int(open(SECOND_PIDFILE).read().strip())
    except ValueError:
        pass

if tracked and pid_alive(tracked) and RESTART_SECOND:
    subprocess.run(['kill', str(tracked)]); time.sleep(3)
    print('stopped pid', tracked)
    tracked = None

if tracked and pid_alive(tracked):
    print('already running, pid', tracked)
else:
    stray = subprocess.run(['pgrep', '-f', f'run_mc_sweep.py {SECOND_MODEL}'],
                           capture_output=True, text=True).stdout.split()
    if stray and not os.path.exists(SECOND_LOG):
        print(f'A {SECOND_MODEL} sweep is running (pid {stray}) but was not started by this '
              f'cell -- set RESTART_SECOND = True and re-run to replace it. Nothing is lost.')
    else:
        subprocess.Popen(
            f'cd {SECOND_DIR} && nohup python -u run_mc_sweep.py {SECOND_MODEL} '
            f'> {SECOND_LOG} 2>&1 & echo $! > {SECOND_PIDFILE}',
            shell=True, env=env)
        print('launched', SECOND_MODEL, 'in', SECOND_DIR)

time.sleep(25)
try:
    tail = open(SECOND_LOG).read()[-3000:]
except FileNotFoundError:
    tail = ''
print(tail or '(no log yet -- re-run this cell in a few seconds)')
""")

md("""
### Progress (second model)
""")

code("""
import subprocess, glob, json, os
print(subprocess.run(['tail', '-25', SECOND_LOG], capture_output=True, text=True).stdout)
print('--- banked so far ---')
for f in sorted(glob.glob(os.path.join(SECOND_DIR, 'Results/metrics/.mc_sweep_checkpoints/*.json'))):
    d = json.load(open(f))
    n = f.split('/')[-1]
    print(' ', n, '->', f"{len(d['per_rep_means'])} reps" if 'per_rep_means' in d else d)
try:
    pid = int(open(SECOND_PIDFILE).read().strip())
    os.kill(pid, 0)
    alive = True
except (FileNotFoundError, ValueError, ProcessLookupError, OSError):
    alive = False
print('\\nprocess:', f'running (pid {pid})' if alive else 'NOT running (finished or died)')
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
