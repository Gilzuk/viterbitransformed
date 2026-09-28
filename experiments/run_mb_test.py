import os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, REPO)
os.environ['MC_SWEEP_NO_GIT'] = '1'
os.chdir(REPO)
import torch
torch.set_num_threads(1)
from run_mc_sweep import run_point

trainer_model, minibatches, snr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
detector_method = sys.argv[4] if len(sys.argv) > 4 else 'ModelBased'
suffix = '' if detector_method == 'ModelBased' else f'_{detector_method}'
no_online = os.environ.get('NO_ONLINE') == '1'
if no_online:
    suffix += '_noonline'
kw = {'train_minibatch_num': minibatches}
online_iters = os.environ.get('ONLINE_ITERS')
if online_iters is not None:
    kw['self_supervised_iterations'] = int(online_iters)
    suffix += f'_on{online_iters}'
if no_online:
    kw['self_supervised'] = False
t0 = time.time()
print(f'=== {trainer_model} {detector_method} snr={snr} minibatches={minibatches} ===', flush=True)
row = run_point(
    model_name=f'{trainer_model}_mb{minibatches}{suffix}_test',
    detector_method=detector_method, snr=snr, min_reps=20, max_bits=40000, step=1,
    trainer_model_name=trainer_model,
    trainer_kwargs=kw,
)
print('=== RESULT ===', flush=True)
print(row, flush=True)
print(f'Total wall time: {time.time()-t0:.1f}s', flush=True)
