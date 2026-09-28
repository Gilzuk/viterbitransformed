"""Latency benchmark for ViterbiNet MLP topologies, single CPU thread.

Per topology, on one received word (136 samples, 16 states):
  nn_fwd     NN forward pass only (state priors)
  detect     full detection: NN forward + 136-step Viterbi trellis
  online_it  one online-adaptation iteration (forward + loss + backward + Adam step)
Median of many repetitions; weights are random (timing does not depend on them).
"""
import os, sys, time, statistics
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch
torch.set_num_threads(1)
from Code.models import ViterbiNetMLP
from Code.detector import Detector

T, N_STATES = 136, 16
SPECS = {'affine': (), '4': (4,), '8': (8,), '16': (16,), '32': (32,), '64': (64,),
         '8-8': (8, 8), '16-16': (16, 16), '32-16': (32, 16), '100-58': (100, 58)}


def med_time(fn, reps):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6  # microseconds


def macs(hidden):
    dims = [1, *hidden, N_STATES]
    return sum(a * b for a, b in zip(dims, dims[1:]))


y = torch.randn(1, T)
labels = torch.randint(0, N_STATES, (T,))
print(f'{"topology":8s} {"params":>6s} {"MACs/sym":>8s} {"nn_fwd_us":>10s} {"detect_us":>10s} {"online_it_us":>12s}')
rows = []
for name, hidden in SPECS.items():
    m = ViterbiNetMLP(hidden, N_STATES)
    det = Detector(m, 'ModelBased')
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def fwd():
        with torch.no_grad():
            m(y)

    def detect():
        with torch.no_grad():
            det(y, 'val')

    def online_it():
        opt.zero_grad()
        loss = loss_fn(det(y, 'train').reshape(-1, N_STATES), labels)
        loss.backward(); opt.step()

    params = sum(p.numel() for p in m.parameters())
    r = (name, params, macs(hidden), med_time(fwd, 300), med_time(detect, 30), med_time(online_it, 200))
    rows.append(r)
    print(f'{r[0]:8s} {r[1]:6d} {r[2]:8d} {r[3]:10.1f} {r[4]:10.1f} {r[5]:12.1f}', flush=True)

out = os.path.join(REPO, 'Results', 'metrics', 'vnet_topology_latency.csv')
with open(out, 'w') as f:
    f.write('topology,params,macs_per_symbol,nn_forward_us_per_word,detect_us_per_word,online_iter_us_per_word\n')
    for r in rows:
        f.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{r[5]:.1f}\n')
print('saved', out)
