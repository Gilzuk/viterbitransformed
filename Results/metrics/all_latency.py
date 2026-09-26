import os, sys, time, statistics
os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,'/home/user/viterbitransformed')
import torch; torch.set_num_threads(1)
from Code.models import ViterbiNetMLP, ECC_TransformerV2, ViterbiTransformerV3, ViterbiTransformerV4, ViT1D
from Code.detector import Detector
T=136
def med(fn,reps):
    for _ in range(3): fn()
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1e3
models={
 'VNet affine (32)': lambda: ViterbiNetMLP((),16),
 'VNet 4 (88)': lambda: ViterbiNetMLP((4,),16),
 'ViterbiNet 100-58 (7,002)': lambda: ViterbiNetMLP((100,58),16),
 'TransformerV2 (6,960)': lambda: ECC_TransformerV2(4,16,2,2,16),
 'ViterbiTransformerV4 (6,880)': lambda: ViterbiTransformerV4(4,16,2,16),
 'ViT overlap (6,976)': lambda: ViT1D(4,16,2,2,16,mlp_ratio=4,overlapping=True),
}
y=torch.randn(1,T); lab=torch.randint(0,16,(T,))
print(f'{"model":30s} {"nn_fwd_ms":>9s} {"trellis_ms":>10s} {"detect_ms":>9s} {"online_it_ms":>12s}')
for name,mk in models.items():
    m=mk(); det=Detector(m,'ModelBased'); opt=torch.optim.Adam(m.parameters(),1e-3); lf=torch.nn.CrossEntropyLoss()
    def fwd():
        with torch.no_grad(): det(y,'train')
    def dec():
        with torch.no_grad(): det(y,'val')
    def it():
        opt.zero_grad(); lf(det(y,'train').reshape(-1,16),lab).backward(); opt.step()
    f=med(fwd,100); d=med(dec,20); o=med(it,50)
    print(f'{name:30s} {f:9.3f} {d-f:10.2f} {d:9.2f} {o:12.3f}', flush=True)
