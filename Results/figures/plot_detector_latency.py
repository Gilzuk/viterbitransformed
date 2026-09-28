"""Per-detector latency on one 136-sample word, single CPU thread
(Results/metrics/detector_latency_symbol_budget.csv, from experiments/all_latency.py).

Left: detection time per word split into NN forward (state priors) and Viterbi
trellis. Right: one online-adaptation step per word. All at ~7k parameters
except the small ViterbiNet variants.
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, '..', 'metrics', 'detector_latency_symbol_budget.csv'))))
OUT = os.path.join(HERE, 'detector_latency.png')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
BLUE, GREY, ORANGE = '#2a78d6', '#c9c8c2', '#eb6834'
NAMES = {'VNet_affine': 'VNet affine', 'VNet_4': 'VNet 4 hidden', 'ViterbiNet_100-58': 'ViterbiNet 100-58',
         'TransformerV2': 'TransformerV2', 'ViterbiTransformerV4': 'ViterbiTransformerV4',
         'ViT_overlap': 'ViT overlap', 'Mamba2': 'Mamba2'}
rows.sort(key=lambda r: float(r['detect_ms_per_word']))
labels = [f"{NAMES.get(r['model'], r['model'])} ({int(r['params']):,})" for r in rows]
nn = [float(r['nn_forward_ms_per_word']) for r in rows]
tr = [float(r['trellis_ms_per_word']) for r in rows]
on = [float(r['online_iter_ms_per_word']) for r in rows]
y = range(len(rows))

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF, gridspec_kw={'wspace': 0.08}, sharey=True)
a.barh(y, tr, color=GREY, label='Viterbi trellis', zorder=3)
a.barh(y, nn, left=tr, color=BLUE, label='NN forward (state priors)', zorder=3)
for i, (t, n, r) in enumerate(zip(tr, nn, rows)):
    a.text(t + n + 0.4, i, f"{t + n:.1f} ms  ({float(r['detect_us_per_symbol']):.0f} us/symbol)", va='center',
           fontsize=8.5, color=INK2)
a.set_xlim(0, max(t + n for t, n in zip(tr, nn)) * 1.45)
a.set_title('Detection per word = trellis + NN forward', color=INK, loc='left', fontsize=11.5)
a.set_xlabel('Milliseconds per 136-sample word', color=INK2)
a.legend(frameon=False, fontsize=8.5, loc='lower right')

b.barh(y, on, color=ORANGE, zorder=3)
for i, o in enumerate(on):
    b.text(o * 1.08, i, f'{o:.2f} ms', va='center', fontsize=8.5, color=INK2)
b.set_xscale('log')
b.set_xlim(min(on) * 0.6, max(on) * 4)
b.set_title('One online-adaptation step per word (log)', color=INK, loc='left', fontsize=11.5)
b.set_xlabel('Milliseconds per step (log)', color=INK2)

a.set_yticks(list(y)); a.set_yticklabels(labels, color=INK)
for ax in (a, b):
    ax.set_facecolor(SURF)
    ax.grid(True, axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
fig.suptitle('Detector latency, single CPU thread (the trellis dominates; sequence models add NN and adaptation cost)',
             color=INK, x=0.01, ha='left', fontsize=12.5, y=1.02)
fig.text(0.01, -0.06, 'Median of repeated runs on random weights; measured under load (load average about 8 on 4 cores), '
         'so absolute times are pessimistic. Script: experiments/all_latency.py.', color=MUTED, fontsize=8.5)
fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved', OUT)
