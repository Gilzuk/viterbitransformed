import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

REPO = '/home/user/viterbitransformed'
OUT = os.path.join(REPO, 'Results/figures/snr7_all_trials.png')

INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
C = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'magenta': '#e87ba4', 'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948'}

# ---- training-budget curves: (minibatches, ser, ci) per model ----
curves = {}
def add(model, mb, ser, ci):
    curves.setdefault(model, {})[mb] = (ser, ci)

for r in csv.DictReader(open(os.path.join(REPO, 'Results/metrics/minibatch_search_snr7.csv'))):
    add(r['model'], int(r['train_minibatches']), float(r['ser_mean']), float(r['ser_ci95']))
# one-off fixed-budget runs (same protocol: 20 reps / 40k bits)
add('ViterbiNet', 250, 0.03008, 0.00104); add('ViterbiNet', 1000, 0.02982, 0.00099)
add('TransformerV2', 25, 0.03910, 0.00100); add('TransformerV2', 250, 0.03656, 0.00065)
add('TransformerV2', 1000, 0.03631, 0.00105)
add('ViT_overlap', 250, 0.03211, 0.00078); add('ViT', 250, 0.10322, 0.00362)

vnet = []
for r in csv.DictReader(open(os.path.join(REPO, 'Results/metrics/vnet_topology_study_snr7.csv'))):
    vnet.append((r['topology'], int(r['params']), int(r['train_minibatches']), int(r['online_iters']),
                 float(r['ser_mean']), float(r['ser_ci95'])))

CLASSIC = 0.02161
SERIES = [  # (key, label, color) -- fixed order
    ('ViterbiNet', 'ViterbiNet MLP (7,002)', C['blue']),
    ('TransformerV2', 'TransformerV2 (6,960)', C['orange']),
    ('ViterbiTransformerV3', 'ViterbiTransformerV3 (7,248)', C['aqua']),
    ('ViterbiTransformerV4', 'ViterbiTransformerV4 (6,880)', C['yellow']),
    ('ViT_overlap', 'ViT overlap (6,976)', C['magenta']),
    ('ViT', 'ViT patch (7,376)', C['violet']),
]

# ---- best-per-variant list for the ranking panel ----
best = []  # (label, ser, ci, family)
def bestof(key):
    return min(curves[key].items(), key=lambda kv: kv[1][0]) if key in curves else None
for key, label, _ in SERIES:
    b = bestof(key)
    if b:
        best.append((f'{label.split(" (")[0]}  @{b[0]} mb', b[1][0], b[1][1],
                     'MLP' if key == 'ViterbiNet' else ('ViT' if key.startswith('ViT_') or key == 'ViT' else 'Transformer')))
for topo, params, mb, on, s, ci in vnet:
    if mb == 25 and on == 200 and topo != '100-58':
        best.append((f'VNet {topo} ({params} params)', s, ci, 'MLP'))
best += [
    ('Transformer original (8 heads, causal)', 0.03862, 0.00111, 'Transformer'),
    ('TransformerV3 wide (101,520 params)', 0.03829, 0.00078, 'Transformer'),
    ('TransformerV2 no trellis (EndToEnd)', 0.05934, 0.00177, 'Transformer'),
    ('TransformerV2 1-sample input', 0.11414, 0.00554, 'Transformer'),
    ('ViT patch, no online updates', 0.03841, 0.00135, 'ViT'),
    ('ClassicViterbi, perfect CSI', CLASSIC, 0.00034, 'Reference'),
]
FAM = {'Reference': INK2, 'MLP': C['blue'], 'Transformer': C['orange'], 'ViT': C['magenta']}

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8.4), facecolor=SURF,
                               gridspec_kw={'width_ratios': [1.15, 1], 'wspace': 0.5})
yt = [0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.12]

# ---- panel A: SER vs training minibatches ----
ax1.set_facecolor(SURF)
for key, label, color in SERIES:
    if key not in curves:
        continue
    pts = sorted(curves[key].items())
    xs = [p[0] for p in pts]; ys = [p[1][0] for p in pts]; es = [p[1][1] for p in pts]
    ax1.errorbar(xs, ys, yerr=es, color=color, lw=2, marker='o', ms=6, capsize=0, elinewidth=1,
                 mec=SURF, mew=1.2, label=label, zorder=3)
aff = [v for v in vnet if v[0] == 'affine' and v[2] == 25 and v[3] == 200]
if aff:
    ax1.errorbar([25], [aff[0][4]], yerr=[aff[0][5]], color=C['green'], marker='D', ms=9, lw=0,
                 elinewidth=1, mec=SURF, mew=1.2, label='VNet affine, 32 params (25 mb)', zorder=4)
ax1.axhline(CLASSIC, color=INK2, lw=1.2, ls='--', zorder=2)
ax1.text(900, CLASSIC * 1.015, 'ClassicViterbi, perfect CSI', color=INK2, va='bottom', ha='right', fontsize=9)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.xaxis.set_major_locator(FixedLocator([7, 13, 25, 50, 100, 250, 1000]))
ax1.xaxis.set_major_formatter(FixedFormatter(['7', '13', '25', '50', '100', '250', '1000']))
ax1.xaxis.set_minor_locator(NullLocator())
ax1.yaxis.set_major_locator(FixedLocator(yt)); ax1.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
ax1.yaxis.set_minor_locator(NullLocator())
ax1.set_xlim(5, 1100); ax1.set_ylim(0.019, 0.13)
ax1.set_xlabel('Offline training minibatches (log)'); ax1.set_ylabel('Symbol error rate (log)')
ax1.set_title('SER vs training budget, SNR 7 dB', color=INK, loc='left', fontsize=12, pad=10)
ax1.grid(True, color=GRID, lw=0.6); ax1.set_axisbelow(True)
for s in ('top', 'right'):
    ax1.spines[s].set_visible(False)
leg = ax1.legend(frameon=False, fontsize=8.8, loc='center right', bbox_to_anchor=(1.0, 0.56))
for t in leg.get_texts():
    t.set_color(INK)

# ---- panel B: best result per variant, ranked ----
ax2.set_facecolor(SURF)
best.sort(key=lambda b: b[1], reverse=True)
for i, (label, s, ci, fam) in enumerate(best):
    ax2.plot([s - ci, s + ci], [i, i], color=FAM[fam], lw=1.4, alpha=0.6, zorder=2)
    ax2.plot(s, i, 'o', color=FAM[fam], ms=7, mec=SURF, mew=1.2, zorder=3)
    ax2.text(0.135, i, f'{s:.4f}', va='center', ha='left', color=INK2, fontsize=8.5)
ax2.set_yticks(range(len(best))); ax2.set_yticklabels([b[0] for b in best], fontsize=8.8, color=INK)
ax2.set_xscale('log')
ax2.xaxis.set_major_locator(FixedLocator(yt)); ax2.xaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
ax2.xaxis.set_minor_locator(NullLocator())
ax2.set_xlim(0.019, 0.13)
ax2.axvline(CLASSIC, color=INK2, lw=1, ls='--', zorder=1)
ax2.set_xlabel('Symbol error rate (log), dot = mean, bar = 95% CI')
ax2.set_title('Best result per variant (lower is better)', color=INK, loc='left', fontsize=12, pad=10)
ax2.grid(True, axis='x', color=GRID, lw=0.6); ax2.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis='y', length=0)
handles = [plt.Line2D([], [], marker='o', lw=0, color=FAM[f], ms=7, label=f) for f in FAM]
leg2 = ax2.legend(handles=handles, frameon=False, fontsize=8.8, loc='upper right')
for t in leg2.get_texts():
    t.set_color(INK)

fig.text(0.01, 0.01, 'All trials: 20 reps x 2,000 bits = 40k bits per point; COST2100 fading channel, memory 4; '
         'online adaptation on during evaluation unless noted. Searches for ViT overlap / V4 / MLP sizes still running.',
         color=MUTED, fontsize=8.5)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved', OUT, len(best), 'variants,', sum(len(v) for v in curves.values()) + len(vnet), 'trial points')
