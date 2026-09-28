"""ViterbiNet MLP topology study at SNR 7 dB (experiments/vnet_study.py,
experiments/vnet_online_extra.py, experiments/vnet_latency.py).

A  SER vs parameter count, default budget (25 offline minibatches, 200 online steps/word)
B  SER vs online adaptation steps per word (25 offline minibatches)
C  SER vs offline training minibatches (200 online steps/word)
D  latency per word vs parameter count: NN forward and one online step (single CPU thread)
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
MET = os.path.join(HERE, '..', 'metrics')
OUT = os.path.join(HERE, 'vnet_topology_snr7.png')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
BLUE, GREEN, ORANGE = '#2a78d6', '#008300', '#eb6834'

st = [dict(r, params=int(r['params']), mb=int(r['train_minibatches']), on=int(r['online_iters']),
           ser=float(r['ser_mean']), ci=float(r['ser_ci95']))
      for r in csv.DictReader(open(os.path.join(MET, 'vnet_topology_study_snr7.csv')))]
lat = [dict(r, params=int(r['params'])) for r in csv.DictReader(open(os.path.join(MET, 'vnet_topology_latency.csv')))]
ref = {r['model']: float(r['ser_mean']) for r in csv.DictReader(open(os.path.join(MET, 'mc_sweep_validation.csv')))
       if r['snr'] == '7' and r['model'] in ('ClassicViterbi', 'ClassicViterbi_LS')}

plt.rcParams.update({'font.size': 9.5})
fig, axs = plt.subplots(2, 2, figsize=(13, 9.2), facecolor=SURF)
(a, b), (c, d) = axs


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, loc='left', fontsize=11.5, pad=8)
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def refs(ax):
    if 'ClassicViterbi' in ref:
        ax.axhline(ref['ClassicViterbi'], color=INK2, ls='--', lw=1)
        ax.text(ax.get_xlim()[1], ref['ClassicViterbi'], ' perfect CSI', color=INK2, va='top', ha='right', fontsize=8)
    if 'ClassicViterbi_LS' in ref:
        ax.axhline(ref['ClassicViterbi_LS'], color='#e34948', ls=':', lw=1.2)
        ax.text(ax.get_xlim()[1], ref['ClassicViterbi_LS'], ' classical LS, no CSI', color='#e34948', va='bottom', ha='right', fontsize=8)


# A: SER vs params at the default budget
pts = sorted((r['params'], r['ser'], r['ci'], r['topology']) for r in st if r['mb'] == 25 and r['on'] == 200)
one = [p for p in pts if '-' not in p[3]]; two = [p for p in pts if '-' in p[3]]
for grp, col, lab in ((one, BLUE, 'one hidden layer (or affine)'), (two, ORANGE, 'two hidden layers')):
    a.errorbar([p[0] for p in grp], [p[1] for p in grp], yerr=[p[2] for p in grp], color=col, lw=0, marker='o', ms=7,
               elinewidth=1, mec=SURF, mew=1.2, label=lab, zorder=3)
    for p in grp:
        a.annotate(p[3], (p[0], p[1]), xytext=(4, 5), textcoords='offset points', fontsize=7.5, color=INK2)
on5 = [r for r in st if r['mb'] == 25 and r['on'] == 5]
a.errorbar([r['params'] for r in on5], [r['ser'] for r in on5], yerr=[r['ci'] for r in on5], color=GREEN, lw=0,
           marker='D', ms=7, elinewidth=1, mec=SURF, mew=1.2, label='same, 5 online steps/word', zorder=4)
a.set_xscale('log')
a.set_xlim(20, 12000)
refs(a)
style(a, 'A  SER vs size (25 offline minibatches, 200 online steps)', 'Trainable parameters (log)', 'SER at 7 dB')
a.legend(frameon=False, fontsize=8.5, loc='upper left')

# B: SER vs online iterations; C: SER vs offline minibatches
for ax, key, fixed, xs_label, title in ((b, 'on', ('mb', 25), 'Online adaptation steps per word',
                                         'B  SER vs online adaptation (25 offline minibatches)'),
                                        (c, 'mb', ('on', 200), 'Offline training minibatches',
                                         'C  SER vs offline training (200 online steps/word)')):
    for topo, col, lab in (('affine', GREEN, 'affine, 32 params'), ('100-58', BLUE, 'ViterbiNet 100-58, 7,002 params')):
        p = sorted((r[key], r['ser'], r['ci']) for r in st if r['topology'] == topo and r[fixed[0]] == fixed[1])
        xs = [max(x, 0.5) for x, _, _ in p]  # plot 0 online steps at 0.5 on the log axis
        ax.errorbar(xs, [q[1] for q in p], yerr=[q[2] for q in p], color=col, lw=2, marker='o', ms=6,
                    elinewidth=1, mec=SURF, mew=1.2, label=lab, zorder=3)
    ticks = sorted({max(r[key], 0.5) for r in st if r['topology'] in ('affine', '100-58') and r[fixed[0]] == fixed[1]})
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(['0' if t == 0.5 else f'{t:g}' for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(ticks[0] * 0.7, ticks[-1] * 1.4)
    refs(ax)
    style(ax, title, xs_label, 'SER at 7 dB')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')

# D: latency vs params
lat.sort(key=lambda r: r['params'])
d.plot([r['params'] for r in lat], [float(r['nn_forward_us_per_word']) for r in lat], color=BLUE, marker='o', lw=2,
       ms=6, mec=SURF, label='NN forward (state priors)')
d.plot([r['params'] for r in lat], [float(r['online_iter_us_per_word']) for r in lat], color=ORANGE, marker='s', lw=2,
       ms=6, mec=SURF, label='one online adaptation step')
trel = [float(r['detect_us_per_word']) - float(r['nn_forward_us_per_word']) for r in lat]
d.axhline(sum(trel) / len(trel), color=INK2, ls='--', lw=1)
d.text(lat[0]['params'], sum(trel) / len(trel), f'  Viterbi trellis, about {sum(trel) / len(trel) / 1000:.1f} ms/word (any topology)',
       color=INK2, va='bottom', fontsize=8)
d.set_xscale('log'); d.set_yscale('log')
style(d, 'D  Latency per 136-sample word, single CPU thread', 'Trainable parameters (log)', 'Microseconds per word (log)')
d.legend(frameon=False, fontsize=8.5, loc='center right')

fig.suptitle('ViterbiNet MLP topology study, BPSK COST2100 memory 4, SNR 7 dB', color=INK, x=0.01, ha='left',
             fontsize=13, y=1.0)
fig.text(0.01, -0.01, '20 reps x 2,000 bits per point (bars = 95% CI); reference lines are the full-protocol SNR-7 sweep points. '
         'Scripts: experiments/vnet_study.py, vnet_online_extra.py, vnet_latency.py.', color=MUTED, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved', OUT)
