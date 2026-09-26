import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
rows = list(csv.DictReader(open('Results/metrics/qpsk_sweep.csv')))
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
S = [('classic_csi', 'ClassicViterbi, perfect CSI (bound)', INK2, 'o', '--'),
     ('classic_ls', 'ClassicViterbi, pilot LS + decision-directed tracking (7 params est.)', '#2a78d6', 'o', '-'),
     ('tied@5', 'Learned tied-tap affine, 7 params, 5 steps/word', '#1baf7a', 's', '-'),
     ('tied@10', 'Learned tied-tap affine, 7 params, 10 steps/word', '#008300', 's', '-'),
     ('affine@10', 'Learned per-class affine, 192 params, 10 steps/word', '#eb6834', 'D', '-'),
     ('mlp@10', 'ViterbiNet-style MLP, 9,934 params, 10 steps/word', '#e34948', '^', '-')]
fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor=SURF); ax.set_facecolor(SURF)
for key, label, c, mk, ls in S:
    pts = sorted((int(r['es_n0_db']), int(r['symbol_errors']), int(r['symbols'])) for r in rows if r['receiver'] == key)
    xs = [p[0] for p in pts if p[1] > 0]; ys = [p[1] / p[2] for p in pts if p[1] > 0]
    ax.plot(xs, ys, ls, color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, label=label, zorder=3)
    cens = [(p[0], 3 / p[2]) for p in pts if p[1] == 0]
    if cens:
        ax.plot(*zip(*cens), 'v', color=c, ms=9, mfc='none', mew=1.4, zorder=4)
ax.set_yscale('log'); ax.set_ylim(5e-6, 1)
ax.set_xlabel('Es/N0 (dB)', color=INK2); ax.set_ylabel('Symbol error rate (log)', color=INK2)
ax.set_title('QPSK, 3-tap complex fading ISI channel (AR(1) rho=0.99 per word), 1 pilot word of 120 symbols',
             color=INK, loc='left', fontsize=11.5, pad=10)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
leg = ax.legend(frameon=False, fontsize=9, loc='lower left')
for t in leg.get_texts(): t.set_color(INK)
fig.text(0.01, 0.005, '64 frames x 24 data words x 120 symbols = 184k symbols per point; hollow triangle = 0 errors (rule-of-three bound). '
         'All receivers use the same 16-state Viterbi with traceback.', color=MUTED, fontsize=8)
fig.savefig('Results/figures/qpsk_sweep.png', dpi=150, facecolor=SURF, bbox_inches='tight'); print('saved')
