"""QPSK receivers vs normalized Doppler on the fast-fading channel
(Results/metrics/qpsk_fast_fading.csv, written by run_qpsk_fast_fading.py)."""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [r for r in csv.DictReader(open(os.path.join(HERE, '..', 'metrics', 'qpsk_fast_fading.csv')))
        if r['receiver'] != 'receiver']
OUT = os.path.join(HERE, 'qpsk_fast_fading.png')
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
S = [('classic_csi', 'ClassicViterbi, true per-sample taps (genie)', INK2, 'o', '--'),
     ('classic_ls', 'ClassicViterbi, LS per word (pilot + own decisions)', '#2a78d6', 'o', '-'),
     ('tied@5', 'Trellis + structured affine, 7 params', '#008300', 's', '-'),
     ('tied_mlp@5', 'Trellis + structured MLP, 88 params', '#1baf7a', 'D', '-'),
     ('le_ls', 'Linear equalizer, LS re-solved per word, 9 samples', '#5b95dd', 'o', ':'),
     ('sym_eq@30', '4-class classifier, equalizer structure, DD-LMS', '#e87ba4', 'P', '-')]
snrs = sorted({int(r['es_n0_db']) for r in rows})
fds = sorted({float(r['doppler']) for r in rows})
T_word = 122  # samples per word (120 symbols + L-1)
fig, axes = plt.subplots(1, len(snrs), figsize=(6.2 * len(snrs), 5.6), facecolor=SURF, sharey=True)
axes = [axes] if len(snrs) == 1 else axes
for ax, snr in zip(axes, snrs):
    ax.set_facecolor(SURF)
    for key, label, c, mk, ls in S:
        pts = sorted((float(r['doppler']), float(r['ser'])) for r in rows
                     if r['receiver'] == key and int(r['es_n0_db']) == snr and float(r['ser']) > 0)
        if pts:
            ax.plot(*zip(*pts), ls, color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, label=label, zorder=3)
    ax.axhline(0.75, color=MUTED, lw=1, ls='--', zorder=1)
    ax.text(fds[0], 0.8, 'random guessing (0.75)', color=MUTED, fontsize=8.5, va='bottom')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(fds))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{f:g}\n({f * T_word:.2g}/word)' for f in fds]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(3e-4, 1.3)
    ax.set_xlabel('Normalized Doppler f_D x T_symbol  (Doppler cycles per word)', color=INK2)
    ax.set_title(f'Es/N0 = {snr} dB', color=INK, loc='left', fontsize=12, pad=8)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8.5)
axes[0].set_ylabel('Symbol error rate (log)', color=INK2)
leg = fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc='upper center',
                 bbox_to_anchor=(0.5, 0.02), ncol=3)
for t in leg.get_texts():
    t.set_color(INK)
fig.suptitle('QPSK, 3-tap complex Rayleigh channel, taps vary sample by sample (Jakes); 1 pilot word per 25',
             color=INK, x=0.01, ha='left', fontsize=12.5)
fig.text(0.01, -0.1, '64 frames x 24 data words x 120 symbols per point; all receivers see the same channel and data. '
         'Block receivers estimate one channel per word, so they fail once the taps rotate appreciably within a word.',
         color=MUTED, fontsize=8.5)
fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved', OUT)
