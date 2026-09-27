"""SER vs normalized Doppler at SNR 7 dB on the fast-fading channel
(Results/metrics/fast_fading.csv, written by run_fast_fading.py)."""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, '..', 'metrics', 'fast_fading.csv')
OUT = os.path.join(HERE, 'fast_fading_snr7.png')

INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
SERIES = [  # key, label, color, dashed
    ('ClassicViterbi_genie', 'Classic Viterbi, true taps every sample (genie)', INK2, True),
    ('ClassicViterbi_LS', 'Classic Viterbi, LS on last word (no CSI)', '#e34948', False),
    ('ClassicViterbi_PSP', 'Classic Viterbi, per-survivor LMS tracking (no CSI)', '#eda100', False),
    ('ClassicViterbi_RLS', 'Classic Viterbi, LS with forgetting 0.8 (no CSI)', '#4a3aa7', False),
    ('ViterbiNet_on5', 'ViterbiNet MLP, 5 online steps (7,002 params)', '#2a78d6', False),
    ('VNet_affine', 'VNet affine (32 params)', '#1baf7a', False),
]

rows = list(csv.DictReader(open(CSV)))
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURF)
ax.set_facecolor(SURF)
fds = sorted({float(r['doppler']) for r in rows})
for key, label, color, dashed in SERIES:
    pts = sorted((float(r['doppler']), float(r['ser_mean']), float(r['ser_ci95']))
                 for r in rows if r['model'] == key)
    if not pts:
        continue
    xs, ys, es = zip(*pts)
    ax.errorbar(xs, ys, yerr=es, color=color, lw=2, ls='--' if dashed else '-', marker='o', ms=6,
                capsize=0, elinewidth=1, mec=SURF, mew=1.2, label=label, zorder=3)
ax.set_xscale('log'); ax.set_yscale('log')
ax.xaxis.set_major_locator(FixedLocator(fds))
ax.xaxis.set_major_formatter(FixedFormatter([f'{f:g}' for f in fds]))
ax.xaxis.set_minor_locator(NullLocator())
yt = [0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_ylim(0.055, 0.21)
ax.set_xlabel('Normalized Doppler  f_D x T_symbol  (log)   --   0.01 = channel changes over ~40 symbols')
ax.set_ylabel('Symbol error rate (log)')
ax.set_title('Fast fading within a word, SNR 7 dB, Rician K = 3', color=INK, loc='left', fontsize=12, pad=10)
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
leg = ax.legend(frameon=False, fontsize=8.8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
for t in leg.get_texts():
    t.set_color(INK)
ax.text(0.0, -0.33, '50 repetitions x 125 words per point; every detector sees the same transmissions '
         '(paired). Taps = COST2100 x (sqrt(K/(K+1)) + sqrt(1/(K+1)) x Jakes process). Bars = 95% CI.',
         color=MUTED, fontsize=8.5, transform=ax.transAxes)
fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved', OUT)
