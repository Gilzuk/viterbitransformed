"""SER vs SNR for every detector in the MC sweep CSV (+ Mamba2 from its branch)."""
import csv, io, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = 'Results/metrics/mc_sweep_validation.csv'
rows = list(csv.DictReader(open(P)))
try:
    mb = subprocess.check_output(['git', 'show', 'origin/claude/mamba2-vs-viterbinet:' + P]).decode()
    rows += [r for r in csv.DictReader(io.StringIO(mb)) if r['model'] == 'Mamba2']
except Exception:
    pass

INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8983', '#e4e3dd', '#fcfcfb'
S = [  # (label in CSV, legend, color, marker, linestyle)
    ('ClassicViterbi', 'ClassicViterbi, perfect CSI (bound)', INK2, 'o', '--'),
    ('ClassicViterbi_csi25', 'ClassicViterbi, CSI error 25%', '#8fb8ea', 'D', ':'),
    ('ClassicViterbi_csi50', 'ClassicViterbi, CSI error 50%', '#5b95dd', 'D', ':'),
    ('ClassicViterbi_csi75', 'ClassicViterbi, CSI error 75%', '#3a78c9', 'D', ':'),
    ('ClassicViterbi_csi100', 'ClassicViterbi, CSI error 100%', '#1f5fae', 'D', ':'),
    ('ViterbiNet_on5', 'ViterbiNet, 5 online steps (recommended)', '#1baf7a', 's', '-'),
    ('VNet_affine', 'Affine VNet, 32 params, 200 steps', '#008300', 'P', '-'),
    ('ViterbiNet', 'ViterbiNet, 200 online steps (default)', '#eda100', '^', '-'),
    ('Transformer', 'Transformer', '#eb6834', 'v', '-'),
    ('Mamba2', 'Mamba2', '#e87ba4', 'X', '-'),
]
fig, ax = plt.subplots(figsize=(11, 7.2), facecolor=SURF); ax.set_facecolor(SURF)
cens_label = True
for key, label, c, mk, ls in S:
    pts = sorted((int(r['snr']), float(r['ser_mean']), int(r.get('censored') or 0), float(r['bits_run']))
                 for r in rows if r['model'] == key)
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [3.0 / p[3] if p[2] else p[1] for p in pts]
    ax.plot(xs, ys, ls, color=c, lw=1.8, zorder=2)
    m = [(x, y) for x, y, p in zip(xs, ys, pts) if not p[2]]
    if m:
        ax.plot(*zip(*m), mk, color=c, ms=6.5, mec=SURF, mew=1, lw=0, label=label, zorder=3)
    cz = [(x, y) for x, y, p in zip(xs, ys, pts) if p[2]]
    if cz:
        ax.plot(*zip(*cz), 'v', color=c, ms=9, mfc='none', mew=1.4, lw=0, zorder=4,
                label='0 errors: 95% upper bound' if cens_label else None)
        cens_label = False
ax.set_yscale('log')
ax.set_xlabel('SNR (dB)', color=INK2); ax.set_ylabel('Symbol error rate (log)', color=INK2)
ax.set_title('SER vs SNR, BPSK, COST2100 fading ISI channel (memory 4)', color=INK, loc='left', fontsize=12.5, pad=10)
ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
leg = ax.legend(frameon=False, fontsize=9, loc='lower left')
for t in leg.get_texts(): t.set_color(INK)
fig.text(0.01, 0.005, 'Monte Carlo: >=20 reps x 2,000 bits per point (ClassicViterbi >=100 reps). Transformer/ViterbiNet 200-step rows from the Colab-GPU run.',
         color=MUTED, fontsize=8)
fig.savefig('Results/figures/ser_vs_snr_all.png', dpi=150, facecolor=SURF, bbox_inches='tight')
print('saved')
