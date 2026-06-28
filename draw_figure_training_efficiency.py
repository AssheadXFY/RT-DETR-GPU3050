"""
Figure: Training Dynamics & Efficiency — clean, CV-paper style
IEEEtran double-column: 7.16 in × 3.6 in
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})

# ── Colors ───────────────────────────────────────────────────────
BLUE  = '#2166AC'   # FreqSpatial
ORANGE = '#D6604D'  # Baseline
GREY  = '#999999'

# ── Data ─────────────────────────────────────────────────────────
# From server logs, every epoch mAP (×100 for %)
epochs_72 = np.arange(0, 72)

base_map = np.array([14.4, 24.1, 27.4, 29.8, 32.1, 33.9, 35.3, 36.5, 37.5,
    38.2, 39.0, 39.6, 40.2, 40.5, 40.9, 41.2, 41.5, 41.8, 42.0, 42.3,
    42.5, 42.6, 42.9, 43.0, 43.2, 43.4, 43.6, 43.7, 43.8, 44.0, 44.0,
    44.1, 44.3, 44.4, 44.5, 44.8, 44.8, 44.9, 44.9, 45.0, 45.1, 45.1,
    45.2, 45.4, 45.5, 45.5, 45.6, 45.6, 45.7, 45.7, 45.8, 45.9, 45.8,
    45.8, 45.8, 46.0, 46.1, 46.2, 46.2, 46.1, 46.0, 46.1, 46.2, 46.2,
    46.2, 46.3, 46.4, 46.4, 46.4, 46.5, 46.6, 46.7])

fs_map = np.array([17.8, 26.8, 30.0, 32.3, 34.4, 36.0, 37.3, 38.4, 39.5,
    40.2, 40.9, 41.4, 41.8, 42.3, 42.6, 43.0, 43.4, 43.7, 43.9, 44.1,
    44.3, 44.5, 44.8, 44.8, 45.0, 45.3, 45.4, 45.5, 45.7, 45.8, 46.0,
    46.0, 46.1, 46.1, 46.2, 46.4, 46.4, 46.6, 46.6, 46.8, 46.9, 46.9,
    47.0, 47.1, 47.0, 47.1, 47.2, 47.2, 47.3, 47.4, 47.5, 47.6, 47.6,
    47.6, 47.6, 47.7, 47.7, 47.7, 47.7, 47.8, 47.9, 48.0, 47.9, 48.0,
    48.1, 48.1, 48.1, 48.2, 48.1, 48.1, 48.1, 48.2])

# Subsample for markers (~10 per curve)
mk_idx = np.arange(0, 72, 8)  # 0,8,16,24,32,40,48,56,64,71

models_eff = ['Baseline', 'FreqSpatial']
map_eff    = [46.73, 48.18]
fps_eff    = [66.4,  38.5]

# ── Figure ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.4))

# ══════════════════════════════════════════════════════════════
# PANEL (a): Training Curve
# ══════════════════════════════════════════════════════════════
# FreqSpatial — solid line + markers at 8-epoch intervals
ax1.plot(epochs_72, fs_map, '-', color=BLUE, lw=1.6, alpha=0.85, zorder=3)
ax1.scatter(epochs_72[mk_idx], fs_map[mk_idx], s=28, color=BLUE,
            marker='o', edgecolors='white', linewidths=0.5, zorder=6,
            label='FreqSpatial (Ours)')

# Baseline — solid line + markers at 8-epoch intervals
ax1.plot(epochs_72, base_map, '-', color=ORANGE, lw=1.6, alpha=0.85, zorder=2)
ax1.scatter(epochs_72[mk_idx], base_map[mk_idx], s=28, color=ORANGE,
            marker='D', edgecolors='white', linewidths=0.5, zorder=5,
            label='Baseline')

# Horizontal reference lines
ax1.axhline(46.73, color=GREY, lw=0.7, ls=':', alpha=0.45, zorder=1)
ax1.text(68, 43.5, '46.73', fontsize=7, color=GREY, ha='right')
ax1.axhline(48.18, color=GREY, lw=0.7, ls=':', alpha=0.45, zorder=1)
ax1.text(68, 49.5, '48.18', fontsize=7, color=GREY, ha='right')

# Grid
ax1.grid(True, alpha=0.15, lw=0.5)

ax1.set_xlabel('Epoch', fontweight='medium')
ax1.set_ylabel('mAP (%)', fontweight='medium')
ax1.set_xlim(-4, 76)
ax1.set_ylim(5, 54)
ax1.legend(loc='lower right', framealpha=0.9, fancybox=False,
           edgecolor='#cccccc', handlelength=1.8)

# ══════════════════════════════════════════════════════════════
# PANEL (b): Efficiency Scatter
# ══════════════════════════════════════════════════════════════
for i, name in enumerate(models_eff):
    c = BLUE if i == 1 else ORANGE
    m = 'o'   if i == 1 else 'D'
    ax2.scatter(map_eff[i], fps_eff[i], s=140, color=c, marker=m,
                edgecolors='white', linewidths=0.9, zorder=5)
    # Label
    dx = 0.10 if i == 0 else -0.10
    dy = -6   if i == 0 else 5
    ha = 'left' if i == 0 else 'right'
    ax2.text(map_eff[i] + dx, fps_eff[i] + dy, name,
             fontsize=8.5, fontweight='bold', color=c, ha=ha, va='center')

# Grid
ax2.grid(True, alpha=0.15, lw=0.5)

ax2.set_xlabel('mAP (%)', fontweight='medium')
ax2.set_ylabel('FPS', fontweight='medium')
ax2.set_xlim(46.3, 48.6)
ax2.set_ylim(5, 78)

# ══════════════════════════════════════════════════════════════
# Panel labels
# ══════════════════════════════════════════════════════════════
for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

plt.tight_layout(pad=1.0, w_pad=3.0)
fig.savefig('figure_training_efficiency.pdf', format='pdf', dpi=300)
fig.savefig('figure_training_efficiency.png', format='png', dpi=300)
print('Saved: figure_training_efficiency.pdf / .png')
