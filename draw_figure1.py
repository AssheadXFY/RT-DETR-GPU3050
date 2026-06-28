#!/usr/bin/env python3
"""Draw Figure 1: FreqSpatial module architecture using matplotlib.
Output: paper_fig1.pdf (vector), paper_fig1.png (preview)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# ---- Style ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
})

FIG_W, FIG_H = 14, 5.5

# Colors
C_BLUE  = '#3B78C8'
C_BLUEL = '#E0EDFA'
C_ORANGE = '#D87828'
C_ORANGEL = '#FFF0E0'
C_GREY  = '#555555'
C_GREYL = '#F5F5F0'
C_PURPLE = '#8844BB'
C_PURPLEL = '#F5EDFF'

# ---- Coordinate system ----
# y positions
Y_SPATIAL = 0.75
Y_FREQ    = 0.26
Y_MID     = 0.52
Y_BYPASS  = 0.93

# Block widths & heights
BW, BH = 0.078, 0.075   # regular block (in figure coords)
LW, LH = 0.096, 0.075   # larger block (F_spatial, F_freq, Output, Conv1x1)

def block(ax, x,  y, w, h, facecolor, edgecolor, text, fontweight='normal'):
    """Draw a rounded rectangle block with text."""
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.02", linewidth=0.8,
                        facecolor=facecolor, edgecolor=edgecolor)
    ax.add_patch(r)
    ax.text(x, y, text, ha='center', va='center', fontsize=8.5,
            fontweight=fontweight, linespacing=1.1)


def arrow(ax, x1, y1, x2, y2, color='#666666'):
    """Draw a simple arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.0, connectionstyle='arc3,rad=0'))

# Path-based arrow with explicit bends
def polyarrow(ax, points, color='#666666'):
    """Draw arrow through a series of points, arrowhead at last point."""
    x, y = zip(*points)
    ax.plot(x[:-1], y[:-1], '-', color=color, lw=1.0)
    # arrowhead
    dx = points[-1][0] - points[-2][0]
    dy = points[-1][1] - points[-2][1]
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0))


def dashed_box(ax, x1, y1, x2, y2, color, label):
    """Draw dashed rounded rect with label."""
    r = FancyBboxPatch((x1, y1), x2-x1, y2-y1,
                        boxstyle="round,pad=0.015", linewidth=0.8,
                        linestyle='--', facecolor='none', edgecolor=color)
    ax.add_patch(r)
    ax.text(x1 + 0.012, y2 + 0.01, label, fontsize=7.5, fontweight='bold',
            color=color, va='bottom', ha='left')


# ---- Build figure ----
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ====== SPATIAL BRANCH blocks ======
bx = 0.18
spacing = 0.10

block(ax, bx, Y_SPATIAL, BW, BH, C_BLUEL, C_BLUE, 'Scharr\nEdge')
bx += spacing
block(ax, bx, Y_SPATIAL, BW, BH, C_BLUEL, C_BLUE, 'Conv 3×3\nSiLU')
bx += spacing
# + circle (residual add)
circle = Circle((bx, Y_SPATIAL), 0.018, facecolor='white', edgecolor=C_GREY, lw=0.8, zorder=3)
ax.add_patch(circle)
ax.text(bx, Y_SPATIAL, '+', ha='center', va='center', fontsize=8, fontweight='bold', color=C_GREY)
sp_x = bx   # save for residual arrow
bx += spacing
block(ax, bx, Y_SPATIAL, BW, BH, C_BLUEL, C_BLUE, 'Conv 3×3\nSiLU')
bx += spacing
block(ax, bx, Y_SPATIAL, LW, LH, C_BLUEL, C_BLUE, r'$\mathbf{F_{spatial}}$', 'bold')
so_right = bx + LW/2   # right edge of F_spatial

# Spatial arrows
s_start = 0.18 - spacing
for sx in [0.18, 0.18+spacing, 0.18+spacing-spacing+0.02]:
    # Draw arrows between blocks
    pass

# ====== FREQUENCY BRANCH blocks ======
fx = 0.12
fspacing = 0.091

blocks_f = [
    ('rFFT2D', C_ORANGEL, C_ORANGE, BW, BH),
    ('Re/Im\nSplit', C_ORANGEL, C_ORANGE, BW, BH),
    ('Concat\n2C', C_ORANGEL, C_ORANGE, BW, BH),
    ('Conv 3×3\nSiLU', C_ORANGEL, C_ORANGE, BW, BH),
    ('irFFT2D', C_ORANGEL, C_ORANGE, BW, BH),
    ('Conv 3×3\nSiLU', C_ORANGEL, C_ORANGE, BW, BH),
]

for i, (txt, fc, ec, w, h) in enumerate(blocks_f):
    block(ax, fx, Y_FREQ, w, h, fc, ec, txt)
    fx += fspacing

# F_freq
block(ax, fx, Y_FREQ, LW, LH, C_ORANGEL, C_ORANGE, r'$\mathbf{F_{freq}}$', 'bold')
fo_right = fx + LW/2         # right edge of F_freq
fo_center_x = fx

# ====== INPUT block ======
in_x = 0.035
in_w = 0.10
block(ax, in_x, Y_MID, in_w + 0.01, BH + 0.005, C_GREYL, C_GREY,
      r'$X \in \mathbb{R}^{B\times C\times H\times W}$')

# ====== FUSION blocks ======
ad_x = 0.875
block(ax, ad_x, Y_MID, 0.020, 0.020, None, None, r'$\oplus$', 'bold')  # just text
# Actually use circle
ad_circle = Circle((ad_x, Y_MID), 0.022, facecolor='white', edgecolor=C_GREY, lw=0.9, zorder=3)
ax.add_patch(ad_circle)
ax.text(ad_x, Y_MID, r'$\mathbf{\oplus}$', ha='center', va='center', fontsize=9, color=C_GREY)

cv_x = ad_x + 0.055
block(ax, cv_x, Y_MID, 0.064, BH, C_PURPLEL, C_PURPLE, 'Conv\n1×1 SiLU')

out_x = cv_x + 0.058
block(ax, out_x, Y_MID, 0.068, BH, C_GREYL, C_GREY, 'Output')

# ====== ARROWS within spatial branch ======
s_positions = [0.18, 0.18 + spacing, sp_x, sp_x + spacing, so_right]
s_y = Y_SPATIAL
for i in range(len(s_positions) - 1):
    arrow(ax, s_positions[i] + BW/2, s_y, s_positions[i+1], s_y)

# But for + to next conv: the "+" is at sp_x,conv1 at sp_x+spacing
# We need to handle the arrows more carefully
# Rewriting arrow logic manually

def seg_arrow(ax, x1, y1, x2, y2, color='#777777'):
    """Arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0))


s_centers_x = [0.18, 0.28, sp_x, 0.48, 0.60]  # manual x centeers
for i in range(3):
    seg_arrow(ax, s_centers_x[i] + BW/2, Y_SPATIAL, s_centers_x[i+1] - 0.018, Y_SPATIAL)

# Scharr → Conv1
seg_arrow(ax, 0.18 + BW/2, Y_SPATIAL, 0.28 - 0.018, Y_SPATIAL)
# Conv1 → +
seg_arrow(ax, 0.28 + BW/2, Y_SPATIAL, sp_x - 0.018, Y_SPATIAL)
# + → Conv2
seg_arrow(ax, sp_x + 0.018, Y_SPATIAL, 0.48 - 0.018, Y_SPATIAL)
# Conv2 → F_spatial
seg_arrow(ax, 0.48 + BW/2, Y_SPATIAL, 0.60 - LW/2, Y_SPATIAL)

# ====== ARROWS within frequency branch ======
f_positions = [0.12, 0.211, 0.302, 0.393, 0.484, 0.575, fo_center_x]
for i in range(len(f_positions)-1):
    seg_arrow(ax, f_positions[i] + BW/2, Y_FREQ, f_positions[i+1] - 0.018, Y_FREQ)

# ====== INPUT → BRANCHES ======
# Fork point
fork_x = 0.095
fork_y = Y_MID

# Input → fork
seg_arrow(ax, in_x + in_w/2, Y_MID, fork_x, Y_MID)

# Fork → Scharr (up-right)
seg_arrow(ax, fork_x, fork_y, fork_x, Y_SPATIAL)        # go UP
seg_arrow(ax, fork_x, Y_SPATIAL, 0.18 - BW/2, Y_SPATIAL)  # go RIGHT to Scharr

# Fork → rFFT (down-right)
seg_arrow(ax, fork_x, fork_y, fork_x, Y_FREQ)           # go DOWN
seg_arrow(ax, fork_x, Y_FREQ, 0.12 - BW/2, Y_FREQ)      # go RIGHT to rFFT

# ====== RESIDUAL: fork → right → up → ↓ into + circle ======
sp_center_x = sp_x
sp_y = Y_SPATIAL
# Route: go right from fork, then up past the spatial row, then turn left and go down into the + circle
residual_route = [
    (fork_x, fork_y),           # start
    (sp_center_x, fork_y),      # go right to same x as + circle
    (sp_center_x, Y_BYPASS),    # go up above spatial
    (sp_center_x, Y_SPATIAL + 0.022), # go down to just above + circle
]
for i in range(len(residual_route)-1):
    ax.plot([residual_route[i][0], residual_route[i+1][0]],
            [residual_route[i][1], residual_route[i+1][1]],
            '-', color='#777777', lw=1.0)
# Arrowhead
seg_arrow(ax, sp_center_x, Y_SPATIAL + 0.022, sp_center_x, Y_SPATIAL + 0.018)

# ====== F_spatial, F_freq → fusion ======
# F_spatial → adder (go right, turn down to adder top)
mh = 0.835  # midpoint horizontal
seg_arrow(ax, 0.60 + LW/2, Y_SPATIAL, mh, Y_SPATIAL)
seg_arrow(ax, mh, Y_SPATIAL, mh, Y_MID + 0.022)

# F_freq → adder (go right, turn up to adder bottom)
seg_arrow(ax, fo_center_x + LW/2, Y_FREQ, mh, Y_FREQ)
seg_arrow(ax, mh, Y_FREQ, mh, Y_MID - 0.022)

# ====== Adder → Conv → Output ======
seg_arrow(ax, ad_x + 0.022, Y_MID, cv_x - 0.032, Y_MID)
seg_arrow(ax, cv_x + 0.032, Y_MID, out_x - 0.034, Y_MID)

# ====== DASHED BOXES ======
# Spatial branch
dashed_box(ax, 0.15 - 0.008, Y_SPATIAL - 0.055, 0.68, 0.125, C_BLUE, 'Spatial Branch')
# Frequency branch
dashed_box(ax, 0.08 - 0.008, Y_FREQ - 0.055, fo_right - 0.08 + 0.035, 0.125, C_ORANGE, 'Frequency Branch')

# ====== Save ======
plt.tight_layout(pad=0)
fig.savefig('paper_fig1.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('paper_fig1.png', dpi=200, bbox_inches='tight', facecolor='white')
print('Saved: paper_fig1.pdf, paper_fig1.png')
plt.close()
