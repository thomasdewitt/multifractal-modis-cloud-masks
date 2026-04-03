"""
Visualize a parent 8192^2 cloud mask with 6 randomly chosen children
(1024x2048 sub-scenes) highlighted, showing where they came from.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from load_masks import load_large_masks, subset_masks

np.random.seed(42)

# Load one parent
print("Loading parent mask...")
parent = load_large_masks(n=1)[0]

# Grid info
nrows, ncols = 4, 8
h, w = parent.shape
sh, sw = h // nrows, w // ncols  # 2048 x 1024

# Pick 6 random children
n_children = 6
total_cells = nrows * ncols
chosen = np.random.choice(total_cells, size=n_children, replace=False)
chosen_rc = [(idx // ncols, idx % ncols) for idx in chosen]

# Colormap: deep sky blue (clear) and white (cloud)
cmap = ListedColormap(["#0A6EBD", "#FFFFFF"])

# Layout: parent on left, children on right in 3x2 grid
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], wspace=0.08, hspace=0.3)

# Parent panel (spans all 3 rows)
ax_parent = fig.add_subplot(gs[:, 0])
ax_parent.imshow(parent, cmap=cmap, interpolation="none", aspect="equal",
                 origin="upper")
ax_parent.set_title("Parent $8192^2$ cloud mask", fontsize=13)
ax_parent.set_xticks([])
ax_parent.set_yticks([])

# Draw grid lines for all 4x8 sub-scenes (faint)
for r in range(1, nrows):
    ax_parent.axhline(r * sh, color="gray", lw=0.3, alpha=0.4)
for c in range(1, ncols):
    ax_parent.axvline(c * sw, color="gray", lw=0.3, alpha=0.4)

# Highlight chosen children with colored rectangles
child_colors = ["#E74C3C", "#F39C12", "#2ECC71", "#9B59B6", "#E67E22", "#1ABC9C"]

for i, (r, c) in enumerate(chosen_rc):
    rect = patches.Rectangle(
        (c * sw, r * sh), sw, sh,
        linewidth=2.5, edgecolor=child_colors[i], facecolor="none"
    )
    ax_parent.add_patch(rect)
    # Label in corner of rectangle
    ax_parent.text(
        c * sw + sw / 2, r * sh + sh / 2,
        str(i + 1), color=child_colors[i], fontsize=14, fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=child_colors[i], alpha=0.85),
    )

# Children panels (3 rows x 2 cols on right side)
for i, (r, c) in enumerate(chosen_rc):
    row = i // 2
    col = 1 + i % 2
    ax = fig.add_subplot(gs[row, col])
    child = parent[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw]
    ax.imshow(child, cmap=cmap, interpolation="none", aspect="equal", origin="upper")
    ax.set_title(f"Scene {i+1}", fontsize=11, color=child_colors[i], fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(child_colors[i])
        spine.set_linewidth(2.5)

fig.suptitle("Parent simulation and selected MODIS-like sub-scenes", fontsize=14)
plt.tight_layout()
outpath = "../figures/parent_children_example.png"
plt.savefig(outpath, dpi=400, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
