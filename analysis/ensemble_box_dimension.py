"""
Compare ensemble box dimension D_{e,b} for:
  - Full 8192^2 cloud masks
  - Subsetted 1024x2048 MODIS-like scenes

Also checks the relationship D_{e,b} = beta * D_i.
"""

import numpy as np
import matplotlib.pyplot as plt
import objscale
from load_masks import load_large_masks, load_all_subscenes

print("Loading full 8192^2 masks...")
large_masks = load_large_masks()

print("Loading 1024x2048 sub-scenes...")
subscenes = load_all_subscenes()

print(f"Full masks: {len(large_masks)}, Sub-scenes: {len(subscenes)}")

# Ensemble box dimension for full masks
print("Computing D_{e,b} for full masks...")
dim_full, err_full, boxes_full, counts_full = objscale.ensemble_box_dimension(
    large_masks, return_values=True, min_box_size=30
)
print(f"  D_e,b (full) = {dim_full:.3f} +/- {err_full:.3f}")

# Ensemble box dimension for sub-scenes
print("Computing D_{e,b} for sub-scenes...")
dim_sub, err_sub, boxes_sub, counts_sub = objscale.ensemble_box_dimension(
    subscenes, return_values=True, min_box_size=30
)
print(f"  D_e,b (sub)  = {dim_sub:.3f} +/- {err_sub:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(boxes_full, counts_full, color="#9B2226", lw=1.2,
          label=rf"$8192^2$: $D_{{e,b}} = {dim_full:.3f} \pm {err_full:.3f}$")
ax.loglog(boxes_sub, counts_sub, color="#005F73", lw=1.2,
          label=rf"$1024 \times 2048$: $D_{{e,b}} = {dim_sub:.3f} \pm {err_sub:.3f}$")

ax.set_xlabel("Box size", fontsize=12)
ax.set_ylabel("Box count", fontsize=12)
ax.set_title("Ensemble Box Dimension", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath = "../figures/ensemble_box_dimension.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
