"""
Compare ensemble correlation dimension D_{e,c} for:
  - Full 8192^2 cloud masks
  - Subsetted 1024x2048 MODIS-like scenes

Plots C(l) curves for both domain sizes with fitted slopes.
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

# Ensemble correlation dimension for full masks
print("Computing D_{e,c} for full masks...")
dim_full, err_full, bins_full, Cl_full = objscale.ensemble_correlation_dimension(
    large_masks, return_C_l=True, point_reduction_factor=10000
)
print(f"  D_e,c (full) = {dim_full:.3f} +/- {err_full:.3f}")

# Ensemble correlation dimension for sub-scenes
print("Computing D_{e,c} for sub-scenes...")
dim_sub, err_sub, bins_sub, Cl_sub = objscale.ensemble_correlation_dimension(
    subscenes, return_C_l=True, point_reduction_factor=10000
)
print(f"  D_e,c (sub)  = {dim_sub:.3f} +/- {err_sub:.3f}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(bins_full, Cl_full, color="#9B2226", lw=1.2, label=rf"$8192^2$: $D_{{e,c}} = {dim_full:.3f} \pm {err_full:.3f}$")
ax.loglog(bins_sub, Cl_sub, color="#005F73", lw=1.2, label=rf"$1024 \times 2048$: $D_{{e,c}} = {dim_sub:.3f} \pm {err_sub:.3f}$")

ax.set_xlabel(r"Pixel separation $l$", fontsize=12)
ax.set_ylabel(r"$C(l)$", fontsize=12)
ax.set_title("Ensemble Correlation Dimension", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath = "../figures/ensemble_correlation_dimension.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
