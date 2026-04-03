"""
Individual correlation dimension of the largest cloud in each image for:
  - Full 8192^2 cloud masks
  - Subsetted 1024x2048 MODIS-like scenes

Uses objscale.individual_correlation_dimension which isolates the Nth largest
structure internally and computes its correlation dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import objscale
from load_masks import load_large_masks, load_all_subscenes

print("Loading masks...")
large_masks = load_large_masks()
subscenes = load_all_subscenes()

# Compute individual correlation dimension for largest cloud in each mask
print("Computing individual correlation dimension (full masks)...")
Cl_full_list = []
for i, m in enumerate(large_masks):
    print(f"  Full mask {i+1}/{len(large_masks)}...", flush=True)
    dim, err, bins, Cl = objscale.individual_correlation_dimension(
        m, n=1, return_C_l=True, point_reduction_factor=10000
    )
    Cl_full_list.append((dim, err, bins, Cl))
    print(f"    D_c = {dim:.3f} +/- {err:.3f}")

print("\nComputing individual correlation dimension (sub-scenes)...")
Cl_sub_list = []
n_skipped = 0
for i, m in enumerate(subscenes):
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  Sub-scene {i+1}/{len(subscenes)}...", flush=True)
    try:
        dim, err, bins, Cl = objscale.individual_correlation_dimension(
            m, n=1, return_C_l=True, point_reduction_factor=10000
        )
        Cl_sub_list.append((dim, err, bins, Cl))
    except ValueError:
        n_skipped += 1
print(f"  Skipped {n_skipped}/{len(subscenes)} sub-scenes (largest cloud too small)")

# Summary statistics (filter nans)
dims_full = np.array([x[0] for x in Cl_full_list])
dims_sub = np.array([x[0] for x in Cl_sub_list])
dims_full_valid = dims_full[np.isfinite(dims_full)]
dims_sub_valid = dims_sub[np.isfinite(dims_sub)]
print(f"\nD_c (full):  mean={dims_full_valid.mean():.3f}, std={dims_full_valid.std():.3f} (n={len(dims_full_valid)})")
print(f"D_c (sub):   mean={dims_sub_valid.mean():.3f}, std={dims_sub_valid.std():.3f} (n={len(dims_sub_valid)})")

# Plot: overlay all C(l) curves
fig, ax = plt.subplots(figsize=(7, 5))

labeled_full = False
for dim, err, bins, Cl in Cl_full_list:
    if not np.isfinite(dim):
        continue
    label = rf"$8192^2$ (mean $D_c = {dims_full_valid.mean():.3f}$)" if not labeled_full else None
    ax.loglog(bins, Cl, color="#9B2226", lw=0.6, alpha=1.0, label=label)
    labeled_full = True

labeled_sub = False
for dim, err, bins, Cl in Cl_sub_list:
    if not np.isfinite(dim):
        continue
    label = rf"$1024 \times 2048$ (mean $D_c = {dims_sub_valid.mean():.3f}$)" if not labeled_sub else None
    ax.loglog(bins, Cl, color="#005F73", lw=0.4, alpha=1.0, label=label)
    labeled_sub = True

ax.set_xlabel(r"Pixel separation $l$", fontsize=12)
ax.set_ylabel(r"$C(l)$", fontsize=12)
ax.set_title("Individual Correlation Dimension (Largest Cloud)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath = "../figures/individual_correlation_dimension.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
