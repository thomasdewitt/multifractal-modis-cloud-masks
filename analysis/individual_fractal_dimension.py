"""
Individual fractal dimension D_i via perimeter-area scaling for:
  - Full 8192^2 cloud masks
  - Subsetted 1024x2048 MODIS-like scenes

Plots log10(perimeter) vs log10(sqrt(area)) with fitted slopes.
"""

import numpy as np
import matplotlib.pyplot as plt
import objscale
from load_masks import load_large_masks, load_all_subscenes

print("Loading masks...")
large_masks = load_large_masks()
subscenes = load_all_subscenes()

# Full masks
print("Computing D_i for full masks...")
dim_full, err_full, log_sqrtA_full, log_P_full = (
    objscale.individual_fractal_dimension(large_masks, return_values=True)
)
print(f"  D_i (full) = {dim_full:.3f} +/- {err_full:.3f}")

# Sub-scenes
print("Computing D_i for sub-scenes...")
dim_sub, err_sub, log_sqrtA_sub, log_P_sub = (
    objscale.individual_fractal_dimension(subscenes, return_values=True)
)
print(f"  D_i (sub)  = {dim_sub:.3f} +/- {err_sub:.3f}")

# Convert to raw values for log-scaled axes
sqrtA_full, P_full = 10**log_sqrtA_full, 10**log_P_full
sqrtA_sub, P_sub = 10**log_sqrtA_sub, 10**log_P_sub

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

# Fit lines first (behind scatter)
for log_x, log_y, dim, err, color, label in [
    (log_sqrtA_full, log_P_full, dim_full, err_full, "k",
     rf"$8192^2$: $D_i = {dim_full:.3f} \pm {err_full:.3f}$"),
    (log_sqrtA_sub, log_P_sub, dim_sub, err_sub, "k",
     rf"$1024 \times 2048$: $D_i = {dim_sub:.3f} \pm {err_sub:.3f}$"),
]:
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    (slope, intercept), _ = objscale.linear_regression(log_x[valid], log_y[valid])
    x_fit = np.linspace(log_x[valid].min(), log_x[valid].max(), 100)
    ax.plot(10**x_fit, 10**(slope * x_fit + intercept), color=color, lw=0.8,
            alpha=1.0, zorder=1)

# Scatter on top
ax.loglog(sqrtA_full, P_full, "o", color="#9B2226", ms=5, alpha=0.8, zorder=2,
          label=rf"$8192^2$: $D_i = {dim_full:.3f} \pm {err_full:.3f}$")
ax.loglog(sqrtA_sub, P_sub, "s", color="#005F73", ms=5, alpha=0.8, zorder=2,
          label=rf"$1024 \times 2048$: $D_i = {dim_sub:.3f} \pm {err_sub:.3f}$")

ax.set_xlabel(r"$\sqrt{a}$", fontsize=12)
ax.set_ylabel(r"$p$", fontsize=12)
ax.set_title("Individual Fractal Dimension (Perimeter-Area)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath = "../figures/individual_fractal_dimension.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
