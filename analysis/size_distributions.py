"""
Size distributions of cloud areas and nested perimeters for:
  - Full 8192^2 cloud masks
  - Subsetted 1024x2048 MODIS-like scenes

Uses finite_array_powerlaw_exponent to account for domain truncation.
Exponents: alpha (area), beta (nested perimeter).
"""

import numpy as np
import matplotlib.pyplot as plt
import objscale
from load_masks import load_large_masks, load_all_subscenes

print("Loading masks...")
large_masks = load_large_masks()
subscenes = load_all_subscenes()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

variables = [
    ("area", ax1, r"\alpha"),
    ("nested perimeter", ax2, r"\beta"),
]

for var, ax, greek in variables:
    print(f"\n--- {var.upper()} ---")

    # Full masks
    print(f"  Full masks...")
    (exp_full, err_full), (log_bins_full, log_counts_full) = (
        objscale.finite_array_powerlaw_exponent(large_masks, var, return_counts=True)
    )
    print(f"  Exponent (full) = {exp_full:.3f} +/- {err_full:.3f}")

    # Sub-scenes
    print(f"  Sub-scenes...")
    (exp_sub, err_sub), (log_bins_sub, log_counts_sub) = (
        objscale.finite_array_powerlaw_exponent(subscenes, var, return_counts=True)
    )
    print(f"  Exponent (sub)  = {exp_sub:.3f} +/- {err_sub:.3f}")

    # Convert to raw values
    bins_full, counts_full = 10**log_bins_full, 10**log_counts_full
    bins_sub, counts_sub = 10**log_bins_sub, 10**log_counts_sub

    # Fit lines first (behind scatter)
    for log_b, log_c in [(log_bins_full, log_counts_full), (log_bins_sub, log_counts_sub)]:
        valid = np.isfinite(log_b) & np.isfinite(log_c)
        (slope, intercept), _ = objscale.linear_regression(log_b[valid], log_c[valid])
        x_fit = np.linspace(log_b[valid].min(), log_b[valid].max(), 100)
        ax.loglog(10**x_fit, 10**(slope * x_fit + intercept), "k-", lw=0.8,
                  alpha=1.0, zorder=1)

    # Scatter on top
    ax.loglog(bins_full, counts_full, "o", color="#9B2226", ms=6, zorder=2,
              label=rf"$8192^2$: ${greek} = {exp_full:.3f} \pm {err_full:.3f}$")
    ax.loglog(bins_sub, counts_sub, "s", color="#005F73", ms=6, zorder=2,
              label=rf"$1024 \times 2048$: ${greek} = {exp_sub:.3f} \pm {err_sub:.3f}$")

    var_label = "$a$" if var == "area" else "$p$"
    title_label = "Area" if var == "area" else "Nested Perimeter"
    ax.set_xlabel(var_label, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{title_label} Distribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
outpath = "../figures/size_distributions.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nSaved: {outpath}")
plt.close()
