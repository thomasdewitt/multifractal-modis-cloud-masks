"""
Filled vs. unfilled individual fractal dimension.

Computes the perimeter-area fractal dimension D_i two ways:
  1. Standard (filled): holes in structures are filled before computing P and A
  2. Unfilled: holes are NOT filled, so P and A reflect the actual boundary

Processes 20 large 8192^2 FIF arrays.
Produces:
  - P-A scatter plot (60 bins) with global regressions
  - D_i vs sqrt(a) plot showing local slopes in scale bands
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

import numpy as np
import matplotlib.pyplot as plt
import objscale

LARGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'large')
N_MASKS = 20
BINS = 60

# Scale bands in sqrt(a) for local regressions
SCALE_BANDS = [
    (3, 10),
    (10, 30),
    (30, 100),
    (100, 300),
    (300, 1000),
]


def fit_scale_bands(masks, filled, bands=SCALE_BANDS):
    """Regress D_i within each scale band (defined in sqrt(a) space)."""
    results = []
    for lo, hi in bands:
        dim, err, lx, ly = objscale.individual_fractal_dimension(
            masks, min_a=lo**2, max_a=hi**2, bins=None,
            return_values=True, filled=filled,
        )
        center = np.sqrt(lo * hi)
        if len(lx) < 5:
            results.append((center, np.nan, np.nan))
        else:
            results.append((center, dim, err))
    return results


def load_masks(n=N_MASKS):
    masks = []
    for i in range(n):
        path = os.path.join(LARGE_DIR, f'fif_{i:03d}.npy')
        arr = np.load(path)
        masks.append((arr > arr.mean()).astype(np.float64))
        print(f"  loaded {i+1}/{n}", end='\r')
    print()
    return masks


# --- Main ---
print(f"Loading {N_MASKS} large masks...")
masks = load_masks()

print("Computing D_i (filled)...")
dim_f, err_f, lx_f, ly_f = objscale.individual_fractal_dimension(
    masks, bins=BINS, return_values=True, filled=True)
print(f"  D_i (filled)   = {dim_f:.3f} +/- {err_f:.3f}")

print("Computing D_i (unfilled)...")
dim_u, err_u, lx_u, ly_u = objscale.individual_fractal_dimension(
    masks, bins=BINS, return_values=True, filled=False)
print(f"  D_i (unfilled) = {dim_u:.3f} +/- {err_u:.3f}")

# Scale-band regressions
print("Computing scale-band regressions (filled)...")
bands_f = fit_scale_bands(masks, filled=True)
print("Computing scale-band regressions (unfilled)...")
bands_u = fit_scale_bands(masks, filled=False)

print("\nScale-band regressions:")
print(f"  {'sqrt(a) range':>16s}  {'Filled D_i':>12s}  {'Unfilled D_i':>14s}")
for (lo, hi), (_, df, ef), (_, du, eu) in zip(SCALE_BANDS, bands_f, bands_u):
    print(f"  {lo:>6.0f} – {hi:<6.0f}   {df:>6.3f} ± {ef:.3f}   {du:>6.3f} ± {eu:.3f}")

# ============================================================
# Figure 1: P-A scatter (60 bins)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

for lx, ly, dim, err, color, marker, tag in [
    (lx_f, ly_f, dim_f, err_f, '#9B2226', 'o', 'Filled'),
    (lx_u, ly_u, dim_u, err_u, '#005F73', 's', 'Unfilled'),
]:
    valid = np.isfinite(lx) & np.isfinite(ly)
    (slope, intercept), _ = objscale.linear_regression(lx[valid], ly[valid])
    x_fit = np.linspace(lx[valid].min(), lx[valid].max(), 100)
    ax.plot(10**x_fit, 10**(slope * x_fit + intercept), color='k', lw=0.8, zorder=1)
    ax.loglog(10**lx, 10**ly, marker, color=color, ms=5, alpha=0.85, zorder=2,
              label=rf"{tag}: $D_i = {dim:.3f} \pm {err:.3f}$")

ax.set_xlabel(r"$\sqrt{a}$", fontsize=12)
ax.set_ylabel(r"$p$", fontsize=12)
ax.set_title("Individual Fractal Dimension: Filled vs. Unfilled Holes", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

outpath1 = os.path.join(os.path.dirname(__file__), '..', 'figures',
                        'filled_vs_unfilled_fractal_dim.png')
plt.savefig(outpath1, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath1}")
plt.close()

# ============================================================
# Figure 2: D_i vs sqrt(a) across scale bands
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

centers_f = [c for c, _, _ in bands_f]
dims_f = [d for _, d, _ in bands_f]
errs_f = [e for _, _, e in bands_f]

centers_u = [c for c, _, _ in bands_u]
dims_u = [d for _, d, _ in bands_u]
errs_u = [e for _, _, e in bands_u]

ax.errorbar(centers_f, dims_f, yerr=errs_f, fmt='o-', color='#9B2226',
            ms=7, capsize=4, lw=1.5, label='Filled')
ax.errorbar(centers_u, dims_u, yerr=errs_u, fmt='s-', color='#005F73',
            ms=7, capsize=4, lw=1.5, label='Unfilled')

for lo, hi in SCALE_BANDS:
    ax.axvspan(lo, hi, alpha=0.06, color='grey')

ax.set_xscale('log')
ax.set_xlabel(r"$\sqrt{a}$", fontsize=12)
ax.set_ylabel(r"$D_i$ (local slope)", fontsize=12)
ax.set_title("Scale-Dependent Fractal Dimension", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

outpath2 = os.path.join(os.path.dirname(__file__), '..', 'figures',
                        'filled_vs_unfilled_scale_bands.png')
plt.savefig(outpath2, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath2}")
plt.close()
