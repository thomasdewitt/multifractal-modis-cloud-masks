"""
Ensemble box dimension: log-spaced box sizes vs. strictly dyadic.

Compares:
  1. Log-spaced box sizes (including non-power-of-2 values that don't evenly
     tile the 8192^2 arrays, causing partial edge superpixels)
  2. Strictly dyadic: only powers of 2 that exactly divide 8192

Processes 20 large 8192^2 FIF arrays.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))

import numpy as np
import matplotlib.pyplot as plt
import objscale

LARGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'large')
N_MASKS = 20
ARRAY_SIZE = 8192


def load_masks(n=N_MASKS):
    masks = []
    for i in range(n):
        path = os.path.join(LARGE_DIR, f'fif_{i:03d}.npy')
        arr = np.load(path)
        masks.append((arr > arr.mean()).astype(np.float64))
        print(f"  loaded {i+1}/{n}", end='\r')
    print()
    return masks


# Strictly dyadic: powers of 2 that exactly divide 8192
dyadic_sizes = np.array([2**k for k in range(1, 14) if ARRAY_SIZE % (2**k) == 0])

# Log-spaced: ~30 sizes from 2 to 8192, many of which don't divide 8192 evenly
logspaced_sizes = np.unique(np.logspace(np.log10(2), np.log10(ARRAY_SIZE), 30).astype(int))

print(f"Dyadic sizes ({len(dyadic_sizes)}): {dyadic_sizes}")
print(f"Log-spaced sizes ({len(logspaced_sizes)}): {logspaced_sizes}")

print(f"\nLoading {N_MASKS} large masks...")
masks = load_masks()

print("Computing box dimension (log-spaced)...")
dim_log, err_log, bs_log, nb_log = objscale.ensemble_box_dimension(
    masks, box_sizes=logspaced_sizes, return_values=True)
print(f"  D_b (log-spaced) = {dim_log:.3f} +/- {err_log:.3f}")

print("Computing box dimension (dyadic only)...")
dim_dya, err_dya, bs_dya, nb_dya = objscale.ensemble_box_dimension(
    masks, box_sizes=dyadic_sizes, return_values=True)
print(f"  D_b (dyadic)     = {dim_dya:.3f} +/- {err_dya:.3f}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(7, 5))

for bs, nb, dim, err, color, marker, tag in [
    (bs_log, nb_log, dim_log, err_log, '#9B2226', 'o', 'Log-spaced'),
    (bs_dya, nb_dya, dim_dya, err_dya, '#005F73', 's', 'Dyadic'),
]:
    valid = np.isfinite(np.log10(nb))
    log_bs = np.log10(bs[valid])
    log_nb = np.log10(nb[valid])
    (slope, intercept), _ = objscale.linear_regression(log_bs, log_nb)
    x_fit = np.linspace(log_bs.min(), log_bs.max(), 100)
    ax.plot(10**x_fit, 10**(slope * x_fit + intercept), color='k', lw=0.8, zorder=1)
    ax.loglog(bs[valid], nb[valid], marker, color=color, ms=6, alpha=0.85, zorder=2,
              label=rf"{tag}: $D_b = {-slope:.3f} \pm {err:.3f}$")

ax.set_xlabel("Box size (pixels)", fontsize=12)
ax.set_ylabel("Number of occupied boxes", fontsize=12)
ax.set_title("Ensemble Box Dimension: Log-Spaced vs. Dyadic Box Sizes", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()

outpath = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'box_dimension_dyadic.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
plt.close()
