"""
Part 2b: D_e,c bootstrap from subdivided 8192^2 FIF fields.

Generate 2 FIF at 16384^2, subsample to 8192^2, split each into
4x4 = 16 sub-arrays of 2048^2. Total: 32 sub-arrays.

Same bootstrap analysis as Part 2 but sub-arrays are spatially
correlated (from same parent). Append results to part2_results.csv.
"""

import numpy as np
import pandas as pd
import scaleinvariance
import objscale
import time
import os
import warnings

warnings.filterwarnings("ignore")

OUTDIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(OUTDIR, "part2_results.csv")

H = 0.25
C1_fif = 0.05
ALPHA = 1.8
N_PARENTS = 2
N_BOOTSTRAP = 30
N_COUNTS = [1, 2, 4, 8, 16, 32]
PRF = 2000


def generate_subdivided_masks():
    """Generate 2 parents at 16384^2, subsample to 8192^2, split 4x4."""
    masks = []
    for i in range(N_PARENTS):
        print(f"  Generating parent {i+1}/{N_PARENTS}...", end=" ", flush=True)
        t0 = time.time()
        field = scaleinvariance.FIF_ND(
            (16384, 16384), alpha=ALPHA, C1=C1_fif, H=H,
            kernel_construction_method_flux="LS2010",
            kernel_construction_method_observable="spectral",
            periodic=True,
        )
        sub = field[::2, ::2]  # 8192^2
        del field
        print(f"{time.time()-t0:.0f}s", flush=True)

        # Split 4x4
        h, w = sub.shape  # 8192, 8192
        sh, sw = h // 4, w // 4  # 2048, 2048
        for r in range(4):
            for c in range(4):
                chunk = sub[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                masks.append((chunk > chunk.mean()).astype(np.float64))
        del sub

    return masks


def compute_ens_corr_dim(masks):
    try:
        dim, err = objscale.ensemble_correlation_dimension(
            masks, point_reduction_factor=PRF
        )
        return dim, err
    except Exception:
        return np.nan, np.nan


def bootstrap_uncertainty(all_masks, n_use, n_bootstrap=N_BOOTSTRAP):
    values = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(all_masks), size=n_use, replace=True)
        sample = [all_masks[i] for i in indices]
        val, _ = compute_ens_corr_dim(sample)
        if np.isfinite(val):
            values.append(val)
    if len(values) < 5:
        return np.nan
    return np.std(values)


print("Part 2b: Subdivided FIF bootstrap", flush=True)
print(f"Backend: {scaleinvariance.get_backend()}", flush=True)
print(flush=True)

print("--- Generating subdivided masks ---", flush=True)
t0 = time.time()
all_masks = generate_subdivided_masks()
print(f"  {len(all_masks)} sub-arrays in {time.time()-t0:.0f}s\n", flush=True)

for n in N_COUNTS:
    print(f"=== FIF_sub n={n} ===", flush=True)
    masks_subset = all_masks[:n]

    t0 = time.time()
    val, obj_unc = compute_ens_corr_dim(masks_subset)
    print(f"  D_e,c = {val:.4f} +/- {obj_unc:.4f} ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    boot_unc = bootstrap_uncertainty(all_masks, n)
    print(f"  Bootstrap err = {boot_unc:.4f} ({time.time()-t0:.1f}s)", flush=True)

    row = pd.DataFrame([{
        "field_type": "FIF_sub",
        "size": 2048,
        "n_arrays": n,
        "metric": "ens_corr_dim",
        "value": val,
        "objscale_uncertainty": obj_unc,
        "bootstrap_uncertainty": boot_unc,
    }])
    row.to_csv(CSV_PATH, mode="a", header=False, index=False)

del all_masks
print("\nPart 2b complete!", flush=True)
