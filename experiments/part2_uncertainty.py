"""
Part 2: Uncertainty vs number of arrays, with bootstrapping.
D_e,c only — the fast metric.

Size: 2048^2. Both FIF and fBm.
Generate 32 arrays, compute D_e,c for n=1,2,4,8,16,32.
Report objscale uncertainty and bootstrap uncertainty (50 resamples).
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
SIZE = 2048
N_MAX = 32
N_BOOTSTRAP = 30
N_COUNTS = [1, 2, 4, 8, 16, 32]
PRF = 2000


def generate_fif(n):
    sim_size = (2 * SIZE, 2 * SIZE)
    masks = []
    for i in range(n):
        field = scaleinvariance.FIF_ND(
            sim_size, alpha=ALPHA, C1=C1_fif, H=H,
            kernel_construction_method_flux="LS2010",
            kernel_construction_method_observable="spectral",
            periodic=True,
        )
        sub = field[::2, ::2]
        masks.append((sub > sub.mean()).astype(np.float64))
        del field, sub
    return masks


def generate_fbm(n):
    masks = []
    for i in range(n):
        field = scaleinvariance.fBm_ND_circulant((SIZE, SIZE), H=H)
        masks.append((field > field.mean()).astype(np.float64))
        del field
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


print(f"Part 2: D_e,c uncertainty vs n_arrays")
print(f"Size: {SIZE}x{SIZE}, max arrays: {N_MAX}")
print(f"Bootstrap resamples: {N_BOOTSTRAP}")
print(f"Backend: {scaleinvariance.get_backend()}")
print(flush=True)

if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

for field_type in ["fBm", "FIF"]:
    print(f"\n--- Generating {N_MAX} {field_type} arrays ---", flush=True)
    t0 = time.time()
    if field_type == "FIF":
        all_masks = generate_fif(N_MAX)
    else:
        all_masks = generate_fbm(N_MAX)
    print(f"  Generated in {time.time() - t0:.1f}s\n", flush=True)

    for n in N_COUNTS:
        print(f"=== {field_type} n={n} ===", flush=True)
        masks_subset = all_masks[:n]

        t0 = time.time()
        val, obj_unc = compute_ens_corr_dim(masks_subset)
        print(f"  D_e,c = {val:.4f} +/- {obj_unc:.4f} ({time.time()-t0:.1f}s)", flush=True)

        t0 = time.time()
        boot_unc = bootstrap_uncertainty(all_masks, n)
        print(f"  Bootstrap err = {boot_unc:.4f} ({time.time()-t0:.1f}s)", flush=True)

        row = pd.DataFrame([{
            "field_type": field_type,
            "size": SIZE,
            "n_arrays": n,
            "metric": "ens_corr_dim",
            "value": val,
            "objscale_uncertainty": obj_unc,
            "bootstrap_uncertainty": boot_unc,
        }])
        if os.path.exists(CSV_PATH):
            row.to_csv(CSV_PATH, mode="a", header=False, index=False)
        else:
            row.to_csv(CSV_PATH, index=False)

    del all_masks

print("\nPart 2 complete!", flush=True)
print(f"Results: {CSV_PATH}")
