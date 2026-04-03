"""
Part 1: Parameter convergence with domain size.

For FIF (H=0.25, C1=0.05, alpha=1.8) and fBm (C1=0, H=0.25):
  - Sizes: 512^2, 1024^2, 2048^2, 4096^2, 8192^2
  - 10 realizations each
  - Metrics: ensemble corr dim, ind corr dim (mean), p-a ind dim,
             area exponent, nested perimeter exponent

FIF: generate at 2x size, subsample 2x.
fBm: generate directly at target size.
Threshold at mean for binary masks.

Results saved incrementally to part1_results.csv.
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
CSV_PATH = os.path.join(OUTDIR, "part1_results.csv")

# Parameters
H = 0.25
C1_fif = 0.05
ALPHA = 1.8
SIZES = [512, 1024, 2048, 4096, 8192]
N_REAL = 10

# Point reduction factors scaled by size
PRF = {512: 100, 1024: 500, 2048: 2000, 4096: 5000, 8192: 10000}


def generate_fif(size, n):
    """Generate n FIF realizations at 2x size, subsample, threshold."""
    sim_size = (2 * size, 2 * size)
    masks = []
    for i in range(n):
        field = scaleinvariance.FIF_ND(
            sim_size, alpha=ALPHA, C1=C1_fif, H=H,
            kernel_construction_method_flux="LS2010",
            kernel_construction_method_observable="spectral",
            periodic=True,
        )
        sub = field[::2, ::2]
        mask = (sub > sub.mean()).astype(np.float64)
        masks.append(mask)
        del field, sub
    return masks


def generate_fbm(size, n):
    """Generate n fBm realizations directly, threshold at mean (0)."""
    masks = []
    for i in range(n):
        field = scaleinvariance.fBm_ND_circulant((size, size), H=H)
        mask = (field > field.mean()).astype(np.float64)
        masks.append(mask)
        del field
    return masks


def compute_metrics(masks, size):
    """Compute all metrics for a list of binary masks."""
    prf = PRF[size]
    results = {}

    # 1. Ensemble correlation dimension
    try:
        dim, err = objscale.ensemble_correlation_dimension(
            masks, point_reduction_factor=prf
        )
        results["ens_corr_dim"] = (dim, err)
    except Exception as e:
        print(f"    ens_corr_dim failed: {e}")
        results["ens_corr_dim"] = (np.nan, np.nan)

    # 2. Individual correlation dimension (per-image, report mean/std)
    ind_dims = []
    for m in masks:
        try:
            dim, err = objscale.individual_correlation_dimension(
                m, n=1, point_reduction_factor=prf
            )
            if np.isfinite(dim):
                ind_dims.append(dim)
        except (ValueError, Exception):
            pass
    if ind_dims:
        results["ind_corr_dim"] = (np.mean(ind_dims), np.std(ind_dims))
        results["ind_corr_dim_n"] = len(ind_dims)
    else:
        results["ind_corr_dim"] = (np.nan, np.nan)
        results["ind_corr_dim_n"] = 0

    # 3. Individual fractal dimension (perimeter-area)
    try:
        dim, err = objscale.individual_fractal_dimension(masks)
        results["ind_frac_dim"] = (dim, err)
    except Exception as e:
        print(f"    ind_frac_dim failed: {e}")
        results["ind_frac_dim"] = (np.nan, np.nan)

    # 4. Area size distribution exponent
    try:
        exp, err = objscale.finite_array_powerlaw_exponent(masks, "area")
        results["area_exp"] = (exp, err)
    except Exception as e:
        print(f"    area_exp failed: {e}")
        results["area_exp"] = (np.nan, np.nan)

    # 5. Nested perimeter size distribution exponent
    try:
        exp, err = objscale.finite_array_powerlaw_exponent(masks, "nested perimeter")
        results["perim_exp"] = (exp, err)
    except Exception as e:
        print(f"    perim_exp failed: {e}")
        results["perim_exp"] = (np.nan, np.nan)

    return results


def save_row(field_type, size, metrics):
    """Append one row per metric to CSV."""
    rows = []
    for metric_name, val in metrics.items():
        if metric_name.endswith("_n"):
            continue  # skip count fields
        value, uncertainty = val
        row = {
            "field_type": field_type,
            "size": size,
            "n_arrays": N_REAL,
            "metric": metric_name,
            "value": value,
            "uncertainty": uncertainty,
        }
        # Add count for ind_corr_dim
        if metric_name == "ind_corr_dim":
            row["n_valid"] = metrics.get("ind_corr_dim_n", N_REAL)
        rows.append(row)

    df = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)


# ============================================================
# Main experiment loop
# ============================================================
print(f"Part 1: Convergence experiment")
print(f"FIF: H={H}, C1={C1_fif}, alpha={ALPHA}")
print(f"fBm: H={H}, C1=0")
print(f"Sizes: {SIZES}, N={N_REAL}")
print(f"Backend: {scaleinvariance.get_backend()}")
print()

# Remove old results
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

for field_type in ["fBm", "FIF"]:
    for size in SIZES:
        print(f"=== {field_type} {size}x{size} ===")

        # Generate
        t0 = time.time()
        if field_type == "FIF":
            masks = generate_fif(size, N_REAL)
        else:
            masks = generate_fbm(size, N_REAL)
        t_gen = time.time() - t0
        print(f"  Generated {N_REAL} realizations in {t_gen:.1f}s")

        # Compute metrics
        t0 = time.time()
        metrics = compute_metrics(masks, size)
        t_analysis = time.time() - t0

        for name, val in metrics.items():
            if name.endswith("_n"):
                continue
            print(f"  {name}: {val[0]:.4f} +/- {val[1]:.4f}")
        print(f"  Analysis time: {t_analysis:.1f}s")

        # Save
        save_row(field_type, size, metrics)
        del masks
        print()

print("Part 1 complete!")
print(f"Results: {CSV_PATH}")
