"""
Plot experiment results from Part 1 and Part 2 CSVs.

Part 1: Parameter convergence vs domain size.
Part 2: Uncertainty (objscale vs bootstrap) vs n_arrays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.dirname(__file__)

METRIC_LABELS = {
    "ens_corr_dim": r"$D_{e,c}$",
    "ind_corr_dim": r"$D_c$ (largest)",
    "ind_frac_dim": r"$D_i$ (P-A)",
    "area_exp": r"$\alpha$ (area)",
    "perim_exp": r"$\beta$ (perimeter)",
}

COLORS = {"fBm": "#005F73", "FIF": "#9B2226"}

# ============================================================
# Part 1: Convergence with domain size
# ============================================================
csv1 = os.path.join(OUTDIR, "part1_results.csv")
if os.path.exists(csv1):
    df1 = pd.read_csv(csv1)

    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for ft in ["fBm", "FIF"]:
            sub = df1[(df1["metric"] == metric) & (df1["field_type"] == ft)]
            sub = sub.sort_values("size")
            valid = sub["value"].notna()
            ax.errorbar(
                sub.loc[valid, "size"], sub.loc[valid, "value"],
                yerr=sub.loc[valid, "uncertainty"],
                fmt="o-", color=COLORS[ft], ms=6, lw=1.5, capsize=4,
                label=ft,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Domain size (pixels)", fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(METRIC_LABELS[metric], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([512, 1024, 2048, 4096, 8192])
        ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])

    # Hide unused subplot
    axes[5].set_visible(False)

    fig.suptitle("Part 1: Parameter Convergence with Domain Size (n=10)", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "part1_convergence.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()
else:
    print(f"Part 1 CSV not found: {csv1}")


# ============================================================
# Part 2: Uncertainty vs n_arrays
# ============================================================
csv2 = os.path.join(OUTDIR, "part2_results.csv")
if os.path.exists(csv2):
    df2 = pd.read_csv(csv2)

    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for ft in ["fBm", "FIF"]:
            sub = df2[(df2["metric"] == metric) & (df2["field_type"] == ft)]
            sub = sub.sort_values("n_arrays")
            valid = sub["value"].notna()

            # Objscale uncertainty
            ax.plot(
                sub.loc[valid, "n_arrays"], sub.loc[valid, "objscale_uncertainty"],
                "o-", color=COLORS[ft], ms=6, lw=1.5,
                label=f"{ft} (objscale)",
            )
            # Bootstrap uncertainty
            ax.plot(
                sub.loc[valid, "n_arrays"], sub.loc[valid, "bootstrap_uncertainty"],
                "s--", color=COLORS[ft], ms=6, lw=1.5, alpha=0.7,
                label=f"{ft} (bootstrap)",
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of arrays", fontsize=11)
        ax.set_ylabel("Uncertainty", fontsize=11)
        ax.set_title(METRIC_LABELS[metric], fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])

    axes[5].set_visible(False)

    fig.suptitle("Part 2: Uncertainty vs Number of Arrays (2048², bootstrap=50)", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "part2_uncertainty.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()

    # Also plot parameter values vs n_arrays
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for ft in ["fBm", "FIF"]:
            sub = df2[(df2["metric"] == metric) & (df2["field_type"] == ft)]
            sub = sub.sort_values("n_arrays")
            valid = sub["value"].notna()
            ax.errorbar(
                sub.loc[valid, "n_arrays"], sub.loc[valid, "value"],
                yerr=sub.loc[valid, "bootstrap_uncertainty"],
                fmt="o-", color=COLORS[ft], ms=6, lw=1.5, capsize=4,
                label=ft,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of arrays", fontsize=11)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=12)
        ax.set_title(METRIC_LABELS[metric], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])

    axes[5].set_visible(False)

    fig.suptitle("Part 2: Parameter Values vs Number of Arrays (2048², bootstrap CI)", fontsize=14)
    plt.tight_layout()
    outpath = os.path.join(OUTDIR, "part2_values.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close()
else:
    print(f"Part 2 CSV not found: {csv2}")
