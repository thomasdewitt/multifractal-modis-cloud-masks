"""
1D power spectra and first-order structure functions of the continuous
(non-binary) FIF fields, averaged over 10 parents.

Theoretical reference slopes:
  - Spectral: beta = 1 + 2H - K(2)
  - Structure function: xi(1) = H - K(1) = H
"""

import numpy as np
import matplotlib.pyplot as plt
import scaleinvariance
import os

LARGE_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "large")
N_ANALYSIS = 10

# Parameters
H = 0.25
C1 = 0.05
alpha = 1.8

K1 = scaleinvariance.K(1, C1, alpha)
K2 = scaleinvariance.K(2, C1, alpha)
beta_theory = 1 + 2 * H - K2
xi1_theory = H - K1  # = H

print(f"H={H}, C1={C1}, alpha={alpha}")
print(f"K(1)={K1:.6f}, K(2)={K2:.6f}")
print(f"Theoretical beta  = 1+2H-K(2) = {beta_theory:.4f}")
print(f"Theoretical xi(1) = H         = {xi1_theory:.4f}")

# Accumulate spectra and structure functions
spec_accum = None
sf_accum = None

for i in range(N_ANALYSIS):
    print(f"  [{i+1}/{N_ANALYSIS}]", end=" ", flush=True)
    field = np.load(os.path.join(LARGE_DIR, f"fif_{i:03d}.npy"))

    freqs, psd = scaleinvariance.spectral_analysis(field, axis=1)
    lags, sf = scaleinvariance.structure_function_analysis(field, order=1, axis=1)

    if spec_accum is None:
        spec_accum = psd.copy()
        sf_accum = sf.copy()
        spec_freqs = freqs
        sf_lags = lags
    else:
        spec_accum += psd
        sf_accum += sf

    del field
    print("done")

spec_avg = spec_accum / N_ANALYSIS
sf_avg = sf_accum / N_ANALYSIS

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# -- Power spectrum --
# Reference line first (behind)
mid = len(spec_freqs) // 2
f0, p0 = spec_freqs[mid], spec_avg[mid]
ref_spec = p0 * (spec_freqs / f0) ** (-beta_theory)
ax1.loglog(spec_freqs, ref_spec, "k-", lw=0.8, alpha=1.0, zorder=1)

# Data on top
ax1.loglog(spec_freqs, spec_avg, "o", color="#2D6A4F", ms=5, zorder=2,
           label=rf"Measured PSD")
# Invisible line for legend entry
ax1.plot([], [], "k-", lw=0.8,
         label=rf"$\beta = 1+2H-K(2) = {beta_theory:.3f}$")

ax1.set_xlabel("Frequency", fontsize=12)
ax1.set_ylabel("PSD", fontsize=12)
ax1.set_title("Power Spectrum (1D, along axis)", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which="both")

# -- Structure function --
# Reference line first
mid = len(sf_lags) // 2
r0, s0 = sf_lags[mid], sf_avg[mid]
ref_sf = s0 * (sf_lags / r0) ** xi1_theory
ax2.loglog(sf_lags, ref_sf, "k-", lw=0.8, alpha=1.0, zorder=1)

# Data on top
ax2.loglog(sf_lags, sf_avg, "o", color="#E07A5F", ms=5, zorder=2,
           label=r"Measured $S_1(r)$")
ax2.plot([], [], "k-", lw=0.8,
         label=rf"$\xi(1) = H = {xi1_theory:.3f}$")

ax2.set_xlabel(r"Lag $r$", fontsize=12)
ax2.set_ylabel(r"$S_1(r)$", fontsize=12)
ax2.set_title("First-Order Structure Function", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle(
    rf"Continuous FIF fields ($n={N_ANALYSIS}$, $8192^2$): "
    rf"$H={H}$, $C_1={C1}$, $\alpha={alpha}$",
    fontsize=14,
)
plt.tight_layout()
outpath = "../figures/scaling_analysis.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
