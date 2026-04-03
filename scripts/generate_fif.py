"""
Generate 30 FIF_ND 16384^2 simulations, subsample 2x to 8192^2, save as .npy.

Parameters: H=0.4, C1=0.05, alpha=1.8
Kernels: LS2010 flux, spectral observable
"""

import numpy as np
import scaleinvariance
import time
import os

# ---------- Parameters ----------
H = 0.25
C1 = 0.05
alpha = 1.8
size = (16384, 16384)
n_realizations = 30

outdir = os.path.join(os.path.dirname(__file__), "..", "output", "large")
os.makedirs(outdir, exist_ok=True)

print(f"Parameters: H={H}, C1={C1}, alpha={alpha}")
print(f"Simulation size: {size}, subsampled to {size[0]//2}x{size[1]//2}")
print(f"Kernel: LS2010 flux, spectral observable")
print(f"Backend: {scaleinvariance.get_backend()}")
print(f"Output: {outdir}")
print()

t_total = time.time()
for i in range(n_realizations):
    outfile = os.path.join(outdir, f"fif_{i:03d}.npy")
    if os.path.exists(outfile):
        print(f"[{i+1:2d}/{n_realizations}] {outfile} exists, skipping.")
        continue

    t0 = time.time()
    print(f"[{i+1:2d}/{n_realizations}] Generating...", end=" ", flush=True)

    field = scaleinvariance.FIF_ND(
        size, alpha=alpha, C1=C1, H=H,
        kernel_construction_method_flux="LS2010",
        kernel_construction_method_observable="spectral",
        periodic=True,
    )

    # Subsample 2x
    subsampled = field[::2, ::2]
    del field

    np.save(outfile, subsampled.astype(np.float32))
    del subsampled

    dt = time.time() - t0
    print(f"{dt:.0f}s  -> {outfile}")

print(f"\nDone. Total time: {time.time() - t_total:.0f}s")
