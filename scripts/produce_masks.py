"""
Produce bool child cloud masks from the large FIF simulations.

For each parent fif_NNN.npy in output/large/:
  - threshold at the parent's mean (cloudy where value > mean)
  - split into 4 x 8 = 32 sub-scenes of shape 1024 x 2048
  - save each sub-scene as a bool .npy in output/masks/

One parent is held in memory at a time.
"""

import numpy as np
import os
import time

LARGE_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "large")
MASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "masks")
N_REALIZATIONS = 50
NROWS, NCOLS = 4, 8

os.makedirs(MASKS_DIR, exist_ok=True)

print(f"Input:  {LARGE_DIR}")
print(f"Output: {MASKS_DIR}")
print(f"Grid:   {NROWS} x {NCOLS} children per parent\n")

t_total = time.time()
written = 0
skipped = 0
for i in range(N_REALIZATIONS):
    infile = os.path.join(LARGE_DIR, f"fif_{i:03d}.npy")
    if not os.path.exists(infile):
        print(f"[{i+1:2d}/{N_REALIZATIONS}] {infile} missing, skipping.")
        continue

    t0 = time.time()
    arr = np.load(infile)
    mask = arr > arr.mean()
    del arr

    h, w = mask.shape
    sh, sw = h // NROWS, w // NCOLS

    for r in range(NROWS):
        for c in range(NCOLS):
            outfile = os.path.join(MASKS_DIR, f"child_{i:03d}_r{r}_c{c}.npy")
            if os.path.exists(outfile):
                skipped += 1
                continue
            np.save(outfile, mask[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw])
            written += 1

    del mask
    dt = time.time() - t0
    print(f"[{i+1:2d}/{N_REALIZATIONS}] wrote {NROWS*NCOLS} children ({dt:.1f}s)")

print(
    f"\nDone. Wrote {written} files, skipped {skipped} existing. "
    f"Total: {time.time() - t_total:.0f}s"
)
