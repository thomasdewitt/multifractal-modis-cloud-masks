"""
Shared loader: reads the 30 FIF arrays from output/large/,
thresholds at the mean to produce binary cloud masks,
and optionally subsets into 4x8 MODIS-like scenes (1024x2048).
"""

import numpy as np
import os

LARGE_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "large")
N_REALIZATIONS = 30
N_ANALYSIS = 10  # number of parents to load for analysis


def load_large_masks(n=N_ANALYSIS):
    """Load n full 8192^2 binary cloud masks (thresholded at mean)."""
    masks = []
    for i in range(n):
        arr = np.load(os.path.join(LARGE_DIR, f"fif_{i:03d}.npy"))
        masks.append((arr > arr.mean()).astype(np.float64))
    return masks


def subset_masks(mask, nrows=4, ncols=8):
    """Split an 8192^2 mask into nrows x ncols sub-scenes (1024 x 2048)."""
    h, w = mask.shape
    sh, sw = h // nrows, w // ncols
    scenes = []
    for r in range(nrows):
        for c in range(ncols):
            scenes.append(mask[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw])
    return scenes


def load_all_subscenes(n=N_ANALYSIS):
    """Load n x 32 MODIS-like 1024x2048 sub-scenes."""
    scenes = []
    for mask in load_large_masks(n):
        scenes.extend(subset_masks(mask))
    return scenes
