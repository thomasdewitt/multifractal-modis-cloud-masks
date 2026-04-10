"""
Shared loader: reads bool child cloud masks from output/masks/ and
exposes the original API used by the analysis scripts.

Each child on disk is a 1024 x 2048 bool sub-scene. Full 8192^2 parent
masks are reconstructed on demand by stitching each parent's 4 x 8 grid
of children.
"""

import numpy as np
import os

MASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "masks")
N_ANALYSIS = 50  # number of parents to load for analysis
NROWS, NCOLS = 4, 8


def load_large_masks(n=N_ANALYSIS):
    """Load n full 8192^2 bool cloud masks (stitched from their children)."""
    masks = []
    for i in range(n):
        rows = [
            np.concatenate(
                [
                    np.load(os.path.join(MASKS_DIR, f"child_{i:03d}_r{r}_c{c}.npy"))
                    for c in range(NCOLS)
                ],
                axis=1,
            )
            for r in range(NROWS)
        ]
        masks.append(np.concatenate(rows, axis=0))
    return masks


def subset_masks(mask, nrows=NROWS, ncols=NCOLS):
    """Split an 8192^2 mask into nrows x ncols sub-scenes (1024 x 2048)."""
    h, w = mask.shape
    sh, sw = h // nrows, w // ncols
    scenes = []
    for r in range(nrows):
        for c in range(ncols):
            scenes.append(mask[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw])
    return scenes


def load_all_subscenes(n=N_ANALYSIS):
    """Load all n x 32 MODIS-like 1024 x 2048 child sub-scenes directly."""
    scenes = []
    for i in range(n):
        for r in range(NROWS):
            for c in range(NCOLS):
                scenes.append(
                    np.load(os.path.join(MASKS_DIR, f"child_{i:03d}_r{r}_c{c}.npy"))
                )
    return scenes
