"""Spatial block folds for a single large raster (an alternative, stricter proxy: whole
regions unseen, not just segments)."""
from __future__ import annotations
import numpy as np


def block_folds(height: int, width: int, block_px: int = 512, n_folds: int = 5,
                seed: int = 0, buffer_px: int = 0) -> np.ndarray:
    """Fold id in [0, n_folds) per pixel, assigned by square blocks; -1 within buffer_px of a block edge."""
    rng = np.random.default_rng(seed)
    nby, nbx = -(-height // block_px), -(-width // block_px)
    block_fold = rng.permutation(np.arange(nby * nbx) % n_folds).reshape(nby, nbx)
    folds = np.repeat(np.repeat(block_fold, block_px, axis=0), block_px, axis=1)[:height, :width]
    if buffer_px > 0:
        yy, xx = np.mgrid[:height, :width]
        near_edge = ((yy % block_px) < buffer_px) | ((yy % block_px) >= block_px - buffer_px) | \
                    ((xx % block_px) < buffer_px) | ((xx % block_px) >= block_px - buffer_px)
        folds = np.where(near_edge, -1, folds)
    return folds.astype(np.int8)
