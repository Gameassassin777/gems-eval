"""Post-processing that the metric rewards when the scored faults are sparse."""
from __future__ import annotations
import numpy as np


def keep_top_fraction(pred: np.ndarray, frac: float, valid: np.ndarray | None = None,
                      binarise: bool = False) -> np.ndarray:
    """Zero every valid pixel below the (1 - frac) quantile of the valid predictions.
    With binarise=True the kept pixels become 1.0. NaNs are preserved."""
    p = np.array(pred, dtype=np.float32, copy=True)
    finite = np.isfinite(p)
    v = finite if valid is None else (finite & valid)
    if not v.any():
        return p
    t = float(np.quantile(p[v], 1.0 - frac))
    keep = v & (p >= t) & (p > 0)
    out = np.where(keep, 1.0 if binarise else p, 0.0).astype(np.float32)
    out[~finite] = np.nan
    return out


def skeleton_top_fraction(pred: np.ndarray, frac: float, valid: np.ndarray | None = None,
                          exclude: np.ndarray | None = None, exclude_dilate_px: int = 1) -> np.ndarray:
    """The post-processing the metric rewards most in our tests: optionally zero an exclusion
    mask (e.g. known faults, dilated), keep the top `frac` of the remaining valid pixels,
    thin the result to one-pixel lines (skimage skeletonize) and set them to 1.0.
    NaNs are preserved. Needs scikit-image."""
    from scipy import ndimage
    from skimage.morphology import skeletonize
    p = np.array(pred, dtype=np.float32, copy=True)
    finite = np.isfinite(p)
    v = finite if valid is None else (finite & valid)
    if exclude is not None:
        m = ndimage.binary_dilation(exclude, iterations=exclude_dilate_px) if exclude_dilate_px > 0 else exclude
        v = v & ~m
    q = np.where(v, p, 0.0)
    t = float(np.quantile(q[v], 1.0 - frac))
    out = skeletonize(v & (q >= t) & (q > 0)).astype(np.float32)      # never select zero-probability pixels
    out[~finite] = np.nan
    return out
