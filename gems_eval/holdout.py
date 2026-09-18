"""Held-out-segment protocol.

The leaderboard scores only faults that are NOT in the training labels, and the training
labels are excluded from scoring. A model that re-draws the known faults therefore scores
near zero no matter how good it looks locally. This protocol imitates the board:

  1. split the known faults into connected segments;
  2. hold out a fraction of the segments (default 30%) from training;
  3. score predictions on the held-out segments only, with a buffer around the remaining
     (training) faults excluded from scoring, as the organizer excludes known faults.

Evaluation raster codes: 1 = held-out truth, 2 = excluded buffer, 0 = background, -1 = nodata.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from .dti import dti


def split_segments(labels: np.ndarray, frac: float = 0.3, seed: int = 0,
                   exclude_buffer_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """labels: int array, >=1 fault, 0 background, <0 nodata.
    Returns (train_labels, eval_labels) as int8 arrays with the codes described above."""
    known = labels >= 1
    comp, n = ndimage.label(known, structure=np.ones((3, 3)))
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    rng.shuffle(ids)
    held = ids[: int(round(frac * n))]
    held_mask = np.isin(comp, held)
    train = np.where(labels < 0, -1, np.where(known & ~held_mask, 1, 0)).astype(np.int8)
    excl = ndimage.binary_dilation(known & ~held_mask, iterations=exclude_buffer_px) if exclude_buffer_px > 0 \
        else (known & ~held_mask)
    ev = np.where(labels < 0, -1, np.where(held_mask, 1, np.where(excl, 2, 0))).astype(np.int8)
    return train, ev


def holdout_dti(pred: np.ndarray, eval_labels: np.ndarray, **kw) -> float:
    """DTI on held-out segments only; buffer and nodata pixels are excluded."""
    valid = (eval_labels == 0) | (eval_labels == 1)
    return dti(pred, eval_labels == 1, valid=valid, **kw)


def summary(eval_labels: np.ndarray) -> str:
    n_held = int((eval_labels == 1).sum()); n_buf = int((eval_labels == 2).sum())
    n_bg = int((eval_labels == 0).sum())
    return f"held-out px {n_held}, excluded buffer px {n_buf}, scoreable background px {n_bg}"
