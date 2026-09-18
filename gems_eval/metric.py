"""Compatibility shim. The metric lives in gems_eval.dti; an earlier draft of this module
computed a tolerant Dice (symmetric, binary-dilation based), which is NOT the competition's
distance-weighted Tversky index and is removed."""
from .dti import dti as compute_dti  # noqa: F401
