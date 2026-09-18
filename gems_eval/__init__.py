"""gems-eval: local scoring for the DOE GEMS fault-detection challenge (DrivenData 306).

Everything here works on plain numpy arrays; rasterio is only needed for the file helpers.
"""
from .dti import dti
from .holdout import split_segments, holdout_dti
from .blocks import block_folds
from .postprocess import keep_top_fraction, skeleton_top_fraction

__all__ = ["dti", "split_segments", "holdout_dti", "block_folds", "keep_top_fraction", "skeleton_top_fraction"]
__version__ = "0.1.0"
