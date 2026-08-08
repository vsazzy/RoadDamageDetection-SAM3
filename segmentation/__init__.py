"""Utilities for converting detector boxes into segmentation overlays."""

from .masks import (
    combine_masks,
    make_mask_overlay,
    make_proposal_overlay,
    mask_to_png_bytes,
    scale_box_xyxy,
)

__all__ = [
    "combine_masks",
    "make_mask_overlay",
    "make_proposal_overlay",
    "mask_to_png_bytes",
    "scale_box_xyxy",
]