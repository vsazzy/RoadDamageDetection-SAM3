"""Utilities for converting detector boxes into segmentation overlays."""

from .masks import combine_masks, make_mask_overlay, mask_to_png_bytes, scale_box_xyxy
from .sam3_client import Sam3Client, Sam3ServiceError

__all__ = [
    "Sam3Client",
    "Sam3ServiceError",
    "combine_masks",
    "make_mask_overlay",
    "mask_to_png_bytes",
    "scale_box_xyxy",
]
