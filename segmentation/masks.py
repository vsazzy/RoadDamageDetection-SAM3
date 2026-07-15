"""Mask visualization helpers shared by the Streamlit application and tests."""

from io import BytesIO
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MASK_COLORS = (
    (255, 59, 48),
    (52, 199, 89),
    (0, 122, 255),
    (255, 149, 0),
    (175, 82, 222),
    (90, 200, 250),
)


def scale_box_xyxy(
    box_xyxy: Sequence[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale an XYXY box between images expressed as (width, height)."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Source and target dimensions must be positive.")
    box = np.asarray(box_xyxy, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError("box_xyxy must contain exactly four values.")
    scale = np.asarray(
        [
            target_width / source_width,
            target_height / source_height,
            target_width / source_width,
            target_height / source_height,
        ],
        dtype=np.float32,
    )
    return box * scale


def _as_mask_stack(masks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0:
        return np.empty((0, *image_shape), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(image_shape):
        raise ValueError(
            f"Expected masks shaped (N, {image_shape[0]}, {image_shape[1]}), "
            f"received {masks.shape}."
        )
    return masks


def combine_masks(masks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Return a uint8 binary mask containing every predicted instance."""
    mask_stack = _as_mask_stack(masks, image_shape)
    if len(mask_stack) == 0:
        return np.zeros(image_shape, dtype=np.uint8)
    return np.any(mask_stack, axis=0).astype(np.uint8) * 255


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a binary/label mask as a lossless PNG."""
    buffer = BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(buffer, "PNG")
    return buffer.getvalue()


def make_mask_overlay(
    image_rgb: np.ndarray,
    masks: np.ndarray,
    boxes: Iterable[Sequence[float]],
    labels: Sequence[str],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend instance masks onto an RGB image and redraw their detector boxes."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3).")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    height, width = image_rgb.shape[:2]
    mask_stack = _as_mask_stack(masks, (height, width))
    boxes = list(boxes)
    if not (len(mask_stack) == len(boxes) == len(labels)):
        raise ValueError("masks, boxes, and labels must contain the same number of items.")

    overlay = image_rgb.astype(np.float32).copy()
    for index, mask in enumerate(mask_stack):
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

    rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    for index, (box, label) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = (float(value) for value in box)
        color = MASK_COLORS[index % len(MASK_COLORS)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text_box = draw.textbbox((x0, y0), label, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        text_y = max(0.0, y0 - text_height - 6)
        draw.rectangle((x0, text_y, x0 + text_width + 8, text_y + text_height + 6), fill=color)
        draw.text((x0 + 4, text_y + 3), label, fill=(255, 255, 255), font=font)

    return np.asarray(rendered)
