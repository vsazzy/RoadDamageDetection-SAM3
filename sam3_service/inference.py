"""Thin wrapper around Meta's official SAM 3 image processor."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def xyxy_to_normalized_cxcywh(
    box_xyxy: Sequence[float], width: int, height: int
) -> list[float]:
    """Convert an absolute XYXY box to SAM 3's normalized CXCYWH format."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if len(box_xyxy) != 4:
        raise ValueError("Each box must contain x0, y0, x1, and y1.")
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    x0, x1 = sorted((max(0.0, min(x0, width)), max(0.0, min(x1, width))))
    y0, y1 = sorted((max(0.0, min(y0, height)), max(0.0, min(y1, height))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Box has no area after clipping: {box_xyxy}.")
    return [
        ((x0 + x1) / 2.0) / width,
        ((y0 + y1) / 2.0) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]


def select_box_mask(
    candidate_masks: np.ndarray,
    candidate_scores: np.ndarray,
    box_xyxy: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Select the candidate best aligned with a detector box."""
    masks = np.asarray(candidate_masks, dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or len(masks) == 0:
        raise ValueError("SAM 3 returned no candidate masks for a detector box.")
    scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    if len(scores) != len(masks):
        raise ValueError("SAM 3 returned different candidate mask and score counts.")

    height, width = masks.shape[1:]
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    x0, x1 = sorted((int(np.floor(x0)), int(np.ceil(x1))))
    y0, y1 = sorted((int(np.floor(y0)), int(np.ceil(y1))))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)

    inside_pixels = masks[:, y0:y1, x0:x1].sum(axis=(1, 2), dtype=np.float64)
    total_pixels = masks.sum(axis=(1, 2), dtype=np.float64)
    inside_fraction = np.divide(
        inside_pixels,
        total_pixels,
        out=np.zeros_like(inside_pixels),
        where=total_pixels > 0,
    )
    ranking = scores + inside_fraction.astype(np.float32)
    best_index = int(np.argmax(ranking))
    return masks[best_index], float(scores[best_index])


def clip_mask_to_padded_box(
    mask: np.ndarray, box_xyxy: Sequence[float], padding: float
) -> np.ndarray:
    """Suppress unrelated instances outside a detector box plus relative padding."""
    if padding < 0.0:
        raise ValueError("padding must be non-negative.")
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    box_width, box_height = abs(x1 - x0), abs(y1 - y0)
    left = max(0, int(np.floor(min(x0, x1) - box_width * padding)))
    right = min(width, int(np.ceil(max(x0, x1) + box_width * padding)))
    top = max(0, int(np.floor(min(y0, y1) - box_height * padding)))
    bottom = min(height, int(np.ceil(max(y0, y1) + box_height * padding)))
    clipped = np.zeros_like(mask)
    clipped[top:bottom, left:right] = mask[top:bottom, left:right]
    return clipped


class Sam3BoxSegmenter:
    """Loads SAM 3 once and produces one instance mask per YOLO box."""

    def __init__(self, confidence_threshold: float = 0.05):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("The official SAM 3 service requires a CUDA-capable GPU.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self._autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        self._autocast.__enter__()
        self.processor = Sam3Processor(
            build_sam3_image_model(), confidence_threshold=confidence_threshold
        )

    def segment(
        self,
        image: Image.Image,
        boxes_xyxy: Iterable[Sequence[float]],
        box_padding: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = image.convert("RGB")
        width, height = image.size
        state = self.processor.set_image(image)
        masks = []
        scores = []
        for box in boxes_xyxy:
            self.processor.reset_all_prompts(state)
            state = self.processor.add_geometric_prompt(
                state=state,
                box=xyxy_to_normalized_cxcywh(box, width, height),
                label=True,
            )
            # SAM 3 runs under BF16 autocast. NumPy cannot represent BF16, so
            # normalize tensors to supported CPU dtypes before conversion.
            candidates = state["masks"].detach().bool().cpu().numpy()
            candidate_scores = state["scores"].detach().float().cpu().numpy()
            mask, score = select_box_mask(candidates, candidate_scores, box)
            masks.append(clip_mask_to_padded_box(mask, box, box_padding))
            scores.append(score)

        if not masks:
            return (
                np.empty((0, height, width), dtype=bool),
                np.empty(0, dtype=np.float32),
            )
        return np.stack(masks), np.asarray(scores, dtype=np.float32)
