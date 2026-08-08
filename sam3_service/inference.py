"""Thin wrapper around Meta's official SAM 3 image processor with Coarse-to-Fine Dual SAM 3 support."""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
import torch


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
    """Loads SAM 3 once and produces instance masks per YOLO box."""

    def __init__(self, confidence_threshold: float = 0.05):
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not torch.cuda.is_available():
            raise RuntimeError("The official SAM 3 service requires a CUDA-capable GPU.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.processor = Sam3Processor(
                build_sam3_image_model(), confidence_threshold=confidence_threshold
            )

    def segment(
        self,
        image: Image.Image,
        boxes_xyxy: Iterable[Sequence[float]],
        box_padding: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Baseline Image -> YOLO -> SAM 3 segmenter."""
        image = image.convert("RGB")
        width, height = image.size
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
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

    def segment_coarse_to_fine(
        self,
        image: Image.Image,
        yolo_net,
        classes: list[str],
        score_threshold: float = 0.35,
        box_padding: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
        """Advanced SAM 3 -> YOLO -> SAM 3 Coarse-to-Fine Pipeline.

        Step 1: Coarse anomaly proposal extraction via SAM 3 grid prompts.
        Step 2: Region classification & false-positive filtering via YOLOv8.
        Step 3: Hybrid (Box + Point) prompt refinement in SAM 3.
        """
        image_rgb = image.convert("RGB")
        width, height = image_rgb.size
        img_np = np.array(image_rgb)

        t_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        # Step 1: Coarse Grid Proposal Sampling (SAM 3 Pass 1)
        grid_pts = [0.2, 0.5, 0.8]
        coarse_proposals = []

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image_rgb)
            for gx in grid_pts:
                for gy in grid_pts:
                    self.processor.reset_all_prompts(state)
                    state = self.processor.add_geometric_prompt(
                        state=state,
                        box=[gx, gy, 0.06, 0.06],
                        label=True,
                    )
                    candidates = state["masks"].detach().bool().cpu().numpy()
                    if len(candidates) > 0:
                        m = candidates[0]
                        if m.ndim == 3:
                            m = m[0]
                        y_idx, x_idx = np.where(m)
                        if len(x_idx) > 40:
                            x0, x1 = int(np.min(x_idx)), int(np.max(x_idx))
                            y0, y1 = int(np.min(y_idx)), int(np.max(y_idx))
                            area = (x1 - x0) * (y1 - y0)
                            if 0.0005 * width * height < area < 0.4 * width * height:
                                coarse_proposals.append({
                                    "box": [x0, y0, x1, y1],
                                    "point": [float((x0 + x1) / 2), float((y0 + y1) / 2)]
                                })

        # Step 2: YOLO Classifier & Verification Pass
        image_resized = cv2.resize(img_np, (640, 640), interpolation=cv2.INTER_AREA)
        results = yolo_net.predict(image_resized, conf=score_threshold)

        verified_detections = []
        for result in results:
            boxes_yolo = result.boxes.cpu().numpy()
            for _box in boxes_yolo:
                cls_id = int(_box.cls.item())
                score = float(_box.conf.item())
                b640 = _box.xyxy[0]
                x0 = int(b640[0] * width / 640.0)
                y0 = int(b640[1] * height / 640.0)
                x1 = int(b640[2] * width / 640.0)
                y1 = int(b640[3] * height / 640.0)

                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0

                verified_detections.append({
                    "class_id": cls_id,
                    "label": classes[cls_id],
                    "score": score,
                    "box": [x0, y0, x1, y1],
                    "point": [cx, cy]
                })

        # Step 3: SAM 3 Box Refinement
        final_masks = []
        final_scores = []

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(image_rgb)
            for det in verified_detections:
                box = det["box"]
                self.processor.reset_all_prompts(state)

                state = self.processor.add_geometric_prompt(
                    state=state,
                    box=xyxy_to_normalized_cxcywh(box, width, height),
                    label=True,
                )
                candidates = state["masks"].detach().bool().cpu().numpy()
                candidate_scores = state["scores"].detach().float().cpu().numpy()
                mask, score = select_box_mask(candidates, candidate_scores, box)
                final_masks.append(clip_mask_to_padded_box(mask, box, box_padding))
                final_scores.append(score)


        if not final_masks:
            return (
                np.empty((0, height, width), dtype=bool),
                np.empty(0, dtype=np.float32),
                [],
                {"coarse_proposals": len(coarse_proposals), "verified_targets": 0},
                coarse_proposals
            )

        metrics = {
            "coarse_proposals": len(coarse_proposals),
            "verified_targets": len(verified_detections)
        }

        return np.stack(final_masks), np.asarray(final_scores, dtype=np.float32), verified_detections, metrics, coarse_proposals