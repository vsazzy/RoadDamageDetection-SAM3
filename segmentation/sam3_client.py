"""HTTP client for the isolated official SAM 3 GPU service."""

import json
from io import BytesIO
from typing import Iterable, Sequence

import numpy as np
import requests
from PIL import Image


class Sam3ServiceError(RuntimeError):
    """Raised when the SAM 3 service cannot produce masks."""


class Sam3Client:
    def __init__(self, base_url: str, timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes_xyxy: Iterable[Sequence[float]],
        box_padding: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one boolean mask and one SAM score per detector box."""
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        boxes = [[float(value) for value in box] for box in boxes_xyxy]
        if not boxes:
            return np.empty((0, *image_rgb.shape[:2]), dtype=bool), np.empty(0, dtype=np.float32)

        image_buffer = BytesIO()
        Image.fromarray(image_rgb, mode="RGB").save(image_buffer, format="PNG")

        try:
            response = requests.post(
                f"{self.base_url}/segment",
                files={"image": ("image.png", image_buffer.getvalue(), "image/png")},
                data={"boxes": json.dumps(boxes), "box_padding": str(box_padding)},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise Sam3ServiceError(
                f"Could not reach the SAM 3 service at {self.base_url}: {exc}"
            ) from exc

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise Sam3ServiceError(f"SAM 3 service returned HTTP {response.status_code}: {detail}")

        try:
            with np.load(BytesIO(response.content), allow_pickle=False) as payload:
                masks = payload["masks"].astype(bool)
                scores = payload["scores"].astype(np.float32)
        except (KeyError, ValueError, OSError) as exc:
            raise Sam3ServiceError("SAM 3 service returned an invalid mask payload.") from exc

        expected_shape = (len(boxes), *image_rgb.shape[:2])
        if masks.shape != expected_shape or scores.shape != (len(boxes),):
            raise Sam3ServiceError(
                f"Unexpected SAM 3 output shapes: masks={masks.shape}, scores={scores.shape}; "
                f"expected {expected_shape} and {(len(boxes),)}."
            )
        return masks, scores
