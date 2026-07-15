"""FastAPI entry point for official SAM 3 box-prompt inference."""

import json
import os
from io import BytesIO
from threading import Lock

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from .inference import Sam3BoxSegmenter


app = FastAPI(title="Road Damage SAM 3 Service", version="1.0.0")
_segmenter = None
_segmenter_lock = Lock()


def get_segmenter() -> Sam3BoxSegmenter:
    global _segmenter
    if _segmenter is None:
        with _segmenter_lock:
            if _segmenter is None:
                threshold = float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.05"))
                _segmenter = Sam3BoxSegmenter(confidence_threshold=threshold)
    return _segmenter


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _segmenter is not None}


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    box_padding: float = Form(0.05),
) -> Response:
    try:
        parsed_boxes = json.loads(boxes)
        if not isinstance(parsed_boxes, list) or any(
            not isinstance(box, list) or len(box) != 4 for box in parsed_boxes
        ):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="boxes must be a JSON list of XYXY boxes.")
    if not 0.0 <= box_padding <= 0.5:
        raise HTTPException(status_code=400, detail="box_padding must be between 0 and 0.5.")

    try:
        source_image = Image.open(BytesIO(await image.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image is not a valid supported image.") from exc

    try:
        masks, scores = get_segmenter().segment(source_image, parsed_boxes, box_padding)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload = BytesIO()
    np.savez_compressed(payload, masks=masks.astype(np.uint8), scores=scores)
    return Response(payload.getvalue(), media_type="application/x-npz")
