import logging
import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file
from segmentation import (
    Sam3Client,
    Sam3ServiceError,
    combine_masks,
    make_mask_overlay,
    mask_to_png_bytes,
    scale_box_xyxy,
)

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

use_sam3 = st.checkbox(
    "Generate precise SAM 3 masks from YOLO bounding boxes",
    value=False,
    help="Requires the separate CUDA SAM 3 service described in sam3_service/README.md.",
)
sam3_service_url = os.getenv("SAM3_SERVICE_URL", "http://127.0.0.1:8001")
if use_sam3:
    st.caption(f"SAM 3 service: {sam3_service_url}")
    mask_opacity = st.slider(
        "Mask Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )
    box_padding = st.slider(
        "SAM Box Padding", min_value=0.0, max_value=0.25, value=0.05, step=0.01,
        help="Allows the mask to extend slightly beyond YOLO's box.",
    )

if image_file is not None:

    # Load the image
    image = Image.open(image_file).convert("RGB")
    
    # Perform inference
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls.item()),
               label=CLASSES[int(_box.cls.item())],
               score=float(_box.conf.item()),
               # Keep the model's original stretched-square preprocessing, then
               # map the detector box back onto the untouched source for SAM 3.
               box=scale_box_xyxy(
                   _box.xyxy[0], source_size=(640, 640), target_size=(w_ori, h_ori)
               ).astype(int),
            )
            for _box in boxes
        ]

    # Ultralytics returns the plotted ndarray in BGR order.
    annotated_bgr = cv2.resize(
        results[0].plot(), (w_ori, h_ori), interpolation=cv2.INTER_AREA
    )
    _image_pred = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    columns = st.columns(3 if use_sam3 else 2)

    # Original Image
    with columns[0]:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with columns[1]:
        st.write("#### YOLO Boxes")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )

    if use_sam3:
        with columns[2]:
            st.write("#### SAM 3 Masks")
            if not detections:
                st.info("YOLO found no road damage to use as a SAM 3 box prompt.")
            else:
                boxes = [detection.box.tolist() for detection in detections]
                labels = [
                    f"{detection.label} · YOLO {detection.score:.2f}"
                    for detection in detections
                ]
                try:
                    with st.spinner("Refining YOLO boxes into SAM 3 masks..."):
                        masks, sam_scores = Sam3Client(sam3_service_url).segment(
                            _image, boxes, box_padding=box_padding
                        )
                    labels = [
                        f"{label} · SAM {sam_score:.2f}"
                        for label, sam_score in zip(labels, sam_scores)
                    ]
                    overlay = make_mask_overlay(
                        _image, masks, boxes, labels, alpha=mask_opacity
                    )
                    st.image(overlay)

                    overlay_buffer = BytesIO()
                    Image.fromarray(overlay).save(overlay_buffer, format="PNG")
                    st.download_button(
                        label="Download SAM 3 Overlay",
                        data=overlay_buffer.getvalue(),
                        file_name="RDD_SAM3_Overlay.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                    combined_mask = combine_masks(masks, _image.shape[:2])
                    st.download_button(
                        label="Download Binary Mask",
                        data=mask_to_png_bytes(combined_mask),
                        file_name="RDD_SAM3_Mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Sam3ServiceError as exc:
                    st.error(str(exc))
                    st.caption(
                        "Start the CUDA service using sam3_service/README.md, then set "
                        "SAM3_SERVICE_URL before launching Streamlit."
                    )
