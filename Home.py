import logging
import os
from pathlib import Path
from typing import NamedTuple
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

from sample_utils.download import download_file
from segmentation import (
    combine_masks,
    make_mask_overlay,
    mask_to_png_bytes,
    scale_box_xyxy,
    make_proposal_overlay
)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Road Damage Detection & Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)

# Ensure YOLO model checkpoint is downloaded
MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Load YOLO model into session cache
@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_LOCAL_PATH)

net = load_yolo_model()

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

# 🚀 Load direct in-process SAM 3 segmenter
@st.cache_resource
def get_sam3_segmenter():
    import importlib
    import sam3_service.inference
    importlib.reload(sam3_service.inference)
    from sam3_service.inference import Sam3BoxSegmenter
    return Sam3BoxSegmenter()


# Title & Subtitle
st.title("🛣️ Road Damage Detection & SAM 3 Segmentation")
st.caption("A multi-tier pipeline for road damage detection and zero-shot pixel mask refinement.")

# Sidebar Configuration Controls
st.sidebar.header("⚙️ Configuration Controls")

pipeline_mode = st.sidebar.radio(
    "Pipeline Technique",
    [
        "Baseline: Image ➔ YOLO ➔ SAM 3",
        "Advanced: SAM 3 ➔ YOLO ➔ SAM 3"
    ],
    help="Choose between standard fast baseline or advanced dual-pass hybrid prompting."
)

score_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
    help="Lower threshold if damage is missed; increase if false positives occur."
)

use_sam3 = st.sidebar.checkbox(
    "Enable SAM 3 Pixel Mask Refinement",
    value=True,
    help="Runs Meta SAM 3 directly inside Streamlit GPU memory."
)

if use_sam3:
    mask_opacity = st.sidebar.slider(
        "Mask Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05
    )
    box_padding = st.sidebar.slider(
        "SAM Box Padding", min_value=0.0, max_value=0.25, value=0.05, step=0.01,
        help="Allows the SAM 3 mask to extend slightly beyond YOLO's box."
    )
else:
    mask_opacity = 0.45
    box_padding = 0.05

with st.expander("ℹ️ About the Dataset & Architecture Modes"):
    st.markdown("""
    * **Dataset Source**: Trained on public **CRDDC2022 / RDD2022** dataset (Japan & India subsets).
    * **Damage Categories**: `D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Potholes.
    * **Pipeline Modes**:
      * **Baseline ($Image \rightarrow YOLO \rightarrow SAM 3$)**: Fast candidate filtering via YOLO followed by box-prompted SAM 3 segmentation (~1 FPS).
      * **Advanced ($SAM 3 \rightarrow YOLO \rightarrow SAM 3$)**: Coarse anomaly discovery pass, YOLO target verification, and final Hybrid (Box + Point) prompt refinement.
    """)

# Image Upload Widget
image_file = st.file_uploader("Upload Road Image", type=['png', 'jpg', 'jpeg'])

if image_file is not None:
    # 1. Load Original Image
    image = Image.open(image_file).convert("RGB")
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]

    is_coarse_fine_mode = "Advanced" in pipeline_mode

    if not is_coarse_fine_mode:
        # ⚡ BASELINE MODE (Image -> YOLO -> SAM 3)
        image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)

        start_yolo = time.perf_counter()
        results = net.predict(image_resized, conf=score_threshold)
        yolo_latency_ms = (time.perf_counter() - start_yolo) * 1000.0

        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            detections.extend([
                Detection(
                    class_id=int(_box.cls.item()),
                    label=CLASSES[int(_box.cls.item())],
                    score=float(_box.conf.item()),
                    box=scale_box_xyxy(
                        _box.xyxy[0], source_size=(640, 640), target_size=(w_ori, h_ori)
                    ).astype(int),
                )
                for _box in boxes
            ])

        annotated_bgr = cv2.resize(
            results[0].plot(), (w_ori, h_ori), interpolation=cv2.INTER_AREA
        )
        _image_pred = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        columns = st.columns(3 if use_sam3 else 2)

        with columns[0]:
            st.write("#### Input Image")
            st.image(_image, use_container_width=True)

        with columns[1]:
            st.write("####  YOLOv8 Bounding Boxes")
            st.image(_image_pred, use_container_width=True)

            buffer = BytesIO()
            Image.fromarray(_image_pred).save(buffer, format="PNG")
            st.download_button(
                label="Download Prediction Image",
                data=buffer.getvalue(),
                file_name="RDD_YOLO_Prediction.png",
                mime="image/png",
                use_container_width=True
            )

        sam3_latency_ms = 0.0
        sam3_masks_count = 0

        if use_sam3:
            with columns[2]:
                st.write("####  SAM 3 Pixel Masks")
                if not detections:
                    st.info("YOLO found no road damage to prompt SAM 3.")
                else:
                    boxes = [detection.box.tolist() for detection in detections]
                    labels = [
                        f"{detection.label} · YOLO {detection.score:.2f}"
                        for detection in detections
                    ]
                    try:
                        start_sam3 = time.perf_counter()
                        with st.spinner("Refining YOLO boxes into SAM 3 masks in GPU memory..."):
                            segmenter = get_sam3_segmenter()
                            masks, sam_scores = segmenter.segment(
                                image, boxes, box_padding=box_padding
                            )

                        sam3_latency_ms = (time.perf_counter() - start_sam3) * 1000.0
                        sam3_masks_count = len(masks)

                        labels = [
                            f"{label} · SAM {sam_score:.2f}"
                            for label, sam_score in zip(labels, sam_scores)
                        ]
                        overlay = make_mask_overlay(
                            _image, masks, boxes, labels, alpha=mask_opacity
                        )
                        st.image(overlay, use_container_width=True)

                        overlay_buffer = BytesIO()
                        Image.fromarray(overlay).save(overlay_buffer, format="PNG")
                        st.download_button(
                            label="Download SAM 3 Overlay",
                            data=overlay_buffer.getvalue(),
                            file_name="RDD_SAM3_Overlay.png",
                            mime="image/png",
                            use_container_width=True
                        )

                        combined_mask = combine_masks(masks, _image.shape[:2])
                        st.download_button(
                            label="Download Binary Mask",
                            data=mask_to_png_bytes(combined_mask),
                            file_name="RDD_SAM3_Binary_Mask.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as exc:
                        st.error(f"SAM 3 Direct Segmentation Error: {exc}")

        total_latency_ms = yolo_latency_ms + sam3_latency_ms
        fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

        st.markdown("---")
        st.write("### Baseline Performance & Latency Benchmark")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("YOLO Detection", f"{yolo_latency_ms:.1f} ms")
        if use_sam3 and detections:
            m2.metric("SAM 3 Segmentation", f"{sam3_latency_ms:.1f} ms", delta=f"{sam3_masks_count} masks")
        elif use_sam3:
            m2.metric("SAM 3 Segmentation", "0.0 ms", delta="No boxes")
        else:
            m2.metric("SAM 3 Segmentation", "Disabled")

        m3.metric("Total Pipeline Latency", f"{total_latency_ms:.1f} ms")
        m4.metric("Inference Throughput", f"{fps:.1f} FPS")

    else:
        # 🔥 ADVANCED MODE (SAM 3 -> YOLO -> SAM 3 Coarse-to-Fine)
        st.write("### Coarse-to-Fine Dual SAM 3 Pipeline ($SAM 3 ➔ YOLO ➔ SAM 3$)")
        
        start_pipeline = time.perf_counter()
        try:
            with st.spinner("Running SAM 3 Pass 1 (Coarse Anomaly Discovery) ➔ YOLO Pass (Verification) ➔ SAM 3 Pass 2 (Point+Box Hybrid Refinement)..."):
                segmenter = get_sam3_segmenter()
                masks, sam_scores, verified_dets, metrics_dict, coarse_proposals = segmenter.segment_coarse_to_fine(
                    image, net, CLASSES, score_threshold=score_threshold, box_padding=box_padding
                )
            
            total_latency_ms = (time.perf_counter() - start_pipeline) * 1000.0
            fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

            columns = st.columns(3)
            with columns[0]:
                st.write("#### Input Image")
                st.image(_image, use_container_width=True)

            with columns[1]:
                st.write("#### Step 1 & 2: Verified Proposals")
                st.info(f"SAM 3 Discovered **{metrics_dict.get('coarse_proposals', 0)}** coarse anomalies (Gold). YOLO verified **{len(verified_dets)}** targets (Green).")
                proposal_img = make_proposal_overlay(_image, coarse_proposals, verified_dets)
                st.image(proposal_img, use_container_width=True)

            with columns[2]:
                st.write("####  Step 3: Hybrid Point+Box Masks")
                if len(verified_dets) == 0:
                    st.info("No damage targets verified.")
                else:
                    boxes = [det["box"] for det in verified_dets]
                    labels = [f"{det['label']} · SAM {score:.2f}" for det, score in zip(verified_dets, sam_scores)]
                    overlay = make_mask_overlay(_image, masks, boxes, labels, alpha=mask_opacity)
                    st.image(overlay, use_container_width=True)

                    overlay_buffer = BytesIO()
                    Image.fromarray(overlay).save(overlay_buffer, format="PNG")
                    st.download_button(
                        label="Download Coarse-to-Fine SAM 3 Overlay",
                        data=overlay_buffer.getvalue(),
                        file_name="RDD_SAM3_CoarseToFine_Overlay.png",
                        mime="image/png",
                        use_container_width=True
                    )

                    combined_mask = combine_masks(masks, _image.shape[:2])
                    st.download_button(
                        label="Download Binary Mask",
                        data=mask_to_png_bytes(combined_mask),
                        file_name="RDD_SAM3_CoarseToFine_Mask.png",
                        mime="image/png",
                        use_container_width=True
                    )

            st.markdown("---")
            st.write("### Advanced Coarse-to-Fine Latency & Recall Benchmark")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SAM 3 Coarse Proposals", f"{metrics_dict.get('coarse_proposals', 0)} regions")
            m2.metric("YOLO Verified Targets", f"{len(verified_dets)} targets")
            m3.metric("Total Dual-Pass Latency", f"{total_latency_ms:.1f} ms")
            m4.metric("Throughput", f"{fps:.2f} FPS")

        except Exception as exc:
            st.error(f"Coarse-to-Fine Pipeline Execution Error: {exc}")
