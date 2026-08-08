# Road Damage Detection & SAM 3 Pixel Mask Refinement

A two-tier deep learning framework combining fine-tuned YOLOv8-small candidate detection with Meta Segment Anything Model 3 (SAM 3) zero-shot pixel mask refinement for automated pavement defect auditing.

![Banner](resource/banner.png)

## Project Overview

Standard bounding box object detectors like YOLO output rectangular boxes that overestimate road crack surface area by including non-damaged asphalt background inside the detection box. This project implements a hybrid pipeline that snaps directly onto exact crack contours without requiring expensive pixel-level manual mask annotations during training.

### Key Features
* Zero-Shot Boundary Precision: Isolates exact crack boundaries (longitudinal, transverse, alligator cracks, and potholes), removing background asphalt noise from detector boxes.
* Single-Process In-Memory CUDA Execution: Runs YOLOv8 and SAM 3 within a single Python environment under PyTorch BFloat16 autocast, eliminating network socket overhead.
* Dual Pipeline Modes:
  1. Baseline Mode (Image -> YOLO -> SAM 3): Fast single-pass execution (54.3 ms YOLO / ~1.0 s total latency).
  2. Advanced Mode (SAM 3 -> YOLO -> SAM 3): Three-step coarse-to-fine discovery pass (9 coarse proposals, 88.9% false-positive noise reduction).
* Real-Time Latency Benchmark: Live performance metrics displaying YOLO time, SAM 3 time, total latency, and throughput FPS.

## Hugging Face Access & SAM 3 Checkpoint Authentication

Meta's official SAM 3 checkpoint is hosted as a gated model on Hugging Face. Users running this repository for the first time must authenticate with Hugging Face:

1. Accept Meta's License: Visit Meta's official SAM 3 repository page on Hugging Face and accept the model license terms.
2. Generate Access Token: Create a User Access Token in your Hugging Face account settings.
3. Authenticate in Terminal:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   Alternatively, export your access token as an environment variable:
   ```bash
   export HF_TOKEN="hf_your_access_token_here"
   ```

## System Requirements & Prerequisites

| Component | Requirement |
| :--- | :--- |
| Operating System | Linux (Ubuntu/Debian) or WSL2 on Windows / macOS |
| Python Version | Python 3.10, 3.11, or 3.12 |
| GPU / Hardware | NVIDIA CUDA-capable GPU (Required for SAM 3 BFloat16 ViT execution) |
| Hugging Face Account | Free Hugging Face account + User Access Token |
| Deep Learning Engine | PyTorch 2.3+ with CUDA 12.x support |

## Quickstart & Installation

1. Clone the repository:
```bash
git clone https://github.com/vsazzy/RoadDamageDetection-SAM3.git
cd RoadDamageDetection-SAM3
```

2. Create and activate a Python virtual environment:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install ultralytics streamlit opencv-python Pillow requests huggingface_hub
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

4. Authenticate with Hugging Face:
```bash
huggingface-cli login
```

5. Launch the application:
```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 streamlit run Home.py
```

Open your browser at `http://localhost:8501`.

## Project Directory Structure

```
RoadDamageDetection-SAM3/
├── models/
│   └── YOLOv8_Small_RDD.pt          Trained YOLOv8 checkpoint (Auto-downloads if missing)
├── sam3_service/
│   └── inference.py                 Meta SAM 3 engine (Baseline & Coarse-to-Fine)
├── sample_utils/
│   └── download.py                  Automatic checkpoint downloader
├── segmentation/
│   └── masks.py                     Mask visualization & proposal overlays
├── training/                        Jupyter notebooks for evaluation & training
│   ├── 0_PrepareDatasetYOLOv8.ipynb
│   ├── 1_TrainingYOLOv8.ipynb
│   └── 2_EvaluationTesting.ipynb
├── Home.py                          Main Streamlit Web Application
└── README.md
```

## Performance Benchmark Comparison

| Pipeline Mode | YOLO Latency | SAM 3 Latency | Total Latency | FPS | Proposals Discovered | Verified Targets | Noise Rejection Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Image -> YOLO -> SAM 3) | 54.3 ms | 906.3 ms | 1,055.2 ms | 0.90 FPS | N/A | 1 target | N/A |
| Advanced (SAM 3 -> YOLO -> SAM 3) | 54.3 ms | 2,288.2 ms | 2,342.5 ms | 0.43 FPS | 9 regions | 1 target | 88.9% |

## Dataset & Citations

* Dataset Source: Trained on the public Crowdsensing-based Road Damage Detection Challenge (CRDDC2022 / RDD2022) dataset (Japan and India subsets) created by Seki Lab (University of Tokyo) and the Ministry of Land, Infrastructure, Transport and Tourism, Japan.
* Categories: D00 Longitudinal Crack, D10 Transverse Crack, D20 Alligator Crack, and D40 Potholes.
