# SAM 3 GPU service

Meta's official SAM 3.1 release requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+, a CUDA-capable GPU, and approved access to the gated checkpoint. It cannot share this application's original Python 3.10 / PyTorch 2.0 environment.

On a CUDA machine:

```bash
git clone https://github.com/facebookresearch/sam3.git ../sam3
cd ../sam3
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r ../RoadDamageDetection/sam3_service/requirements.txt
hf auth login
cd ../RoadDamageDetection
uvicorn sam3_service.app:app --host 0.0.0.0 --port 8001
```

Request access to the checkpoint from Meta's official Hugging Face SAM 3 repository before running `hf auth login`.

Point the Streamlit application at this service:

```bash
export SAM3_SERVICE_URL=http://GPU_HOST:8001
streamlit run Home.py
```

For a remote machine, place the service behind HTTPS and authentication instead of exposing port 8001 publicly.
