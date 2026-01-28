#!/bin/bash
# Vast.ai Setup Script for Plura Grounding
# Template: PyTorch (Vast)
# GPU: A40 (45GB VRAM)

set -e

echo "=========================================="
echo "Plura Grounding - Vast.ai Setup"
echo "=========================================="

# 1. Update system
echo "[1/6] Updating system..."
apt-get update -qq

# 2. Install system dependencies
echo "[2/6] Installing system dependencies..."
apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 git

# 3. Clone/upload your code (adjust the path)
echo "[3/6] Setting up code..."
# Option A: Clone from git (if you have a repo)
# git clone https://github.com/yourusername/plura-backend.git /workspace/Backend
# Option B: Upload via Vast.ai file manager or scp
# The code should be at /workspace/Backend

# 4. Install Python dependencies
echo "[4/6] Installing Python dependencies..."
pip install --quiet --upgrade pip

pip install --quiet \
    numpy \
    pillow \
    opencv-python-headless \
    scipy \
    easyocr \
    transformers \
    torch \
    torchvision \
    sentence-transformers \
    datasets \
    huggingface_hub

# 5. Pre-download models (so they're cached)
echo "[5/6] Pre-downloading models..."
python3 -c "
print('Downloading LayoutLMv3...')
from transformers import LayoutLMv3Processor, LayoutLMv3Model
LayoutLMv3Processor.from_pretrained('microsoft/layoutlmv3-base')
LayoutLMv3Model.from_pretrained('microsoft/layoutlmv3-base')

print('Downloading Sentence Transformer...')
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')

print('Downloading EasyOCR...')
import easyocr
reader = easyocr.Reader(['en'], gpu=True)

print('All models downloaded!')
"

# 6. Test GPU
echo "[6/6] Testing GPU..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=========================================="
echo "Setup complete! Run benchmark with:"
echo "  cd /workspace/Backend/Backend"
echo "  python scripts/run_enhanced_benchmark.py --limit 100 --verbose"
echo "=========================================="
