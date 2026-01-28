#!/bin/bash
# =============================================================================
# OcuMamba-Lite Training Script for Vast.ai
# =============================================================================
# Usage:
#   1. Start Vast.ai instance with GPU (A40 recommended)
#   2. Upload Backend folder to /workspace/
#   3. Run: bash /workspace/Backend/scripts/train_ocumamba.sh
# =============================================================================

set -e

echo "========================================"
echo "OcuMamba-Lite Training Setup"
echo "========================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet torch torchvision pillow numpy pandas

# Set up environment
export PYTHONPATH=/workspace

# Create checkpoints directory
mkdir -p /workspace/checkpoints

# Run training
echo ""
echo "========================================"
echo "Starting OcuMamba-Lite Training"
echo "========================================"

cd /workspace

python3 -c "
import sys
sys.path.insert(0, '/workspace')

from Backend.indexing.ocumamba_lite.trainer import train_ocumamba_lite

# Train the model
model = train_ocumamba_lite(
    model_size='tiny',       # Start with tiny for faster iteration
    epochs=10,               # 10 epochs for initial run
    batch_size=8,            # Adjust based on GPU memory
    lr=1e-4,
    data_dir=None,           # Use synthetic data only for now
    device='cuda',
    checkpoint_dir='/workspace/checkpoints',
)

print('\\nTraining complete!')
print(f'Model saved to /workspace/checkpoints/')
"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Checkpoints saved in: /workspace/checkpoints/"
echo ""
echo "To benchmark, run:"
echo "  python3 -m Backend.indexing.ocumamba_lite.benchmark"
