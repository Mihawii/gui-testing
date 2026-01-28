# Vast.ai GPU Deployment Guide

## Quick Start

### 1. Rent the GPU
- Go to [Vast.ai](https://vast.ai)
- Select: **1x A40** (45GB VRAM, ~$0.33/hr)
- Template: **PyTorch (Vast)**
- Click **RENT**

### 2. Upload Your Code
Option A - Via Vast.ai File Manager:
```
Upload the entire Backend/ folder to /workspace/
```

Option B - Via SSH/SCP:
```bash
scp -r Backend/ root@<vast-ip>:/workspace/
```

### 3. Run Setup
SSH into the instance and run:
```bash
cd /workspace/Backend/Backend
chmod +x scripts/vastai_setup.sh
./scripts/vastai_setup.sh
```

This will:
- Install all Python dependencies
- Download LayoutLMv3, EasyOCR, SentenceTransformer models
- Verify GPU is working

### 4. Run Benchmark
```bash
./scripts/vastai_run_benchmark.sh
```

Or manually:
```bash
export PYTHONPATH=/workspace/Backend
python3 scripts/run_enhanced_benchmark.py --limit 100 --verbose
```

## Expected Results

| Metric | CPU (Mac) | GPU (A40) |
|--------|-----------|-----------|
| Latency | 5-12s | <0.5s |
| 100 samples | ~10 min | ~1 min |

## Cost Estimate

- Setup: ~5 min = ~$0.03
- 100 sample benchmark: ~2 min = ~$0.01
- Full experimentation session: ~2 hrs = ~$0.65

## Troubleshooting

**CUDA not available:**
```bash
nvidia-smi  # Check GPU is visible
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Models not downloading:**
```bash
export HF_HUB_OFFLINE=0
python -c "from transformers import LayoutLMv3Model; LayoutLMv3Model.from_pretrained('microsoft/layoutlmv3-base')"
```

**Permission denied:**
```bash
chmod +x scripts/*.sh
```
