#!/bin/bash
# Run Enhanced Benchmark on Vast.ai GPU
# Expected: <0.5s/sample on A40

set -e

cd /workspace/Backend/Backend

echo "=========================================="
echo "Plura Enhanced Grounding Benchmark"
echo "=========================================="

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)')"

# Set environment
export PYTHONPATH=/workspace/Backend
export CUDA_VISIBLE_DEVICES=0

# Run benchmark
echo ""
echo "Running benchmark (100 samples)..."
python3 scripts/run_enhanced_benchmark.py \
    --limit 100 \
    --verbose

echo ""
echo "Results saved to enhanced_benchmark_results.json"
echo "=========================================="
