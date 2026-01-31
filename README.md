This repository contains the experimental engine for Plura, a research-focused VLM (Visual Language Model) designed to audit and outperform GPT-5.2 in specific UI navigation tasks.

## Key Modules for Reviewers
* **Core Logic:** `plura_engine.py` - The main orchestration pipeline.
* **Visual Physics:** `indexing/visual_physics/` - Modules for "Spectral Saliency" and click refinement (auditing perception artifacts).
* **Architecture:** `indexing/ocumamba_lite/` - Implementation of Mamba-based vision encoders for high-efficiency inference.
* **Benchmarking:** `scripts/gpt52_benchmark_fixed.py` - The evaluation harness used to compare speed/cost against SOTA.

## Infrastructure
* See `scripts/VASTAI_DEPLOY.md` for GPU cluster deployment notes.

*Note: This is an active research repo. You will see failed experiment scripts (e.g., `_v1`, `_debug`) which are preserved for audit trails.*

# GUI Testing Project

Research and benchmarking for GUI visual grounding models.

## Project Structure

- `scripts/` - Benchmark and evaluation scripts
- `indexing/` - Active Inference GUI grounding implementation
- `docs/` - Research reports and documentation
- `security/` - Rate limiting and security utilities

## Key Scripts

- `gpt52_cot_benchmark.py` - GPT-5.2 benchmark with Chain-of-Thought reasoning
- `gpt_benchmark.py` - GPT-4o benchmark script
- `active_gui_grounding.py` - OcuMamba-Lite + Active Inference implementation

## Setup

```bash
pip install openai datasets pillow
export OPENAI_API_KEY="your-key"
```

## Running Benchmarks

```bash
python scripts/gpt52_cot_benchmark.py
```

## Research

See `docs/research_report.md` for full analysis.
