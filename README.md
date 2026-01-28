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
