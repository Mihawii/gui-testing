"""
OcuMamba: Vision Mamba integration for native 4K GUI processing.

This module provides the foundation for the OcuMamba architecture,
which combines:
1. Vision Mamba (Vim) - Linear O(N) processing for 4K images
2. Active Inference - Optimal saccadic search
3. Visual Physics Grounding - Deterministic anchoring

GPU Requirements:
- Minimum: A10G (24GB VRAM) or T4 (16GB, with mixed precision)  
- Recommended: A100 (40GB+)

Status: SKELETON - Full implementation requires mamba-ssm package and GPU
"""

from Backend.indexing.mamba.vision_mamba import VisionMambaEncoder
from Backend.indexing.mamba.active_inference import ActiveInferenceController

__all__ = [
    "VisionMambaEncoder",
    "ActiveInferenceController",
]
