"""
OcuMamba-Lite: Lightweight GUI Grounding Model.

A specialized model for GUI element grounding, designed to be small (~12-37M params)
while achieving competitive accuracy through architectural innovations.

Key Components:
- Mamba-2 Visual Encoder: Linear O(N) complexity for high-res processing
- Instruction Encoder: Lightweight transformer for instruction understanding
- Multi-Scale Fusion: Instruction-conditioned feature fusion
- Icon-Aware Detection Head: Anchor-free detection for tiny targets
"""

from Backend.indexing.ocumamba_lite.mamba_visual_encoder import MambaVisualEncoder
from Backend.indexing.ocumamba_lite.instruction_encoder import InstructionEncoder
from Backend.indexing.ocumamba_lite.multiscale_fusion import MultiScaleFusion
from Backend.indexing.ocumamba_lite.detection_head import IconDetectionHead
from Backend.indexing.ocumamba_lite.model import OcuMambaLite

__all__ = [
    "MambaVisualEncoder",
    "InstructionEncoder",
    "MultiScaleFusion",
    "IconDetectionHead",
    "OcuMambaLite",
]

# Training utilities (imported on demand)
def get_trainer():
    from Backend.indexing.ocumamba_lite.trainer import Trainer, train_ocumamba_lite
    return train_ocumamba_lite

def get_datasets():
    from Backend.indexing.ocumamba_lite.dataset import (
        ScreenSpotProDataset, SyntheticIconDataset, create_dataloaders
    )
    return ScreenSpotProDataset, SyntheticIconDataset, create_dataloaders

def get_benchmark():
    from Backend.indexing.ocumamba_lite.benchmark import run_benchmark
    return run_benchmark
