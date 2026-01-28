"""
Visual Physics Engine for deterministic GUI grounding.

This module provides the "hard reality" anchor for OcuMamba predictions,
using classical computer vision algorithms to snap model predictions to
actual visual boundaries and prevent hallucination.

Key Components:
    1. Spectral Residual Saliency (enhanced for 4K)
    2. Edge-based element detection  
    3. Semantic region verification
    4. Physics-based click target refinement

The Visual Physics Engine ensures that predicted click locations
always correspond to real, interactable UI elements.
"""

from Backend.indexing.visual_physics.spectral_saliency import (
    compute_4k_saliency,
    detect_visual_anchors,
)
from Backend.indexing.visual_physics.click_refinement import (
    refine_click_target,
    snap_to_nearest_edge,
)

__all__ = [
    "compute_4k_saliency",
    "detect_visual_anchors", 
    "refine_click_target",
    "snap_to_nearest_edge",
]
