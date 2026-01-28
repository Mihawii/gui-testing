"""
Mathematical utility functions.

These functions provide common numerical operations used throughout
the vision and analysis pipelines.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def clamp01(x: float) -> float:
    """
    Clamp a value to the [0, 1] range.
    
    Args:
        x: Input value.
        
    Returns:
        Value clamped between 0.0 and 1.0.
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array to the [0, 1] range.
    
    Uses min-max normalization. Returns zeros if the array has
    constant values (max - min < 1e-6).
    
    Args:
        arr: Input array of any shape.
        
    Returns:
        Normalized array with values in [0, 1], dtype float32.
    """
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def sigmoid(x: float) -> float:
    """
    Compute the sigmoid function: 1 / (1 + exp(-x)).
    
    Handles extreme values to avoid overflow.
    
    Args:
        x: Input value.
        
    Returns:
        Sigmoid output in (0, 1).
    """
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v >= 60.0:
        return 1.0
    if v <= -60.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-v)))


def normalized_entropy(weights: list[float]) -> float:
    """
    Compute the normalized entropy of a weight distribution.
    
    Entropy is normalized by log(n) to produce a value in [0, 1].
    
    Args:
        weights: List of non-negative weights.
        
    Returns:
        Normalized entropy, 0 for deterministic, 1 for uniform.
    """
    w = np.array([max(0.0, float(x)) for x in weights], dtype=np.float32)
    s = float(np.sum(w))
    if s <= 1e-9:
        return 0.0

    p = w / s
    p = np.clip(p, 1e-9, 1.0)

    ent = float(-np.sum(p * np.log(p)))
    n = int(p.size)
    if n <= 1:
        return 0.0

    return float(ent / math.log(float(n)))
