"""
Common utilities shared across Backend modules.

This module consolidates duplicate helper functions that were previously
scattered across multiple files.
"""

from Backend.common.cv_utils import maybe_cv2, maybe_pil
from Backend.common.math_utils import clamp01, normalize_01, sigmoid
from Backend.common.text_utils import (
    normalize_text,
    tokens,
    char_trigrams,
    levenshtein_distance,
    levenshtein_similarity,
    jaccard,
)

__all__ = [
    # CV utilities
    "maybe_cv2",
    "maybe_pil",
    # Math utilities
    "clamp01",
    "normalize_01",
    "sigmoid",
    # Text utilities
    "normalize_text",
    "tokens",
    "char_trigrams",
    "levenshtein_distance",
    "levenshtein_similarity",
    "jaccard",
]
