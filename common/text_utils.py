"""
Text processing utility functions.

These functions provide common text normalization and comparison
operations used for instruction parsing and OCR matching.
"""

from __future__ import annotations

import re
from typing import Sequence, Set, List

from Backend.common.math_utils import clamp01


def normalize_text(s: str) -> str:
    """
    Normalize text for comparison: lowercase, alphanumeric only, single spaces.
    
    Args:
        s: Input string.
        
    Returns:
        Normalized string.
    """
    t = str(s or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(s: str) -> List[str]:
    """
    Tokenize normalized text into words.
    
    Args:
        s: Input string (will be normalized).
        
    Returns:
        List of non-empty word tokens.
    """
    t = normalize_text(s)
    return [x for x in t.split(" ") if x]


def char_trigrams(s: str) -> Set[str]:
    """
    Extract character trigrams from normalized text.
    
    Args:
        s: Input string.
        
    Returns:
        Set of 3-character substrings.
    """
    t = normalize_text(s).replace(" ", "")
    if not t:
        return set()
    if len(t) < 3:
        return {t}
    return {t[i : i + 3] for i in range(0, len(t) - 2)}


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """
    Compute Levenshtein (edit) distance between two token sequences.
    
    Uses dynamic programming with space optimization.
    
    Args:
        a: First sequence of tokens.
        b: Second sequence of tokens.
        
    Returns:
        Minimum number of edits to transform a into b.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)

    for i, ca in enumerate(a, start=1):
        cur[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev

    return int(prev[len(b)])


def levenshtein_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Compute normalized Levenshtein similarity in [0, 1].
    
    Args:
        a: First sequence of tokens.
        b: Second sequence of tokens.
        
    Returns:
        1.0 for identical sequences, 0.0 for completely different.
    """
    n = max(len(a), len(b))
    if n <= 0:
        return 0.0
    return float(clamp01(1.0 - float(levenshtein_distance(a, b)) / float(n)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    """
    Compute Jaccard similarity coefficient.
    
    Args:
        a: First set.
        b: Second set.
        
    Returns:
        Intersection size / union size, in [0, 1].
    """
    if not a or not b:
        return 0.0
    inter = float(len(a.intersection(b)))
    union = float(len(a.union(b)))
    if union <= 1e-9:
        return 0.0
    return float(clamp01(inter / union))
