"""
OpenCV and PIL utility functions.

These functions provide safe imports for optional image processing dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


def maybe_cv2() -> Optional[Any]:
    """
    Attempt to import OpenCV. Returns None if unavailable.
    
    This allows code to gracefully degrade when cv2 is not installed,
    falling back to PIL or pure NumPy implementations.
    
    Returns:
        cv2 module if available, None otherwise.
    """
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def maybe_pil() -> Optional[Any]:
    """
    Attempt to import PIL Image. Returns None if unavailable.
    
    Returns:
        PIL.Image module if available, None otherwise.
    """
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None
