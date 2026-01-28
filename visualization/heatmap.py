from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2, maybe_pil
from Backend.common.math_utils import normalize_01


def generate_heatmap_png(
    *,
    image_rgb: np.ndarray,
    attention_map: np.ndarray,
    ignore_threshold: float = 0.15,
    over_attention_threshold: float = 0.65,
    alpha: float = 0.55,
) -> bytes:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")
    if attention_map.ndim != 2:
        raise ValueError("attention_map must be HxW")

    h, w = image_rgb.shape[:2]
    if attention_map.shape[0] != h or attention_map.shape[1] != w:
        raise ValueError("attention_map must match image dimensions")

    att = normalize_01(attention_map)
    overlay = _zone_overlay(att, ignore_threshold=ignore_threshold, over_threshold=over_attention_threshold)
    blended = _blend(image_rgb.astype(np.float32), overlay.astype(np.float32), alpha=float(alpha))
    blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)

    cv2 = maybe_cv2()
    if cv2 is not None:
        bgr = cv2.cvtColor(blended_u8, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.png', bgr)
        if not ok:
            raise RuntimeError("Failed to encode PNG with OpenCV")
        return buf.tobytes()

    pil = maybe_pil()
    if pil is not None:
        Image = pil
        img = Image.fromarray(blended_u8, mode='RGB')
        out = io.BytesIO()
        img.save(out, format='PNG')
        return out.getvalue()

    raise ImportError("To generate heatmap PNG, install opencv-python (or opencv-python-headless) or pillow")


def _zone_overlay(att: np.ndarray, *, ignore_threshold: float, over_threshold: float) -> np.ndarray:
    h, w = att.shape
    out = np.zeros((h, w, 3), dtype=np.float32)

    ignored = att < float(ignore_threshold)
    over = att >= float(over_threshold)
    clarity = (~ignored) & (~over)

    out[ignored] = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    out[clarity] = np.array([0.0, 255.0, 0.0], dtype=np.float32)
    out[over] = np.array([255.0, 0.0, 0.0], dtype=np.float32)

    intensity = att[:, :, None]
    out = out * (0.35 + 0.65 * intensity)

    return np.clip(out, 0.0, 255.0)


def _blend(base: np.ndarray, overlay: np.ndarray, *, alpha: float) -> np.ndarray:
    a = float(alpha)
    if a <= 0.0:
        return base
    if a >= 1.0:
        return overlay
    return (1.0 - a) * base + a * overlay

