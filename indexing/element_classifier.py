from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from Backend.common.math_utils import clamp01


def classify_elements(
    elements: List[Dict[str, Any]],
    saliency_map: np.ndarray,
    *,
    image_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if saliency_map.ndim != 2:
        raise ValueError("saliency_map must be HxW")

    h, w = image_shape
    if saliency_map.shape[0] != h or saliency_map.shape[1] != w:
        raise ValueError("saliency_map shape must match image_shape")

    updated: List[Dict[str, Any]] = []
    for el in elements:
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, bw, bh = [int(v) for v in bbox]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        patch = saliency_map[y : y + bh, x : x + bw]
        saliency_score = float(np.mean(patch)) if patch.size else 0.0

        contrast = float(el.get("contrast", 0.0))
        area_ratio = float(el.get("area_ratio", (bw * bh) / float(h * w)))
        hierarchy_level = int(el.get("hierarchy_level", 0))

        cx = (x + bw / 2.0) / float(w)
        cy = (y + bh / 2.0) / float(h)

        aspect = (bw / float(bh)) if bh > 0 else 1.0
        aspect_score = 1.0 - min(abs(aspect - 3.0) / 5.0, 1.0)

        size_score = _triangular_score(area_ratio, peak=0.02, width=0.06)
        contrast_score = clamp01((contrast - 0.05) / 0.35)
        sal_score = clamp01(saliency_score)

        center_dist = ((cx - 0.5) ** 2 + (cy - 0.4) ** 2) ** 0.5
        center_score = 1.0 - min(center_dist / 0.75, 1.0)

        cta_score = clamp01(
            0.35 * size_score + 0.25 * contrast_score + 0.25 * sal_score + 0.15 * center_score
        )
        cta_score = clamp01(cta_score * (0.75 + 0.25 * aspect_score))

        el_type = _infer_type(area_ratio=area_ratio, contrast=contrast, cta_score=cta_score, hierarchy_level=hierarchy_level)

        importance_score = clamp01(0.55 * sal_score + 0.25 * contrast_score + 0.20 * (1.0 - min(hierarchy_level / 6.0, 1.0)))

        enriched = {
            **el,
            "bbox": [x, y, bw, bh],
            "saliency_score": saliency_score,
            "cta_score": cta_score,
            "type": el_type,
            "importance_score": float(importance_score),
        }
        updated.append(enriched)

    return updated


def _infer_type(*, area_ratio: float, contrast: float, cta_score: float, hierarchy_level: int) -> str:
    if cta_score >= 0.62:
        return "cta"
    if area_ratio >= 0.20:
        return "container"
    if contrast <= 0.06 and area_ratio >= 0.03:
        return "text_block"
    if hierarchy_level >= 3 and area_ratio <= 0.01:
        return "icon_or_control"
    return "ui_element"


def _triangular_score(value: float, *, peak: float, width: float) -> float:
    if width <= 0:
        return 0.0
    dist = abs(value - peak)
    return clamp01(1.0 - dist / width)

