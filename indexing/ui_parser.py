from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2


def parse_ui(image_rgb: np.ndarray, *, max_elements: int = 64) -> List[Dict[str, Any]]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    h, w = image_rgb.shape[:2]
    image_area = float(h * w)

    cv2 = maybe_cv2()
    if cv2 is None:
        return _fallback_grid_elements(h, w)

    gray = _to_gray_u8(image_rgb)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    min_area = max(64.0, image_area * 0.0008)
    max_area = image_area * 0.85

    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area < min_area or area > max_area:
            continue
        if bw < 8 or bh < 8:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))

    boxes = _nms(boxes, iou_threshold=0.90)

    boxes.sort(key=lambda b: (b[2] * b[3]), reverse=True)
    boxes = boxes[:max_elements]

    elements: List[Dict[str, Any]] = []
    for idx, (x, y, bw, bh) in enumerate(boxes):
        contrast = _box_contrast(gray, x, y, bw, bh)
        elements.append(
            {
                "id": f"el_{idx}",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area_ratio": float((bw * bh) / image_area),
                "contrast": float(contrast),
            }
        )

    _assign_hierarchy(elements)
    return elements


def _assign_hierarchy(elements: List[Dict[str, Any]]) -> None:
    if not elements:
        return

    boxes = [(e["bbox"][0], e["bbox"][1], e["bbox"][2], e["bbox"][3]) for e in elements]

    levels = [0 for _ in elements]
    for i, (x, y, w, h) in enumerate(boxes):
        parent = None
        parent_area = None
        for j, (px, py, pw, ph) in enumerate(boxes):
            if i == j:
                continue
            if _contains((px, py, pw, ph), (x, y, w, h)):
                area = pw * ph
                if parent is None or area < parent_area:
                    parent = j
                    parent_area = area
        if parent is not None:
            levels[i] = levels[parent] + 1

    for e, lvl in zip(elements, levels):
        e["hierarchy_level"] = int(lvl)


def _contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ix >= ox and iy >= oy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)


def _box_contrast(gray_u8: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    region = gray_u8[y : y + h, x : x + w]
    if region.size == 0:
        return 0.0
    return float(np.std(region.astype(np.float32)) / 255.0)


def _to_gray_u8(image_rgb: np.ndarray) -> np.ndarray:
    image_f = image_rgb.astype(np.float32)
    gray = (0.299 * image_f[:, :, 0] + 0.587 * image_f[:, :, 1] + 0.114 * image_f[:, :, 2]).astype(np.float32)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _fallback_grid_elements(h: int, w: int) -> List[Dict[str, Any]]:
    elements: List[Dict[str, Any]] = []
    rows = 4
    cols = 4
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0 = int(round((c / cols) * w))
            y0 = int(round((r / rows) * h))
            x1 = int(round(((c + 1) / cols) * w))
            y1 = int(round(((r + 1) / rows) * h))
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            elements.append(
                {
                    "id": f"el_{idx}",
                    "bbox": [x0, y0, bw, bh],
                    "area_ratio": float((bw * bh) / float(h * w)),
                    "contrast": 0.0,
                    "hierarchy_level": 0,
                }
            )
            idx += 1

    return elements


def _nms(boxes: List[Tuple[int, int, int, int]], *, iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
    y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = np.argsort(areas)[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        order = rest[iou <= float(iou_threshold)]

    return [boxes[k] for k in keep]

