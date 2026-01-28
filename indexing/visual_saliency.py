from typing import Any, Dict, List, Optional

import numpy as np

from Backend.common.cv_utils import maybe_cv2
from Backend.common.math_utils import normalize_01


def compute_saliency(image_rgb: np.ndarray, *, center_bias_strength: float = 0.25) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    image_f = image_rgb.astype(np.float32)
    gray = (0.299 * image_f[:, :, 0] + 0.587 * image_f[:, :, 1] + 0.114 * image_f[:, :, 2]).astype(np.float32)

    sal = _spectral_residual_saliency(gray)
    if float(np.max(sal)) <= 1e-6:
        sal = _gradient_magnitude(gray)
    sal = normalize_01(sal)

    if center_bias_strength <= 0:
        return sal

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma = 0.35 * float(min(h, w))
    sigma = max(sigma, 1.0)
    center = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)).astype(np.float32)
    center = normalize_01(center)

    saliency = (1.0 - center_bias_strength) * sal + center_bias_strength * center
    return normalize_01(saliency)


def compute_perceptual_field(
    image_rgb: np.ndarray,
    *,
    center_bias_strength: float = 0.25,
    include_text_density: bool = True,
) -> Dict[str, np.ndarray]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    image_f = image_rgb.astype(np.float32)
    gray = (0.299 * image_f[:, :, 0] + 0.587 * image_f[:, :, 1] + 0.114 * image_f[:, :, 2]).astype(np.float32)

    saliency = compute_saliency(image_rgb, center_bias_strength=center_bias_strength)
    contrast = _contrast_map(image_rgb)
    edges = _edge_structure_map(gray)
    text_density = _text_density_map(gray) if include_text_density else np.zeros_like(gray, dtype=np.float32)

    return {
        "saliency": saliency.astype(np.float32),
        "contrast": contrast.astype(np.float32),
        "edges": edges.astype(np.float32),
        "text_density": text_density.astype(np.float32),
    }


def compute_attention_surface(
    image_rgb: np.ndarray,
    *,
    center_bias_strength: float = 0.25,
    include_text_density: bool = True,
    w_saliency: float = 0.55,
    w_contrast: float = 0.25,
    w_edges: float = 0.20,
    text_penalty_strength: float = 0.35,
    smooth_sigma_ratio: float = 0.01,
    min_sigma: float = 1.0,
) -> np.ndarray:
    field = compute_perceptual_field(
        image_rgb,
        center_bias_strength=center_bias_strength,
        include_text_density=include_text_density,
    )

    sal = field["saliency"]
    con = field["contrast"]
    edg = field["edges"]
    txt = field["text_density"]

    att = (
        float(w_saliency) * sal.astype(np.float32)
        + float(w_contrast) * con.astype(np.float32)
        + float(w_edges) * edg.astype(np.float32)
    ).astype(np.float32)

    if include_text_density and float(text_penalty_strength) > 0.0:
        att = att * (1.0 - float(text_penalty_strength) * txt.astype(np.float32))

    att = np.clip(att, 0.0, None).astype(np.float32)

    h, w = att.shape
    sigma = max(float(min_sigma), float(smooth_sigma_ratio) * float(min(h, w)))
    cv2 = maybe_cv2()
    if cv2 is not None and sigma > 0.0:
        att = cv2.GaussianBlur(att, (0, 0), sigmaX=float(sigma)).astype(np.float32)
    else:
        att = _box_filter_3x3(att)
        att = _box_filter_3x3(att)

    return normalize_01(att)


def compute_attention_frames(
    image_rgb: np.ndarray,
    *,
    intent_kind: Optional[str] = None,
    elements: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, np.ndarray]:
    field = compute_perceptual_field(
        image_rgb,
        center_bias_strength=0.25,
        include_text_density=True,
    )

    sal = field["saliency"].astype(np.float32)
    con = field["contrast"].astype(np.float32)
    edg = field["edges"].astype(np.float32)
    txt = field["text_density"].astype(np.float32)

    orienting = (0.55 * sal + 0.25 * con + 0.20 * edg).astype(np.float32)
    orienting = orienting * (1.0 - 0.35 * txt)
    orienting = np.clip(orienting, 0.0, None).astype(np.float32)
    orienting = _smooth_map(orienting, smooth_sigma_ratio=0.01, min_sigma=1.0)
    orienting = normalize_01(orienting)

    sal_imm = compute_saliency(image_rgb, center_bias_strength=0.40).astype(np.float32)
    immediate = (0.60 * sal_imm + 0.30 * con + 0.10 * edg).astype(np.float32)
    immediate = np.clip(immediate, 0.0, None).astype(np.float32)
    immediate = _smooth_map(immediate, smooth_sigma_ratio=0.008, min_sigma=1.0)
    immediate = normalize_01(immediate)

    scanning_base = (0.45 * sal + 0.15 * con + 0.15 * edg + 0.25 * txt).astype(np.float32)
    scanning_base = np.clip(scanning_base, 0.0, None).astype(np.float32)

    influence = _element_influence_map(elements, image_shape=(int(sal.shape[0]), int(sal.shape[1])), intent_kind=intent_kind)
    if influence is None:
        scanning = scanning_base
    else:
        scanning = (0.65 * scanning_base + 0.35 * influence).astype(np.float32)
    scanning = np.clip(scanning, 0.0, None).astype(np.float32)
    scanning = _smooth_map(scanning, smooth_sigma_ratio=0.014, min_sigma=1.0)
    scanning = normalize_01(scanning)

    return {
        "immediate": immediate.astype(np.float32),
        "orienting": orienting.astype(np.float32),
        "scanning": scanning.astype(np.float32),
    }


def _spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    fft = np.fft.fft2(g)
    amp = np.abs(fft).astype(np.float32)
    log_amp = np.log(amp + 1e-8).astype(np.float32)
    phase = np.angle(fft).astype(np.float32)

    cv2 = maybe_cv2()
    if cv2 is not None:
        avg = cv2.blur(log_amp, (3, 3))
    else:
        avg = _box_filter_3x3(log_amp)

    spectral_residual = (log_amp - avg).astype(np.float32)
    exp = np.exp(spectral_residual + 1j * phase)
    sal = np.abs(np.fft.ifft2(exp)) ** 2
    sal = sal.astype(np.float32)

    if cv2 is not None:
        sal = cv2.GaussianBlur(sal, (0, 0), sigmaX=2.5)

    return sal.astype(np.float32)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    cv2 = maybe_cv2()
    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def _contrast_map(image_rgb: np.ndarray) -> np.ndarray:
    cv2 = maybe_cv2()
    if cv2 is None:
        image_f = image_rgb.astype(np.float32)
        gray = (0.299 * image_f[:, :, 0] + 0.587 * image_f[:, :, 1] + 0.114 * image_f[:, :, 2]).astype(np.float32)
        return normalize_01(_gradient_magnitude(gray))

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b).astype(np.float32)

    std_l = _local_std(l, ksize=9)
    std_c = _local_std(chroma, ksize=9)
    contrast = (0.70 * std_l + 0.30 * std_c).astype(np.float32)
    return normalize_01(contrast)


def _edge_structure_map(gray: np.ndarray) -> np.ndarray:
    mag = _gradient_magnitude(gray)
    cv2 = maybe_cv2()
    if cv2 is not None:
        mag = cv2.GaussianBlur(mag.astype(np.float32), (0, 0), sigmaX=1.6)
    return normalize_01(mag)


def _text_density_map(gray: np.ndarray) -> np.ndarray:
    cv2 = maybe_cv2()
    if cv2 is None:
        return np.zeros_like(gray, dtype=np.float32)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = gray_u8.shape[:2]

    max_dim = int(max(h, w))
    scale = 1.0
    if max_dim > 960:
        scale = 960.0 / float(max_dim)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        g = cv2.resize(gray_u8, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        g = gray_u8

    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
    grad = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    abs_grad = cv2.convertScaleAbs(grad)
    _, th = cv2.threshold(abs_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours_result = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    mask = np.zeros_like(th, dtype=np.uint8)
    img_area = float(th.shape[0] * th.shape[1])
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        if box_area < 50.0:
            continue
        if bh < 6 or bh > int(0.20 * th.shape[0]):
            continue
        if bw < 12:
            continue
        if (bw / float(max(bh, 1))) < 1.2:
            continue
        if box_area > img_area * 0.35:
            continue
        cv2.rectangle(mask, (int(x), int(y)), (int(x + bw), int(y + bh)), 255, thickness=-1)

    density = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=7.0)
    density = normalize_01(density)
    if scale != 1.0:
        density = cv2.resize(density.astype(np.float32), (int(w), int(h)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return density.astype(np.float32)


def _local_std(arr: np.ndarray, *, ksize: int) -> np.ndarray:
    a = arr.astype(np.float32)
    cv2 = maybe_cv2()
    if cv2 is None:
        return np.zeros_like(a, dtype=np.float32)

    k = max(1, int(ksize))
    mean = cv2.blur(a, (k, k))
    mean_sq = cv2.blur(a * a, (k, k))
    var = np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32)
    return np.sqrt(var).astype(np.float32)


def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def _box_filter_3x3(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr.astype(np.float32), ((1, 1), (1, 1)), mode="edge")
    out = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return (out / 9.0).astype(np.float32)


def _smooth_map(arr: np.ndarray, *, smooth_sigma_ratio: float, min_sigma: float) -> np.ndarray:
    a = arr.astype(np.float32)
    h, w = a.shape
    sigma = max(float(min_sigma), float(smooth_sigma_ratio) * float(min(h, w)))
    cv2 = maybe_cv2()
    if cv2 is not None and sigma > 0.0:
        return cv2.GaussianBlur(a, (0, 0), sigmaX=float(sigma)).astype(np.float32)
    out = a
    out = _box_filter_3x3(out)
    out = _box_filter_3x3(out)
    return out.astype(np.float32)


def _element_influence_map(
    elements: Optional[List[Dict[str, Any]]],
    *,
    image_shape: tuple[int, int],
    intent_kind: Optional[str],
) -> Optional[np.ndarray]:
    if not elements:
        return None

    h, w = image_shape
    influence = np.zeros((h, w), dtype=np.float32)

    for el in elements:
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, bw, bh = [int(v) for v in bbox]
        except Exception:
            continue
        if bw <= 0 or bh <= 0:
            continue

        x = max(0, min(int(x), int(w - 1)))
        y = max(0, min(int(y), int(h - 1)))
        bw = max(1, min(int(bw), int(w - x)))
        bh = max(1, min(int(bh), int(h - y)))

        el_type = str(el.get("type") or "ui_element")
        importance = float(el.get("importance_score", 0.0) or 0.0)
        cta = float(el.get("cta_score", 0.0) or 0.0)
        sal = float(el.get("saliency_score", 0.0) or 0.0)

        base_weight = max(0.0, importance)
        if base_weight <= 1e-6:
            base_weight = 0.55 * max(0.0, sal) + 0.45 * max(0.0, cta)

        type_weight = 1.0
        if el_type == "cta":
            type_weight = 1.25 if intent_kind == "action" else 0.95
        elif el_type == "text_block":
            type_weight = 1.20 if intent_kind in ("info", "trust") else 0.90
        elif el_type == "icon_or_control":
            type_weight = 0.70
        elif el_type == "container":
            type_weight = 0.55

        weight = float(base_weight) * float(type_weight)
        if weight <= 0.0:
            continue
        influence[y : y + bh, x : x + bw] += float(weight)

    if float(np.max(influence)) <= 1e-9:
        return None
    return normalize_01(influence.astype(np.float32))

