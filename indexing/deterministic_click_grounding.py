from __future__ import annotations

import base64
import importlib.util
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests

import ueyes_eval
from Backend.indexing.ui_parser import parse_ui
from Backend.indexing.visual_saliency import compute_perceptual_field


CONFIG_VERSION = "det_click_grounding_v1"


DEFAULT_CONFIG: Dict[str, Any] = {
    "weights": {
        "saliency_mean": 0.20,
        "saliency_max": 0.10,
        "saliency_percentile": 0.05,
        "text_token_overlap": 0.20,
        "text_levenshtein": 0.10,
        "text_trigram_jaccard": 0.10,
        "keyword_presence": 0.10,
        "text_exact_match": 0.15,
        "text_phrase_match": 0.10,
        "search_bar_prior": 0.08,
        "icon_text_penalty": -0.08,
        "icon_size_prior": 0.18,
        "icon_aspect_prior": 0.08,
        "icon_likelihood": 0.14,
        "vlm_support": 0.18,
        "spatial_prior": 0.10,
        "edge_density": 0.05,
        "circularity": 0.05,
        "symmetry": 0.05,
        "color_contrast": 0.05,
        "area_penalty": -0.10,
        "center_penalty": -0.10,
    },
    "proposals": {
        "max_regions": 128,
        "merge_iou_threshold": 0.40,
        "min_area_ratio": 0.0006,
        "max_area_ratio": 0.65,
        "min_side_px": 10,
        "edge_cc": {"enabled": True, "use_otsu": True, "fixed_threshold": 0.35, "dilate_iters": 1},
        "saliency_cc": {"enabled": True, "use_otsu": False, "fixed_threshold": 0.62, "dilate_iters": 1, "max_cc": 96},
        "text_regions": {"enabled": True, "threshold": 0.55, "close_kernel": [11, 5], "close_iters": 1},
        "ocr_words": {"enabled": True, "max_words": 72, "min_conf": 0.35, "pad_px": 2},
        "ocr_lines": {"enabled": True, "max_lines": 48, "min_conf": 0.35, "pad_px": 2, "close_kernel": [18, 6], "close_iters": 1},
        "ocr_buttons": {
            "enabled": True,
            "max_buttons": 48,
            "pad_x_mult": 0.55,
            "pad_y_mult": 0.75,
            "min_pad_px": 2,
            "max_pad_px": 24,
        },
        "icon_anchors": {"enabled": True, "sizes": [0.032, 0.048, 0.070]},
    },
    "ocr": {
        "enabled": True,
        "engine": "easyocr",
        "max_dim": 960,
        "lang": "eng",
        "psm": 11,
        "timeout_s": 6,
        "gpu": False,
    },
    "vlm": {
        "enabled": False,
        "provider": "owlvit",
        "model": "google/owlvit-base-patch32",
        "device": "auto",
        "score_threshold": 0.005,
        "max_queries": 8,
        "local_files_only": False,
        "max_dim": 1280,
        "max_candidates": 24,
        "timeout_s": 120,
        "cache": True,
        "min_icon_mode_strength": 0.55,
        "tiling": {
            "enabled": True,
            "grid": [2, 2],
            "overlap": 0.18,
            "max_dim": 960,
            "max_candidates_per_tile": 8,
            "base_share": 0.55,
        },
        "verify": {
            "min_saliency_mean": 0.10,
            "min_saliency_max": 0.28,
            "min_edge_mean": 0.08,
            "min_contrast": 0.10,
        },
    },
    "scoring": {"area_target": 0.02, "area_width": 0.08, "top_k": 20},
    "utility": {
        "enabled": False,
        "saliency_key": "saliency_mean",
        "p_intent": {
            "source": "deterministic",
            "deterministic": {
                "text_overlap": 0.38,
                "text_levenshtein": 0.18,
                "text_trigram_jaccard": 0.12,
                "keyword_presence": 0.10,
                "text_exact_match": 0.14,
                "text_phrase_match": 0.08,
            },
            "ranker": {"weight": 0.45},
        },
    },
    "ranker": {
        "enabled": False,
        "path": "",
        "feature_order": [
            "saliency_mean",
            "saliency_max",
            "saliency_p90",
            "edge_density",
            "text_density",
            "spatial_prior",
            "text_overlap",
            "text_levenshtein",
            "text_trigram_jaccard",
            "keyword_presence",
            "text_exact_match",
            "text_phrase_match",
            "search_bar_prior",
            "icon_text_penalty",
            "icon_size_prior",
            "icon_aspect_prior",
            "icon_likelihood",
            "vlm_support",
            "circularity",
            "symmetry",
            "color_contrast",
            "ocr_conf",
            "area_ratio",
            "center_pen",
        ],
    },
    "vlm_fallback": {
        "enabled": False,
        "min_confidence": 0.18,
        "min_p_intent": 0.26,
        "min_candidates": 14,
        "min_icon_mode_strength": 0.0,
        "max_extra_candidates": 24,
    },
    "export": {"return_candidates": False},
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    p = str(path or "").strip()
    if not p:
        return cfg

    file_path = Path(p).expanduser().resolve()
    if not file_path.exists():
        return cfg

    text = file_path.read_text(encoding="utf-8")
    loaded: Any = None
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = None

    if isinstance(loaded, dict):
        _deep_update(cfg, loaded)
    return cfg


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def _maybe_cv2() -> Optional[object]:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _normalize_text(s: str) -> str:
    t = str(s or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _tokens(s: str) -> List[str]:
    t = _normalize_text(s)
    return [x for x in t.split(" ") if x]


def _char_trigrams(s: str) -> Set[str]:
    t = _normalize_text(s).replace(" ", "")
    if not t:
        return set()
    if len(t) < 3:
        return {t}
    return {t[i : i + 3] for i in range(0, len(t) - 2)}


def _levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
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


def _levenshtein_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    n = max(len(a), len(b))
    if n <= 0:
        return 0.0
    return float(_clamp01(1.0 - float(_levenshtein_distance(a, b)) / float(n)))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = float(len(a.intersection(b)))
    union = float(len(a.union(b)))
    if union <= 1e-9:
        return 0.0
    return float(_clamp01(inter / union))


def _cache_root(cache_dir: Optional[str]) -> Path:
    if cache_dir:
        p = Path(cache_dir).expanduser().resolve()
    else:
        p = Path(__file__).resolve().parents[1] / "cache" / "screenspot_det"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(*, image_bytes: bytes, suffix: str) -> str:
    h = hashlib.md5()
    h.update(image_bytes)
    h.update(b"|")
    h.update(str(CONFIG_VERSION).encode("utf-8"))
    h.update(b"|")
    h.update(str(suffix).encode("utf-8"))
    return h.hexdigest()


_OCR_MEM_CACHE: Dict[str, Dict[str, Any]] = {}
_EASYOCR_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool], Any] = {}
_VLM_MEM_CACHE: Dict[str, Dict[str, Any]] = {}
_OWL_VIT_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_YOLO_WORLD_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_RANKER_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _sigmoid(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    if v >= 60.0:
        return 1.0
    if v <= -60.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-v)))


def _linear_ranker_predict_proba(
    *,
    model: Dict[str, Any],
    features: Dict[str, float],
) -> float:
    if not isinstance(model, dict) or not isinstance(features, dict):
        return 0.0
    order = model.get("feature_order")
    wmap = model.get("weights")
    if not isinstance(order, list) or not isinstance(wmap, dict):
        return 0.0
    try:
        bias = float(model.get("bias", 0.0) or 0.0)
    except Exception:
        bias = 0.0
    z = float(bias)
    for name in order:
        if not isinstance(name, str) or not name:
            continue
        try:
            w = float(wmap.get(name, 0.0) or 0.0)
        except Exception:
            w = 0.0
        try:
            x = float(features.get(name, 0.0) or 0.0)
        except Exception:
            x = 0.0
        z += float(w * x)
    return float(_clamp01(_sigmoid(z)))


def _load_ranker(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(cfg, dict):
        return None
    r_cfg = cfg.get("ranker") if isinstance(cfg.get("ranker"), dict) else {}
    if not bool(r_cfg.get("enabled", False)):
        return None

    path = str(r_cfg.get("path") or "").strip()
    if not path:
        return None
    try:
        file_path = Path(path).expanduser().resolve()
    except Exception:
        return None
    if not file_path.exists() or not file_path.is_file():
        return None
    cache_key = str(file_path)
    cached = _RANKER_MODEL_CACHE.get(cache_key)
    try:
        mtime = float(file_path.stat().st_mtime)
    except Exception:
        mtime = 0.0
    if isinstance(cached, dict) and float(cached.get("_mtime", -1.0) or -1.0) == float(mtime):
        return cached

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    feature_order = data.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        feature_order = r_cfg.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        return None
    feature_order2 = [str(x) for x in feature_order if isinstance(x, str) and str(x).strip()]

    weights = data.get("weights")
    if isinstance(weights, list):
        wmap = {}
        for i, name in enumerate(feature_order2):
            try:
                wmap[name] = float(weights[i] or 0.0)
            except Exception:
                wmap[name] = 0.0
        weights = wmap
    if not isinstance(weights, dict):
        return None

    try:
        bias = float(data.get("bias", 0.0) or 0.0)
    except Exception:
        bias = 0.0

    model = {
        "type": str(data.get("type") or "linear"),
        "feature_order": feature_order2,
        "weights": {str(k): float(v or 0.0) for k, v in weights.items() if isinstance(k, str)},
        "bias": float(bias),
        "path": cache_key,
        "_mtime": float(mtime),
    }
    _RANKER_MODEL_CACHE[cache_key] = model
    return model


def _compute_p_intent_deterministic(
    *,
    features: Dict[str, Any],
    util_cfg: Dict[str, Any],
) -> float:
    if not isinstance(features, dict):
        return 0.0
    p_cfg = util_cfg.get("p_intent") if isinstance(util_cfg.get("p_intent"), dict) else {}
    d_cfg = p_cfg.get("deterministic") if isinstance(p_cfg.get("deterministic"), dict) else {}
    weights = {
        "text_overlap": float(d_cfg.get("text_overlap", 0.38) or 0.0),
        "text_levenshtein": float(d_cfg.get("text_levenshtein", 0.18) or 0.0),
        "text_trigram_jaccard": float(d_cfg.get("text_trigram_jaccard", 0.12) or 0.0),
        "keyword_presence": float(d_cfg.get("keyword_presence", 0.10) or 0.0),
        "text_exact_match": float(d_cfg.get("text_exact_match", 0.14) or 0.0),
        "text_phrase_match": float(d_cfg.get("text_phrase_match", 0.08) or 0.0),
    }
    w_sum = float(sum(abs(float(v)) for v in weights.values()))
    if w_sum <= 1e-9:
        return 0.0
    s = 0.0
    for k, w in weights.items():
        try:
            x = float(features.get(k, 0.0) or 0.0)
        except Exception:
            x = 0.0
        s += float(w * _clamp01(x))
    return float(_clamp01(s / w_sum))


def _compute_p_intent(
    *,
    features: Dict[str, Any],
    util_cfg: Dict[str, Any],
    ranker_model: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    p_det = float(_compute_p_intent_deterministic(features=features, util_cfg=util_cfg))
    p_rank = 0.0
    if isinstance(ranker_model, dict):
        p_rank = float(_linear_ranker_predict_proba(model=ranker_model, features=features))
    p_cfg = util_cfg.get("p_intent") if isinstance(util_cfg.get("p_intent"), dict) else {}
    r_cfg = p_cfg.get("ranker") if isinstance(p_cfg.get("ranker"), dict) else {}
    try:
        w_rank = float(r_cfg.get("weight", 0.45) or 0.0)
    except Exception:
        w_rank = 0.0
    w_rank = float(_clamp01(w_rank))
    if not isinstance(ranker_model, dict):
        w_rank = 0.0
    p_blend = float(_clamp01((1.0 - w_rank) * p_det + w_rank * p_rank))
    mode = str(r_cfg.get("mode") or "clamp_max").strip().lower()
    if mode in {"blend", "raw", "linear"}:
        p = float(p_blend)
    elif mode in {"gate_boost", "boost_gate"}:
        try:
            min_det_to_boost = float(r_cfg.get("min_det_to_boost", 0.20) or 0.0)
        except Exception:
            min_det_to_boost = 0.20
        min_det_to_boost = float(_clamp01(min_det_to_boost))
        try:
            max_boost = float(r_cfg.get("max_boost", 0.0) or 0.0)
        except Exception:
            max_boost = 0.0
        max_boost = float(max(0.0, max_boost))

        if float(p_blend) > float(p_det) and float(p_det) < float(min_det_to_boost):
            p = float(p_det)
        else:
            p = float(p_blend)

        if float(max_boost) > 1e-9 and float(p) > float(p_det):
            p = float(min(float(p), float(p_det) + float(max_boost)))
    elif mode in {"scaled_boost", "scale_boost", "mul_boost", "multiplicative_boost"}:
        r = float(_clamp01(p_rank))
        signed = float((float(r) - 0.5) * 2.0)
        signed = float(max(-1.0, min(1.0, signed)))
        p = float(_clamp01(float(p_det) * float(1.0 + float(w_rank) * float(signed))))
    elif mode in {"smart_boost", "icon_smart_boost", "icon_gate_boost"}:
        try:
            icon_mode_strength = float(features.get("icon_mode_strength", 0.0) or 0.0)
        except Exception:
            icon_mode_strength = 0.0
        try:
            icon_likelihood = float(features.get("icon_likelihood", 0.0) or 0.0)
        except Exception:
            icon_likelihood = 0.0
        try:
            text_density = float(features.get("text_density", 0.0) or 0.0)
        except Exception:
            text_density = 0.0

        icon_mode_strength = float(_clamp01(icon_mode_strength))
        icon_likelihood = float(_clamp01(icon_likelihood))
        text_density = float(_clamp01(text_density))

        try:
            icon_mode_strength_min = float(r_cfg.get("icon_mode_strength_min", 0.65) or 0.65)
        except Exception:
            icon_mode_strength_min = 0.65
        try:
            icon_likelihood_min = float(r_cfg.get("icon_likelihood_min", 0.55) or 0.55)
        except Exception:
            icon_likelihood_min = 0.55
        try:
            icon_text_density_max = float(r_cfg.get("icon_text_density_max", 0.40) or 0.40)
        except Exception:
            icon_text_density_max = 0.40
        icon_mode_strength_min = float(_clamp01(icon_mode_strength_min))
        icon_likelihood_min = float(_clamp01(icon_likelihood_min))
        icon_text_density_max = float(_clamp01(icon_text_density_max))

        try:
            min_det_to_boost = float(r_cfg.get("min_det_to_boost", 0.20) or 0.0)
        except Exception:
            min_det_to_boost = 0.20
        min_det_to_boost = float(_clamp01(min_det_to_boost))
        try:
            max_boost = float(r_cfg.get("max_boost", 0.0) or 0.0)
        except Exception:
            max_boost = 0.0
        max_boost = float(max(0.0, max_boost))

        allow_boost = False
        if float(icon_mode_strength) >= float(icon_mode_strength_min):
            if float(icon_likelihood) >= float(icon_likelihood_min) and float(text_density) <= float(icon_text_density_max):
                allow_boost = True
        else:
            if float(p_det) >= float(min_det_to_boost):
                allow_boost = True

        if float(p_blend) > float(p_det) and not bool(allow_boost):
            p = float(p_det)
        else:
            p = float(p_blend)

        if float(max_boost) > 1e-9 and float(p) > float(p_det):
            p = float(min(float(p), float(p_det) + float(max_boost)))
    else:
        p = float(min(float(p_det), float(p_blend)))
    return {"p_intent": float(p), "p_intent_det": float(p_det), "p_intent_ranker": float(p_rank)}


def _compute_uncertainty(
    *,
    candidates: List[Dict[str, Any]],
    conf_margin: float,
    p_intent_top: float,
    min_candidates: int,
) -> Dict[str, Any]:
    n = int(len(candidates))
    weak_recall = bool(n < int(max(0, min_candidates)))
    u_margin = float(_clamp01(1.0 - float(_clamp01(conf_margin))))
    u_intent = float(_clamp01(1.0 - float(_clamp01(p_intent_top))))
    u_recall = 1.0 if weak_recall else 0.0
    u = float(_clamp01(0.55 * u_margin + 0.35 * u_intent + 0.10 * u_recall))
    return {
        "uncertainty": float(u),
        "uncertainty_margin": float(u_margin),
        "uncertainty_intent": float(u_intent),
        "weak_recall": bool(weak_recall),
        "candidate_count": int(n),
    }


def _resize_rgb_max_dim(rgb: np.ndarray, *, max_dim: int) -> Tuple[np.ndarray, float]:
    if max_dim <= 0:
        return rgb, 1.0

    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return rgb, 1.0

    m = int(max(h, w))
    if m <= int(max_dim):
        return rgb, 1.0

    scale = float(max_dim) / float(max(1, m))
    nh = int(max(1, round(float(h) * scale)))
    nw = int(max(1, round(float(w) * scale)))

    cv2 = _maybe_cv2()
    if cv2 is not None:
        resized = cv2.resize(rgb.astype(np.uint8), (int(nw), int(nh)), interpolation=cv2.INTER_AREA)
        return resized, float(scale)

    try:
        from PIL import Image  # type: ignore

        im = Image.fromarray(rgb.astype(np.uint8))
        im2 = im.resize((int(nw), int(nh)), resample=Image.BILINEAR)
        return np.array(im2), float(scale)
    except Exception:
        return rgb, 1.0


def _encode_rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    cv2 = _maybe_cv2()
    if cv2 is not None:
        try:
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                return bytes(buf.tobytes())
        except Exception:
            pass

    import io

    try:
        from PIL import Image  # type: ignore

        im = Image.fromarray(rgb.astype(np.uint8))
        b = io.BytesIO()
        im.save(b, format="PNG")
        return b.getvalue()
    except Exception:
        return b""


def _has_easyocr() -> bool:
    try:
        return importlib.util.find_spec("easyocr") is not None
    except Exception:
        return False


def _easyocr_langs(lang: str) -> List[str]:
    raw = str(lang or "").strip().lower()
    if not raw:
        return ["en"]

    parts = re.split(r"[,+ ]+", raw)
    mapping = {
        "eng": "en",
        "en": "en",
        "deu": "de",
        "ger": "de",
        "de": "de",
        "fra": "fr",
        "fre": "fr",
        "fr": "fr",
        "spa": "es",
        "es": "es",
        "ita": "it",
        "it": "it",
        "por": "pt",
        "pt": "pt",
        "rus": "ru",
        "ru": "ru",
        "jpn": "ja",
        "ja": "ja",
        "kor": "ko",
        "ko": "ko",
        "zho": "ch_sim",
        "chi": "ch_sim",
        "zh": "ch_sim",
    }

    out: List[str] = []
    for p in parts:
        p = str(p or "").strip().lower()
        if not p:
            continue
        out.append(mapping.get(p, p))

    seen: Set[str] = set()
    out2: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        out2.append(p)

    return out2 or ["en"]


def _get_easyocr_reader(*, langs: List[str], gpu: bool) -> Tuple[Optional[Any], Dict[str, Any]]:
    try:
        import certifi  # type: ignore

        ca = str(certifi.where() or "").strip()
        if ca:
            if os.getenv("SSL_CERT_FILE") is None:
                os.environ["SSL_CERT_FILE"] = ca
            if os.getenv("REQUESTS_CA_BUNDLE") is None:
                os.environ["REQUESTS_CA_BUNDLE"] = ca
            if os.getenv("CURL_CA_BUNDLE") is None:
                os.environ["CURL_CA_BUNDLE"] = ca
    except Exception:
        pass
    try:
        import easyocr  # type: ignore
    except Exception as e:
        return None, {"available": False, "engine": "easyocr", "error": f"import_failed: {e}"}

    key = (tuple(str(x) for x in langs), bool(gpu))
    cached = False
    reader = _EASYOCR_READER_CACHE.get(key)
    if reader is not None:
        cached = True
    else:
        try:
            reader = easyocr.Reader(list(langs), gpu=bool(gpu))
        except Exception as e:
            return None, {
                "available": True,
                "engine": "easyocr",
                "langs": list(langs),
                "gpu": bool(gpu),
                "error": str(e),
            }
        _EASYOCR_READER_CACHE[key] = reader

    return reader, {
        "available": True,
        "engine": "easyocr",
        "langs": list(langs),
        "gpu": bool(gpu),
        "reader_cached": bool(cached),
    }


def _run_easyocr_ocr(
    *,
    rgb: np.ndarray,
    lang: str,
    gpu: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    langs = _easyocr_langs(lang)
    reader, meta0 = _get_easyocr_reader(langs=langs, gpu=bool(gpu))
    if reader is None:
        return [], meta0

    img = rgb.astype(np.uint8)
    cv2 = _maybe_cv2()
    if cv2 is not None:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            img = rgb.astype(np.uint8)

    try:
        results = reader.readtext(img)
    except Exception as e:
        m = dict(meta0)
        m["error"] = str(e)
        return [], m

    words: List[Dict[str, Any]] = []
    for rec in results:
        if not isinstance(rec, (list, tuple)) or len(rec) < 3:
            continue
        bbox = rec[0]
        text = rec[1]
        conf = rec[2]
        norm = _normalize_text(str(text or ""))
        if not norm:
            continue
        try:
            conf01 = float(_clamp01(float(conf)))
        except Exception:
            conf01 = 0.0

        try:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            xs: List[float] = []
            ys: List[float] = []
            for p in bbox:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            if not xs or not ys:
                continue
            x0 = int(round(min(xs)))
            y0 = int(round(min(ys)))
            x1 = int(round(max(xs)))
            y1 = int(round(max(ys)))
        except Exception:
            continue

        bw = int(x1 - x0)
        bh = int(y1 - y0)
        if bw <= 0 or bh <= 0:
            continue
        words.append({"bbox_xywh": [int(x0), int(y0), int(bw), int(bh)], "text": norm, "conf": float(conf01)})

    m2: Dict[str, Any] = dict(meta0)
    m2["words"] = int(len(words))
    return words, m2


def _which_tesseract() -> Optional[str]:
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for cand in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"):
        try:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        except Exception:
            continue
    return None


def _run_tesseract_ocr(
    *,
    rgb: np.ndarray,
    lang: str,
    psm: int,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cmd = _which_tesseract()
    if not cmd:
        return [], {"available": False, "engine": "tesseract"}

    png_bytes = _encode_rgb_to_png_bytes(rgb)
    if not png_bytes:
        return [], {"available": True, "engine": "tesseract", "error": "encode_failed"}

    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "img.png"
        img_path.write_bytes(png_bytes)
        args = [
            str(cmd),
            str(img_path),
            "stdout",
            "-l",
            str(lang or "eng"),
            "--psm",
            str(int(psm)),
            "tsv",
        ]

        try:
            proc = subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(max(timeout_s, 0.1)),
                check=False,
            )
        except Exception as e:
            return [], {"available": True, "engine": "tesseract", "error": str(e)}

    try:
        returncode = int(getattr(proc, "returncode", 1))
    except Exception:
        returncode = 1
    if returncode != 0:
        err = str(getattr(proc, "stderr", "") or "").strip()
        return [], {"available": True, "engine": "tesseract", "error": err or f"tesseract_failed_{returncode}"}

    tsv = str(getattr(proc, "stdout", "") or "")
    words: List[Dict[str, Any]] = []
    for i, line in enumerate(tsv.splitlines()):
        if i == 0:
            continue
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        try:
            level = int(cols[0])
        except Exception:
            continue
        if level != 5:
            continue
        try:
            left = int(float(cols[6]))
            top = int(float(cols[7]))
            width = int(float(cols[8]))
            height = int(float(cols[9]))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        try:
            conf_raw = float(cols[10])
        except Exception:
            conf_raw = -1.0
        conf01 = float(_clamp01(conf_raw / 100.0)) if conf_raw >= 0.0 else 0.0
        raw_text = str(cols[11] or "")
        norm = _normalize_text(raw_text)
        if not norm:
            continue
        words.append({"bbox_xywh": [int(left), int(top), int(width), int(height)], "text": norm, "conf": float(conf01)})

    meta_out: Dict[str, Any] = {"available": True, "engine": "tesseract", "words": int(len(words))}
    stderr_text = str(getattr(proc, "stderr", "") or "").strip()
    if stderr_text:
        meta_out["stderr"] = stderr_text
    return words, meta_out


def _get_ocr_words(
    *,
    image_bytes: bytes,
    rgb: np.ndarray,
    cfg: Dict[str, Any],
    cache_dir: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    ocr_cfg = cfg.get("ocr") if isinstance(cfg.get("ocr"), dict) else {}
    enabled = bool(ocr_cfg.get("enabled", True))
    if not enabled:
        return [], {"enabled": False}

    requested_engine = str(ocr_cfg.get("engine", "easyocr") or "easyocr").strip().lower()
    engine = requested_engine
    if engine in ("", "auto"):
        engine = "easyocr"
    if engine in ("easy", "eocr"):
        engine = "easyocr"
    if engine == "tesseract":
        engine = "easyocr"
    max_dim = int(ocr_cfg.get("max_dim", 960) or 960)
    lang = str(ocr_cfg.get("lang", "eng") or "eng")
    psm = int(ocr_cfg.get("psm", 11) or 11)
    timeout_s = float(ocr_cfg.get("timeout_s", 6) or 6)
    gpu = bool(ocr_cfg.get("gpu", False))

    cache_root = _cache_root(cache_dir)
    if engine == "tesseract":
        suffix = f"ocr_{engine}_{max_dim}_{lang}_{psm}"
    elif engine == "easyocr":
        suffix = f"ocr_{engine}_{max_dim}_{lang}_gpu{int(bool(gpu))}"
    else:
        suffix = f"ocr_{engine}_{max_dim}_{lang}"
    key = _cache_key(image_bytes=image_bytes, suffix=suffix)

    mem = _OCR_MEM_CACHE.get(key)
    if isinstance(mem, dict) and isinstance(mem.get("words"), list):
        words = mem.get("words")
        return words, {
            "enabled": True,
            "engine": engine,
            "requested_engine": requested_engine,
            "cached": "memory",
            "words": int(len(words)),
        }

    path = cache_root / "ocr" / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                meta0 = data.get("meta") if isinstance(data.get("meta"), dict) else {}
                if isinstance(meta0, dict) and (meta0.get("error") or meta0.get("available") is False):
                    pass
                else:
                    words = data.get("words")
                    _OCR_MEM_CACHE[key] = {"words": words}
                    return words, {
                        "enabled": True,
                        "engine": engine,
                        "requested_engine": requested_engine,
                        "cached": "disk",
                        "words": int(len(words)),
                    }
        except Exception:
            pass

    rgb_small, scale = _resize_rgb_max_dim(rgb, max_dim=int(max_dim))
    if engine == "tesseract":
        raw_words, meta = _run_tesseract_ocr(rgb=rgb_small, lang=lang, psm=int(psm), timeout_s=float(timeout_s))
    elif engine == "easyocr":
        raw_words, meta = _run_easyocr_ocr(rgb=rgb_small, lang=lang, gpu=bool(gpu))
    else:
        return [], {"enabled": True, "engine": engine, "requested_engine": requested_engine, "available": False}

    h0, w0 = rgb.shape[:2]
    inv = 1.0 / float(max(scale, 1e-9))
    words: List[Dict[str, Any]] = []
    for wrec in raw_words:
        bbox = wrec.get("bbox_xywh")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, bw, bh = [int(v) for v in bbox]
        except Exception:
            continue
        if bw <= 0 or bh <= 0:
            continue
        x0 = int(round(float(x) * inv))
        y0 = int(round(float(y) * inv))
        bw0 = int(max(1, round(float(bw) * inv)))
        bh0 = int(max(1, round(float(bh) * inv)))
        x0 = max(0, min(int(x0), int(max(w0 - 1, 0))))
        y0 = max(0, min(int(y0), int(max(h0 - 1, 0))))
        bw0 = max(1, min(int(bw0), int(max(w0 - x0, 1))))
        bh0 = max(1, min(int(bh0), int(max(h0 - y0, 1))))
        text = _normalize_text(str(wrec.get("text") or ""))
        if not text:
            continue
        conf = float(wrec.get("conf", 0.0) or 0.0)
        words.append({"bbox_xywh": [int(x0), int(y0), int(bw0), int(bh0)], "text": text, "conf": float(_clamp01(conf))})

    out = {
        "engine": engine,
        "max_dim": int(max_dim),
        "lang": lang,
        "psm": int(psm),
        "image_hw": {"h": int(h0), "w": int(w0)},
        "words": words,
        "meta": meta,
    }
    try:
        (cache_root / "ocr").mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    _OCR_MEM_CACHE[key] = {"words": words}
    m: Dict[str, Any] = {
        "enabled": True,
        "engine": engine,
        "requested_engine": requested_engine,
        "cached": False,
        "words": int(len(words)),
    }
    if isinstance(meta, dict):
        m.update(meta)
    return words, m


def _extract_json(text: str) -> Dict[str, Any]:
    t = str(text or "").strip()
    if not t:
        raise ValueError("Empty response")
    try:
        out = json.loads(t)
        if not isinstance(out, dict):
            raise ValueError("Response JSON is not an object")
        return out
    except Exception:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            out = json.loads(t[start : end + 1])
            if not isinstance(out, dict):
                raise ValueError("Response JSON is not an object")
            return out
        raise


def _openai_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_s: float,
) -> Tuple[Dict[str, Any], float]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    t0 = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=float(max(timeout_s, 10.0)))
    latency = time.time() - t0
    if int(resp.status_code) != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("OpenAI API returned non-JSON response")
    return data, float(latency)


def _resolve_torch_device(*, requested: str, torch: Any) -> str:
    d = str(requested or "").strip().lower()
    if d in ("", "auto"):
        try:
            if bool(torch.cuda.is_available()):
                return "cuda"
        except Exception:
            pass
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and bool(mps.is_available()):
                return "mps"
        except Exception:
            pass
        return "cpu"
    return d


def _get_owlvit_bundle(*, model_name: str, device: str, local_files_only: bool) -> Tuple[Optional[Tuple[Any, Any, str]], Dict[str, Any]]:
    try:
        import torch  # type: ignore

        from transformers import OwlViTForObjectDetection, OwlViTProcessor  # type: ignore
    except Exception as e:
        return None, {"available": False, "provider": "owlvit", "model": str(model_name), "error": f"import_failed: {e}"}

    dev = _resolve_torch_device(requested=str(device), torch=torch)
    key = (str(model_name), str(dev))
    bundle = _OWL_VIT_MODEL_CACHE.get(key)
    if bundle is not None:
        return bundle, {
            "available": True,
            "provider": "owlvit",
            "model": str(model_name),
            "device": str(dev),
            "model_cached": True,
        }

    local_only_env = bool(os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1")
    try:
        proc = OwlViTProcessor.from_pretrained(str(model_name), local_files_only=True)
        mdl = OwlViTForObjectDetection.from_pretrained(str(model_name), local_files_only=True)
    except Exception as e_local:
        if bool(local_files_only) or bool(local_only_env):
            return None, {
                "available": False,
                "provider": "owlvit",
                "model": str(model_name),
                "device": str(dev),
                "error": f"model_not_available_offline: {e_local}",
            }
        try:
            proc = OwlViTProcessor.from_pretrained(str(model_name))
            mdl = OwlViTForObjectDetection.from_pretrained(str(model_name))
        except Exception as e:
            return None, {"available": False, "provider": "owlvit", "model": str(model_name), "device": str(dev), "error": str(e)}

    dev2 = str(dev)
    try:
        mdl = mdl.to(dev2)
    except Exception:
        dev2 = "cpu"
        try:
            mdl = mdl.to(dev2)
        except Exception:
            pass
    try:
        mdl.eval()
    except Exception:
        pass

    bundle = (proc, mdl, dev2)
    _OWL_VIT_MODEL_CACHE[(str(model_name), str(dev2))] = bundle
    return bundle, {
        "available": True,
        "provider": "owlvit",
        "model": str(model_name),
        "device": str(dev2),
        "model_cached": False,
    }


def _get_yoloworld_bundle(*, model_name: str, device: str, local_files_only: bool) -> Tuple[Optional[Tuple[Any, str]], Dict[str, Any]]:
    try:
        import certifi  # type: ignore

        ca = str(certifi.where() or "").strip()
        if ca:
            if os.getenv("SSL_CERT_FILE") is None:
                os.environ["SSL_CERT_FILE"] = ca
            if os.getenv("REQUESTS_CA_BUNDLE") is None:
                os.environ["REQUESTS_CA_BUNDLE"] = ca
            if os.getenv("CURL_CA_BUNDLE") is None:
                os.environ["CURL_CA_BUNDLE"] = ca
    except Exception:
        pass

    try:
        import torch  # type: ignore

        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        return None, {"available": False, "provider": "yoloworld", "model": str(model_name), "error": f"import_failed: {e}"}

    dev = _resolve_torch_device(requested=str(device), torch=torch)
    key = (str(model_name), str(dev))
    bundle = _YOLO_WORLD_MODEL_CACHE.get(key)
    if bundle is not None:
        return bundle, {
            "available": True,
            "provider": "yoloworld",
            "model": str(model_name),
            "device": str(dev),
            "model_cached": True,
        }

    if bool(local_files_only):
        try:
            p = Path(str(model_name)).expanduser()
            if not p.exists():
                return None, {
                    "available": False,
                    "provider": "yoloworld",
                    "model": str(model_name),
                    "device": str(dev),
                    "error": "model_not_available_offline",
                }
        except Exception:
            return None, {
                "available": False,
                "provider": "yoloworld",
                "model": str(model_name),
                "device": str(dev),
                "error": "model_not_available_offline",
            }

    try:
        mdl = YOLO(str(model_name))
    except Exception as e:
        return None, {"available": False, "provider": "yoloworld", "model": str(model_name), "device": str(dev), "error": str(e)}

    try:
        mdl.to(str(dev))
    except Exception:
        pass

    bundle2 = (mdl, str(dev))
    _YOLO_WORLD_MODEL_CACHE[(str(model_name), str(dev))] = bundle2
    return bundle2, {
        "available": True,
        "provider": "yoloworld",
        "model": str(model_name),
        "device": str(dev),
        "model_cached": False,
    }


def _get_vlm_candidates(
    *,
    image_bytes: bytes,
    rgb: np.ndarray,
    instruction: str,
    cfg: Dict[str, Any],
    cache_dir: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    vlm_cfg = cfg.get("vlm") if isinstance(cfg.get("vlm"), dict) else {}
    enabled = bool(vlm_cfg.get("enabled", False))
    if not enabled:
        return [], {"enabled": False}

    provider = str(vlm_cfg.get("provider", "owlvit") or "owlvit").strip().lower()
    model = str(vlm_cfg.get("model", "google/owlvit-base-patch32") or "google/owlvit-base-patch32").strip()
    max_dim = int(vlm_cfg.get("max_dim", 1280) or 1280)
    max_candidates = int(vlm_cfg.get("max_candidates", 24) or 24)
    timeout_s = float(vlm_cfg.get("timeout_s", 120) or 120)
    use_cache = bool(vlm_cfg.get("cache", True))

    _score_thr_raw = vlm_cfg.get("score_threshold", 0.005)
    if _score_thr_raw is None:
        _score_thr_raw = 0.005
    if isinstance(_score_thr_raw, str) and not _score_thr_raw.strip():
        _score_thr_raw = 0.005
    try:
        score_threshold = float(_score_thr_raw)
    except Exception:
        score_threshold = 0.005
    max_queries = int(vlm_cfg.get("max_queries", 8) or 8)
    device = str(vlm_cfg.get("device", "auto") or "auto")
    local_files_only = bool(vlm_cfg.get("local_files_only", False))

    tiling_cfg = vlm_cfg.get("tiling") if isinstance(vlm_cfg.get("tiling"), dict) else {}
    tile_enabled = bool(tiling_cfg.get("enabled", False))
    grid_raw = tiling_cfg.get("grid")
    if isinstance(grid_raw, (list, tuple)) and len(grid_raw) == 2:
        try:
            tile_grid_x = int(grid_raw[0])
            tile_grid_y = int(grid_raw[1])
        except Exception:
            tile_grid_x, tile_grid_y = 2, 2
    else:
        tile_grid_x, tile_grid_y = 2, 2
    tile_grid_x = int(max(1, min(6, int(tile_grid_x))))
    tile_grid_y = int(max(1, min(6, int(tile_grid_y))))

    _ov_raw = tiling_cfg.get("overlap", 0.18)
    if _ov_raw is None:
        _ov_raw = 0.18
    if isinstance(_ov_raw, str) and not _ov_raw.strip():
        _ov_raw = 0.18
    try:
        tile_overlap = float(_ov_raw)
    except Exception:
        tile_overlap = 0.18
    tile_overlap = float(_clamp01(tile_overlap))

    tile_max_dim = int(tiling_cfg.get("max_dim", 960) or 960)
    tile_max_candidates = int(tiling_cfg.get("max_candidates_per_tile", 8) or 8)
    tile_max_candidates = int(max(1, min(32, int(tile_max_candidates))))

    _bs_raw = tiling_cfg.get("base_share", 0.55)
    if _bs_raw is None:
        _bs_raw = 0.55
    if isinstance(_bs_raw, str) and not _bs_raw.strip():
        _bs_raw = 0.55
    try:
        tile_base_share = float(_bs_raw)
    except Exception:
        tile_base_share = 0.55
    tile_base_share = float(_clamp01(tile_base_share))

    if provider in ("owl-vit", "owl_vit", "owl"):
        provider = "owlvit"
    if provider in ("yolo-world", "yolo_world", "yolo", "yoloworld", "yolov8-world", "yolov8_world"):
        provider = "yoloworld"
    if provider == "owlvit":
        ml = str(model or "").strip().lower()
        if ml.endswith(".pt") and "world" in ml:
            provider = "yoloworld"
    if provider not in ("owlvit", "yoloworld"):
        return [], {"enabled": True, "available": False, "provider": provider, "model": model, "error": "unsupported_provider"}

    cache_root = _cache_root(cache_dir)
    inst_norm = _normalize_text(instruction)
    inst_hash = hashlib.md5(inst_norm.encode("utf-8")).hexdigest()[:10]
    thr_tag = f"{float(score_threshold):.6f}"
    if tile_enabled:
        tile_tag = (
            f"tile{int(tile_grid_x)}x{int(tile_grid_y)}_ov{float(tile_overlap):.2f}_"
            f"md{int(tile_max_dim)}_mc{int(tile_max_candidates)}_bs{float(tile_base_share):.2f}"
        )
    else:
        tile_tag = "tile0"
    suffix = f"vlm_{provider}_{model}_{max_dim}_{max_candidates}_{thr_tag}_{max_queries}_{tile_tag}_{inst_hash}"
    key = _cache_key(image_bytes=image_bytes, suffix=suffix)

    mem = _VLM_MEM_CACHE.get(key)
    if use_cache and isinstance(mem, dict) and isinstance(mem.get("candidates"), list):
        cands = mem.get("candidates")
        return cands, {
            "enabled": True,
            "available": True,
            "provider": provider,
            "model": model,
            "cached": "memory",
            "candidates": int(len(cands)),
        }

    path = cache_root / "vlm" / f"{key}.json"
    if use_cache and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("candidates"), list):
                meta0 = data.get("meta") if isinstance(data.get("meta"), dict) else {}
                if isinstance(meta0, dict) and meta0.get("error"):
                    pass
                else:
                    cands = data.get("candidates")
                    _VLM_MEM_CACHE[key] = {"candidates": cands}
                    return cands, {
                        "enabled": True,
                        "available": True,
                        "provider": provider,
                        "model": model,
                        "cached": "disk",
                        "candidates": int(len(cands)),
                    }
        except Exception:
            pass

    out_meta: Dict[str, Any] = {
        "enabled": True,
        "available": True,
        "provider": str(provider),
        "model": model,
        "cached": False,
        "max_dim": int(max_dim),
        "max_candidates": int(max_candidates),
        "score_threshold": float(score_threshold),
        "max_queries": int(max_queries),
    }

    groups = _instruction_groups(instruction)
    targets = _extract_target_phrases(instruction)

    queries: List[str] = []
    for t in targets:
        nt = _normalize_text(t)
        if nt:
            queries.append(nt)

    group_to_queries: Dict[str, List[str]] = {
        "close": ["close button", "close icon", "x button", "x icon"],
        "back": ["back button", "back arrow", "left arrow icon", "arrow left"],
        "menu": ["menu icon", "hamburger menu", "three lines icon"],
        "more": ["more icon", "ellipsis icon", "three dots icon"],
        "settings": ["settings icon", "gear icon", "cog icon"],
        "search": ["search icon", "magnifying glass icon"],
        "refresh": ["refresh icon", "reload icon"],
        "share": ["share icon", "share button"],
        "copy": ["copy icon", "copy button"],
        "paste": ["paste icon", "paste button"],
        "filter": ["filter icon", "funnel icon"],
        "sort": ["sort icon", "sort button"],
        "location": ["location icon", "map pin icon", "pin icon"],
        "history": ["history icon", "clock icon"],
        "home": ["home icon", "home button"],
        "profile": ["profile icon", "person icon", "user icon"],
        "notifications": ["notification icon", "bell icon"],
        "favorite": ["favorite icon", "heart icon", "star icon"],
    }
    for g in groups:
        qs = group_to_queries.get(str(g))
        if not qs:
            continue
        for q in qs:
            nq = _normalize_text(q)
            if nq:
                queries.append(nq)

    if not queries:
        tin = _normalize_text(instruction)
        if tin:
            queries.append(tin[:64])

    seen: Set[str] = set()
    queries2: List[str] = []
    for q in queries:
        qq = str(q or "").strip()
        if not qq or qq in seen:
            continue
        seen.add(qq)
        queries2.append(qq)
    queries = queries2[: max(1, int(max_queries))]
    out_meta["queries"] = queries

    if provider == "owlvit":
        bundle, meta_model = _get_owlvit_bundle(model_name=model, device=device, local_files_only=bool(local_files_only))
        if isinstance(meta_model, dict):
            out_meta.update(meta_model)
        if bundle is None:
            out_meta["available"] = False
            return [], out_meta
        proc, mdl, dev = bundle
        yolo_mdl = None
    else:
        bundle, meta_model = _get_yoloworld_bundle(model_name=model, device=device, local_files_only=bool(local_files_only))
        if isinstance(meta_model, dict):
            out_meta.update(meta_model)
        if bundle is None:
            out_meta["available"] = False
            return [], out_meta
        yolo_mdl, dev = bundle
        proc, mdl = None, None
        try:
            yolo_mdl.set_classes([str(q) for q in queries if str(q or "").strip()])
        except Exception:
            pass

    h0, w0 = rgb.shape[:2]

    if provider == "owlvit":
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            out_meta["error"] = str(e)
            return [], out_meta

        try:
            import torch  # type: ignore
        except Exception as e:
            out_meta["error"] = str(e)
            return [], out_meta

        def _infer_rgb(
            *,
            rgb_in: np.ndarray,
            offset_x: int,
            offset_y: int,
            max_dim_local: int,
            max_out: int,
        ) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
            rgb_small, scale = _resize_rgb_max_dim(rgb_in, max_dim=int(max_dim_local))
            inv = 1.0 / float(max(scale, 1e-9))
            try:
                im = Image.fromarray(rgb_small.astype(np.uint8))
            except Exception as e_img:
                return [], 0.0, str(e_img)

            try:
                inputs = proc(text=[queries], images=im, return_tensors="pt")
                for k, v in list(getattr(inputs, "items", lambda: [])()):
                    try:
                        inputs[k] = v.to(str(dev))
                    except Exception:
                        continue
                t0 = time.time()
                with torch.no_grad():
                    outputs = mdl(**inputs)
                latency = float(time.time() - t0)
                _ = float(timeout_s)

                target_sizes = torch.tensor([[int(im.size[1]), int(im.size[0])]], device=str(dev))
                results = proc.post_process_object_detection(
                    outputs=outputs,
                    threshold=float(score_threshold),
                    target_sizes=target_sizes,
                )
                det0 = results[0] if isinstance(results, list) and results else {}
                boxes = det0.get("boxes")
                scores = det0.get("scores")
                labels = det0.get("labels")
            except Exception as e_inf:
                return [], 0.0, str(e_inf)

            try:
                boxes_list = boxes.detach().cpu().tolist() if boxes is not None else []
            except Exception:
                boxes_list = []
            try:
                scores_list = scores.detach().cpu().tolist() if scores is not None else []
            except Exception:
                scores_list = []
            try:
                labels_list = labels.detach().cpu().tolist() if labels is not None else []
            except Exception:
                labels_list = []

            dets: List[Tuple[float, int, List[float]]] = []
            for i, b in enumerate(boxes_list):
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    continue
                sc = float(scores_list[i]) if i < len(scores_list) else 0.0
                lb = int(labels_list[i]) if i < len(labels_list) else -1
                dets.append((float(sc), int(lb), [float(v) for v in b]))
            dets.sort(key=lambda t: float(t[0]), reverse=True)

            out: List[Dict[str, Any]] = []
            for sc, lb, b in dets[: max(1, int(max_out))]:
                x0s, y0s, x1s, y1s = b
                px0 = float(x0s) * float(inv) + float(offset_x)
                py0 = float(y0s) * float(inv) + float(offset_y)
                px1 = float(x1s) * float(inv) + float(offset_x)
                py1 = float(y1s) * float(inv) + float(offset_y)
                x_min = int(max(0, min(int(round(px0)), int(round(px1)))))
                y_min = int(max(0, min(int(round(py0)), int(round(py1)))))
                x_max = int(min(int(w0), max(int(round(px0)), int(round(px1)))))
                y_max = int(min(int(h0), max(int(round(py0)), int(round(py1)))))
                bw = int(x_max - x_min)
                bh = int(y_max - y_min)
                if bw <= 0 or bh <= 0:
                    continue
                cand: Dict[str, Any] = {
                    "bbox_xywh": [int(x_min), int(y_min), int(bw), int(bh)],
                    "confidence": float(_clamp01(sc)),
                }
                if 0 <= int(lb) < len(queries):
                    cand["label"] = str(queries[int(lb)] or "")[:64]
                out.append(cand)
            return out, float(latency), None
    else:

        def _infer_rgb(
            *,
            rgb_in: np.ndarray,
            offset_x: int,
            offset_y: int,
            max_dim_local: int,
            max_out: int,
        ) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
            rgb_small, scale = _resize_rgb_max_dim(rgb_in, max_dim=int(max_dim_local))
            inv = 1.0 / float(max(scale, 1e-9))
            try:
                t0 = time.time()
                results = yolo_mdl.predict(
                    source=rgb_small,
                    imgsz=int(max_dim_local),
                    conf=float(score_threshold),
                    device=str(dev),
                    verbose=False,
                    max_det=int(max(1, int(max_out))),
                )
                latency = float(time.time() - t0)
            except Exception as e_inf:
                return [], 0.0, str(e_inf)

            res0 = results[0] if isinstance(results, list) and results else None
            if res0 is None:
                return [], float(latency), None

            boxes0 = getattr(res0, "boxes", None)
            if boxes0 is None:
                return [], float(latency), None

            try:
                xyxy = getattr(boxes0, "xyxy", None)
                confs = getattr(boxes0, "conf", None)
                clss = getattr(boxes0, "cls", None)
                xyxy_list = xyxy.detach().cpu().tolist() if xyxy is not None else []
                conf_list = confs.detach().cpu().tolist() if confs is not None else []
                cls_list = clss.detach().cpu().tolist() if clss is not None else []
            except Exception:
                xyxy_list, conf_list, cls_list = [], [], []

            names = getattr(res0, "names", None)

            dets: List[Tuple[float, int, List[float]]] = []
            for i, b in enumerate(xyxy_list):
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    continue
                sc = float(conf_list[i]) if i < len(conf_list) else 0.0
                lb = int(cls_list[i]) if i < len(cls_list) else -1
                dets.append((float(sc), int(lb), [float(v) for v in b]))
            dets.sort(key=lambda t: float(t[0]), reverse=True)

            out: List[Dict[str, Any]] = []
            for sc, lb, b in dets[: max(1, int(max_out))]:
                x0s, y0s, x1s, y1s = b
                px0 = float(x0s) * float(inv) + float(offset_x)
                py0 = float(y0s) * float(inv) + float(offset_y)
                px1 = float(x1s) * float(inv) + float(offset_x)
                py1 = float(y1s) * float(inv) + float(offset_y)
                x_min = int(max(0, min(int(round(px0)), int(round(px1)))))
                y_min = int(max(0, min(int(round(py0)), int(round(py1)))))
                x_max = int(min(int(w0), max(int(round(px0)), int(round(px1)))))
                y_max = int(min(int(h0), max(int(round(py0)), int(round(py1)))))
                bw = int(x_max - x_min)
                bh = int(y_max - y_min)
                if bw <= 0 or bh <= 0:
                    continue
                cand: Dict[str, Any] = {
                    "bbox_xywh": [int(x_min), int(y_min), int(bw), int(bh)],
                    "confidence": float(_clamp01(sc)),
                }
                label = ""
                try:
                    if isinstance(names, dict) and int(lb) in names:
                        label = str(names.get(int(lb)) or "")
                    elif isinstance(names, (list, tuple)) and 0 <= int(lb) < len(names):
                        label = str(names[int(lb)] or "")
                    elif 0 <= int(lb) < len(queries):
                        label = str(queries[int(lb)] or "")
                except Exception:
                    label = ""
                if label:
                    cand["label"] = str(label)[:64]
                out.append(cand)
            return out, float(latency), None

    base_cands, base_lat, base_err = _infer_rgb(
        rgb_in=rgb,
        offset_x=0,
        offset_y=0,
        max_dim_local=int(max_dim),
        max_out=int(max_candidates),
    )
    if base_err:
        out_meta["error"] = str(base_err)
        return [], out_meta

    def _conf(c: Dict[str, Any]) -> float:
        try:
            return float(c.get("confidence", 0.0) or 0.0)
        except Exception:
            return 0.0

    base_sorted = sorted(base_cands, key=_conf, reverse=True)

    candidates: List[Dict[str, Any]] = []
    lat_total = float(base_lat)
    if tile_enabled:
        tile_lat_sum = 0.0
        tile_errors = 0
        tile_cands: List[Dict[str, Any]] = []
        for gy in range(int(tile_grid_y)):
            y0 = int(round(float(gy) * float(h0) / float(max(1, int(tile_grid_y)))))
            y1 = int(round(float(gy + 1) * float(h0) / float(max(1, int(tile_grid_y)))))
            for gx in range(int(tile_grid_x)):
                x0 = int(round(float(gx) * float(w0) / float(max(1, int(tile_grid_x)))))
                x1 = int(round(float(gx + 1) * float(w0) / float(max(1, int(tile_grid_x)))))
                pad_x = int(round(float(tile_overlap) * float(max(1, x1 - x0))))
                pad_y = int(round(float(tile_overlap) * float(max(1, y1 - y0))))
                xx0 = int(max(0, x0 - pad_x))
                yy0 = int(max(0, y0 - pad_y))
                xx1 = int(min(w0, x1 + pad_x))
                yy1 = int(min(h0, y1 + pad_y))
                bw_t = int(xx1 - xx0)
                bh_t = int(yy1 - yy0)
                if bw_t <= 2 or bh_t <= 2:
                    continue
                patch = rgb[int(yy0) : int(yy1), int(xx0) : int(xx1)]
                c2, lat2, err2 = _infer_rgb(
                    rgb_in=patch,
                    offset_x=int(xx0),
                    offset_y=int(yy0),
                    max_dim_local=int(tile_max_dim),
                    max_out=int(tile_max_candidates),
                )
                if err2:
                    tile_errors += 1
                    continue
                tile_lat_sum += float(lat2)
                tile_cands.extend(c2)

        lat_total += float(tile_lat_sum)
        out_meta["tiling"] = {
            "enabled": True,
            "grid": [int(tile_grid_x), int(tile_grid_y)],
            "overlap": float(tile_overlap),
            "max_dim": int(tile_max_dim),
            "max_candidates_per_tile": int(tile_max_candidates),
            "base_share": float(tile_base_share),
            "tile_errors": int(tile_errors),
        }

        base_keep = int(round(float(max_candidates) * float(tile_base_share)))
        base_keep = int(max(1, min(int(max_candidates), int(base_keep))))
        dedup_iou = 0.65

        kept: List[Dict[str, Any]] = []
        kept_boxes: List[Tuple[int, int, int, int]] = []

        def _try_add(cand: Dict[str, Any]) -> bool:
            bb = cand.get("bbox_xywh") if isinstance(cand, dict) else None
            if not isinstance(bb, list) or len(bb) != 4:
                return False
            try:
                box = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            except Exception:
                return False
            if box[2] <= 0 or box[3] <= 0:
                return False
            for ob in kept_boxes:
                if float(_bbox_iou_xywh(box, ob)) >= float(dedup_iou):
                    return False
            kept.append(cand)
            kept_boxes.append(box)
            return True

        for c0 in base_sorted[: int(base_keep)]:
            _try_add(c0)
        tile_sorted = sorted(tile_cands, key=_conf, reverse=True)
        for c1 in tile_sorted:
            if len(kept) >= int(max_candidates):
                break
            _try_add(c1)
        for c2 in base_sorted[int(base_keep) :]:
            if len(kept) >= int(max_candidates):
                break
            _try_add(c2)

        candidates = kept[: int(max_candidates)]
        out_meta["latency_seconds"] = float(lat_total)
        out_meta["latency_base_seconds"] = float(base_lat)
        out_meta["latency_tiles_seconds"] = float(tile_lat_sum)
        out_meta["base_candidates"] = int(len(base_cands))
        out_meta["tile_candidates"] = int(len(tile_cands))
    else:
        candidates = base_sorted[: max(1, int(max_candidates))]
        out_meta["latency_seconds"] = float(base_lat)

    out = {"candidates": candidates, "meta": out_meta}
    try:
        (cache_root / "vlm").mkdir(parents=True, exist_ok=True)
        if use_cache:
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    _VLM_MEM_CACHE[key] = {"candidates": candidates}
    out_meta["candidates"] = int(len(candidates))
    return candidates, out_meta


def _verify_vlm_candidate(
    *,
    bbox_xywh: Tuple[int, int, int, int],
    sal: np.ndarray,
    edges: np.ndarray,
    rgb: np.ndarray,
    min_saliency_mean: float,
    min_saliency_max: float,
    min_edge_mean: float,
    min_contrast: float,
) -> Tuple[bool, Dict[str, float]]:
    x, y, bw, bh = bbox_xywh
    ps = sal[int(y) : int(y + bh), int(x) : int(x + bw)]
    pe = edges[int(y) : int(y + bh), int(x) : int(x + bw)]
    sal_mean = float(np.mean(ps)) if ps.size else 0.0
    sal_max = float(np.max(ps)) if ps.size else 0.0
    edge_mean = float(np.mean(pe)) if pe.size else 0.0
    contrast = float(_color_contrast_score(rgb=rgb, x=int(x), y=int(y), bw=int(bw), bh=int(bh)))

    ok = (
        float(sal_mean) >= float(min_saliency_mean)
        or float(sal_max) >= float(min_saliency_max)
        or (float(edge_mean) >= float(min_edge_mean) and float(contrast) >= float(min_contrast))
    )
    return bool(ok), {
        "saliency_mean": float(_clamp01(sal_mean)),
        "saliency_max": float(_clamp01(sal_max)),
        "edge_mean": float(_clamp01(edge_mean)),
        "contrast": float(_clamp01(contrast)),
    }


def _bbox_iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax, ay, ax + aw, ay + ah
    bx0, by0, bx1, by1 = bx, by, bx + bw, by + bh
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, aw) * max(0, ah))
    area_b = float(max(0, bw) * max(0, bh))
    union = float(area_a + area_b - inter)
    if union <= 1e-9:
        return 0.0
    return float(_clamp01(inter / union))


def _merge_boxes_xywh(boxes: List[Tuple[int, int, int, int]], *, iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes2 = [tuple(int(v) for v in b) for b in boxes if len(b) == 4]
    boxes2 = [b for b in boxes2 if b[2] > 0 and b[3] > 0]
    if not boxes2:
        return []
    boxes2.sort(key=lambda b: int(b[2] * b[3]), reverse=True)
    out: List[Tuple[int, int, int, int]] = []
    used = [False for _ in boxes2]
    for i, b in enumerate(boxes2):
        if used[i]:
            continue
        mx, my, mw, mh = b
        used[i] = True
        changed = True
        while changed:
            changed = False
            merged = (mx, my, mw, mh)
            for j, other in enumerate(boxes2):
                if used[j]:
                    continue
                if float(_bbox_iou_xywh(merged, other)) <= float(iou_threshold):
                    continue
                ox, oy, ow, oh = other
                x0 = min(mx, ox)
                y0 = min(my, oy)
                x1 = max(mx + mw, ox + ow)
                y1 = max(my + mh, oy + oh)
                mx, my, mw, mh = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                used[j] = True
                changed = True
        out.append((int(mx), int(my), int(mw), int(mh)))
    out.sort(key=lambda b: int(b[2] * b[3]), reverse=True)
    return out


def _bbox_xyxy_from_xywh_pixels(*, bbox_xywh: Sequence[int], w: int, h: int) -> List[float]:
    x, y, bw, bh = [int(v) for v in bbox_xywh]
    if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = float(x) / float(w)
    y0 = float(y) / float(h)
    x1 = float(x + bw) / float(w)
    y1 = float(y + bh) / float(h)
    return [
        float(np.clip(x0, 0.0, 1.0)),
        float(np.clip(y0, 0.0, 1.0)),
        float(np.clip(x1, 0.0, 1.0)),
        float(np.clip(y1, 0.0, 1.0)),
    ]


def _triangular_score(value: float, *, peak: float, width: float) -> float:
    if width <= 0.0:
        return 0.0
    dist = abs(float(value) - float(peak))
    return float(_clamp01(1.0 - dist / float(width)))


def _gaussian_score(*, x: float, y: float, mx: float, my: float, sigma: float) -> float:
    s = float(max(float(sigma), 1e-6))
    dx = float(x) - float(mx)
    dy = float(y) - float(my)
    return float(math.exp(-0.5 * (dx * dx + dy * dy) / (s * s)))


def _extract_target_phrases(instruction: str) -> List[str]:
    raw = str(instruction or "")
    out: List[str] = []
    for m in re.finditer(r"\"([^\"]{1,80})\"", raw):
        p = _normalize_text(m.group(1))
        if p and p not in out:
            out.append(p)
    for m in re.finditer(r"'([^']{1,80})'", raw):
        p = _normalize_text(m.group(1))
        if p and p not in out:
            out.append(p)

    t = _normalize_text(raw)
    for pat in (
        r"(?:labeled|labelled) ([a-z0-9][a-z0-9 ]{0,40})",
        r"(?:called|named) ([a-z0-9][a-z0-9 ]{0,40})",
        r"(?:with text|text) ([a-z0-9][a-z0-9 ]{0,40})",
        r"(?:that says|says) ([a-z0-9][a-z0-9 ]{0,40})",
    ):
        m = re.search(pat, t)
        if not m:
            continue
        p = _normalize_text(m.group(1))
        if p and p not in out:
            out.append(p)

    icon_words = {
        "close",
        "back",
        "settings",
        "menu",
        "refresh",
        "share",
        "copy",
        "paste",
        "filter",
        "sort",
        "location",
        "history",
        "home",
        "profile",
        "notifications",
        "favorite",
        "more",
    }
    for pat in (
        r"(?:click|tap|press|select|open|choose|pick|hit) (?:on )?(?:the )?([a-z0-9][a-z0-9 ]{0,60})",
        r"(?:change|set|enable|disable|turn on|turn off|create|delete|remove|add) (?:the )?([a-z0-9][a-z0-9 ]{0,60})",
    ):
        m = re.search(pat, t)
        if not m:
            continue
        p_raw = str(m.group(1) or "")
        for sep in (
            " based on ",
            " in ",
            " on ",
            " at ",
            " to ",
            " from ",
            " within ",
            " inside ",
            " under ",
            " above ",
            " below ",
            " for ",
            " with ",
            " of ",
        ):
            j = p_raw.find(sep)
            if j > 0:
                p_raw = p_raw[:j]
                break
        p = _normalize_text(p_raw)
        if not p or p in out:
            continue
        if p in icon_words and len(_tokens(p)) <= 2:
            continue
        out.append(p)
    return out[:4]


def _extract_spatial_hints(instruction: str) -> List[str]:
    t = _normalize_text(instruction)
    out: List[str] = []
    for phrase, lab in (
        ("top left", "top_left"),
        ("upper left", "top_left"),
        ("top right", "top_right"),
        ("upper right", "top_right"),
        ("bottom left", "bottom_left"),
        ("lower left", "bottom_left"),
        ("bottom right", "bottom_right"),
        ("lower right", "bottom_right"),
    ):
        if phrase in t and lab not in out:
            out.append(lab)
    if out:
        return out
    if "header" in t or re.search(r"\btop\b", t) or re.search(r"\bupper\b", t):
        out.append("top")
    if "footer" in t or re.search(r"\bbottom\b", t) or re.search(r"\blower\b", t):
        out.append("bottom")
    if re.search(r"\bleft\b", t):
        out.append("left")
    if re.search(r"\bright\b", t):
        out.append("right")
    if re.search(r"\bcenter\b", t) or re.search(r"\bmiddle\b", t):
        out.append("center")
    return out


def _spatial_hints_prior(*, hints: List[str], cx: float, cy: float) -> float:
    if not hints:
        return 0.0
    best = 0.0
    for h in hints:
        if h == "top_left":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.08, my=0.12, sigma=0.24))
        elif h == "top_right":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.92, my=0.12, sigma=0.24))
        elif h == "bottom_left":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.08, my=0.92, sigma=0.26))
        elif h == "bottom_right":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.92, my=0.92, sigma=0.26))
        elif h == "top":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.50, my=0.12, sigma=0.30))
        elif h == "bottom":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.50, my=0.92, sigma=0.32))
        elif h == "left":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.08, my=0.52, sigma=0.34))
        elif h == "right":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.92, my=0.52, sigma=0.34))
        elif h == "center":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.50, my=0.52, sigma=0.40))
    return float(_clamp01(best))


def _phrase_match_score(*, target: str, region_tokens: Sequence[str]) -> float:
    tt = _tokens(target)
    if not tt or not region_tokens:
        return 0.0
    if len(tt) == 1:
        return 1.0 if tt[0] in set(region_tokens) else 0.0
    best = 0
    for i in range(0, max(0, len(region_tokens) - 1)):
        k = 0
        while k < len(tt) and (i + k) < len(region_tokens) and region_tokens[i + k] == tt[k]:
            k += 1
        best = max(best, k)
        if best >= len(tt):
            break
    return float(_clamp01(float(best) / float(max(len(tt), 1))))


def _search_bar_prior(*, cx: float, cy: float, area_ratio: float, aspect: float) -> float:
    if aspect <= 0.0:
        return 0.0
    asp = float(_triangular_score(float(aspect), peak=7.0, width=6.0))
    pos = float(_gaussian_score(x=float(cx), y=float(cy), mx=0.50, my=0.14, sigma=0.22))
    area = float(_triangular_score(float(area_ratio), peak=0.035, width=0.10))
    return float(_clamp01(0.50 * asp + 0.35 * pos + 0.15 * area))


def _icon_toolbar_prior(*, cx: float, cy: float) -> float:
    top = max(
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.05, my=0.06, sigma=0.18)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.95, my=0.06, sigma=0.18)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.08, my=0.08, sigma=0.22)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.92, my=0.08, sigma=0.22)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.92, my=0.06, sigma=0.22)),
    )
    bottom = max(
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.10, my=0.92, sigma=0.25)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.50, my=0.92, sigma=0.25)),
        float(_gaussian_score(x=float(cx), y=float(cy), mx=0.90, my=0.92, sigma=0.25)),
    )
    return float(_clamp01(max(top, bottom)))


def _symmetry_score(patch: np.ndarray) -> float:
    if not isinstance(patch, np.ndarray) or patch.ndim != 2 or patch.size < 9:
        return 0.0
    p = np.clip(patch.astype(np.float32), 0.0, 1.0)
    h, w = p.shape
    m = int(max(h, w))
    if m > 64:
        step = int(max(1, math.ceil(float(m) / 64.0)))
        p = p[::step, ::step]
    fx = np.flip(p, axis=1)
    fy = np.flip(p, axis=0)
    sx = 1.0 - float(np.mean(np.abs(p - fx)))
    sy = 1.0 - float(np.mean(np.abs(p - fy)))
    return float(_clamp01(0.5 * sx + 0.5 * sy))


def _color_contrast_score(*, rgb: np.ndarray, x: int, y: int, bw: int, bh: int) -> float:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or int(rgb.shape[2]) != 3:
        return 0.0
    if bw <= 1 or bh <= 1:
        return 0.0
    h, w = rgb.shape[:2]
    pad = int(max(2, round(float(min(bw, bh)) * 0.15)))
    pad = int(min(pad, 16))
    x0 = max(0, int(x) - int(pad))
    y0 = max(0, int(y) - int(pad))
    x1 = min(int(w), int(x + bw + pad))
    y1 = min(int(h), int(y + bh + pad))
    if x0 >= x1 or y0 >= y1:
        return 0.0
    if x0 == int(x) and y0 == int(y) and x1 == int(x + bw) and y1 == int(y + bh):
        return 0.0

    outer = rgb[int(y0) : int(y1), int(x0) : int(x1)].astype(np.float32)
    inner = rgb[int(y) : int(y + bh), int(x) : int(x + bw)].astype(np.float32)
    if outer.size <= 0 or inner.size <= 0:
        return 0.0
    outer_l = (0.299 * outer[:, :, 0] + 0.587 * outer[:, :, 1] + 0.114 * outer[:, :, 2]).astype(np.float32)
    inner_l = (0.299 * inner[:, :, 0] + 0.587 * inner[:, :, 1] + 0.114 * inner[:, :, 2]).astype(np.float32)
    outer_sum = float(np.sum(outer_l))
    inner_sum = float(np.sum(inner_l))
    ring_n = int(outer_l.size - inner_l.size)
    if ring_n <= 0:
        return 0.0
    ring_mean = float((outer_sum - inner_sum) / float(max(ring_n, 1)))
    inner_mean = float(inner_sum / float(max(int(inner_l.size), 1)))
    return float(_clamp01(abs(inner_mean - ring_mean) / 255.0))


def _circularity_score(edge_patch: np.ndarray) -> float:
    cv2 = _maybe_cv2()
    if cv2 is None:
        return 0.0
    if not isinstance(edge_patch, np.ndarray) or edge_patch.ndim != 2 or edge_patch.size < 25:
        return 0.0
    p = np.clip(edge_patch.astype(np.float32), 0.0, 1.0)
    h, w = p.shape
    m = int(max(h, w))
    if m > 96:
        scale = 96.0 / float(max(1, m))
        nh = int(max(1, round(float(h) * scale)))
        nw = int(max(1, round(float(w) * scale)))
        p = cv2.resize(p, (int(nw), int(nh)), interpolation=cv2.INTER_AREA)
    bw = (p > 0.35).astype(np.uint8) * 255
    contours_result = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    if not contours:
        return 0.0
    cnt = max(contours, key=lambda c: float(cv2.contourArea(c)))
    area = float(cv2.contourArea(cnt))
    per = float(cv2.arcLength(cnt, True))
    if area <= 1e-6 or per <= 1e-6:
        return 0.0
    circ = float(4.0 * math.pi * area / (per * per))
    return float(_clamp01(circ))


def _instruction_groups(instruction: str) -> List[str]:
    t = _normalize_text(str(instruction or "").lower())
    if not t:
        return ["generic"]
    groups: List[str] = []

    def _has(*phrases: str) -> bool:
        return any(str(p) in t for p in phrases if p)

    if _has("search", "search bar", "find", "lookup"):
        groups.append("search")
    if _has("setting", "settings", "options", "preference", "config"):
        groups.append("settings")
    if _has("menu", "hamburger", "navigation drawer"):
        groups.append("menu")
    if _has("close", "dismiss", "exit"):
        groups.append("close")
    if _has("back", "previous", "return"):
        groups.append("back")
    if _has("refresh", "reload"):
        groups.append("refresh")
    if _has("share", "send", "forward", "export"):
        groups.append("share")
    if _has("copy", "clipboard"):
        groups.append("copy")
    if _has("paste"):
        groups.append("paste")
    if _has("filter"):
        groups.append("filter")
    if _has("sort", "order"):
        groups.append("sort")
    if _has("location", "locate", "pin", "map", "gps"):
        groups.append("location")
    if _has("history", "recent", "recents"):
        groups.append("history")
    if _has("home"):
        groups.append("home")
    if _has("profile", "account", "avatar"):
        groups.append("profile")
    if _has("notification", "notifications", "inbox", "alerts"):
        groups.append("notifications")
    if _has("favorite", "favourite", "star", "like", "heart", "bookmark", "saved"):
        groups.append("favorite")
    if _has("more", "overflow", "ellipsis", "three dots", "three dot"):
        groups.append("more")
    if _has("download", "install") or ("save" in t and "saved" not in t):
        groups.append("download")
    if _has("upload", "import"):
        groups.append("upload")
    if _has("sign in", "log in", "login"):
        groups.append("sign_in")
    if _has("cancel"):
        groups.append("cancel")
    if _has("done", "confirm", "submit", "ok", "okay"):
        groups.append("done")
    if _has("add", "new", "create", "plus"):
        groups.append("add")
    if _has("next", "continue"):
        groups.append("next")

    return groups if groups else ["generic"]


def _spatial_prior(*, groups: List[str], cx: float, cy: float) -> float:
    if not groups:
        return 0.0
    best = 0.0
    for g in groups:
        if g == "close":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.95, my=0.06, sigma=0.18))
        elif g == "back":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.05, my=0.06, sigma=0.20))
        elif g == "refresh":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.92, my=0.06, sigma=0.22))
        elif g in ("settings", "menu"):
            best = max(
                best,
                _gaussian_score(x=cx, y=cy, mx=0.08, my=0.08, sigma=0.22),
                _gaussian_score(x=cx, y=cy, mx=0.92, my=0.08, sigma=0.22),
            )
        elif g == "search":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.50, my=0.14, sigma=0.30))
        elif g in ("share", "copy", "paste", "filter", "sort", "location", "history", "notifications", "favorite", "more"):
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.92, my=0.08, sigma=0.24))
        elif g == "home":
            best = max(
                best,
                _gaussian_score(x=cx, y=cy, mx=0.10, my=0.92, sigma=0.25),
                _gaussian_score(x=cx, y=cy, mx=0.50, my=0.92, sigma=0.25),
            )
        elif g == "profile":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.90, my=0.92, sigma=0.25))
        elif g == "cancel":
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.28, my=0.88, sigma=0.28))
        elif g in ("done", "next"):
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.72, my=0.88, sigma=0.28))
        else:
            best = max(best, _gaussian_score(x=cx, y=cy, mx=0.50, my=0.52, sigma=0.42))
    return float(_clamp01(best))


def predict_click(
    *,
    image_bytes: bytes,
    instruction: str,
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    return_candidates: bool = False,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else DEFAULT_CONFIG

    rgb = ueyes_eval._decode_image_bytes(image_bytes)
    rgb = ueyes_eval._normalize_image_rgb(rgb, canonical_width=1440)
    h, w = rgb.shape[:2]
    img_area = float(max(1, h * w))

    field = compute_perceptual_field(rgb, center_bias_strength=0.0, include_text_density=True)
    sal = field.get("saliency")
    if not isinstance(sal, np.ndarray) or sal.shape[:2] != (h, w):
        sal = np.zeros((int(h), int(w)), dtype=np.float32)
    sal = np.clip(sal.astype(np.float32), 0.0, 1.0)

    edges = field.get("edges")
    if not isinstance(edges, np.ndarray) or edges.shape[:2] != (h, w):
        edges = np.zeros((int(h), int(w)), dtype=np.float32)
    edges = np.clip(edges.astype(np.float32), 0.0, 1.0)

    text_density = field.get("text_density")
    if not isinstance(text_density, np.ndarray) or text_density.shape[:2] != (h, w):
        text_density = np.zeros((int(h), int(w)), dtype=np.float32)
    text_density = np.clip(text_density.astype(np.float32), 0.0, 1.0)

    ocr_words, ocr_meta = _get_ocr_words(image_bytes=image_bytes, rgb=rgb, cfg=cfg, cache_dir=cache_dir)
    instr_tokens = _tokens(instruction)
    stop = {
        "a",
        "an",
        "and",
        "at",
        "click",
        "in",
        "on",
        "open",
        "press",
        "select",
        "tap",
        "the",
        "to",
        "with",
    }
    instr_kw = [t for t in instr_tokens if t and t not in stop]
    if not instr_kw:
        instr_kw = instr_tokens
    instr_kw_set = set(instr_kw)
    instr_trigrams = _char_trigrams(instruction)

    groups = _instruction_groups(instruction)
    spatial_hints = _extract_spatial_hints(instruction)
    targets = _extract_target_phrases(instruction)
    icon_intent = (not targets) and any(
        g
        in (
            "close",
            "back",
            "settings",
            "menu",
            "refresh",
            "share",
            "copy",
            "paste",
            "filter",
            "sort",
            "location",
            "history",
            "home",
            "profile",
            "notifications",
            "favorite",
            "more",
        )
        for g in groups
    )
    search_intent = ("search" in groups) or ("search" in _normalize_text(instruction))

    tnorm = _normalize_text(instruction)
    icon_mode_strength = 1.0 if icon_intent else 0.0
    if not icon_intent:
        strength = 0.0
        icon_groups = {
            "close",
            "back",
            "settings",
            "menu",
            "refresh",
            "share",
            "copy",
            "paste",
            "filter",
            "sort",
            "location",
            "history",
            "home",
            "profile",
            "notifications",
            "favorite",
            "more",
        }
        if any(g in icon_groups for g in groups):
            strength = max(strength, 0.45)

        has_icon_word = any(p in tnorm for p in (" icon ", " symbol ", " logo ", " avatar "))
        if has_icon_word:
            strength = max(strength, 0.70)

        if any(
            p in tnorm
            for p in (
                "gear",
                "hamburger",
                "ellipsis",
                "three dots",
                "three dot",
                "chevron",
                "caret",
                "arrow",
                "plus",
            )
        ) or re.search(r"\bx\b", tnorm):
            strength = max(strength, 0.60)

        if targets:
            norm_targets = [_normalize_text(t) for t in targets if t]
            if any(len(nt) <= 2 for nt in norm_targets):
                strength = max(strength, 0.70)
            if (not has_icon_word) and any(len(_tokens(nt)) >= 2 or len(nt) >= 4 for nt in norm_targets):
                strength = min(strength, 0.25)

        if (not has_icon_word) and any(p in tnorm for p in ("with text", "that says", "labeled", "labelled", "text", "type")):
            strength = min(strength, 0.20)

        icon_mode_strength = float(_clamp01(strength))

    prop_cfg = cfg.get("proposals") if isinstance(cfg.get("proposals"), dict) else {}
    max_regions = int(prop_cfg.get("max_regions", 128) or 128)
    merge_thr = float(prop_cfg.get("merge_iou_threshold", 0.40) or 0.40)
    min_area_ratio = float(prop_cfg.get("min_area_ratio", 0.0006) or 0.0006)
    max_area_ratio = float(prop_cfg.get("max_area_ratio", 0.65) or 0.65)
    min_side = int(prop_cfg.get("min_side_px", 10) or 10)

    def _select_spread_by_area(
        src: List[Tuple[int, int, int, int]],
        *,
        k: int,
    ) -> List[Tuple[int, int, int, int]]:
        if not src:
            return []
        k2 = int(max(0, k))
        if k2 <= 0 or len(src) <= k2:
            return src
        ordered = sorted(src, key=lambda b: int(max(0, b[2]) * max(0, b[3])))
        if k2 == 1:
            return [ordered[len(ordered) // 2]]
        out: List[Tuple[int, int, int, int]] = []
        n = int(len(ordered))
        for i in range(int(k2)):
            idx = int(round(float(i) * float(max(n - 1, 1)) / float(max(k2 - 1, 1))))
            idx = max(0, min(idx, n - 1))
            out.append(ordered[idx])
        seen: Set[Tuple[int, int, int, int]] = set()
        out2: List[Tuple[int, int, int, int]] = []
        for b in out:
            if b in seen:
                continue
            seen.add(b)
            out2.append(b)
        return out2

    boxes: List[Tuple[int, int, int, int]] = []
    proposal_stats: Dict[str, int] = {}
    try:
        ui = parse_ui(rgb, max_elements=max(64, min(192, max_regions)))
        for el in ui:
            bbox = el.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, bw, bh = [int(v) for v in bbox]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append((x, y, bw, bh))
        proposal_stats["ui"] = int(len(boxes))
    except Exception:
        boxes = []
        proposal_stats["ui"] = 0

    vlm_meta: Dict[str, Any] = {"enabled": False}
    vlm_boxes: List[Tuple[int, int, int, int]] = []
    vlm_cfg = cfg.get("vlm") if isinstance(cfg.get("vlm"), dict) else {}
    soft_icon_pre = float(_clamp01(icon_mode_strength))

    def _maybe_add_vlm(*, reason: str) -> None:
        nonlocal boxes, vlm_boxes, vlm_meta, proposal_stats
        if not isinstance(vlm_cfg, dict) or not bool(vlm_cfg.get("enabled", False)):
            proposal_stats["vlm_raw"] = 0
            proposal_stats["vlm_accepted"] = 0
            vlm_meta = {"enabled": False, "used": False, "reason": str(reason)}
            return

        raw_vlm, vm = _get_vlm_candidates(
            image_bytes=image_bytes,
            rgb=rgb,
            instruction=instruction,
            cfg=cfg,
            cache_dir=cache_dir,
        )

        verify_cfg = vlm_cfg.get("verify") if isinstance(vlm_cfg.get("verify"), dict) else {}
        min_sal_mean = float(verify_cfg.get("min_saliency_mean", 0.10) or 0.10)
        min_sal_max = float(verify_cfg.get("min_saliency_max", 0.28) or 0.28)
        min_edge_mean = float(verify_cfg.get("min_edge_mean", 0.08) or 0.08)
        min_contrast = float(verify_cfg.get("min_contrast", 0.10) or 0.10)

        verified: List[Tuple[int, int, int, int]] = []
        verify_feats: List[Dict[str, Any]] = []
        for rec in raw_vlm:
            bb = rec.get("bbox_xywh") if isinstance(rec, dict) else None
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            try:
                x0, y0, bw0, bh0 = [int(v) for v in bb]
            except Exception:
                continue
            if bw0 <= 0 or bh0 <= 0:
                continue
            if bw0 < min_side or bh0 < min_side:
                continue
            area_ratio = float((bw0 * bh0) / img_area)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            ok, feats = _verify_vlm_candidate(
                bbox_xywh=(int(x0), int(y0), int(bw0), int(bh0)),
                sal=sal,
                edges=edges,
                rgb=rgb,
                min_saliency_mean=float(min_sal_mean),
                min_saliency_max=float(min_sal_max),
                min_edge_mean=float(min_edge_mean),
                min_contrast=float(min_contrast),
            )
            if ok:
                verified.append((int(x0), int(y0), int(bw0), int(bh0)))
                d: Dict[str, Any] = {"bbox_xywh": [int(x0), int(y0), int(bw0), int(bh0)]}
                if isinstance(rec, dict) and rec.get("label") is not None:
                    d["label"] = str(rec.get("label") or "")[:64]
                if isinstance(rec, dict) and rec.get("confidence") is not None:
                    try:
                        d["confidence"] = float(_clamp01(float(rec.get("confidence", 0.0) or 0.0)))
                    except Exception:
                        pass
                d["verify"] = feats
                verify_feats.append(d)

        vlm_boxes = verified
        proposal_stats["vlm_raw"] = int(len(raw_vlm))
        proposal_stats["vlm_accepted"] = int(len(vlm_boxes))
        if vlm_boxes:
            boxes.extend(vlm_boxes)

        try:
            vm2 = dict(vm)
        except Exception:
            vm2 = {"enabled": True}
        vm2["used"] = True
        vm2["reason"] = str(reason)
        vm2["icon_mode_strength"] = float(_clamp01(soft_icon_pre))
        vm2["raw"] = int(len(raw_vlm))
        vm2["accepted"] = int(len(vlm_boxes))
        vm2["accepted_candidates"] = verify_feats[: min(8, len(verify_feats))]
        vlm_meta = vm2

    if bool(vlm_cfg.get("enabled", False)):
        _min_icon_raw = vlm_cfg.get("min_icon_mode_strength", 0.0)
        if _min_icon_raw is None:
            _min_icon_raw = 0.0
        if isinstance(_min_icon_raw, str) and not _min_icon_raw.strip():
            _min_icon_raw = 0.0
        try:
            min_icon = float(_min_icon_raw)
        except Exception:
            min_icon = 0.0
        if float(soft_icon_pre) >= float(min_icon):
            _maybe_add_vlm(reason="icon_mode_strength")
        else:
            proposal_stats["vlm_raw"] = 0
            proposal_stats["vlm_accepted"] = 0
            vlm_meta = {
                "enabled": True,
                "used": False,
                "reason": "icon_mode_strength",
                "icon_mode_strength": float(_clamp01(soft_icon_pre)),
                "min_icon_mode_strength": float(min_icon),
            }
    else:
        proposal_stats["vlm_raw"] = 0
        proposal_stats["vlm_accepted"] = 0

    cv2 = _maybe_cv2()
    if cv2 is not None:
        s_u8 = np.clip(sal * 255.0, 0.0, 255.0).astype(np.uint8)
        e_u8 = np.clip(edges * 255.0, 0.0, 255.0).astype(np.uint8)

        sal_cc = prop_cfg.get("saliency_cc") if isinstance(prop_cfg.get("saliency_cc"), dict) else {}
        if bool(sal_cc.get("enabled", True)):
            if bool(sal_cc.get("use_otsu", False)):
                _, sm = cv2.threshold(s_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                thr = float(sal_cc.get("fixed_threshold", 0.62) or 0.62)
                _, sm = cv2.threshold(s_u8, int(round(thr * 255.0)), 255, cv2.THRESH_BINARY)
            iters = int(sal_cc.get("dilate_iters", 1) or 0)
            if iters > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                sm = cv2.dilate(sm, k, iterations=int(iters))
            n, _labels, stats, _ = cv2.connectedComponentsWithStats(sm, connectivity=8)
            sal_cc_boxes: List[Tuple[int, int, int, int]] = []
            for i in range(1, int(n)):
                x, y, bw2, bh2, _area = [int(v) for v in stats[i].tolist()]
                if bw2 < min_side or bh2 < min_side:
                    continue
                area_ratio = float((bw2 * bh2) / img_area)
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                sal_cc_boxes.append((int(x), int(y), int(bw2), int(bh2)))
            max_cc = int(sal_cc.get("max_cc", 96) or 96)
            sal_cc_boxes = _select_spread_by_area(sal_cc_boxes, k=int(max(1, max_cc)))
            proposal_stats["saliency_cc"] = int(len(sal_cc_boxes))
            boxes.extend(sal_cc_boxes)

        edge_cc = prop_cfg.get("edge_cc") if isinstance(prop_cfg.get("edge_cc"), dict) else {}
        if bool(edge_cc.get("enabled", True)):
            if bool(edge_cc.get("use_otsu", True)):
                _, bw = cv2.threshold(e_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                thr = float(edge_cc.get("fixed_threshold", 0.35) or 0.35)
                _, bw = cv2.threshold(e_u8, int(round(thr * 255.0)), 255, cv2.THRESH_BINARY)
            iters = int(edge_cc.get("dilate_iters", 1) or 0)
            if iters > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                bw = cv2.dilate(bw, k, iterations=int(iters))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            edge_boxes: List[Tuple[int, int, int, int]] = []
            for i in range(1, int(n)):
                x, y, bw2, bh2, _area = [int(v) for v in stats[i].tolist()]
                if bw2 < min_side or bh2 < min_side:
                    continue
                area_ratio = float((bw2 * bh2) / img_area)
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                edge_boxes.append((int(x), int(y), int(bw2), int(bh2)))
            edge_boxes = _select_spread_by_area(edge_boxes, k=int(max(1, max_regions)))
            proposal_stats["edge_cc"] = int(len(edge_boxes))
            boxes.extend(edge_boxes)

        text_cfg = prop_cfg.get("text_regions") if isinstance(prop_cfg.get("text_regions"), dict) else {}
        if bool(text_cfg.get("enabled", True)):
            thr = float(text_cfg.get("threshold", 0.55) or 0.55)
            tmask = (text_density >= float(thr)).astype(np.uint8) * 255
            ksz = text_cfg.get("close_kernel")
            if isinstance(ksz, (list, tuple)) and len(ksz) == 2:
                kx, ky = int(ksz[0]), int(ksz[1])
                kx = max(1, kx)
                ky = max(1, ky)
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kx), int(ky)))
                iters = int(text_cfg.get("close_iters", 1) or 1)
                tmask = cv2.morphologyEx(tmask, cv2.MORPH_CLOSE, k, iterations=int(max(1, iters)))
            contours_result = cv2.findContours(tmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
            text_boxes: List[Tuple[int, int, int, int]] = []
            for cnt in contours:
                x, y, bw2, bh2 = cv2.boundingRect(cnt)
                if bw2 < min_side or bh2 < min_side:
                    continue
                area_ratio = float((bw2 * bh2) / img_area)
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    continue
                text_boxes.append((int(x), int(y), int(bw2), int(bh2)))
            text_boxes = _select_spread_by_area(text_boxes, k=int(max(1, max_regions)))
            proposal_stats["text_regions"] = int(len(text_boxes))
            boxes.extend(text_boxes)

        if ocr_words:
            ocr_words_cfg = prop_cfg.get("ocr_words") if isinstance(prop_cfg.get("ocr_words"), dict) else {}
            if bool(ocr_words_cfg.get("enabled", True)):
                max_words = int(ocr_words_cfg.get("max_words", 72) or 72)
                min_conf = float(ocr_words_cfg.get("min_conf", 0.35) or 0.35)
                pad_px = int(ocr_words_cfg.get("pad_px", 2) or 0)
                word_boxes: List[Tuple[int, int, int, int]] = []
                for wr in ocr_words:
                    bb = wr.get("bbox_xywh")
                    if not isinstance(bb, list) or len(bb) != 4:
                        continue
                    try:
                        x, y, bw2, bh2 = [int(v) for v in bb]
                    except Exception:
                        continue
                    if bw2 <= 0 or bh2 <= 0:
                        continue
                    try:
                        conf0 = float(wr.get("conf", 0.0) or 0.0)
                    except Exception:
                        conf0 = 0.0
                    if float(conf0) < float(min_conf):
                        continue
                    x0 = max(0, int(x) - int(pad_px))
                    y0 = max(0, int(y) - int(pad_px))
                    x1 = min(int(w), int(x + bw2 + pad_px))
                    y1 = min(int(h), int(y + bh2 + pad_px))
                    bw3 = int(x1 - x0)
                    bh3 = int(y1 - y0)
                    if bw3 < min_side or bh3 < min_side:
                        continue
                    area_ratio = float((bw3 * bh3) / img_area)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    word_boxes.append((int(x0), int(y0), int(bw3), int(bh3)))
                word_boxes.sort(key=lambda b: int(b[2] * b[3]))
                word_boxes = word_boxes[: int(max(0, max_words))]
                proposal_stats["ocr_words"] = int(len(word_boxes))
                boxes.extend(word_boxes)

            ocr_lines_cfg = prop_cfg.get("ocr_lines") if isinstance(prop_cfg.get("ocr_lines"), dict) else {}
            ocr_buttons_cfg = prop_cfg.get("ocr_buttons") if isinstance(prop_cfg.get("ocr_buttons"), dict) else {}
            line_boxes: List[Tuple[int, int, int, int]] = []
            if bool(ocr_lines_cfg.get("enabled", True)):
                max_lines = int(ocr_lines_cfg.get("max_lines", 48) or 48)
                min_conf = float(ocr_lines_cfg.get("min_conf", 0.35) or 0.35)
                pad_px = int(ocr_lines_cfg.get("pad_px", 2) or 0)
                close_kernel = ocr_lines_cfg.get("close_kernel")
                close_iters = int(ocr_lines_cfg.get("close_iters", 1) or 1)
                mask = np.zeros((int(h), int(w)), dtype=np.uint8)
                for wr in ocr_words:
                    bb = wr.get("bbox_xywh")
                    if not isinstance(bb, list) or len(bb) != 4:
                        continue
                    try:
                        x, y, bw2, bh2 = [int(v) for v in bb]
                    except Exception:
                        continue
                    if bw2 <= 0 or bh2 <= 0:
                        continue
                    try:
                        conf0 = float(wr.get("conf", 0.0) or 0.0)
                    except Exception:
                        conf0 = 0.0
                    if float(conf0) < float(min_conf):
                        continue
                    x0 = max(0, int(x) - int(pad_px))
                    y0 = max(0, int(y) - int(pad_px))
                    x1 = min(int(w), int(x + bw2 + pad_px))
                    y1 = min(int(h), int(y + bh2 + pad_px))
                    if x0 >= x1 or y0 >= y1:
                        continue
                    mask[int(y0) : int(y1), int(x0) : int(x1)] = 255
                if int(np.sum(mask)) > 0:
                    if isinstance(close_kernel, (list, tuple)) and len(close_kernel) == 2:
                        kx, ky = int(close_kernel[0]), int(close_kernel[1])
                        kx = max(1, kx)
                        ky = max(1, ky)
                        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kx), int(ky)))
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(max(1, close_iters)))
                    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
                    for cnt in contours:
                        x, y, bw2, bh2 = cv2.boundingRect(cnt)
                        if bw2 < min_side or bh2 < min_side:
                            continue
                        area_ratio = float((bw2 * bh2) / img_area)
                        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                            continue
                        line_boxes.append((int(x), int(y), int(bw2), int(bh2)))
                    line_boxes = _select_spread_by_area(line_boxes, k=int(max(1, max_lines)))
            proposal_stats["ocr_lines"] = int(len(line_boxes))
            boxes.extend(line_boxes)

            if bool(ocr_buttons_cfg.get("enabled", True)) and line_boxes:
                max_buttons = int(ocr_buttons_cfg.get("max_buttons", 48) or 48)
                pad_x_mult = float(ocr_buttons_cfg.get("pad_x_mult", 0.55) or 0.55)
                pad_y_mult = float(ocr_buttons_cfg.get("pad_y_mult", 0.75) or 0.75)
                min_pad_px = int(ocr_buttons_cfg.get("min_pad_px", 2) or 0)
                max_pad_px = int(ocr_buttons_cfg.get("max_pad_px", 24) or 24)
                button_boxes: List[Tuple[int, int, int, int]] = []
                for x, y, bw2, bh2 in line_boxes:
                    base = float(max(1, bh2))
                    px = int(round(float(pad_x_mult) * base))
                    py = int(round(float(pad_y_mult) * base))
                    px = int(max(int(min_pad_px), min(int(px), int(max_pad_px))))
                    py = int(max(int(min_pad_px), min(int(py), int(max_pad_px))))
                    x0 = max(0, int(x) - int(px))
                    y0 = max(0, int(y) - int(py))
                    x1 = min(int(w), int(x + bw2 + px))
                    y1 = min(int(h), int(y + bh2 + py))
                    bw3 = int(x1 - x0)
                    bh3 = int(y1 - y0)
                    if bw3 < min_side or bh3 < min_side:
                        continue
                    area_ratio = float((bw3 * bh3) / img_area)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    button_boxes.append((int(x0), int(y0), int(bw3), int(bh3)))
                button_boxes = _select_spread_by_area(button_boxes, k=int(max(1, max_buttons)))
                proposal_stats["ocr_buttons"] = int(len(button_boxes))
                boxes.extend(button_boxes)
            else:
                proposal_stats["ocr_buttons"] = 0

        anchors_cfg = prop_cfg.get("icon_anchors") if isinstance(prop_cfg.get("icon_anchors"), dict) else {}
        if float(icon_mode_strength) >= 0.55 and bool(anchors_cfg.get("enabled", True)):
            sizes = anchors_cfg.get("sizes")
            if not isinstance(sizes, list) or not sizes:
                sizes = [0.032, 0.048, 0.070]

            centers: List[Tuple[float, float]] = []
            if "close" in groups:
                centers.append((0.95, 0.06))
            if "back" in groups:
                centers.append((0.05, 0.06))
            if "refresh" in groups:
                centers.append((0.92, 0.06))
            if any(g in ("settings", "menu") for g in groups):
                centers.extend([(0.08, 0.08), (0.92, 0.08)])
            if any(
                g in ("share", "copy", "paste", "filter", "sort", "location", "history", "notifications", "favorite", "more")
                for g in groups
            ):
                centers.append((0.92, 0.08))
            if "home" in groups:
                centers.extend([(0.10, 0.92), (0.50, 0.92)])
            if "profile" in groups:
                centers.append((0.90, 0.92))
            if not centers:
                centers.extend([(0.05, 0.06), (0.95, 0.06), (0.08, 0.08), (0.92, 0.08)])

            m = int(max(1, min(int(w), int(h))))
            anchor_boxes: List[Tuple[int, int, int, int]] = []
            for cx0, cy0 in centers:
                for s in sizes:
                    try:
                        sf = float(s)
                    except Exception:
                        continue
                    side = int(round(sf * float(m)))
                    side = int(max(int(min_side), side))
                    x0 = int(round(float(cx0) * float(w) - float(side) / 2.0))
                    y0 = int(round(float(cy0) * float(h) - float(side) / 2.0))
                    x0 = max(0, min(int(x0), int(w - 1)))
                    y0 = max(0, min(int(y0), int(h - 1)))
                    bw2 = int(min(int(side), int(w - x0)))
                    bh2 = int(min(int(side), int(h - y0)))
                    if bw2 < min_side or bh2 < min_side:
                        continue
                    area_ratio = float((bw2 * bh2) / img_area)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    anchor_boxes.append((int(x0), int(y0), int(bw2), int(bh2)))
            proposal_stats["icon_anchors"] = int(len(anchor_boxes))
            boxes.extend(anchor_boxes)
        else:
            proposal_stats["icon_anchors"] = 0

    # Add tiny target proposals for edge icons (toolbar, sidebar)
    # This addresses the 84% missing target issue in ScreenSpot-Pro
    try:
        from Backend.indexing.tiny_target_proposals import generate_tiny_target_proposals
        
        tiny_props = generate_tiny_target_proposals(
            rgb,
            instruction=instruction,
            include_edge_icons=True,
            include_high_contrast=bool(icon_intent or soft_icon_pre > 0.3),
            include_corners=bool(icon_intent or soft_icon_pre > 0.3),
            max_total_proposals=200,
        )
        
        # Convert to (x, y, w, h) format and filter by size
        tiny_boxes = []
        for prop in tiny_props:
            bbox = prop.get("bbox_xywh") or prop.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            
            if len(bbox) == 4 and bbox[2] > bbox[0]:
                # Already (x, y, x2, y2) format, convert to (x, y, w, h)
                tx, ty = int(bbox[0]), int(bbox[1])
                tw = int(bbox[2] - bbox[0]) if bbox[2] > 1 else int(bbox[2])
                th = int(bbox[3] - bbox[1]) if bbox[3] > 1 else int(bbox[3])
            else:
                tx, ty, tw, th = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Allow smaller min_side for tiny icons
            if tw >= 8 and th >= 8 and tw <= 80 and th <= 80:
                tiny_boxes.append((tx, ty, tw, th))
        
        proposal_stats["tiny_proposals"] = int(len(tiny_boxes))
        boxes.extend(tiny_boxes)
    except ImportError:
        proposal_stats["tiny_proposals"] = 0
    except Exception:
        proposal_stats["tiny_proposals"] = 0

    boxes = _merge_boxes_xywh(boxes, iou_threshold=float(merge_thr))
    if len(boxes) > int(max_regions):
        boxes = _select_spread_by_area(list(boxes), k=int(max_regions))

    weights = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
    scoring = cfg.get("scoring") if isinstance(cfg.get("scoring"), dict) else {}
    area_target = float(scoring.get("area_target", 0.02) or 0.02)
    area_width = float(scoring.get("area_width", 0.08) or 0.08)
    top_k = int(scoring.get("top_k", 20) or 20)
    top_k = int(max(1, min(int(top_k), 64)))
    candidates: List[Dict[str, Any]] = []

    util_cfg = cfg.get("utility") if isinstance(cfg.get("utility"), dict) else {}
    export_cfg = cfg.get("export") if isinstance(cfg.get("export"), dict) else {}
    return_all = bool(return_candidates) or bool(export_cfg.get("return_candidates", False))
    ranker_model = _load_ranker(cfg)

    soft_icon = float(_clamp01(icon_mode_strength))
    text_scale = float(max(0.20, 1.0 - 0.80 * soft_icon))
    shape_scale = float(1.0 + 0.60 * soft_icon)
    spatial_scale = float(1.0 + 0.50 * soft_icon)
    icon_pen_scale = float(1.0 + 1.50 * soft_icon)
    center_pen_scale = float(max(0.0, 1.0 - 1.00 * soft_icon))

    for (x, y, bw, bh) in boxes:
        x = max(0, min(int(x), int(w - 1)))
        y = max(0, min(int(y), int(h - 1)))
        bw = max(1, min(int(bw), int(w - x)))
        bh = max(1, min(int(bh), int(h - y)))
        if bw < min_side or bh < min_side:
            continue

        patch = sal[y : y + bh, x : x + bw]
        if patch.size <= 0:
            continue
        sal_mean = float(np.mean(patch))
        sal_max = float(np.max(patch))
        sal_p = float(np.quantile(patch, 0.90)) if patch.size >= 4 else sal_max

        edge_patch = edges[y : y + bh, x : x + bw]
        edge_mean = float(np.mean(edge_patch)) if edge_patch.size else 0.0
        text_patch = text_density[y : y + bh, x : x + bw]
        text_mean = float(np.mean(text_patch)) if text_patch.size else 0.0

        region_words: List[str] = []
        region_confs: List[float] = []
        if ocr_words:
            rx0 = int(x)
            ry0 = int(y)
            rx1 = int(x + bw)
            ry1 = int(y + bh)
            for wr in ocr_words:
                bb = wr.get("bbox_xywh")
                if not isinstance(bb, list) or len(bb) != 4:
                    continue
                try:
                    wx, wy, ww, wh = [int(v) for v in bb]
                except Exception:
                    continue
                if ww <= 0 or wh <= 0:
                    continue
                wcx = float(wx) + float(ww) / 2.0
                wcy = float(wy) + float(wh) / 2.0
                if not (float(rx0) <= wcx < float(rx1) and float(ry0) <= wcy < float(ry1)):
                    continue
                txt = str(wr.get("text") or "")
                if txt:
                    region_words.append(txt)
                    try:
                        region_confs.append(float(wr.get("conf", 0.0) or 0.0))
                    except Exception:
                        region_confs.append(0.0)

        region_text = " ".join(region_words).strip()
        region_tokens = [t for t in region_text.split(" ") if t]
        region_token_set = set(region_tokens)
        ocr_conf = float(np.mean(np.clip(np.array(region_confs, dtype=np.float32), 0.0, 1.0))) if region_confs else 0.0

        token_overlap = 0.0
        keyword_presence = 0.0
        if instr_kw_set and region_token_set:
            inter = instr_kw_set.intersection(region_token_set)
            token_overlap = float(len(inter)) / float(max(len(instr_kw_set), 1))
            keyword_presence = 1.0 if inter else 0.0

        lev = float(_levenshtein_similarity(instr_kw, region_tokens[:24])) if instr_kw and region_tokens else 0.0
        tri = float(_jaccard(instr_trigrams, _char_trigrams(region_text))) if region_text and instr_trigrams else 0.0

        conf_w = float(0.65 + 0.35 * _clamp01(ocr_conf)) if region_confs else 1.0
        token_overlap = float(_clamp01(token_overlap) * conf_w)
        keyword_presence = float(_clamp01(keyword_presence) * conf_w)
        lev = float(_clamp01(lev) * conf_w)
        tri = float(_clamp01(tri) * conf_w)

        text_overlap_score = float(text_mean) if not ocr_words else float(token_overlap)

        cx = (float(x) + float(bw) / 2.0) / float(w)
        cy = (float(y) + float(bh) / 2.0) / float(h)
        area_ratio = float((bw * bh) / img_area)
        aspect = float(bw) / float(max(1, bh))
        spatial_group = float(_spatial_prior(groups=groups, cx=float(cx), cy=float(cy)))
        spatial_hint = float(_spatial_hints_prior(hints=spatial_hints, cx=float(cx), cy=float(cy)))
        spatial = float(max(spatial_group, spatial_hint))

        vlm_support = 0.0
        if vlm_boxes:
            for vb in vlm_boxes:
                vlm_support = max(vlm_support, float(_bbox_iou_xywh((int(x), int(y), int(bw), int(bh)), vb)))

        text_exact = 0.0
        text_phrase = 0.0
        if targets and region_text:
            rnorm = _normalize_text(region_text)
            for tgt in targets:
                if tgt and tgt in rnorm:
                    text_exact = 1.0
                text_phrase = max(text_phrase, float(_phrase_match_score(target=str(tgt), region_tokens=region_tokens)))

        search_prior = float(_search_bar_prior(cx=float(cx), cy=float(cy), area_ratio=float(area_ratio), aspect=float(aspect))) if search_intent else 0.0

        token_ratio = float(_clamp01(float(len(region_tokens)) / 8.0)) if region_tokens else 0.0
        icon_text_pen = float(max(token_ratio, float(_clamp01(text_mean))))
        icon_size_prior = 0.0
        icon_aspect_prior = 0.0
        if icon_intent:
            asp_peak = 1.0
            asp_width = 1.6
            if "back" in groups:
                asp_peak, asp_width = 1.8, 2.3
            elif "menu" in groups:
                asp_peak, asp_width = 1.4, 2.0
            icon_aspect_prior = float(_triangular_score(float(aspect), peak=float(asp_peak), width=float(asp_width)))
            icon_size_prior = float(_triangular_score(float(area_ratio), peak=0.0012, width=0.008))

        circ = 0.0
        sym = 0.0
        col = 0.0
        need_shape = abs(float(weights.get("icon_likelihood", 0.0) or 0.0)) > 1e-9 and float(soft_icon) > 1e-9
        if abs(float(weights.get("circularity", 0.0) or 0.0)) > 1e-9 or need_shape:
            circ = float(_circularity_score(edge_patch))
        if abs(float(weights.get("symmetry", 0.0) or 0.0)) > 1e-9 or need_shape:
            sym = float(_symmetry_score(edge_patch))
        if abs(float(weights.get("color_contrast", 0.0) or 0.0)) > 1e-9 or need_shape:
            col = float(_color_contrast_score(rgb=rgb, x=int(x), y=int(y), bw=int(bw), bh=int(bh)))

        icon_low_text = float(_clamp01(1.0 - float(icon_text_pen)))
        icon_size_score = float(_triangular_score(float(area_ratio), peak=0.0012, width=0.008))
        icon_aspect_score = float(_triangular_score(float(aspect), peak=1.0, width=1.8))
        icon_toolbar = float(_icon_toolbar_prior(cx=float(cx), cy=float(cy)))
        icon_shape = float(_clamp01(0.40 * float(circ) + 0.30 * float(sym) + 0.30 * float(col)))
        icon_likelihood = float(
            _clamp01(
                0.28 * float(icon_low_text)
                + 0.22 * float(icon_size_score)
                + 0.14 * float(icon_aspect_score)
                + 0.16 * float(icon_toolbar)
                + 0.12 * float(_clamp01(edge_mean))
                + 0.08 * float(icon_shape)
            )
        )

        area_ok = float(_triangular_score(area_ratio, peak=float(area_target), width=float(area_width)))
        area_pen = float(1.0 - area_ok)
        center_dist = float(math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
        center_pen = float(_clamp01(center_dist / 0.85))

        score = 0.0
        score += float(weights.get("saliency_mean", 0.0) or 0.0) * float(_clamp01(sal_mean))
        score += float(weights.get("saliency_max", 0.0) or 0.0) * float(_clamp01(sal_max))
        score += float(weights.get("saliency_percentile", 0.0) or 0.0) * float(_clamp01(sal_p))
        score += float(weights.get("edge_density", 0.0) or 0.0) * float(_clamp01(edge_mean)) * float(shape_scale)
        score += float(weights.get("spatial_prior", 0.0) or 0.0) * float(_clamp01(spatial)) * float(spatial_scale)
        score += float(weights.get("text_token_overlap", 0.0) or 0.0) * float(_clamp01(text_overlap_score)) * float(text_scale)
        score += float(weights.get("text_levenshtein", 0.0) or 0.0) * float(_clamp01(lev)) * float(text_scale)
        score += float(weights.get("text_trigram_jaccard", 0.0) or 0.0) * float(_clamp01(tri)) * float(text_scale)
        score += float(weights.get("keyword_presence", 0.0) or 0.0) * float(_clamp01(keyword_presence)) * float(text_scale)
        score += float(weights.get("text_exact_match", 0.0) or 0.0) * float(_clamp01(text_exact)) * float(text_scale)
        score += float(weights.get("text_phrase_match", 0.0) or 0.0) * float(_clamp01(text_phrase)) * float(text_scale)
        score += float(weights.get("search_bar_prior", 0.0) or 0.0) * float(_clamp01(search_prior))
        score += (
            float(weights.get("icon_text_penalty", 0.0) or 0.0)
            * float(_clamp01(icon_text_pen))
            * float(icon_pen_scale)
            * float(soft_icon)
        )
        score += float(weights.get("icon_size_prior", 0.0) or 0.0) * float(_clamp01(icon_size_prior))
        score += float(weights.get("icon_aspect_prior", 0.0) or 0.0) * float(_clamp01(icon_aspect_prior))
        score += (
            float(weights.get("icon_likelihood", 0.0) or 0.0)
            * float(_clamp01(icon_likelihood))
            * float(soft_icon)
        )
        score += float(weights.get("vlm_support", 0.0) or 0.0) * float(_clamp01(vlm_support))
        score += float(weights.get("circularity", 0.0) or 0.0) * float(_clamp01(circ)) * float(shape_scale)
        score += float(weights.get("symmetry", 0.0) or 0.0) * float(_clamp01(sym)) * float(shape_scale)
        score += float(weights.get("color_contrast", 0.0) or 0.0) * float(_clamp01(col)) * float(shape_scale)
        score += float(weights.get("area_penalty", 0.0) or 0.0) * float(_clamp01(area_pen))
        score += float(weights.get("center_penalty", 0.0) or 0.0) * float(_clamp01(center_pen)) * float(center_pen_scale)

        base_features: Dict[str, Any] = {
            "saliency_mean": float(_clamp01(sal_mean)),
            "saliency_max": float(_clamp01(sal_max)),
            "saliency_p90": float(_clamp01(sal_p)),
            "edge_density": float(_clamp01(edge_mean)),
            "text_density": float(_clamp01(text_mean)),
            "spatial_prior": float(_clamp01(spatial)),
            "spatial_group": float(_clamp01(spatial_group)),
            "spatial_hint": float(_clamp01(spatial_hint)),
            "text_overlap": float(_clamp01(text_overlap_score)),
            "text_levenshtein": float(_clamp01(lev)),
            "text_trigram_jaccard": float(_clamp01(tri)),
            "keyword_presence": float(_clamp01(keyword_presence)),
            "text_exact_match": float(_clamp01(text_exact)),
            "text_phrase_match": float(_clamp01(text_phrase)),
            "search_bar_prior": float(_clamp01(search_prior)),
            "icon_text_penalty": float(_clamp01(icon_text_pen)),
            "icon_size_prior": float(_clamp01(icon_size_prior)),
            "icon_aspect_prior": float(_clamp01(icon_aspect_prior)),
            "icon_likelihood": float(_clamp01(icon_likelihood)),
            "icon_low_text": float(_clamp01(icon_low_text)),
            "icon_toolbar_prior": float(_clamp01(icon_toolbar)),
            "icon_size_score": float(_clamp01(icon_size_score)),
            "icon_aspect_score": float(_clamp01(icon_aspect_score)),
            "icon_mode_strength": float(_clamp01(soft_icon)),
            "vlm_support": float(_clamp01(vlm_support)),
            "circularity": float(_clamp01(circ)),
            "symmetry": float(_clamp01(sym)),
            "color_contrast": float(_clamp01(col)),
            "ocr_conf": float(_clamp01(ocr_conf)),
            "area_ratio": float(_clamp01(area_ratio / float(max(max_area_ratio, 1e-9)))),
            "center_pen": float(_clamp01(center_pen)),
        }
        pint = _compute_p_intent(features=base_features, util_cfg=util_cfg, ranker_model=ranker_model)
        for k, v in pint.items():
            base_features[k] = float(v)
        sal_key = str(util_cfg.get("saliency_key") or "saliency_mean")
        try:
            sal_for_util = float(base_features.get(sal_key, base_features.get("saliency_mean", 0.0)) or 0.0)
        except Exception:
            sal_for_util = float(base_features.get("saliency_mean", 0.0) or 0.0)
        utility_score = float(_clamp01(float(_clamp01(sal_for_util)) * float(_clamp01(base_features.get("p_intent", 0.0) or 0.0))))
        base_features["utility"] = float(utility_score)

        cand = {
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
            "bbox": _bbox_xyxy_from_xywh_pixels(bbox_xywh=[int(x), int(y), int(bw), int(bh)], w=int(w), h=int(h)),
            "score": float(score),
            "features": base_features,
            "ocr_text": str(region_text)[:240] if region_text else "",
        }

        candidates.append(cand)

    if not candidates:
        return {
            "x": 0.5,
            "y": 0.5,
            "bbox": None,
            "confidence": 0.0,
            "meta": {"version": CONFIG_VERSION, "candidates": 0, "fallback": "center"},
        }

    candidates.sort(key=lambda c: float(c.get("score", 0.0) or 0.0), reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = int(i + 1)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    best_features = best.get("features") if isinstance(best.get("features"), dict) else {}
    best_icon_like = float(best_features.get("icon_likelihood", 0.0) or 0.0)
    icon_click = bool(icon_intent) or (float(soft_icon) >= 0.65 and float(best_icon_like) >= 0.55)

    bx, by, bw, bh = [int(v) for v in best.get("bbox_xywh")]
    click_method = "saliency_peak"
    px = int(bx + bw // 2)
    py = int(by + bh // 2)

    if icon_click:
        click_method = "bbox_center"
    else:
        query_terms = targets if targets else instr_kw[:8]
        word_bbox: Optional[Tuple[int, int, int, int]] = None
        word_score = 0.0
        if query_terms and ocr_words:
            rx0 = int(bx)
            ry0 = int(by)
            rx1 = int(bx + bw)
            ry1 = int(by + bh)
            for wr in ocr_words:
                bb = wr.get("bbox_xywh")
                if not isinstance(bb, list) or len(bb) != 4:
                    continue
                try:
                    wx, wy, ww, wh = [int(v) for v in bb]
                except Exception:
                    continue
                if ww <= 0 or wh <= 0:
                    continue
                wcx = float(wx) + float(ww) / 2.0
                wcy = float(wy) + float(wh) / 2.0
                if not (float(rx0) <= wcx < float(rx1) and float(ry0) <= wcy < float(ry1)):
                    continue
                wt = _normalize_text(str(wr.get("text") or ""))
                if not wt:
                    continue

                base = 0.0
                for term in query_terms:
                    tnorm = _normalize_text(str(term or ""))
                    if not tnorm:
                        continue
                    if wt == tnorm:
                        base = max(base, 1.0)
                        continue
                    t_tokens = _tokens(tnorm)
                    if wt in set(t_tokens):
                        base = max(base, 0.92)
                    if tnorm in wt or wt in tnorm:
                        base = max(base, 0.85)
                    base = max(base, float(_jaccard(_char_trigrams(wt), _char_trigrams(tnorm))))

                conf_w = float(0.65 + 0.35 * _clamp01(float(wr.get("conf", 0.0) or 0.0)))
                s = float(_clamp01(base) * conf_w)
                if s > word_score:
                    word_score = float(s)
                    word_bbox = (int(wx), int(wy), int(ww), int(wh))

        if word_bbox is not None and float(word_score) >= 0.55:
            wx, wy, ww, wh = word_bbox
            px = int(wx + ww // 2)
            py = int(wy + wh // 2)
            click_method = "ocr_word_center"
        else:
            patch = sal[by : by + bh, bx : bx + bw]
            if patch.size > 0:
                idx = int(np.argmax(patch))
                py0, px0 = np.unravel_index(idx, patch.shape)
                px = int(bx + int(px0))
                py = int(by + int(py0))
                click_method = "saliency_peak"
            else:
                click_method = "bbox_center"

    x01 = float(px) / float(max(w - 1, 1))
    y01 = float(py) / float(max(h - 1, 1))

    top_s = float(best.get("score", 0.0) or 0.0)
    sec_s = float(second.get("score", 0.0) or 0.0) if second is not None else 0.0
    denom = float(max(abs(top_s), 1e-6))
    conf = float(_clamp01((top_s - sec_s) / denom))

    top_features0 = best.get("features") if isinstance(best.get("features"), dict) else {}
    try:
        top_p_intent0 = float(top_features0.get("p_intent", 0.0) or 0.0)
    except Exception:
        top_p_intent0 = 0.0

    fb_cfg = cfg.get("vlm_fallback") if isinstance(cfg.get("vlm_fallback"), dict) else {}
    fb_enabled = bool(fb_cfg.get("enabled", False))
    fb_min_conf = float(fb_cfg.get("min_confidence", 0.18) or 0.18)
    fb_min_p = float(fb_cfg.get("min_p_intent", 0.26) or 0.26)
    fb_min_cands = int(fb_cfg.get("min_candidates", 14) or 14)
    fb_min_icon = float(fb_cfg.get("min_icon_mode_strength", 0.0) or 0.0)
    fb_max_extra = int(fb_cfg.get("max_extra_candidates", 24) or 24)
    u0 = _compute_uncertainty(candidates=candidates, conf_margin=float(conf), p_intent_top=float(top_p_intent0), min_candidates=int(fb_min_cands))

    fb_trigger = False
    fb_reason = ""
    if fb_enabled and bool(vlm_cfg.get("enabled", False)) and not bool(vlm_meta.get("used", False)):
        if float(soft_icon_pre) >= float(fb_min_icon):
            if bool(u0.get("weak_recall", False)):
                fb_trigger = True
                fb_reason = "weak_recall"
            elif float(top_p_intent0) < float(fb_min_p) and float(conf) < float(fb_min_conf):
                fb_trigger = True
                fb_reason = "low_p_intent"

    if fb_trigger:
        _maybe_add_vlm(reason=str(fb_reason))
        if vlm_boxes:
            boxes = _merge_boxes_xywh(boxes, iou_threshold=float(merge_thr))
            if len(boxes) > int(max_regions):
                boxes = _select_spread_by_area(list(boxes), k=int(max_regions))

            candidates = []
            for (x, y, bw, bh) in boxes:
                x = max(0, min(int(x), int(w - 1)))
                y = max(0, min(int(y), int(h - 1)))
                bw = max(1, min(int(bw), int(w - x)))
                bh = max(1, min(int(bh), int(h - y)))
                if bw < min_side or bh < min_side:
                    continue
                patch = sal[y : y + bh, x : x + bw]
                if patch.size <= 0:
                    continue
                sal_mean = float(np.mean(patch))
                sal_max = float(np.max(patch))
                sal_p = float(np.quantile(patch, 0.90)) if patch.size >= 4 else sal_max

                edge_patch = edges[y : y + bh, x : x + bw]
                edge_mean = float(np.mean(edge_patch)) if edge_patch.size else 0.0
                text_patch = text_density[y : y + bh, x : x + bw]
                text_mean = float(np.mean(text_patch)) if text_patch.size else 0.0

                region_words = []
                region_confs = []
                if ocr_words:
                    rx0 = int(x)
                    ry0 = int(y)
                    rx1 = int(x + bw)
                    ry1 = int(y + bh)
                    for wr in ocr_words:
                        bb = wr.get("bbox_xywh")
                        if not isinstance(bb, list) or len(bb) != 4:
                            continue
                        try:
                            wx, wy, ww, wh = [int(v) for v in bb]
                        except Exception:
                            continue
                        if ww <= 0 or wh <= 0:
                            continue
                        wcx = float(wx) + float(ww) / 2.0
                        wcy = float(wy) + float(wh) / 2.0
                        if not (float(rx0) <= wcx < float(rx1) and float(ry0) <= wcy < float(ry1)):
                            continue
                        txt = str(wr.get("text") or "")
                        if txt:
                            region_words.append(txt)
                            try:
                                region_confs.append(float(wr.get("conf", 0.0) or 0.0))
                            except Exception:
                                region_confs.append(0.0)

                region_text = " ".join(region_words).strip()
                region_tokens = [t for t in region_text.split(" ") if t]
                region_token_set = set(region_tokens)
                ocr_conf = (
                    float(np.mean(np.clip(np.array(region_confs, dtype=np.float32), 0.0, 1.0))) if region_confs else 0.0
                )

                token_overlap = 0.0
                keyword_presence = 0.0
                if instr_kw_set and region_token_set:
                    inter = instr_kw_set.intersection(region_token_set)
                    token_overlap = float(len(inter)) / float(max(len(instr_kw_set), 1))
                    keyword_presence = 1.0 if inter else 0.0
                lev = float(_levenshtein_similarity(instr_kw, region_tokens[:24])) if instr_kw and region_tokens else 0.0
                tri = float(_jaccard(instr_trigrams, _char_trigrams(region_text))) if region_text and instr_trigrams else 0.0
                conf_w = float(0.65 + 0.35 * _clamp01(ocr_conf)) if region_confs else 1.0
                token_overlap = float(_clamp01(token_overlap) * conf_w)
                keyword_presence = float(_clamp01(keyword_presence) * conf_w)
                lev = float(_clamp01(lev) * conf_w)
                tri = float(_clamp01(tri) * conf_w)
                text_overlap_score = float(text_mean) if not ocr_words else float(token_overlap)

                cx = (float(x) + float(bw) / 2.0) / float(w)
                cy = (float(y) + float(bh) / 2.0) / float(h)
                area_ratio = float((bw * bh) / img_area)
                aspect = float(bw) / float(max(1, bh))
                spatial_group = float(_spatial_prior(groups=groups, cx=float(cx), cy=float(cy)))
                spatial_hint = float(_spatial_hints_prior(hints=spatial_hints, cx=float(cx), cy=float(cy)))
                spatial = float(max(spatial_group, spatial_hint))

                vlm_support = 0.0
                if vlm_boxes:
                    for vb in vlm_boxes:
                        vlm_support = max(vlm_support, float(_bbox_iou_xywh((int(x), int(y), int(bw), int(bh)), vb)))

                text_exact = 0.0
                text_phrase = 0.0
                if targets and region_text:
                    rnorm = _normalize_text(region_text)
                    for tgt in targets:
                        if tgt and tgt in rnorm:
                            text_exact = 1.0
                        text_phrase = max(text_phrase, float(_phrase_match_score(target=str(tgt), region_tokens=region_tokens)))

                search_prior = (
                    float(_search_bar_prior(cx=float(cx), cy=float(cy), area_ratio=float(area_ratio), aspect=float(aspect)))
                    if search_intent
                    else 0.0
                )

                token_ratio = float(_clamp01(float(len(region_tokens)) / 8.0)) if region_tokens else 0.0
                icon_text_pen = float(max(token_ratio, float(_clamp01(text_mean))))
                icon_size_prior = 0.0
                icon_aspect_prior = 0.0
                if icon_intent:
                    asp_peak = 1.0
                    asp_width = 1.6
                    if "back" in groups:
                        asp_peak, asp_width = 1.8, 2.3
                    elif "menu" in groups:
                        asp_peak, asp_width = 1.4, 2.0
                    icon_aspect_prior = float(_triangular_score(float(aspect), peak=float(asp_peak), width=float(asp_width)))
                    icon_size_prior = float(_triangular_score(float(area_ratio), peak=0.0012, width=0.008))

                circ = 0.0
                sym = 0.0
                col = 0.0
                need_shape = abs(float(weights.get("icon_likelihood", 0.0) or 0.0)) > 1e-9 and float(soft_icon) > 1e-9
                if abs(float(weights.get("circularity", 0.0) or 0.0)) > 1e-9 or need_shape:
                    circ = float(_circularity_score(edge_patch))
                if abs(float(weights.get("symmetry", 0.0) or 0.0)) > 1e-9 or need_shape:
                    sym = float(_symmetry_score(edge_patch))
                if abs(float(weights.get("color_contrast", 0.0) or 0.0)) > 1e-9 or need_shape:
                    col = float(_color_contrast_score(rgb=rgb, x=int(x), y=int(y), bw=int(bw), bh=int(bh)))

                icon_low_text = float(_clamp01(1.0 - float(icon_text_pen)))
                icon_size_score = float(_triangular_score(float(area_ratio), peak=0.0012, width=0.008))
                icon_aspect_score = float(_triangular_score(float(aspect), peak=1.0, width=1.8))
                icon_toolbar = float(_icon_toolbar_prior(cx=float(cx), cy=float(cy)))
                icon_shape = float(_clamp01(0.40 * float(circ) + 0.30 * float(sym) + 0.30 * float(col)))
                icon_likelihood = float(
                    _clamp01(
                        0.28 * float(icon_low_text)
                        + 0.22 * float(icon_size_score)
                        + 0.14 * float(icon_aspect_score)
                        + 0.16 * float(icon_toolbar)
                        + 0.12 * float(_clamp01(edge_mean))
                        + 0.08 * float(icon_shape)
                    )
                )

                area_ok = float(_triangular_score(area_ratio, peak=float(area_target), width=float(area_width)))
                area_pen = float(1.0 - area_ok)
                center_dist = float(math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
                center_pen = float(_clamp01(center_dist / 0.85))

                score = 0.0
                score += float(weights.get("saliency_mean", 0.0) or 0.0) * float(_clamp01(sal_mean))
                score += float(weights.get("saliency_max", 0.0) or 0.0) * float(_clamp01(sal_max))
                score += float(weights.get("saliency_percentile", 0.0) or 0.0) * float(_clamp01(sal_p))
                score += float(weights.get("edge_density", 0.0) or 0.0) * float(_clamp01(edge_mean)) * float(shape_scale)
                score += float(weights.get("spatial_prior", 0.0) or 0.0) * float(_clamp01(spatial)) * float(spatial_scale)
                score += float(weights.get("text_token_overlap", 0.0) or 0.0) * float(_clamp01(text_overlap_score)) * float(text_scale)
                score += float(weights.get("text_levenshtein", 0.0) or 0.0) * float(_clamp01(lev)) * float(text_scale)
                score += float(weights.get("text_trigram_jaccard", 0.0) or 0.0) * float(_clamp01(tri)) * float(text_scale)
                score += float(weights.get("keyword_presence", 0.0) or 0.0) * float(_clamp01(keyword_presence)) * float(text_scale)
                score += float(weights.get("text_exact_match", 0.0) or 0.0) * float(_clamp01(text_exact)) * float(text_scale)
                score += float(weights.get("text_phrase_match", 0.0) or 0.0) * float(_clamp01(text_phrase)) * float(text_scale)
                score += float(weights.get("search_bar_prior", 0.0) or 0.0) * float(_clamp01(search_prior))
                score += (
                    float(weights.get("icon_text_penalty", 0.0) or 0.0)
                    * float(_clamp01(icon_text_pen))
                    * float(icon_pen_scale)
                    * float(soft_icon)
                )
                score += float(weights.get("icon_size_prior", 0.0) or 0.0) * float(_clamp01(icon_size_prior))
                score += float(weights.get("icon_aspect_prior", 0.0) or 0.0) * float(_clamp01(icon_aspect_prior))
                score += float(weights.get("icon_likelihood", 0.0) or 0.0) * float(_clamp01(icon_likelihood)) * float(soft_icon)
                score += float(weights.get("vlm_support", 0.0) or 0.0) * float(_clamp01(vlm_support))
                score += float(weights.get("circularity", 0.0) or 0.0) * float(_clamp01(circ)) * float(shape_scale)
                score += float(weights.get("symmetry", 0.0) or 0.0) * float(_clamp01(sym)) * float(shape_scale)
                score += float(weights.get("color_contrast", 0.0) or 0.0) * float(_clamp01(col)) * float(shape_scale)
                score += float(weights.get("area_penalty", 0.0) or 0.0) * float(_clamp01(area_pen))
                score += float(weights.get("center_penalty", 0.0) or 0.0) * float(_clamp01(center_pen)) * float(center_pen_scale)

                base_features = {
                    "saliency_mean": float(_clamp01(sal_mean)),
                    "saliency_max": float(_clamp01(sal_max)),
                    "saliency_p90": float(_clamp01(sal_p)),
                    "edge_density": float(_clamp01(edge_mean)),
                    "text_density": float(_clamp01(text_mean)),
                    "spatial_prior": float(_clamp01(spatial)),
                    "spatial_group": float(_clamp01(spatial_group)),
                    "spatial_hint": float(_clamp01(spatial_hint)),
                    "text_overlap": float(_clamp01(text_overlap_score)),
                    "text_levenshtein": float(_clamp01(lev)),
                    "text_trigram_jaccard": float(_clamp01(tri)),
                    "keyword_presence": float(_clamp01(keyword_presence)),
                    "text_exact_match": float(_clamp01(text_exact)),
                    "text_phrase_match": float(_clamp01(text_phrase)),
                    "search_bar_prior": float(_clamp01(search_prior)),
                    "icon_text_penalty": float(_clamp01(icon_text_pen)),
                    "icon_size_prior": float(_clamp01(icon_size_prior)),
                    "icon_aspect_prior": float(_clamp01(icon_aspect_prior)),
                    "icon_likelihood": float(_clamp01(icon_likelihood)),
                    "icon_low_text": float(_clamp01(icon_low_text)),
                    "icon_toolbar_prior": float(_clamp01(icon_toolbar)),
                    "icon_size_score": float(_clamp01(icon_size_score)),
                    "icon_aspect_score": float(_clamp01(icon_aspect_score)),
                    "icon_mode_strength": float(_clamp01(soft_icon)),
                    "vlm_support": float(_clamp01(vlm_support)),
                    "circularity": float(_clamp01(circ)),
                    "symmetry": float(_clamp01(sym)),
                    "color_contrast": float(_clamp01(col)),
                    "ocr_conf": float(_clamp01(ocr_conf)),
                    "area_ratio": float(_clamp01(area_ratio / float(max(max_area_ratio, 1e-9)))),
                    "center_pen": float(_clamp01(center_pen)),
                }
                pint = _compute_p_intent(features=base_features, util_cfg=util_cfg, ranker_model=ranker_model)
                for k, v in pint.items():
                    base_features[k] = float(v)
                sal_key = str(util_cfg.get("saliency_key") or "saliency_mean")
                try:
                    sal_for_util = float(base_features.get(sal_key, base_features.get("saliency_mean", 0.0)) or 0.0)
                except Exception:
                    sal_for_util = float(base_features.get("saliency_mean", 0.0) or 0.0)
                utility_score = float(
                    _clamp01(
                        float(_clamp01(sal_for_util)) * float(_clamp01(base_features.get("p_intent", 0.0) or 0.0))
                    )
                )
                base_features["utility"] = float(utility_score)

                candidates.append(
                    {
                        "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                        "bbox": _bbox_xyxy_from_xywh_pixels(
                            bbox_xywh=[int(x), int(y), int(bw), int(bh)], w=int(w), h=int(h)
                        ),
                        "score": float(score),
                        "features": base_features,
                        "ocr_text": str(region_text)[:240] if region_text else "",
                    }
                )

            candidates.sort(key=lambda c: float(c.get("score", 0.0) or 0.0), reverse=True)
            for i, c in enumerate(candidates):
                c["rank"] = int(i + 1)

            if candidates:
                best = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                top_s = float(best.get("score", 0.0) or 0.0)
                sec_s = float(second.get("score", 0.0) or 0.0) if second is not None else 0.0
                denom = float(max(abs(top_s), 1e-6))
                conf = float(_clamp01((top_s - sec_s) / denom))
                top_features0 = best.get("features") if isinstance(best.get("features"), dict) else {}
                try:
                    top_p_intent0 = float(top_features0.get("p_intent", 0.0) or 0.0)
                except Exception:
                    top_p_intent0 = 0.0
                u0 = _compute_uncertainty(
                    candidates=candidates,
                    conf_margin=float(conf),
                    p_intent_top=float(top_p_intent0),
                    min_candidates=int(fb_min_cands),
                )

                if fb_max_extra > 0 and len(candidates) > int(max(1, top_k)):
                    extra_keep = int(max(0, fb_max_extra))
                    candidates = candidates[: int(max(1, top_k) + extra_keep)]

    use_utility = bool(util_cfg.get("enabled", False))
    selected_by = "score"
    if use_utility and candidates:
        best_u = None
        best_u_val = -1.0
        for c in candidates:
            feats = c.get("features") if isinstance(c.get("features"), dict) else {}
            try:
                uval = float(feats.get("utility", 0.0) or 0.0)
            except Exception:
                uval = 0.0
            if best_u is None or float(uval) > float(best_u_val):
                best_u = c
                best_u_val = float(uval)
        if best_u is not None:
            best = best_u
            selected_by = "utility"

    best_features = best.get("features") if isinstance(best.get("features"), dict) else {}
    best_icon_like = float(best_features.get("icon_likelihood", 0.0) or 0.0)
    icon_click = bool(icon_intent) or (float(soft_icon) >= 0.65 and float(best_icon_like) >= 0.55)

    bx, by, bw, bh = [int(v) for v in best.get("bbox_xywh")]
    click_method = "saliency_peak"
    px = int(bx + bw // 2)
    py = int(by + bh // 2)

    if icon_click:
        click_method = "bbox_center"
    else:
        query_terms = targets if targets else instr_kw[:8]
        word_bbox: Optional[Tuple[int, int, int, int]] = None
        word_score = 0.0
        if query_terms and ocr_words:
            rx0 = int(bx)
            ry0 = int(by)
            rx1 = int(bx + bw)
            ry1 = int(by + bh)
            for wr in ocr_words:
                bb = wr.get("bbox_xywh")
                if not isinstance(bb, list) or len(bb) != 4:
                    continue
                try:
                    wx, wy, ww, wh = [int(v) for v in bb]
                except Exception:
                    continue
                if ww <= 0 or wh <= 0:
                    continue
                wcx = float(wx) + float(ww) / 2.0
                wcy = float(wy) + float(wh) / 2.0
                if not (float(rx0) <= wcx < float(rx1) and float(ry0) <= wcy < float(ry1)):
                    continue
                wt = _normalize_text(str(wr.get("text") or ""))
                if not wt:
                    continue

                base = 0.0
                for term in query_terms:
                    tnorm2 = _normalize_text(str(term or ""))
                    if not tnorm2:
                        continue
                    if wt == tnorm2:
                        base = max(base, 1.0)
                        continue
                    t_tokens = _tokens(tnorm2)
                    if wt in set(t_tokens):
                        base = max(base, 0.92)
                    if tnorm2 in wt or wt in tnorm2:
                        base = max(base, 0.85)
                    base = max(base, float(_jaccard(_char_trigrams(wt), _char_trigrams(tnorm2))))

                conf_w = float(0.65 + 0.35 * _clamp01(float(wr.get("conf", 0.0) or 0.0)))
                s = float(_clamp01(base) * conf_w)
                if s > word_score:
                    word_score = float(s)
                    word_bbox = (int(wx), int(wy), int(ww), int(wh))

        if word_bbox is not None and float(word_score) >= 0.55:
            wx, wy, ww, wh = word_bbox
            px = int(wx + ww // 2)
            py = int(wy + wh // 2)
            click_method = "ocr_word_center"
        else:
            patch = sal[by : by + bh, bx : bx + bw]
            if patch.size > 0:
                idx = int(np.argmax(patch))
                py0, px0 = np.unravel_index(idx, patch.shape)
                px = int(bx + int(px0))
                py = int(by + int(py0))
                click_method = "saliency_peak"
            else:
                click_method = "bbox_center"

    x01 = float(px) / float(max(w - 1, 1))
    y01 = float(py) / float(max(h - 1, 1))

    return {
        "x": float(np.clip(x01, 0.0, 1.0)),
        "y": float(np.clip(y01, 0.0, 1.0)),
        "bbox": best.get("bbox"),
        "confidence": float(conf),
        "meta": {
            "version": CONFIG_VERSION,
            "candidates": int(len(candidates)),
            "proposal_stats": proposal_stats,
            "groups": groups,
            "spatial_hints": spatial_hints,
            "targets": targets,
            "icon_intent": bool(icon_intent),
            "icon_mode_strength": float(_clamp01(soft_icon)),
            "click_method": str(click_method),
            "ocr": ocr_meta,
            "vlm": vlm_meta,
            "ranker": {
                "enabled": bool(ranker_model is not None),
                "path": str(ranker_model.get("path") or "") if isinstance(ranker_model, dict) else "",
            },
            "utility": {"enabled": bool(use_utility), "selected_by": str(selected_by)},
            "uncertainty": u0,
            "selected": best,
            "top_k": candidates[:top_k],
            "all_candidates": candidates if return_all else None,
        },
    }
