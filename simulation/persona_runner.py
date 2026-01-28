from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2
from Backend.common.math_utils import clamp01, normalize_01


def run_personas(
    *,
    elements: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    personas: Optional[List[Dict[str, Any]]] = None,
    max_first_noticed: int = 5,
) -> List[Dict[str, Any]]:
    h, w = image_shape
    if personas is None:
        personas = generate_synthetic_users(elements=elements, image_shape=(h, w))

    runs: List[Dict[str, Any]] = []

    for persona in personas:
        weights = _score_elements_for_persona(elements, persona, image_shape=(h, w))
        order = sorted(range(len(elements)), key=lambda i: weights[i], reverse=True)

        first_noticed_idx = order[: max(1, int(persona.get("patience", max_first_noticed)))]
        first_noticed = [
            {
                "id": elements[i].get("id"),
                "bbox": elements[i].get("bbox"),
                "type": elements[i].get("type"),
                "weight": float(weights[i]),
            }
            for i in first_noticed_idx[:max_first_noticed]
        ]

        confusion = _normalized_entropy([weights[i] for i in first_noticed_idx])
        trust_score, trust_signals = _trust_signals(elements, weights)

        attention = _attention_map(elements, weights, image_shape=(h, w))

        runs.append(
            {
                "persona": {
                    "id": persona.get("id"),
                    "name": persona.get("name"),
                    "archetype": persona.get("archetype"),
                    "focus": persona.get("focus"),
                },
                "first_noticed": first_noticed,
                "confusion": float(confusion),
                "trust": float(trust_score),
                "trust_signals": trust_signals,
                "weights_by_element": {elements[i].get("id"): float(weights[i]) for i in range(len(elements))},
                "attention_map": attention,
            }
        )

    return runs


def generate_synthetic_users(
    *,
    elements: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    max_users: int = 6,
) -> List[Dict[str, Any]]:
    h, w = image_shape
    if max_users <= 0:
        return []

    def _center_norm(el: Dict[str, Any]) -> Tuple[float, float]:
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            return (0.5, 0.4)
        x, y, bw, bh = [int(v) for v in bbox]
        cx = (x + bw / 2.0) / float(max(w, 1))
        cy = (y + bh / 2.0) / float(max(h, 1))
        return (float(cx), float(cy))

    cta_elements = [el for el in elements if float(el.get("cta_score", 0.0)) >= 0.62]
    text_elements = [el for el in elements if str(el.get("type")) == "text_block"]

    dominant_cta = max(cta_elements, key=lambda e: float(e.get("cta_score", 0.0)), default=None)
    dominant_text = max(text_elements, key=lambda e: float(e.get("area_ratio", 0.0)), default=None)
    dominant_saliency = max(elements, key=lambda e: float(e.get("saliency_score", 0.0)), default=None)

    personas: List[Dict[str, Any]] = []

    personas.append(
        {
            "id": "su_scanner",
            "name": "First-time Scanner",
            "archetype": "scanner",
            "focus": "Overall layout and immediate visual hierarchy",
            "patience": 6,
            "weights": {"saliency": 0.35, "cta": 0.25, "contrast": 0.20, "center": 0.20},
            "preferred_center": (0.50, 0.38),
        }
    )

    personas.append(
        {
            "id": "su_cta_seeker",
            "name": "CTA Seeker",
            "archetype": "cta_seeker",
            "focus": "Primary action and next-step clarity",
            "patience": 5,
            "weights": {"saliency": 0.15, "cta": 0.55, "contrast": 0.10, "center": 0.20},
            "preferred_center": _center_norm(dominant_cta) if dominant_cta else (0.55, 0.35),
            "type_bias": {"cta": 1.35},
        }
    )

    personas.append(
        {
            "id": "su_visual_explorer",
            "name": "Visual Explorer",
            "archetype": "visual_explorer",
            "focus": "Visuals, hero sections, and strong imagery",
            "patience": 6,
            "weights": {"saliency": 0.50, "cta": 0.10, "contrast": 0.20, "center": 0.20},
            "preferred_center": _center_norm(dominant_saliency) if dominant_saliency else (0.50, 0.42),
            "type_bias": {"container": 1.20},
        }
    )

    if dominant_text is not None:
        personas.append(
            {
                "id": "su_reader",
                "name": "Content Reader",
                "archetype": "reader",
                "focus": "Copy, value proposition, and explanatory text",
                "patience": 7,
                "weights": {"saliency": 0.35, "cta": 0.10, "contrast": 0.35, "center": 0.20},
                "preferred_center": _center_norm(dominant_text),
                "type_bias": {"text_block": 1.35},
            }
        )

    personas.append(
        {
            "id": "su_trust_seeker",
            "name": "Trust Seeker",
            "archetype": "trust_seeker",
            "focus": "Trust signals (logos, badges, policies) and clarity",
            "patience": 7,
            "weights": {"saliency": 0.25, "cta": 0.10, "contrast": 0.45, "center": 0.20},
            "preferred_center": (0.50, 0.72),
            "type_bias": {"text_block": 1.10, "icon_or_control": 1.15},
        }
    )

    personas.append(
        {
            "id": "su_bouncer",
            "name": "Impatient Bouncer",
            "archetype": "bouncer",
            "focus": "Immediate clarity within the first few seconds",
            "patience": 3,
            "weights": {"saliency": 0.45, "cta": 0.25, "contrast": 0.15, "center": 0.15},
            "preferred_center": (0.50, 0.33),
        }
    )

    return personas[: int(max_users)]


def _default_personas() -> List[Dict[str, Any]]:
    return [
        {
            "id": "p_first_time",
            "name": "First-time Visitor",
            "archetype": "scanner",
            "patience": 6,
            "weights": {"saliency": 0.35, "cta": 0.30, "contrast": 0.20, "center": 0.15},
            "preferred_center": (0.50, 0.38),
        },
        {
            "id": "p_goal_oriented",
            "name": "Goal-Oriented",
            "archetype": "cta_seeker",
            "patience": 5,
            "weights": {"saliency": 0.20, "cta": 0.50, "contrast": 0.15, "center": 0.15},
            "preferred_center": (0.55, 0.35),
        },
        {
            "id": "p_skeptical",
            "name": "Skeptical Buyer",
            "archetype": "trust_seeker",
            "patience": 7,
            "weights": {"saliency": 0.30, "cta": 0.15, "contrast": 0.30, "center": 0.25},
            "preferred_center": (0.50, 0.45),
        },
    ]


def _score_elements_for_persona(
    elements: List[Dict[str, Any]], persona: Dict[str, Any], *, image_shape: Tuple[int, int]
) -> List[float]:
    h, w = image_shape
    pw = persona.get("weights", {})
    w_sal = float(pw.get("saliency", 0.35))
    w_cta = float(pw.get("cta", 0.30))
    w_con = float(pw.get("contrast", 0.20))
    w_ctr = float(pw.get("center", 0.15))

    pcx, pcy = persona.get("preferred_center", (0.5, 0.4))

    raw: List[float] = []
    for el in elements:
        bbox = el.get("bbox") or [0, 0, 1, 1]
        x, y, bw, bh = [int(v) for v in bbox]
        cx = (x + bw / 2.0) / float(max(w, 1))
        cy = (y + bh / 2.0) / float(max(h, 1))

        center_dist = ((cx - float(pcx)) ** 2 + (cy - float(pcy)) ** 2) ** 0.5
        center_score = 1.0 - min(center_dist / 0.85, 1.0)

        saliency = float(el.get("saliency_score", 0.0))
        cta = float(el.get("cta_score", 0.0))
        contrast = float(el.get("contrast", 0.0))
        importance = float(el.get("importance_score", saliency))
        level = int(el.get("hierarchy_level", 0))

        base = (
            w_sal * saliency
            + w_cta * cta
            + w_con * clamp01((contrast - 0.05) / 0.35)
            + w_ctr * center_score
        )

        type_bias = persona.get("type_bias")
        if isinstance(type_bias, dict):
            base = base * float(type_bias.get(str(el.get("type")), 1.0))

        base = base * (0.85 + 0.15 * importance)
        base = base / (1.0 + 0.18 * float(level))
        raw.append(max(0.0, float(base)))

    s = float(sum(raw))
    if s <= 1e-9:
        return [1.0 / float(max(len(elements), 1)) for _ in elements]

    return [v / s for v in raw]


def _attention_map(elements: List[Dict[str, Any]], weights: List[float], *, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    att = np.zeros((h, w), dtype=np.float32)

    for el, wt in zip(elements, weights):
        bbox = el.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        att[y : y + bh, x : x + bw] += float(wt)

    cv2 = maybe_cv2()
    if cv2 is not None:
        att = cv2.GaussianBlur(att, (0, 0), sigmaX=9.0)

    return normalize_01(att)


def _trust_signals(elements: List[Dict[str, Any]], weights: List[float]) -> Tuple[float, List[str]]:
    if not elements:
        return 0.0, ["No detectable UI structure"]

    cta_count = sum(1 for el in elements if float(el.get("cta_score", 0.0)) >= 0.62)
    clutter = min(max((cta_count - 1) / 3.0, 0.0), 1.0)

    focus = 1.0 - _normalized_entropy(weights)

    trust = clamp01(0.65 * focus + 0.35 * (1.0 - clutter))

    signals: List[str] = []
    if cta_count <= 1:
        signals.append("Single dominant action")
    elif cta_count >= 3:
        signals.append("Multiple competing actions")

    if focus >= 0.60:
        signals.append("Clear visual priority")
    elif focus <= 0.35:
        signals.append("Visual priority is unclear")

    return trust, signals


def _normalized_entropy(weights: List[float]) -> float:
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

