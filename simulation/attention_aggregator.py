from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from Backend.common.math_utils import clamp01, normalize_01


def aggregate_attention(
    *,
    persona_runs: List[Dict[str, Any]],
    elements: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    ignore_threshold: float = 0.15,
    over_attention_threshold: float = 0.65,
) -> Dict[str, Any]:
    h, w = image_shape

    attention_maps: List[np.ndarray] = []
    confusions: List[float] = []
    trusts: List[float] = []

    element_ids = [str(el.get("id")) for el in elements]
    per_element_weights: Dict[str, List[float]] = {eid: [] for eid in element_ids}

    persona_summaries: List[Dict[str, Any]] = []

    for run in persona_runs:
        att = run.get("attention_map")
        if isinstance(att, np.ndarray) and att.shape == (h, w):
            attention_maps.append(att.astype(np.float32))

        confusion = float(run.get("confusion", 0.0))
        trust = float(run.get("trust", 0.0))
        confusions.append(confusion)
        trusts.append(trust)

        w_by_el = run.get("weights_by_element") or {}
        if isinstance(w_by_el, dict):
            for eid in element_ids:
                if eid in w_by_el:
                    per_element_weights[eid].append(float(w_by_el[eid]))

        persona = run.get("persona") or {}
        persona_summaries.append(
            {
                "persona": {
                    "id": persona.get("id"),
                    "name": persona.get("name"),
                    "archetype": persona.get("archetype"),
                    "focus": persona.get("focus"),
                },
                "first_noticed": run.get("first_noticed", []),
                "confusion": confusion,
                "trust": trust,
                "trust_signals": run.get("trust_signals", []),
            }
        )

    if attention_maps:
        att_mean = np.mean(np.stack(attention_maps, axis=0), axis=0).astype(np.float32)
    else:
        att_mean = np.zeros((h, w), dtype=np.float32)

    att_mean = normalize_01(att_mean)

    element_weight_avg: Dict[str, float] = {}
    for eid, vals in per_element_weights.items():
        if vals:
            element_weight_avg[eid] = float(sum(vals) / float(len(vals)))
        else:
            element_weight_avg[eid] = 0.0

    mean_confusion = float(sum(confusions) / float(max(len(confusions), 1)))
    mean_trust = float(sum(trusts) / float(max(len(trusts), 1)))

    focus_score = 1.0 - _normalized_entropy(list(element_weight_avg.values()))
    clarity_score = clamp01(1.0 - mean_confusion)

    bxo_score = 100.0 * (
        0.45 * clarity_score + 0.35 * clamp01(mean_trust) + 0.20 * clamp01(focus_score)
    )

    zones = _attention_zones(att_mean, ignore_threshold=ignore_threshold, over_threshold=over_attention_threshold)

    top_elements = _top_elements(elements, element_weight_avg, k=6)

    evaluation = _build_evaluation(
        mean_confusion=mean_confusion,
        mean_trust=mean_trust,
        focus_score=focus_score,
        zones=zones,
        top_elements=top_elements,
    )

    return {
        "attention_map": att_mean,
        "bxo_score": float(bxo_score),
        "plura_index_score": float(bxo_score),
        "metrics": {
            "persona_count": int(len(persona_runs)),
            "mean_confusion": float(mean_confusion),
            "mean_trust": float(mean_trust),
            "focus_score": float(focus_score),
            "clarity_score": float(clarity_score),
            "ignore_threshold": float(ignore_threshold),
            "over_attention_threshold": float(over_attention_threshold),
        },
        "attention_zones": zones,
        "top_elements": top_elements,
        "persona_runs": persona_summaries,
        "evaluation": evaluation,
    }


def _top_elements(
    elements: List[Dict[str, Any]], element_weight_avg: Dict[str, float], *, k: int
) -> List[Dict[str, Any]]:
    ranked = []
    for el in elements:
        eid = str(el.get("id"))
        ranked.append((eid, float(element_weight_avg.get(eid, 0.0)), el))

    ranked.sort(key=lambda x: x[1], reverse=True)

    out: List[Dict[str, Any]] = []
    for eid, wt, el in ranked[: max(0, int(k))]:
        out.append(
            {
                "id": eid,
                "bbox": el.get("bbox"),
                "type": el.get("type"),
                "weight": float(wt),
                "cta_score": float(el.get("cta_score", 0.0)),
                "saliency_score": float(el.get("saliency_score", 0.0)),
            }
        )

    return out


def _attention_zones(
    att: np.ndarray, *, ignore_threshold: float, over_threshold: float
) -> Dict[str, Any]:
    att = att.astype(np.float32)
    h, w = att.shape
    total = float(h * w)

    ignored = att < float(ignore_threshold)
    over = att >= float(over_threshold)
    clarity = (~ignored) & (~over)

    ignored_ratio = float(np.sum(ignored) / total) if total > 0 else 0.0
    clarity_ratio = float(np.sum(clarity) / total) if total > 0 else 0.0
    over_ratio = float(np.sum(over) / total) if total > 0 else 0.0

    return {
        "ignored": {"ratio": ignored_ratio, "pixels": int(np.sum(ignored))},
        "clarity": {"ratio": clarity_ratio, "pixels": int(np.sum(clarity))},
        "over_attention": {"ratio": over_ratio, "pixels": int(np.sum(over))},
    }


def _build_evaluation(
    *,
    mean_confusion: float,
    mean_trust: float,
    focus_score: float,
    zones: Dict[str, Any],
    top_elements: List[Dict[str, Any]],
) -> str:
    confusion_level = "low" if mean_confusion <= 0.33 else "medium" if mean_confusion <= 0.60 else "high"
    trust_level = "high" if mean_trust >= 0.66 else "medium" if mean_trust >= 0.45 else "low"
    focus_level = "high" if focus_score >= 0.66 else "medium" if focus_score >= 0.45 else "low"

    ignored_ratio = float(zones.get("ignored", {}).get("ratio", 0.0))
    over_ratio = float(zones.get("over_attention", {}).get("ratio", 0.0))

    dominant = top_elements[0] if top_elements else None
    dominant_type = dominant.get("type") if dominant else None

    lines: List[str] = []

    lines.append(
        f"Behavioral clarity is {focus_level} with {confusion_level} confusion across personas (trust: {trust_level})."
    )

    if dominant and dominant_type == "cta":
        lines.append("A primary call-to-action is visually dominant, which improves task direction.")
    elif dominant:
        lines.append("The most noticed element is not a clear call-to-action; consider strengthening the primary action.")

    if ignored_ratio >= 0.50:
        lines.append("Large areas are ignored; consider reducing dead space or improving hierarchy and contrast.")
    if over_ratio >= 0.25:
        lines.append("Attention is heavily concentrated; verify the focal region is the intended action and not visual noise.")

    return " ".join(lines)


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

