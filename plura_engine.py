from __future__ import annotations

import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.indexing.element_classifier import classify_elements
from Backend.indexing.ui_parser import parse_ui
from Backend.indexing.visual_saliency import compute_attention_frames, compute_perceptual_field
from Backend.visualization.heatmap import generate_heatmap_png


ALGO_VERSION = "plura_mvp_v6"


def run_plura_analysis(
    *,
    image_bytes: bytes,
    base_dir: Path,
    cache_ttl_seconds: int = 60 * 60 * 24 * 7,
    intent: Optional[str] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    if not image_bytes:
        raise ValueError("image_bytes is empty")

    intent_str = str(intent).strip() if intent is not None else ""
    cache_seed = image_bytes + b"|" + ALGO_VERSION.encode("utf-8") + b"|" + intent_str.encode("utf-8", errors="ignore")
    cache_key = hashlib.md5(cache_seed).hexdigest()
    json_path, png_path = get_cache_paths(cache_key, base_dir=base_dir)
    md_path, xml_path = get_export_paths(cache_key, base_dir=base_dir)

    try:
        _maybe_save_original_image(cache_key=cache_key, base_dir=base_dir, image_bytes=image_bytes)
    except Exception:
        pass

    cached = _try_load_cached(json_path=json_path, png_path=png_path, ttl_seconds=int(cache_ttl_seconds))
    if cached is not None:
        cached["cache_key"] = cache_key
        cached["cached"] = True
        cached.setdefault("version", ALGO_VERSION)
        if intent_str:
            cached.setdefault("intent", intent_str)
        if "preliminary_doc" not in cached or "final_report" not in cached:
            preliminary_doc, final_report = _build_plura_documents(cached)
            cached["preliminary_doc"] = preliminary_doc
            cached["final_report"] = final_report
        export_urls = cached.get("export_urls")
        if not isinstance(export_urls, dict):
            export_urls = {}
        export_urls.setdefault("markdown", f"/api/plura/export/{cache_key}/markdown/")
        export_urls.setdefault("xml", f"/api/plura/export/{cache_key}/xml/")
        export_urls.setdefault("pdf", f"/api/plura/export/{cache_key}/pdf/")
        cached["export_urls"] = export_urls
        if not md_path.exists() or not xml_path.exists():
            try:
                _write_export_files(
                    md_path=md_path,
                    xml_path=xml_path,
                    markdown=_build_plura_export_markdown(cached),
                    xml=_build_plura_export_xml(cached),
                )
            except Exception:
                pass
        return cached

    image_rgb = _decode_image_bytes(image_bytes)
    orig_h, orig_w = image_rgb.shape[:2]

    canonical_width = 1440
    image_rgb = _normalize_image_rgb(image_rgb, canonical_width=int(canonical_width))
    h, w = image_rgb.shape[:2]

    elements_raw = parse_ui(image_rgb)
    intent_kind = _intent_kind(intent_str or None)
    frames_seed = compute_attention_frames(image_rgb, intent_kind=intent_kind)
    attention_for_classification = frames_seed.get("orienting")
    if attention_for_classification is None:
        attention_for_classification = frames_seed.get("scanning")
    if attention_for_classification is None:
        attention_for_classification = frames_seed.get("immediate")
    if attention_for_classification is None:
        attention_for_classification = np.zeros((h, w), dtype=np.float32)

    elements = classify_elements(elements_raw, attention_for_classification, image_shape=(h, w))
    frames = compute_attention_frames(image_rgb, intent_kind=intent_kind, elements=elements)
    attention = frames.get("orienting")
    if attention is None:
        attention = attention_for_classification

    ignore_threshold = 0.15
    over_attention_threshold = 0.65

    heatmap_alpha = 0.55

    heatmap_legend = _heatmap_legend(
        ignore_threshold=float(ignore_threshold),
        over_attention_threshold=float(over_attention_threshold),
        alpha=float(heatmap_alpha),
    )

    attention_points = _attention_points(attention, q=0.90, max_points=5)

    element_weights = _compute_element_weights(elements)
    top_elements = _top_elements(elements, element_weights, k=6)
    zones = _attention_zones(attention, ignore_threshold=ignore_threshold, over_threshold=over_attention_threshold)

    attention_frames_payload: Dict[str, Any] = {}
    for frame_name in ("immediate", "orienting", "scanning"):
        frame_map = frames.get(frame_name)
        if frame_map is None:
            continue
        heatmap_url = (
            f"/api/plura/heatmap/{cache_key}/"
            if frame_name == "orienting"
            else f"/api/plura/heatmap/{cache_key}/?frame={frame_name}"
        )
        attention_frames_payload[frame_name] = {
            "heatmap_url": heatmap_url,
            "attention_zones": _attention_zones(
                frame_map,
                ignore_threshold=ignore_threshold,
                over_threshold=over_attention_threshold,
            ),
            "attention_points": _attention_points(frame_map, q=0.90, max_points=5),
        }

    visual_index = _build_visual_index(image_rgb=image_rgb, elements=elements, intent_kind=intent_kind)

    behavioral = _build_behavioral_index(
        image_rgb=image_rgb,
        attention_map=attention,
        elements=elements,
        element_weights=element_weights,
        intent=intent_str or None,
    )
    scores = behavioral.get("scores") if isinstance(behavioral.get("scores"), dict) else {}
    overall_score = float(scores.get("overall", 0.0) or 0.0)
    evaluation = str(behavioral.get("evaluation") or "").strip()

    heatmap_png = generate_heatmap_png(
        image_rgb=image_rgb,
        attention_map=attention,
        ignore_threshold=ignore_threshold,
        over_attention_threshold=over_attention_threshold,
        alpha=float(heatmap_alpha),
    )

    frame_heatmaps: Dict[str, bytes] = {}
    for frame_name in ("immediate", "scanning"):
        frame_map = frames.get(frame_name)
        if frame_map is None:
            continue
        try:
            frame_heatmaps[frame_name] = generate_heatmap_png(
                image_rgb=image_rgb,
                attention_map=frame_map,
                ignore_threshold=ignore_threshold,
                over_attention_threshold=over_attention_threshold,
                alpha=float(heatmap_alpha),
            )
        except Exception:
            pass

    payload = {
        "success": True,
        "cached": False,
        "cache_key": cache_key,
        "version": ALGO_VERSION,
        "intent": intent_str or None,
        "intent_kind": intent_kind,
        "image": {"width": int(w), "height": int(h)},
        "image_original": {"width": int(orig_w), "height": int(orig_h)},
        "image_normalized": {"width": int(w), "height": int(h)},
        "normalization": {
            "canonical_width": int(canonical_width),
            "aspect_preserved": True,
            "luminance_normalized": True,
            "exif_removed": True,
        },
        "bxo_score": float(overall_score),
        "plura_index_score": float(overall_score),
        "metrics": scores,
        "attention_zones": zones,
        "attention_frames": attention_frames_payload,
        "heatmap_legend": heatmap_legend,
        "attention_points": attention_points,
        "top_elements": top_elements,
        "persona_runs": [],
        "evaluation": evaluation,
        "elements": elements,
        "visual_index": visual_index,
        "behavioral_index": {
            "version": ALGO_VERSION,
            "labels": behavioral.get("labels", []),
            "vector": behavioral.get("vector", []),
        },
    }

    preliminary_doc, final_report = _build_plura_documents(payload)

    payload["preliminary_doc"] = preliminary_doc
    payload["final_report"] = final_report
    payload["export_urls"] = {
        "markdown": f"/api/plura/export/{cache_key}/markdown/",
        "xml": f"/api/plura/export/{cache_key}/xml/",
        "pdf": f"/api/plura/export/{cache_key}/pdf/",
    }

    export_markdown = _build_plura_export_markdown(payload)
    export_xml = _build_plura_export_xml(payload)

    _save_cache(json_path=json_path, png_path=png_path, payload=payload, heatmap_png=heatmap_png)
    try:
        _save_frame_heatmaps(cache_key=cache_key, base_dir=base_dir, frame_heatmaps=frame_heatmaps)
    except Exception:
        pass
    try:
        _write_export_files(md_path=md_path, xml_path=xml_path, markdown=export_markdown, xml=export_xml)
    except Exception:
        pass
    return payload


def _build_plura_documents(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    image = payload.get("image") or {}
    w = int(image.get("width", 0) or 0)
    h = int(image.get("height", 0) or 0)

    elements = payload.get("elements") or []
    if not isinstance(elements, list):
        elements = []

    top_elements = payload.get("top_elements") or []
    if not isinstance(top_elements, list):
        top_elements = []

    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    zones = payload.get("attention_zones") or {}
    if not isinstance(zones, dict):
        zones = {}

    legend = payload.get("heatmap_legend") or {}
    if not isinstance(legend, dict):
        legend = {}

    evaluation = str(payload.get("evaluation") or "").strip()
    intent = str(payload.get("intent") or "").strip()

    type_counts: Dict[str, int] = {}
    cta_count = 0
    for el in elements:
        if not isinstance(el, dict):
            continue
        t = str(el.get("type") or "unknown")
        type_counts[t] = int(type_counts.get(t, 0)) + 1
        if float(el.get("cta_score", 0.0) or 0.0) >= 0.62:
            cta_count += 1

    prelim_lines: List[str] = []
    prelim_lines.append("# Preliminary Screenshot Analysis")
    prelim_lines.append("")
    prelim_lines.append(f"Image: {w}×{h}")
    prelim_lines.append(f"Detected elements: {len(elements)}")
    prelim_lines.append(f"Detected CTAs (cta_score ≥ 0.62): {cta_count}")
    if intent:
        prelim_lines.append(f"Intent: {intent}")

    if type_counts:
        prelim_lines.append("")
        prelim_lines.append("## Indexing Summary")
        for t, n in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            prelim_lines.append(f"- {t}: {n}")

    if top_elements:
        prelim_lines.append("")
        prelim_lines.append("## Top Indexed Elements")
        for el in top_elements[:6]:
            if not isinstance(el, dict):
                continue
            eid = str(el.get("id") or "")
            etype = str(el.get("type") or "")
            bbox = el.get("bbox")
            wt = float(el.get("weight", 0.0) or 0.0)
            prelim_lines.append(f"- {eid} | {etype} | weight={wt:.3f} | bbox={bbox}")

    prelim_lines.append("")
    prelim_lines.append("## Quant Metrics")
    prelim_lines.append(f"BXO: {float(payload.get('bxo_score', 0.0) or 0.0):.1f}")
    prelim_lines.append(f"Plura Index: {float(payload.get('plura_index_score', 0.0) or 0.0):.1f}")

    ignored_ratio = float(zones.get("ignored", {}).get("ratio", 0.0) or 0.0)
    over_ratio = float(zones.get("over_attention", {}).get("ratio", 0.0) or 0.0)
    clarity_ratio = float(zones.get("clarity", {}).get("ratio", 0.0) or 0.0)
    prelim_lines.append(f"Attention zones (ratios): ignored={ignored_ratio:.2f}, clarity={clarity_ratio:.2f}, over_attention={over_ratio:.2f}")

    focus_score = float(metrics.get("focus", 0.0) or 0.0)
    clarity_score = float(metrics.get("clarity", 0.0) or 0.0)
    alignment_score = metrics.get("alignment")
    focal_points = int(metrics.get("focal_points", 0) or 0)
    fragmentation = float(metrics.get("fragmentation", 0.0) or 0.0)
    dominance = float(metrics.get("dominance", 0.0) or 0.0)
    decay = float(metrics.get("decay", 0.0) or 0.0)
    intent_overlap = metrics.get("intent_overlap")

    if alignment_score is None:
        prelim_lines.append(f"Focus: {focus_score:.2f} | Clarity: {clarity_score:.2f}")
    else:
        prelim_lines.append(
            f"Focus: {focus_score:.2f} | Clarity: {clarity_score:.2f} | Alignment: {float(alignment_score or 0.0):.2f}"
        )

    prelim_lines.append(
        f"Focal points: {focal_points} | Dominance: {dominance:.2f} | Fragmentation: {fragmentation:.2f} | Decay: {decay:.2f}"
    )
    if intent_overlap is not None:
        try:
            prelim_lines.append(f"Intent overlap: {float(intent_overlap or 0.0):.2f}")
        except Exception:
            pass

    ignore_thr = legend.get("ignore_threshold")
    over_thr = legend.get("over_attention_threshold")
    if isinstance(ignore_thr, (int, float)) and isinstance(over_thr, (int, float)):
        prelim_lines.append(
            f"Heatmap legend: gray< {float(ignore_thr):.2f}, green in [{float(ignore_thr):.2f}, {float(over_thr):.2f}), red>= {float(over_thr):.2f}"
        )
    else:
        prelim_lines.append("Heatmap legend: gray=low attention, green=mid attention, red=high attention")
    if evaluation:
        prelim_lines.append(f"Evaluation: {evaluation}")

    prelim_content = "\n".join(prelim_lines).strip() + "\n"

    final_lines: List[str] = []
    final_lines.append("# Final Screenshot Report")
    final_lines.append("")

    bxo = float(payload.get("bxo_score", 0.0) or 0.0)
    if bxo >= 75.0:
        clarity_label = "High"
    elif bxo >= 55.0:
        clarity_label = "Medium"
    else:
        clarity_label = "Low"

    final_lines.append("## Executive Summary")
    final_lines.append(f"- Behavioral clarity: {clarity_label} (BXO {bxo:.1f})")
    if top_elements:
        dom = top_elements[0] if isinstance(top_elements[0], dict) else {}
        dom_type = str(dom.get("type") or "")
        dom_id = str(dom.get("id") or "")
        if dom_id:
            final_lines.append(f"- Most compelling element: {dom_id} ({dom_type})")
    final_lines.append(f"- Attention distribution: ignored={ignored_ratio:.2f}, over_attention={over_ratio:.2f}")
    if intent:
        final_lines.append(f"- Intent: {intent}")

    final_lines.append("")
    final_lines.append("## Observations")
    if evaluation:
        final_lines.append(f"- {evaluation}")
    else:
        final_lines.append("- No additional observations were produced.")

    final_content = "\n".join(final_lines).strip() + "\n"

    preliminary_doc = {"format": "markdown", "content": prelim_content}
    final_report = {"format": "markdown", "content": final_content}
    return preliminary_doc, final_report


def _build_plura_export_markdown(payload: Dict[str, Any]) -> str:
    cache_key = str(payload.get("cache_key") or "").strip()
    version = str(payload.get("version") or "").strip()
    intent = payload.get("intent")
    intent_kind = payload.get("intent_kind")

    image = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    orig = payload.get("image_original") if isinstance(payload.get("image_original"), dict) else {}

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    zones = payload.get("attention_zones") if isinstance(payload.get("attention_zones"), dict) else {}
    points = payload.get("attention_points") if isinstance(payload.get("attention_points"), list) else []
    top_elements = payload.get("top_elements") if isinstance(payload.get("top_elements"), list) else []
    elements = payload.get("elements") if isinstance(payload.get("elements"), list) else []
    behavioral_index = payload.get("behavioral_index") if isinstance(payload.get("behavioral_index"), dict) else {}
    attention_frames = payload.get("attention_frames") if isinstance(payload.get("attention_frames"), dict) else {}
    visual_index = payload.get("visual_index") if isinstance(payload.get("visual_index"), dict) else {}

    lines: List[str] = []
    lines.append("# Plura Engine Export")
    lines.append("")
    lines.append("## Metadata")
    if version:
        lines.append(f"- version: {version}")
    if cache_key:
        lines.append(f"- cache_key: {cache_key}")
    if intent:
        lines.append(f"- intent: {str(intent).strip()}")
    if intent_kind:
        lines.append(f"- intent_kind: {str(intent_kind).strip()}")
    if image.get("width") and image.get("height"):
        lines.append(f"- image_normalized: {int(image.get('width'))}x{int(image.get('height'))}")
    if orig.get("width") and orig.get("height"):
        lines.append(f"- image_original: {int(orig.get('width'))}x{int(orig.get('height'))}")
    if cache_key:
        lines.append(f"- heatmap_url: /api/plura/heatmap/{cache_key}/")

    lines.append("")
    lines.append("## Metrics")
    for k in sorted([str(x) for x in metrics.keys()]):
        lines.append(f"- {k}: {metrics.get(k)}")

    lines.append("")
    lines.append("## Attention Zones")
    for key in ("ignored", "clarity", "over_attention"):
        z = zones.get(key) if isinstance(zones.get(key), dict) else {}
        lines.append(f"- {key}: ratio={z.get('ratio')} pixels={z.get('pixels')}")

    lines.append("")
    lines.append("## Attention Points")
    for p in points:
        if not isinstance(p, dict):
            continue
        lines.append(
            f"- {p.get('id')}: x_norm={p.get('x_norm')} y_norm={p.get('y_norm')} mass_share={p.get('mass_share')} bbox={p.get('bbox')}"
        )

    lines.append("")
    lines.append("## Time-Framed Attention")
    for key in ("immediate", "orienting", "scanning"):
        fr = attention_frames.get(key) if isinstance(attention_frames.get(key), dict) else {}
        if not fr:
            continue
        fr_zones = fr.get("attention_zones") if isinstance(fr.get("attention_zones"), dict) else {}
        ignored = fr_zones.get("ignored") if isinstance(fr_zones.get("ignored"), dict) else {}
        clarity = fr_zones.get("clarity") if isinstance(fr_zones.get("clarity"), dict) else {}
        over = fr_zones.get("over_attention") if isinstance(fr_zones.get("over_attention"), dict) else {}
        fr_points = fr.get("attention_points") if isinstance(fr.get("attention_points"), list) else []
        lines.append(
            f"- {key}: heatmap_url={fr.get('heatmap_url')} ignored_ratio={ignored.get('ratio')} clarity_ratio={clarity.get('ratio')} over_ratio={over.get('ratio')}"
        )
        for p in fr_points[:3]:
            if not isinstance(p, dict):
                continue
            lines.append(
                f"  - {p.get('id')}: x_norm={p.get('x_norm')} y_norm={p.get('y_norm')} mass_share={p.get('mass_share')} bbox={p.get('bbox')}"
            )

    lines.append("")
    lines.append("## Top Elements")
    for el in top_elements:
        if not isinstance(el, dict):
            continue
        lines.append(
            f"- {el.get('id')}: type={el.get('type')} weight={el.get('weight')} cta_score={el.get('cta_score')} saliency_score={el.get('saliency_score')} bbox={el.get('bbox')}"
        )

    lines.append("")
    lines.append("## Elements")
    for el in elements:
        if not isinstance(el, dict):
            continue
        lines.append(
            f"- {el.get('id')}: type={el.get('type')} bbox={el.get('bbox')} area_ratio={el.get('area_ratio')} saliency_score={el.get('saliency_score')} cta_score={el.get('cta_score')} importance_score={el.get('importance_score')} hierarchy_level={el.get('hierarchy_level')}"
        )

    lines.append("")
    lines.append("## Visual Index")
    layout_regions = visual_index.get("layout_regions") if isinstance(visual_index.get("layout_regions"), list) else []
    text_clusters = visual_index.get("text_clusters") if isinstance(visual_index.get("text_clusters"), list) else []
    contrast_regions = visual_index.get("contrast_regions") if isinstance(visual_index.get("contrast_regions"), list) else []
    minority_proxies = visual_index.get("minority_proxies") if isinstance(visual_index.get("minority_proxies"), list) else []
    lines.append(f"- layout_regions: {len(layout_regions)}")
    for r in layout_regions[:5]:
        if not isinstance(r, dict):
            continue
        lines.append(f"  - {r.get('id')}: kind={r.get('kind')} bbox={r.get('bbox')} score={r.get('score')}")
    lines.append(f"- text_clusters: {len(text_clusters)}")
    for r in text_clusters[:5]:
        if not isinstance(r, dict):
            continue
        lines.append(f"  - {r.get('id')}: bbox={r.get('bbox')} mean={r.get('mean')} area_ratio={r.get('area_ratio')}")
    lines.append(f"- contrast_regions: {len(contrast_regions)}")
    for r in contrast_regions[:5]:
        if not isinstance(r, dict):
            continue
        lines.append(f"  - {r.get('id')}: bbox={r.get('bbox')} mean={r.get('mean')} area_ratio={r.get('area_ratio')}")
    lines.append(f"- minority_proxies: {len(minority_proxies)}")
    for p in minority_proxies[:5]:
        if not isinstance(p, dict):
            continue
        lines.append(
            f"- {p.get('id')}: type={p.get('type')} risk_score={p.get('risk_score')} element_id={p.get('element_id')} bbox={p.get('bbox')}"
        )

    labels = behavioral_index.get("labels") if isinstance(behavioral_index.get("labels"), list) else []
    vector = behavioral_index.get("vector") if isinstance(behavioral_index.get("vector"), list) else []
    lines.append("")
    lines.append("## Behavioral Index")
    if labels:
        lines.append("Labels: " + ", ".join([str(x) for x in labels]))
    if vector:
        lines.append("Vector: " + ", ".join([str(x) for x in vector]))

    evaluation = str(payload.get("evaluation") or "").strip()
    lines.append("")
    lines.append("## Engine Evaluation")
    lines.append(evaluation if evaluation else "-")

    prelim = payload.get("preliminary_doc") if isinstance(payload.get("preliminary_doc"), dict) else {}
    prelim_content = str(prelim.get("content") or "").strip()
    if prelim_content:
        lines.append("")
        lines.append("## Preliminary Doc")
        lines.append(prelim_content)

    final = payload.get("final_report") if isinstance(payload.get("final_report"), dict) else {}
    final_content = str(final.get("content") or "").strip()
    if final_content:
        lines.append("")
        lines.append("## Final Report")
        lines.append(final_content)

    return "\n".join(lines).strip() + "\n"


def _build_plura_export_xml(payload: Dict[str, Any]) -> str:
    def _set_attr(el: ET.Element, key: str, value: Any) -> None:
        if value is None:
            return
        s = str(value).strip()
        if not s:
            return
        el.set(str(key), s)

    def _bbox_str(bbox: Any) -> str:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return ""
        try:
            return ",".join([str(int(v)) for v in bbox])
        except Exception:
            return ",".join([str(v) for v in bbox])

    root = ET.Element("plura_export")
    _set_attr(root, "version", payload.get("version"))
    _set_attr(root, "cache_key", payload.get("cache_key"))
    _set_attr(root, "intent", payload.get("intent"))
    _set_attr(root, "intent_kind", payload.get("intent_kind"))

    image = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    orig = payload.get("image_original") if isinstance(payload.get("image_original"), dict) else {}

    img_el = ET.SubElement(root, "image")
    _set_attr(img_el, "width", image.get("width"))
    _set_attr(img_el, "height", image.get("height"))

    orig_el = ET.SubElement(root, "image_original")
    _set_attr(orig_el, "width", orig.get("width"))
    _set_attr(orig_el, "height", orig.get("height"))

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics_el = ET.SubElement(root, "metrics")
    for k in sorted([str(x) for x in metrics.keys()]):
        m = ET.SubElement(metrics_el, "metric")
        _set_attr(m, "name", k)
        m.text = str(metrics.get(k))

    zones = payload.get("attention_zones") if isinstance(payload.get("attention_zones"), dict) else {}
    zones_el = ET.SubElement(root, "attention_zones")
    for key in ("ignored", "clarity", "over_attention"):
        z = zones.get(key) if isinstance(zones.get(key), dict) else {}
        z_el = ET.SubElement(zones_el, "zone")
        _set_attr(z_el, "name", key)
        _set_attr(z_el, "ratio", z.get("ratio"))
        _set_attr(z_el, "pixels", z.get("pixels"))

    attention_frames = payload.get("attention_frames") if isinstance(payload.get("attention_frames"), dict) else {}
    frames_el = ET.SubElement(root, "attention_frames")
    for name in ("immediate", "orienting", "scanning"):
        fr = attention_frames.get(name) if isinstance(attention_frames.get(name), dict) else {}
        if not fr:
            continue
        fe = ET.SubElement(frames_el, "frame")
        _set_attr(fe, "name", name)
        _set_attr(fe, "heatmap_url", fr.get("heatmap_url"))

        fr_zones = fr.get("attention_zones") if isinstance(fr.get("attention_zones"), dict) else {}
        fz_el = ET.SubElement(fe, "attention_zones")
        for key in ("ignored", "clarity", "over_attention"):
            z = fr_zones.get(key) if isinstance(fr_zones.get(key), dict) else {}
            z_el = ET.SubElement(fz_el, "zone")
            _set_attr(z_el, "name", key)
            _set_attr(z_el, "ratio", z.get("ratio"))
            _set_attr(z_el, "pixels", z.get("pixels"))

        fr_points = fr.get("attention_points") if isinstance(fr.get("attention_points"), list) else []
        fp_el = ET.SubElement(fe, "attention_points")
        for p in fr_points:
            if not isinstance(p, dict):
                continue
            pe = ET.SubElement(fp_el, "point")
            _set_attr(pe, "id", p.get("id"))
            _set_attr(pe, "x_norm", p.get("x_norm"))
            _set_attr(pe, "y_norm", p.get("y_norm"))
            _set_attr(pe, "mass_share", p.get("mass_share"))
            _set_attr(pe, "bbox", _bbox_str(p.get("bbox")))

    points = payload.get("attention_points") if isinstance(payload.get("attention_points"), list) else []
    points_el = ET.SubElement(root, "attention_points")
    for p in points:
        if not isinstance(p, dict):
            continue
        pe = ET.SubElement(points_el, "point")
        _set_attr(pe, "id", p.get("id"))
        _set_attr(pe, "x_norm", p.get("x_norm"))
        _set_attr(pe, "y_norm", p.get("y_norm"))
        _set_attr(pe, "mass_share", p.get("mass_share"))
        _set_attr(pe, "bbox", _bbox_str(p.get("bbox")))

    top_elements = payload.get("top_elements") if isinstance(payload.get("top_elements"), list) else []
    top_el = ET.SubElement(root, "top_elements")
    for el in top_elements:
        if not isinstance(el, dict):
            continue
        ee = ET.SubElement(top_el, "element")
        _set_attr(ee, "id", el.get("id"))
        _set_attr(ee, "type", el.get("type"))
        _set_attr(ee, "weight", el.get("weight"))
        _set_attr(ee, "cta_score", el.get("cta_score"))
        _set_attr(ee, "saliency_score", el.get("saliency_score"))
        _set_attr(ee, "bbox", _bbox_str(el.get("bbox")))

    elements = payload.get("elements") if isinstance(payload.get("elements"), list) else []
    elements_el = ET.SubElement(root, "elements")
    for el in elements:
        if not isinstance(el, dict):
            continue
        ee = ET.SubElement(elements_el, "element")
        _set_attr(ee, "id", el.get("id"))
        _set_attr(ee, "type", el.get("type"))
        _set_attr(ee, "bbox", _bbox_str(el.get("bbox")))
        _set_attr(ee, "area_ratio", el.get("area_ratio"))
        _set_attr(ee, "saliency_score", el.get("saliency_score"))
        _set_attr(ee, "cta_score", el.get("cta_score"))
        _set_attr(ee, "importance_score", el.get("importance_score"))
        _set_attr(ee, "hierarchy_level", el.get("hierarchy_level"))

    visual_index = payload.get("visual_index") if isinstance(payload.get("visual_index"), dict) else {}
    vi_el = ET.SubElement(root, "visual_index")
    for list_name in ("layout_regions", "text_clusters", "contrast_regions", "minority_proxies"):
        items = visual_index.get(list_name) if isinstance(visual_index.get(list_name), list) else []
        le = ET.SubElement(vi_el, str(list_name))
        for r in items:
            if not isinstance(r, dict):
                continue
            it = ET.SubElement(le, "item")
            _set_attr(it, "id", r.get("id"))
            _set_attr(it, "type", r.get("type") or r.get("kind") or list_name)
            _set_attr(it, "kind", r.get("kind"))
            _set_attr(it, "bbox", _bbox_str(r.get("bbox")))
            _set_attr(it, "score", r.get("score"))
            _set_attr(it, "mean", r.get("mean"))
            _set_attr(it, "area_ratio", r.get("area_ratio"))
            _set_attr(it, "risk_score", r.get("risk_score"))
            _set_attr(it, "saliency_score", r.get("saliency_score"))
            _set_attr(it, "cta_score", r.get("cta_score"))
            _set_attr(it, "element_id", r.get("element_id"))

    behavioral_index = payload.get("behavioral_index") if isinstance(payload.get("behavioral_index"), dict) else {}
    bi_el = ET.SubElement(root, "behavioral_index")
    labels = behavioral_index.get("labels") if isinstance(behavioral_index.get("labels"), list) else []
    vector = behavioral_index.get("vector") if isinstance(behavioral_index.get("vector"), list) else []
    labels_el = ET.SubElement(bi_el, "labels")
    for lab in labels:
        le = ET.SubElement(labels_el, "label")
        le.text = str(lab)
    vector_el = ET.SubElement(bi_el, "vector")
    for v in vector:
        ve = ET.SubElement(vector_el, "value")
        ve.text = str(v)

    evaluation = str(payload.get("evaluation") or "").strip()
    if evaluation:
        ev = ET.SubElement(root, "evaluation")
        ev.text = evaluation

    prelim = payload.get("preliminary_doc") if isinstance(payload.get("preliminary_doc"), dict) else {}
    prelim_content = str(prelim.get("content") or "").strip()
    if prelim_content:
        pe = ET.SubElement(root, "preliminary_doc")
        _set_attr(pe, "format", prelim.get("format"))
        pe.text = prelim_content

    final = payload.get("final_report") if isinstance(payload.get("final_report"), dict) else {}
    final_content = str(final.get("content") or "").strip()
    if final_content:
        fe = ET.SubElement(root, "final_report")
        _set_attr(fe, "format", final.get("format"))
        fe.text = final_content

    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return out.decode("utf-8") + "\n"


def _intent_kind(intent: Optional[str]) -> Optional[str]:
    if not intent:
        return None

    t = str(intent).strip().lower()
    if not t:
        return None

    action_kw = [
        "buy",
        "purchase",
        "checkout",
        "subscribe",
        "sign up",
        "signup",
        "register",
        "download",
        "start",
        "book",
        "add to cart",
    ]
    info_kw = [
        "learn",
        "read",
        "info",
        "information",
        "details",
        "pricing",
        "compare",
        "research",
        "features",
        "documentation",
    ]
    trust_kw = [
        "trust",
        "security",
        "privacy",
        "policy",
        "terms",
        "refund",
        "guarantee",
    ]

    if any(k in t for k in action_kw):
        return "action"
    if any(k in t for k in info_kw):
        return "info"
    if any(k in t for k in trust_kw):
        return "trust"
    return None


def _heatmap_legend(*, ignore_threshold: float, over_attention_threshold: float, alpha: float) -> Dict[str, Any]:
    return {
        "alpha": float(alpha),
        "ignore_threshold": float(ignore_threshold),
        "over_attention_threshold": float(over_attention_threshold),
        "zones": [
            {
                "key": "ignored",
                "label": "Low attention",
                "color_hex": "#808080",
                "color_rgb": [128, 128, 128],
                "range": {"lt": float(ignore_threshold)},
            },
            {
                "key": "clarity",
                "label": "Mid attention",
                "color_hex": "#00FF00",
                "color_rgb": [0, 255, 0],
                "range": {"gte": float(ignore_threshold), "lt": float(over_attention_threshold)},
            },
            {
                "key": "over_attention",
                "label": "High attention",
                "color_hex": "#FF0000",
                "color_rgb": [255, 0, 0],
                "range": {"gte": float(over_attention_threshold)},
            },
        ],
    }


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx - mn <= 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def _attention_zones(att: np.ndarray, *, ignore_threshold: float, over_threshold: float) -> Dict[str, Any]:
    a = _normalize_01(att)
    h, w = a.shape
    total = float(h * w)

    ignored = a < float(ignore_threshold)
    over = a >= float(over_threshold)
    clarity = (~ignored) & (~over)

    ignored_px = int(np.sum(ignored))
    clarity_px = int(np.sum(clarity))
    over_px = int(np.sum(over))

    return {
        "ignored": {"ratio": float(ignored_px / total) if total > 0 else 0.0, "pixels": ignored_px},
        "clarity": {"ratio": float(clarity_px / total) if total > 0 else 0.0, "pixels": clarity_px},
        "over_attention": {"ratio": float(over_px / total) if total > 0 else 0.0, "pixels": over_px},
    }


def _attention_points(att: np.ndarray, *, q: float, max_points: int) -> List[Dict[str, Any]]:
    a = _normalize_01(att)
    h, w = a.shape

    total_mass = float(np.sum(a))
    if total_mass <= 1e-9:
        return []

    try:
        thr = float(np.quantile(a, float(q)))
    except Exception:
        thr = 0.0

    if not (thr > 0.0):
        thr = float(np.max(a) * 0.85)
    if not (thr > 0.0):
        return []

    mask = (a >= float(thr)).astype(np.uint8) * 255
    cv2 = _maybe_cv2()
    points: List[Dict[str, Any]] = []

    if cv2 is not None:
        try:
            num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for lab in range(1, int(num)):
                x, y, bw, bh, area = [int(v) for v in stats[lab].tolist()]
                if area <= 0:
                    continue
                patch = a[y : y + bh, x : x + bw]
                if patch.size == 0:
                    continue
                iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
                px = int(x + int(ix))
                py = int(y + int(iy))
                mass = float(np.sum(patch))
                peak = float(np.max(patch))
                points.append(
                    {
                        "id": f"p{len(points) + 1}",
                        "x_px": int(px),
                        "y_px": int(py),
                        "x_norm": float(px / float(max(w, 1))),
                        "y_norm": float(py / float(max(h, 1))),
                        "mass_share": float(mass / total_mass) if total_mass > 0 else 0.0,
                        "peak_value": float(peak),
                        "bbox": [int(x), int(y), int(bw), int(bh)],
                    }
                )
        except Exception:
            points = []

    if not points:
        flat = a.reshape(-1)
        idxs = np.argsort(flat)[::-1]
        min_dist = int(round(0.08 * float(min(h, w))))
        if min_dist < 6:
            min_dist = 6
        chosen: List[Tuple[int, int]] = []
        for idx in idxs[: min(int(max_points) * 50, idxs.size)]:
            py = int(idx // w)
            px = int(idx % w)
            if a[py, px] < float(thr):
                break
            ok = True
            for cx, cy in chosen:
                if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= min_dist * min_dist:
                    ok = False
                    break
            if not ok:
                continue
            chosen.append((px, py))
            x0 = max(0, px - min_dist)
            y0 = max(0, py - min_dist)
            x1 = min(w, px + min_dist)
            y1 = min(h, py + min_dist)
            patch = a[y0:y1, x0:x1]
            mass = float(np.sum(patch))
            peak = float(a[py, px])
            points.append(
                {
                    "id": f"p{len(points) + 1}",
                    "x_px": int(px),
                    "y_px": int(py),
                    "x_norm": float(px / float(max(w, 1))),
                    "y_norm": float(py / float(max(h, 1))),
                    "mass_share": float(mass / total_mass) if total_mass > 0 else 0.0,
                    "peak_value": float(peak),
                    "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                }
            )
            if len(points) >= int(max_points):
                break

    points.sort(key=lambda p: float(p.get("peak_value", 0.0) or 0.0), reverse=True)
    return points[: max(0, int(max_points))]


def _compute_element_weights(elements: List[Dict[str, Any]]) -> List[float]:
    raw: List[float] = []
    for el in elements:
        if not isinstance(el, dict):
            raw.append(0.0)
            continue
        importance = float(el.get("importance_score", 0.0) or 0.0)
        sal = float(el.get("saliency_score", 0.0) or 0.0)
        cta = float(el.get("cta_score", 0.0) or 0.0)
        area = float(el.get("area_ratio", 0.0) or 0.0)
        base = 0.55 * importance + 0.25 * sal + 0.20 * cta
        base = base * (0.85 + 0.30 * _clamp01(area / 0.08))
        raw.append(float(max(0.0, base)))

    s = float(sum(raw))
    if s <= 1e-9:
        n = int(len(elements))
        if n <= 0:
            return []
        return [float(1.0 / float(n)) for _ in range(n)]
    return [float(x / s) for x in raw]


def _top_elements(elements: List[Dict[str, Any]], element_weights: List[float], *, k: int) -> List[Dict[str, Any]]:
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for el, wt in zip(elements, element_weights):
        if not isinstance(el, dict):
            continue
        ranked.append((float(wt), el))
    ranked.sort(key=lambda t: t[0], reverse=True)

    out: List[Dict[str, Any]] = []
    for wt, el in ranked[: max(0, int(k))]:
        out.append(
            {
                "id": str(el.get("id") or ""),
                "bbox": el.get("bbox"),
                "type": el.get("type"),
                "weight": float(wt),
                "cta_score": float(el.get("cta_score", 0.0) or 0.0),
                "saliency_score": float(el.get("saliency_score", 0.0) or 0.0),
            }
        )
    return out


def _downsample(att: np.ndarray, *, max_side: int) -> np.ndarray:
    a = att.astype(np.float32)
    h, w = a.shape
    if max(h, w) <= int(max_side):
        return a

    scale = float(max_side) / float(max(h, w))
    nh = max(1, int(round(float(h) * scale)))
    nw = max(1, int(round(float(w) * scale)))

    cv2 = _maybe_cv2()
    if cv2 is not None:
        try:
            return cv2.resize(a, (int(nw), int(nh)), interpolation=cv2.INTER_AREA).astype(np.float32)
        except Exception:
            pass

    step_y = max(1, int(round(float(h) / float(nh))))
    step_x = max(1, int(round(float(w) / float(nw))))
    return a[::step_y, ::step_x].astype(np.float32)


def _normalized_entropy_from_array(att: np.ndarray) -> float:
    w = np.clip(att.astype(np.float32).reshape(-1), 0.0, None)
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


def _normalized_entropy_from_weights(weights: List[float]) -> float:
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


def _attention_center(att: np.ndarray) -> Tuple[float, float]:
    a = _normalize_01(att)
    h, w = a.shape
    total = float(np.sum(a))
    if total <= 1e-9:
        return 0.5, 0.4

    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    cx = float(np.sum(xs * a) / total) / float(max(w - 1, 1))
    cy = float(np.sum(ys * a) / total) / float(max(h - 1, 1))
    return _clamp01(cx), _clamp01(cy)


def _top_mass_ratio(att: np.ndarray, *, q: float) -> float:
    a = _normalize_01(att)
    total = float(np.sum(a))
    if total <= 1e-9:
        return 0.0
    try:
        thr = float(np.quantile(a, float(q)))
    except Exception:
        thr = float(np.max(a) * 0.90)
    top = float(np.sum(a[a >= thr]))
    return float(_clamp01(top / total))


def _gini(values: np.ndarray) -> float:
    x = np.clip(values.astype(np.float32).reshape(-1), 0.0, None)
    n = int(x.size)
    if n <= 0:
        return 0.0
    s = float(np.sum(x))
    if s <= 1e-9:
        return 0.0
    x_sorted = np.sort(x)
    cum = np.cumsum(x_sorted)
    g = (float(n + 1) - 2.0 * float(np.sum(cum)) / float(cum[-1])) / float(n)
    return float(_clamp01(g))


def _peak_fragmentation_dominance(att: np.ndarray, *, q: float) -> Tuple[int, float, float]:
    a = _normalize_01(att)
    total = float(np.sum(a))
    if total <= 1e-9:
        return 0, 0.0, 0.0

    try:
        thr = float(np.quantile(a, float(q)))
    except Exception:
        thr = float(np.max(a) * 0.90)

    if not (thr > 0.0):
        return 0, 0.0, 0.0

    mask = (a >= float(thr)).astype(np.uint8)
    cv2 = _maybe_cv2()
    if cv2 is not None:
        try:
            num, labels = cv2.connectedComponents(mask, connectivity=8)
            comps = int(num) - 1
            if comps <= 0:
                return 0, 0.0, 0.0

            masses: List[float] = []
            for lab in range(1, int(num)):
                m = float(np.sum(a[labels == lab]))
                masses.append(float(m / total) if total > 0 else 0.0)

            dominance = float(max(masses)) if masses else 0.0
            fragmentation = float(_clamp01(float(max(comps - 1, 0)) / 6.0))
            return int(comps), float(_clamp01(dominance)), fragmentation
        except Exception:
            pass

    flat = a.reshape(-1)
    idxs = np.argsort(flat)[::-1]
    h, w = a.shape
    min_dist = int(round(0.10 * float(min(h, w))))
    if min_dist < 6:
        min_dist = 6
    chosen: List[Tuple[int, int]] = []
    masses: List[float] = []
    for idx in idxs[: min(250, idxs.size)]:
        py = int(idx // w)
        px = int(idx % w)
        if a[py, px] < float(thr):
            break
        ok = True
        for cx, cy in chosen:
            if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= min_dist * min_dist:
                ok = False
                break
        if not ok:
            continue
        chosen.append((px, py))
        x0 = max(0, px - min_dist)
        y0 = max(0, py - min_dist)
        x1 = min(w, px + min_dist)
        y1 = min(h, py + min_dist)
        masses.append(float(np.sum(a[y0:y1, x0:x1])) / total)
        if len(chosen) >= 7:
            break

    comps = int(len(chosen))
    dominance = float(max(masses)) if masses else 0.0
    fragmentation = float(_clamp01(float(max(comps - 1, 0)) / 6.0))
    return comps, float(_clamp01(dominance)), fragmentation


def _attention_decay(att: np.ndarray) -> float:
    a = _normalize_01(att)
    peak = float(np.max(a))
    if peak <= 1e-9:
        return 0.0
    mean = float(np.mean(a))
    return float(_clamp01(1.0 - (mean / peak)))


def _symmetry(att: np.ndarray) -> Tuple[float, float]:
    a = _normalize_01(att)
    if a.size <= 0:
        return 0.0, 0.0
    diff_x = float(np.mean(np.abs(a - a[:, ::-1])))
    diff_y = float(np.mean(np.abs(a - a[::-1, :])))
    return float(_clamp01(1.0 - diff_x)), float(_clamp01(1.0 - diff_y))


def _to_gray_u8(image_rgb: np.ndarray) -> np.ndarray:
    image_f = image_rgb.astype(np.float32)
    gray = (0.299 * image_f[:, :, 0] + 0.587 * image_f[:, :, 1] + 0.114 * image_f[:, :, 2]).astype(np.float32)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _canny_edges(gray_u8: np.ndarray) -> np.ndarray:
    cv2 = _maybe_cv2()
    if cv2 is not None:
        try:
            return cv2.Canny(gray_u8, 50, 150)
        except Exception:
            pass

    g = gray_u8.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
    gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
    mag = np.clip(gx + gy, 0.0, 255.0)
    return (mag > 30.0).astype(np.uint8) * 255


def _hierarchy_concentration(elements: List[Dict[str, Any]], element_weights: List[float]) -> float:
    if not elements or not element_weights:
        return 0.0
    s = float(sum(element_weights))
    if s <= 1e-9:
        return 0.0

    acc = 0.0
    for el, wt in zip(elements, element_weights):
        if not isinstance(el, dict):
            continue
        lvl = int(el.get("hierarchy_level", 0) or 0)
        lvl_norm = _clamp01(float(lvl) / 6.0)
        acc += float(wt) * (1.0 - lvl_norm)
    return float(_clamp01(acc / s))


def _intent_overlap(att: np.ndarray, *, elements: List[Dict[str, Any]], intent: Optional[str]) -> Optional[float]:
    if not intent:
        return None

    kind = _intent_kind(intent)
    if kind == "action":
        relevant = {"cta"}
    elif kind == "info":
        relevant = {"text_block", "cta"}
    elif kind == "trust":
        relevant = {"text_block"}
    else:
        relevant = {"cta", "text_block"}

    a = _normalize_01(att)
    h, w = a.shape
    total = float(np.sum(a))
    if total <= 1e-9:
        return 0.0

    mask = np.zeros((h, w), dtype=bool)
    for el in elements:
        if not isinstance(el, dict):
            continue
        if str(el.get("type") or "") not in relevant:
            continue
        bbox = el.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x, y, bw, bh = [int(v) for v in bbox]
        except Exception:
            continue
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        mask[y : y + bh, x : x + bw] = True

    if not bool(np.any(mask)):
        return 0.0

    overlap = float(np.sum(a[mask]))
    return float(_clamp01(overlap / total))


def _intent_alignment(intent: Optional[str], *, cta_weight_share: float, text_weight_share: float) -> Optional[float]:
    if not intent:
        return None
    kind = _intent_kind(intent)
    if kind == "action":
        return float(_clamp01(cta_weight_share))
    if kind in ("info", "trust"):
        return float(_clamp01(text_weight_share))
    return float(_clamp01(0.5 * float(cta_weight_share) + 0.5 * float(text_weight_share)))


def _estimate_text_area_ratio(gray_u8: np.ndarray) -> float:
    edges = _canny_edges(gray_u8)
    if edges.size <= 0:
        return 0.0
    density = float(np.mean(edges > 0))
    return float(_clamp01(density * 2.75))


def _colorfulness(image_rgb: np.ndarray) -> float:
    img = image_rgb.astype(np.float32)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))

    raw = math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(mean_rg * mean_rg + mean_yb * mean_yb)
    return float(_clamp01(raw / 100.0))


def _build_evaluation(
    *,
    intent: Optional[str],
    scores: Dict[str, Any],
    dominant_type: Optional[str],
    cta_weight_share: float,
    text_weight_share: float,
) -> str:
    focus = float(scores.get("focus", 0.0) or 0.0)
    clarity = float(scores.get("clarity", 0.0) or 0.0)
    alignment = scores.get("alignment")

    lines: List[str] = []

    if focus >= 0.66:
        lines.append("Attention is concentrated into a small number of hotspots.")
    elif focus >= 0.45:
        lines.append("Attention is moderately distributed across the interface.")
    else:
        lines.append("Attention is diffuse with multiple competing areas.")

    if clarity >= 0.66:
        lines.append("Visual hierarchy appears clear.")
    elif clarity >= 0.45:
        lines.append("Visual hierarchy is serviceable but may compete with secondary elements.")
    else:
        lines.append("Visual hierarchy appears weak or cluttered.")

    if intent:
        if isinstance(alignment, (int, float)):
            a = float(alignment)
            if a >= 0.66:
                lines.append("Attention aligns well with the stated intent.")
            elif a >= 0.45:
                lines.append("Attention partially overlaps intent-relevant regions.")
            else:
                lines.append("Attention is misaligned with the stated intent.")
        else:
            lines.append(
                f"Intent alignment proxy: cta_share={float(cta_weight_share):.2f}, text_share={float(text_weight_share):.2f}."
            )

    if dominant_type == "cta":
        lines.append("A call-to-action is visually dominant.")

    return " ".join([s for s in lines if str(s).strip()]).strip()


def _build_behavioral_index(
    *,
    image_rgb: np.ndarray,
    attention_map: np.ndarray,
    elements: List[Dict[str, Any]],
    element_weights: List[float],
    intent: Optional[str],
) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]

    att = _normalize_01(attention_map)
    att_ds = _downsample(att, max_side=96)

    saliency_entropy = _normalized_entropy_from_array(att_ds)
    saliency_focus = _clamp01(1.0 - saliency_entropy)

    center_x, center_y = _attention_center(att)
    center_dist = float(((center_x - 0.50) ** 2 + (center_y - 0.40) ** 2) ** 0.5)
    center_proximity = _clamp01(1.0 - (center_dist / 0.85))

    top10_mass = _top_mass_ratio(att_ds, q=0.90)
    gini = _gini(att_ds.reshape(-1))

    focal_points, dominance, fragmentation = _peak_fragmentation_dominance(att_ds, q=0.90)
    decay = _attention_decay(att_ds)

    sym_x, sym_y = _symmetry(att)

    gray_u8 = _to_gray_u8(image_rgb)
    edges_u8 = _canny_edges(gray_u8)
    edge_density = float(np.mean(edges_u8.astype(np.float32)) / 255.0) if edges_u8.size else 0.0
    dead_space_ratio = float(np.mean((att < 0.15) & (edges_u8 == 0))) if edges_u8.size else float(np.mean(att < 0.15))

    contrast = float(np.std(gray_u8.astype(np.float32)) / 255.0) if gray_u8.size else 0.0
    colorfulness = _colorfulness(image_rgb)

    element_count_norm = _clamp01(float(len(elements)) / 64.0)
    element_entropy = _normalized_entropy_from_weights(element_weights)
    top_element_weight = float(max(element_weights) if element_weights else 0.0)

    cta_count = sum(1 for el in elements if float(el.get("cta_score", 0.0) or 0.0) >= 0.62)
    cta_count_norm = _clamp01(float(cta_count) / 6.0)

    cta_area_ratio = float(
        sum(float(el.get("area_ratio", 0.0) or 0.0) for el in elements if str(el.get("type")) == "cta")
    )
    cta_area_ratio = _clamp01(cta_area_ratio)

    text_area_ratio = float(
        sum(float(el.get("area_ratio", 0.0) or 0.0) for el in elements if str(el.get("type")) == "text_block")
    )
    text_area_ratio = _clamp01(max(text_area_ratio, _estimate_text_area_ratio(gray_u8)))

    cta_weight_share = float(
        sum(wt for el, wt in zip(elements, element_weights) if str(el.get("type")) == "cta")
    )
    cta_weight_share = _clamp01(cta_weight_share)

    text_weight_share = float(
        sum(wt for el, wt in zip(elements, element_weights) if str(el.get("type")) == "text_block")
    )
    text_weight_share = _clamp01(text_weight_share)

    max_level = max((int(el.get("hierarchy_level", 0) or 0) for el in elements), default=0)
    hierarchy_depth_norm = _clamp01(float(max_level) / 6.0)
    hierarchy_concentration = _hierarchy_concentration(elements, element_weights)

    aspect = float(w) / float(max(h, 1))
    aspect_log_norm = _clamp01((math.log(max(aspect, 1e-6)) + 0.7) / 1.4)

    area = float(max(h * w, 1))
    log_area = math.log10(area)
    log_area_norm = _clamp01((log_area - 4.5) / 2.5)

    labels = [
        "aspect_log_norm",
        "log_area_norm",
        "edge_density",
        "contrast",
        "colorfulness",
        "saliency_entropy",
        "saliency_focus",
        "saliency_center_x",
        "saliency_center_y",
        "saliency_center_proximity",
        "saliency_top10_mass",
        "saliency_gini",
        "symmetry_x",
        "symmetry_y",
        "dead_space_ratio",
        "element_count_norm",
        "element_entropy",
        "top_element_weight",
        "cta_count_norm",
        "cta_area_ratio",
        "cta_weight_share",
        "text_area_ratio",
        "text_weight_share",
        "hierarchy_depth_norm",
        "hierarchy_concentration",
        "focus_fragmentation",
        "focus_dominance",
        "attention_decay",
    ]

    vector = [
        float(aspect_log_norm),
        float(log_area_norm),
        float(_clamp01(edge_density)),
        float(_clamp01(contrast)),
        float(_clamp01(colorfulness)),
        float(_clamp01(saliency_entropy)),
        float(_clamp01(saliency_focus)),
        float(_clamp01(center_x)),
        float(_clamp01(center_y)),
        float(_clamp01(center_proximity)),
        float(_clamp01(top10_mass)),
        float(_clamp01(gini)),
        float(_clamp01(sym_x)),
        float(_clamp01(sym_y)),
        float(_clamp01(dead_space_ratio)),
        float(_clamp01(element_count_norm)),
        float(_clamp01(element_entropy)),
        float(_clamp01(top_element_weight)),
        float(_clamp01(cta_count_norm)),
        float(_clamp01(cta_area_ratio)),
        float(_clamp01(cta_weight_share)),
        float(_clamp01(text_area_ratio)),
        float(_clamp01(text_weight_share)),
        float(_clamp01(hierarchy_depth_norm)),
        float(_clamp01(hierarchy_concentration)),
        float(_clamp01(fragmentation)),
        float(_clamp01(dominance)),
        float(_clamp01(decay)),
    ]

    focus = _clamp01(0.55 * saliency_focus + 0.45 * (1.0 - element_entropy))
    clutter = _clamp01(0.55 * element_count_norm + 0.45 * edge_density)
    clarity = _clamp01(0.45 * focus + 0.35 * (1.0 - clutter) + 0.20 * hierarchy_concentration)

    intent_overlap = _intent_overlap(att, elements=elements, intent=intent)
    intent_alignment = (
        float(intent_overlap)
        if intent_overlap is not None
        else _intent_alignment(intent, cta_weight_share=cta_weight_share, text_weight_share=text_weight_share)
    )

    if intent_alignment is None:
        overall = 100.0 * (0.55 * clarity + 0.45 * focus)
    else:
        overall = 100.0 * (0.45 * clarity + 0.35 * focus + 0.20 * float(intent_alignment))

    scores: Dict[str, Any] = {
        "focus": float(focus),
        "clarity": float(clarity),
        "alignment": intent_alignment,
        "focal_points": int(focal_points),
        "fragmentation": float(fragmentation),
        "dominance": float(dominance),
        "decay": float(decay),
        "intent_overlap": intent_overlap,
        "overall": float(overall),
    }

    dominant = None
    if elements and element_weights:
        best_i = int(max(range(len(elements)), key=lambda i: float(element_weights[i])))
        dominant = elements[best_i] if 0 <= best_i < len(elements) else None

    evaluation = _build_evaluation(
        intent=intent,
        scores=scores,
        dominant_type=str(dominant.get("type")) if isinstance(dominant, dict) else None,
        cta_weight_share=cta_weight_share,
        text_weight_share=text_weight_share,
    )

    return {
        "labels": labels,
        "vector": vector,
        "scores": scores,
        "evaluation": evaluation,
    }


def _build_visual_index(
    *,
    image_rgb: np.ndarray,
    elements: List[Dict[str, Any]],
    intent_kind: Optional[str],
) -> Dict[str, Any]:
    h, w = image_rgb.shape[:2]
    field = compute_perceptual_field(image_rgb, center_bias_strength=0.25, include_text_density=True)
    text_clusters = _extract_regions_from_map(
        field.get("text_density"),
        name="text",
        threshold=0.55,
        min_area_ratio=0.003,
        max_regions=8,
        morph_kernel=(17, 5),
    )
    contrast_regions = _extract_regions_from_map(
        field.get("contrast"),
        name="contrast",
        threshold=0.80,
        min_area_ratio=0.002,
        max_regions=8,
        morph_kernel=(11, 11),
    )
    return {
        "layout_regions": _layout_regions(elements, image_shape=(int(h), int(w)), max_regions=8),
        "text_clusters": text_clusters,
        "contrast_regions": contrast_regions,
        "minority_proxies": _minority_proxies(elements, intent_kind=intent_kind, max_items=10),
    }


def _extract_regions_from_map(
    arr: Any,
    *,
    name: str,
    threshold: float,
    min_area_ratio: float,
    max_regions: int,
    morph_kernel: Tuple[int, int],
) -> List[Dict[str, Any]]:
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return []
    cv2 = _maybe_cv2()
    if cv2 is None:
        return []

    a = np.clip(arr.astype(np.float32), 0.0, 1.0)
    h, w = a.shape
    mask = (a >= float(threshold)).astype(np.uint8) * 255
    kx, ky = morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(kx)), max(1, int(ky))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    min_area = float(min_area_ratio) * float(h * w)
    regions: List[Dict[str, Any]] = []
    for idx, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area < float(min_area):
            continue
        patch = a[int(y) : int(y + bh), int(x) : int(x + bw)]
        mean = float(np.mean(patch)) if patch.size else 0.0
        regions.append(
            {
                "id": f"{name}_{idx}",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "mean": float(mean),
                "area_ratio": float(area / float(max(h * w, 1))),
            }
        )

    regions.sort(key=lambda r: float(r.get("area_ratio", 0.0) or 0.0), reverse=True)
    return regions[: max(0, int(max_regions))]


def _layout_regions(
    elements: List[Dict[str, Any]],
    *,
    image_shape: Tuple[int, int],
    max_regions: int,
) -> List[Dict[str, Any]]:
    h, w = image_shape
    regions: List[Dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        bbox = el.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        area_ratio = float(el.get("area_ratio", 0.0) or 0.0)
        if area_ratio < 0.12 and str(el.get("type")) != "container":
            continue
        try:
            x, y, bw, bh = [int(v) for v in bbox]
        except Exception:
            continue

        x0 = float(x) / float(max(w, 1))
        y0 = float(y) / float(max(h, 1))
        x1 = float(x + bw) / float(max(w, 1))
        y1 = float(y + bh) / float(max(h, 1))

        kind = "region"
        if y0 <= 0.12 and (x1 - x0) >= 0.60:
            kind = "header"
        elif y1 >= 0.88 and (x1 - x0) >= 0.60:
            kind = "footer"
        elif x0 <= 0.12 and (y1 - y0) >= 0.45 and (x1 - x0) <= 0.45:
            kind = "left_sidebar"
        elif x1 >= 0.88 and (y1 - y0) >= 0.45 and (x1 - x0) <= 0.45:
            kind = "right_sidebar"
        elif y0 <= 0.45 and (x1 - x0) >= 0.60 and (y1 - y0) >= 0.22:
            kind = "hero"

        regions.append({"id": str(el.get("id")), "kind": kind, "bbox": bbox, "score": float(area_ratio)})

    regions.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
    return regions[: max(0, int(max_regions))]


def _minority_proxies(
    elements: List[Dict[str, Any]],
    *,
    intent_kind: Optional[str],
    max_items: int,
) -> List[Dict[str, Any]]:
    proxies: List[Dict[str, Any]] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        el_id = str(el.get("id") or "")
        el_type = str(el.get("type") or "")
        bbox = el.get("bbox")
        contrast = float(el.get("contrast", 0.0) or 0.0)
        area_ratio = float(el.get("area_ratio", 0.0) or 0.0)
        saliency = float(el.get("saliency_score", 0.0) or 0.0)
        cta = float(el.get("cta_score", 0.0) or 0.0)
        importance = float(el.get("importance_score", 0.0) or 0.0)

        proxy_type = None
        risk = 0.0
        if el_type == "cta" and contrast < 0.06 and cta >= 0.45:
            proxy_type = "low_contrast_cta"
            risk = _clamp01((0.06 - contrast) / 0.06) * _clamp01(cta)
            if intent_kind == "action":
                risk *= 1.15
        elif el_type == "text_block" and contrast < 0.05 and area_ratio >= 0.02:
            proxy_type = "low_contrast_text"
            risk = _clamp01((0.05 - contrast) / 0.05) * _clamp01(importance)
            if intent_kind in ("info", "trust"):
                risk *= 1.10
        elif el_type == "icon_or_control" and area_ratio <= 0.004 and saliency < 0.15:
            proxy_type = "low_saliency_control"
            risk = _clamp01((0.004 - area_ratio) / 0.004) * _clamp01(1.0 - saliency)

        if proxy_type and float(risk) > 0.0:
            proxies.append(
                {
                    "id": f"proxy_{len(proxies)}",
                    "type": proxy_type,
                    "element_id": el_id,
                    "bbox": bbox,
                    "risk_score": float(risk),
                    "saliency_score": float(saliency),
                    "cta_score": float(cta),
                }
            )

    proxies.sort(key=lambda r: float(r.get("risk_score", 0.0) or 0.0), reverse=True)
    return proxies[: max(0, int(max_items))]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def get_cache_paths(cache_key: str, *, base_dir: Path) -> Tuple[Path, Path]:
    cache_dir = (base_dir / "Backend" / "cache" / "plura_runs").resolve()
    return cache_dir / f"{cache_key}.json", cache_dir / f"{cache_key}.png"


def get_export_paths(cache_key: str, *, base_dir: Path) -> Tuple[Path, Path]:
    cache_dir = (base_dir / "Backend" / "cache" / "plura_runs").resolve()
    return cache_dir / f"{cache_key}.md", cache_dir / f"{cache_key}.xml"


def get_original_image_path(cache_key: str, *, base_dir: Path) -> Optional[Path]:
    cache_dir = (base_dir / "Backend" / "cache" / "plura_runs").resolve()
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        candidate = cache_dir / f"{cache_key}.original{ext}"
        if candidate.exists():
            return candidate
    fallback = cache_dir / f"{cache_key}.original"
    return fallback if fallback.exists() else None


def _infer_image_extension(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if len(image_bytes) >= 12 and image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _maybe_save_original_image(*, cache_key: str, base_dir: Path, image_bytes: bytes) -> Optional[Path]:
    if not cache_key or not image_bytes:
        return None
    existing = get_original_image_path(cache_key, base_dir=base_dir)
    if existing is not None:
        return existing
    cache_dir = (base_dir / "Backend" / "cache" / "plura_runs").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = _infer_image_extension(image_bytes)
    out_path = cache_dir / f"{cache_key}.original{ext}"
    with open(out_path, "wb") as f:
        f.write(image_bytes)
    return out_path


def _save_cache(*, json_path: Path, png_path: Path, payload: Dict[str, Any], heatmap_png: bytes) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    with open(png_path, "wb") as f:
        f.write(heatmap_png)


def _save_frame_heatmaps(*, cache_key: str, base_dir: Path, frame_heatmaps: Dict[str, bytes]) -> None:
    if not frame_heatmaps:
        return
    cache_dir = (base_dir / "Backend" / "cache" / "plura_runs").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for frame, png in frame_heatmaps.items():
        if not png:
            continue
        name = str(frame).strip().lower()
        if name not in ("immediate", "scanning"):
            continue
        out_path = cache_dir / f"{cache_key}.{name}.png"
        with open(out_path, "wb") as f:
            f.write(png)


def _write_export_files(*, md_path: Path, xml_path: Path, markdown: str, xml: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(str(markdown or "") + "\n", encoding="utf-8")
    xml_path.write_text(str(xml or "") + "\n", encoding="utf-8")


def _try_load_cached(*, json_path: Path, png_path: Path, ttl_seconds: int) -> Optional[Dict[str, Any]]:
    try:
        if not json_path.exists() or not png_path.exists():
            return None

        age = (int(json_path.stat().st_mtime))
        now = int(_now_seconds())
        if ttl_seconds > 0 and (now - age) > int(ttl_seconds):
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        data["success"] = True
        return data

    except Exception:
        return None


def _now_seconds() -> float:
    import time

    return time.time()


def _normalize_image_rgb(image_rgb: np.ndarray, *, canonical_width: int = 1440) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    img = image_rgb
    if img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0.0, 255.0).astype(np.uint8)

    img = _resize_rgb_to_width(img, width=int(canonical_width))
    img = _normalize_luminance_rgb_u8(img)
    return img


def _resize_rgb_to_width(image_rgb_u8: np.ndarray, *, width: int) -> np.ndarray:
    if width <= 0:
        return image_rgb_u8

    h, w = image_rgb_u8.shape[:2]
    if h <= 0 or w <= 0:
        return image_rgb_u8
    if int(w) == int(width):
        return image_rgb_u8

    scale = float(width) / float(w)
    nh = max(1, int(round(float(h) * scale)))

    cv2 = _maybe_cv2()
    if cv2 is not None:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(image_rgb_u8, (int(width), int(nh)), interpolation=interp).astype(np.uint8)

    pil = _maybe_pil()
    if pil is not None:
        Image = pil
        img = Image.fromarray(image_rgb_u8)
        resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
        img = img.resize((int(width), int(nh)), resample=resample)
        return np.array(img, dtype=np.uint8)

    return image_rgb_u8


def _normalize_luminance_rgb_u8(
    image_rgb_u8: np.ndarray,
    *,
    target_mean: float = 128.0,
    target_std: float = 64.0,
    min_gain: float = 0.5,
    max_gain: float = 2.5,
) -> np.ndarray:
    cv2 = _maybe_cv2()
    if cv2 is not None:
        ycrcb = cv2.cvtColor(image_rgb_u8, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_f = y.astype(np.float32)

        mean = float(np.mean(y_f))
        std = float(np.std(y_f))
        if std <= 1e-6:
            return image_rgb_u8

        gain = float(target_std) / float(std)
        gain = max(float(min_gain), min(float(max_gain), float(gain)))
        bias = float(target_mean) - float(gain) * float(mean)

        y_n = np.clip(y_f * float(gain) + float(bias), 0.0, 255.0).astype(np.uint8)
        ycrcb_n = cv2.merge((y_n, cr, cb))
        return cv2.cvtColor(ycrcb_n, cv2.COLOR_YCrCb2RGB).astype(np.uint8)

    img_f = image_rgb_u8.astype(np.float32)
    gray = (0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]).astype(np.float32)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if std <= 1e-6:
        return image_rgb_u8

    gain = float(target_std) / float(std)
    gain = max(float(min_gain), min(float(max_gain), float(gain)))
    bias = float(target_mean) - float(gain) * float(mean)
    out = np.clip(img_f * float(gain) + float(bias), 0.0, 255.0).astype(np.uint8)
    return out


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    pil = _maybe_pil()
    if pil is not None:
        try:
            from PIL import ImageOps  # type: ignore

            Image = pil
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                return np.array(img)
        except Exception:
            pass

    cv2 = _maybe_cv2()
    if cv2 is not None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image bytes")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    raise ImportError("To decode uploaded images, install opencv-python (or opencv-python-headless) or pillow")


def _maybe_cv2() -> Optional[object]:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def _maybe_pil() -> Optional[object]:
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:
        return None
