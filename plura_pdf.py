from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional


def _escape_paragraph_text(s: str) -> str:
    from xml.sax.saxutils import escape

    return escape(str(s or "")).replace("\n", "<br/>")


def _scaled_image(*, path: Path, max_width: float, max_height: float):
    from reportlab.platypus import Image

    try:
        from PIL import Image as PilImage

        with PilImage.open(path) as im:
            w, h = im.size
    except Exception:
        w, h = (1, 1)

    if w <= 0 or h <= 0:
        w, h = (1, 1)

    scale = min(float(max_width) / float(w), float(max_height) / float(h))
    scale = max(0.01, min(1.0, scale))

    return Image(str(path), width=float(w) * scale, height=float(h) * scale)


def generate_plura_pdf_report(
    *,
    cache_key: str,
    payload: Dict[str, Any],
    base_dir: Path,
    narrative_text: str = "",
    title: Optional[str] = None,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from Backend.plura_engine import get_cache_paths, get_original_image_path

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=str(title or "Plura Full Report"),
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PluraTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=14,
    )
    h_style = ParagraphStyle(
        "PluraHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "PluraBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#111827"),
    )

    story = []

    intent = payload.get("intent") if isinstance(payload.get("intent"), str) else ""
    overall = payload.get("metrics", {}).get("overall") if isinstance(payload.get("metrics"), dict) else None

    story.append(Paragraph(_escape_paragraph_text(str(title or "Plura Full Report")), title_style))
    meta_lines = [
        f"Cache key: {cache_key}",
        f"Intent: {intent or '—'}",
        f"Overall score: {overall if isinstance(overall, (int, float)) else '—'}",
    ]
    story.append(Paragraph(_escape_paragraph_text("\n".join(meta_lines)), body_style))
    story.append(Spacer(1, 10))

    original_path = get_original_image_path(cache_key, base_dir=base_dir)
    if original_path is not None and original_path.exists():
        story.append(Paragraph(_escape_paragraph_text("Original screenshot"), h_style))
        story.append(_scaled_image(path=original_path, max_width=doc.width, max_height=5.1 * inch))
        story.append(Spacer(1, 12))

    story.append(Paragraph(_escape_paragraph_text("Key metrics"), h_style))

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    rows = [
        ["Metric", "Value", "Plain meaning"],
        ["Focus", str(metrics.get("focus", "—")), "How concentrated attention is."],
        ["Clarity", str(metrics.get("clarity", "—")), "How clean the visual hierarchy appears."],
        ["Alignment", str(metrics.get("alignment", "—")), "Whether attention supports the intended action."],
        ["Fragmentation", str(metrics.get("fragmentation", "—")), "Whether users split attention across many areas."],
        ["Dominance", str(metrics.get("dominance", "—")), "How strongly one area dominates attention."],
        ["Decay", str(metrics.get("decay", "—")), "How quickly attention fades away from hotspots."],
        ["Intent overlap", str(metrics.get("intent_overlap", "—")), "Estimated overlap between intent and attention."],
        ["Hotspots", str(metrics.get("focal_points", "—")), "Number of distinct attention peaks."],
    ]

    tbl = Table(rows, colWidths=[doc.width * 0.22, doc.width * 0.18, doc.width * 0.60])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(tbl)

    story.append(PageBreak())

    story.append(Paragraph(_escape_paragraph_text("Heatmaps"), h_style))

    _json_path, orienting_heatmap_path = get_cache_paths(cache_key, base_dir=base_dir)
    immediate_path = orienting_heatmap_path.parent / f"{cache_key}.immediate.png"
    scanning_path = orienting_heatmap_path.parent / f"{cache_key}.scanning.png"

    def _add_heatmap(label: str, p: Path):
        if not p.exists():
            return
        story.append(Paragraph(_escape_paragraph_text(label), body_style))
        story.append(_scaled_image(path=p, max_width=doc.width, max_height=4.6 * inch))
        story.append(Spacer(1, 10))

    if immediate_path.exists() or orienting_heatmap_path.exists() or scanning_path.exists():
        _add_heatmap("Immediate heatmap", immediate_path)
        _add_heatmap("Orienting heatmap", orienting_heatmap_path)
        _add_heatmap("Scanning heatmap", scanning_path)

    legend = payload.get("heatmap_legend") if isinstance(payload.get("heatmap_legend"), dict) else {}
    if legend:
        legend_lines = [
            f"Ignored threshold: {legend.get('ignore_threshold', '—')}",
            f"Over-attention threshold: {legend.get('over_attention_threshold', '—')}",
            f"Overlay alpha: {legend.get('alpha', '—')}",
        ]
        story.append(Paragraph(_escape_paragraph_text("Legend"), h_style))
        story.append(Paragraph(_escape_paragraph_text("\n".join(legend_lines)), body_style))

    if narrative_text:
        story.append(PageBreak())
        story.append(Paragraph(_escape_paragraph_text("Narrative summary"), h_style))
        trimmed = str(narrative_text)
        if len(trimmed) > 15000:
            trimmed = trimmed[:15000] + "\n\n[Truncated for PDF.]"
        story.append(Paragraph(_escape_paragraph_text(trimmed), body_style))

    doc.build(story)
    return buf.getvalue()
