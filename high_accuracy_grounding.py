#!/usr/bin/env python3
"""
High-Accuracy Visual Grounding - Targeting 50%+ on ScreenSpot-Pro.

Key improvements over baseline:
1. Smarter OCR text matching with fuzzy matching and key phrase extraction
2. Spatial priors (toolbar, sidebar, corner icons)
3. Size-aware scoring (small icons vs buttons)
4. Multi-signal fusion with learned weights

Based on OcuMamba research: R = Saliency × P(SemanticMatch)
"""

from __future__ import annotations

import io
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ============================================================================
# TEXT MATCHING UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def extract_key_phrases(instruction: str) -> List[str]:
    """Extract key phrases from instruction for matching."""
    inst_lower = instruction.lower()
    
    # Pattern 1: quoted text (highest priority)
    quoted = re.findall(r'["\']([^"\']+)["\']', instruction)
    
    # Pattern 2: "X button/icon/tab/menu"
    ui_patterns = re.findall(
        r'(\b\w+(?:\s+\w+)?)\s+(?:button|icon|tab|menu|link|option|toggle|checkbox|slider)',
        inst_lower
    )
    
    # Pattern 3: after action verbs
    action_targets = re.findall(
        r'(?:click|tap|press|select|open|close|toggle|enable|disable|check|uncheck)\s+(?:on\s+)?(?:the\s+)?(["\']?[\w\s]+["\']?)',
        inst_lower
    )
    
    # Clean and combine
    phrases = []
    for p in quoted + ui_patterns + action_targets:
        p = normalize_text(p)
        if p and len(p) > 1 and p not in phrases:
            phrases.append(p)
    
    # Extract individual meaningful words as fallback
    stop_words = {
        'click', 'on', 'the', 'a', 'an', 'in', 'to', 'of', 'and', 'or', 'with',
        'for', 'this', 'that', 'it', 'is', 'at', 'by', 'from', 'button', 'icon',
        'select', 'open', 'close', 'press', 'tap'
    }
    words = [w for w in normalize_text(inst_lower).split() 
             if w not in stop_words and len(w) > 2]
    
    if not phrases and words:
        phrases = words[:3]
    
    return phrases[:5]


def fuzzy_match_score(query: str, text: str) -> float:
    """Compute fuzzy match score between query and text."""
    if not query or not text:
        return 0.0
    
    q = normalize_text(query)
    t = normalize_text(text)
    
    if not q or not t:
        return 0.0
    
    # Exact match (case insensitive) - highest priority
    if q == t:
        return 1.0
    
    # One is complete substring of other
    if q in t:
        # Query in text - good match
        return 0.95 if len(q) >= 3 else 0.7
    if t in q:
        # Text is part of query - partial match
        return 0.85 if len(t) >= 3 else 0.6
    
    # Word-level exact match
    q_words = set(q.split())
    t_words = set(t.split())
    
    if q_words and t_words:
        # Any exact word match
        common = q_words & t_words
        if common:
            # Boost for longer matching words
            max_len = max(len(w) for w in common)
            if max_len >= 4:
                return 0.9
            return 0.75
    
    # Sequence matcher ratio as fallback
    ratio = SequenceMatcher(None, q, t).ratio()
    
    return ratio * 0.8  # Cap fuzzy matches


def compute_text_match_score(instruction: str, ocr_text: str) -> float:
    """Compute overall text match score."""
    if not ocr_text:
        return 0.0
    
    ocr_normalized = normalize_text(ocr_text)
    inst_normalized = normalize_text(instruction)
    
    # Direct exact match check first
    if ocr_normalized and ocr_normalized in inst_normalized:
        return 1.0
    
    key_phrases = extract_key_phrases(instruction)
    if not key_phrases:
        return fuzzy_match_score(instruction, ocr_text)
    
    # Score against each key phrase
    scores = []
    for phrase in key_phrases:
        score = fuzzy_match_score(phrase, ocr_text)
        # Boost exact phrase matches
        if normalize_text(phrase) == ocr_normalized:
            score = 1.0
        scores.append(score)
    
    # Return max score (any phrase matching is good)
    return max(scores) if scores else 0.0


# ============================================================================
# SPATIAL PRIORS
# ============================================================================

def compute_spatial_prior(
    bbox_norm: List[float],
    instruction: str,
) -> float:
    """Compute spatial prior based on expected location."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0
    
    inst_lower = instruction.lower()
    prior = 0.5  # Base prior
    
    # Icon-related instructions favor edges
    icon_keywords = {
        'close', 'cancel', 'back', 'menu', 'settings', 'more', 'share', 
        'home', 'search', 'refresh', 'download', 'upload', 'delete',
        'minimize', 'maximize', 'expand', 'collapse', 'profile', 'notification'
    }
    is_icon_intent = any(kw in inst_lower for kw in icon_keywords)
    
    # Toolbar region (top 15%)
    if cy < 0.15:
        prior += 0.2 if is_icon_intent else 0.1
    
    # Bottom bar (bottom 15%)
    if cy > 0.85:
        prior += 0.15 if is_icon_intent else 0.05
    
    # Left sidebar (left 12%)
    if cx < 0.12:
        prior += 0.15 if is_icon_intent else 0.05
    
    # Right sidebar (right 12%)
    if cx > 0.88:
        prior += 0.15 if is_icon_intent else 0.05
    
    # Small elements are more likely to be icons
    if width < 0.05 and height < 0.05:
        prior += 0.1 if is_icon_intent else 0.0
    
    # Buttons typically have moderate size
    if 0.02 < width < 0.15 and 0.01 < height < 0.05:
        prior += 0.1
    
    return min(1.0, prior)


# ============================================================================
# OCR INTEGRATION
# ============================================================================

def run_ocr(image: Image.Image) -> List[Dict[str, Any]]:
    """Run OCR on image and return detected text boxes."""
    try:
        import easyocr
    except ImportError:
        return []
    
    # Initialize reader (cached globally)
    global _ocr_reader
    if '_ocr_reader' not in globals() or _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    
    # Convert to numpy
    img_np = np.array(image.convert('RGB'))
    
    # Run OCR
    results = _ocr_reader.readtext(img_np)
    
    detections = []
    h, w = img_np.shape[:2]
    
    for bbox, text, conf in results:
        # Convert polygon to rectangle
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xs), max(ys)
        
        # Normalize
        detections.append({
            'bbox': [x0, y0, x1, y1],
            'bbox_norm': [x0/w, y0/h, x1/w, y1/h],
            'text': text,
            'confidence': float(conf),
        })
    
    return detections


# ============================================================================
# MAIN GROUNDING FUNCTION
# ============================================================================

def predict_click_high_accuracy(
    image: Image.Image,
    instruction: str,
    *,
    text_weight: float = 0.6,
    spatial_weight: float = 0.25,
    confidence_weight: float = 0.15,
    min_text_score: float = 0.15,
) -> Dict[str, Any]:
    """
    High-accuracy click prediction using OCR + semantic matching + spatial priors.
    
    Returns:
        Dict with x, y (normalized 0-1), confidence, and metadata
    """
    w, h = image.size
    
    # Run OCR
    ocr_results = run_ocr(image)
    
    if not ocr_results:
        # Fallback to center
        return {
            'x': 0.5,
            'y': 0.5,
            'confidence': 0.0,
            'source': 'fallback',
            'reason': 'no_ocr_results',
        }
    
    # Score each OCR result
    candidates = []
    
    for det in ocr_results:
        text = det['text']
        bbox_norm = det['bbox_norm']
        ocr_conf = det['confidence']
        
        # Text matching score
        text_score = compute_text_match_score(instruction, text)
        if text_score < min_text_score:
            continue
        
        # Spatial prior
        spatial_score = compute_spatial_prior(bbox_norm, instruction)
        
        # Combined score
        combined = (
            text_weight * text_score +
            spatial_weight * spatial_score +
            confidence_weight * ocr_conf
        )
        
        cx = (bbox_norm[0] + bbox_norm[2]) / 2
        cy = (bbox_norm[1] + bbox_norm[3]) / 2
        
        candidates.append({
            'x': cx,
            'y': cy,
            'bbox_norm': bbox_norm,
            'text': text,
            'text_score': text_score,
            'spatial_score': spatial_score,
            'ocr_confidence': ocr_conf,
            'combined_score': combined,
        })
    
    if not candidates:
        return {
            'x': 0.5,
            'y': 0.5,
            'confidence': 0.0,
            'source': 'fallback',
            'reason': 'no_matching_text',
        }
    
    # Sort by combined score
    candidates.sort(key=lambda c: c['combined_score'], reverse=True)
    best = candidates[0]
    
    return {
        'x': best['x'],
        'y': best['y'],
        'confidence': best['combined_score'],
        'source': 'ocr',
        'text': best['text'],
        'text_score': best['text_score'],
        'candidates': len(candidates),
    }


# ============================================================================
# BENCHMARK
# ============================================================================

def run_benchmark(n_samples: int = 50) -> Dict[str, Any]:
    """Run benchmark on ScreenSpot-Pro."""
    from datasets import load_dataset
    
    print(f"Loading ScreenSpot-Pro (streaming, {n_samples} samples)...")
    ds = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test", streaming=True)
    
    hits = 0
    total = 0
    latencies = []
    near_misses = 0  # d < 0.1
    
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        
        img = ex.get("image")
        instr = str(ex.get("instruction") or "")
        bbox = ex.get("bbox")
        img_size = ex.get("img_size")
        
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        if isinstance(img_size, str):
            img_size = json.loads(img_size)
        
        if not bbox or not img_size or not img:
            continue
        
        # Normalize ground truth bbox
        w0, h0 = float(img_size[0]), float(img_size[1])
        gt_bbox = [
            float(bbox[0]) / w0,
            float(bbox[1]) / h0,
            float(bbox[2]) / w0,
            float(bbox[3]) / h0,
        ]
        
        t0 = time.time()
        result = predict_click_high_accuracy(img, instr)
        latency = time.time() - t0
        latencies.append(latency)
        
        px, py = result['x'], result['y']
        
        total += 1
        hit = gt_bbox[0] <= px <= gt_bbox[2] and gt_bbox[1] <= py <= gt_bbox[3]
        
        # Compute distance to center
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
        dist = np.sqrt((px - gt_cx)**2 + (py - gt_cy)**2)
        
        if dist < 0.1:
            near_misses += 1
        
        if hit:
            hits += 1
            status = f"HIT"
        else:
            status = f"MISS d={dist:.3f}"
        
        text_info = result.get('text', '')[:20] if result.get('text') else ''
        print(f"[{i+1:3d}] {status:15s} | {instr[:45]}... | '{text_info}'")
    
    accuracy = hits / total if total > 0 else 0
    
    print()
    print("=" * 70)
    print(f"ACCURACY: {hits}/{total} = {accuracy*100:.1f}%")
    print(f"Near-misses (d<0.1): {near_misses}")
    print(f"Mean latency: {np.mean(latencies):.2f}s")
    print("=" * 70)
    
    return {
        'hits': hits,
        'total': total,
        'accuracy': accuracy,
        'near_misses': near_misses,
        'mean_latency': float(np.mean(latencies)) if latencies else 0,
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(n)
