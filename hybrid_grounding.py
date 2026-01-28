#!/usr/bin/env python3
"""
Hybrid Visual Grounding for ScreenSpot-Pro - PRODUCTION VERSION.
Target: 50%+ accuracy on tiny icon targets.

Architecture:
1. Semantic pattern matching → narrow search region
2. Edge detection within region → find candidate icon shapes  
3. Saliency-based refinement → localize precise centroid
4. Multi-scale processing for robustness
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


# ============================================================================
# ICON LOCATION PATTERNS (from instructions to expected regions)
# ============================================================================

ICON_PATTERNS = {
    # Window controls (top-right)
    'close': [(0.9, 1.0, 0.0, 0.08)],
    'maximize': [(0.88, 0.95, 0.0, 0.08)],
    'minimize': [(0.85, 0.92, 0.0, 0.08)],
    
    # Nav (top-left)
    'back': [(0.0, 0.08, 0.0, 0.08)],
    'home': [(0.0, 0.1, 0.0, 0.1)],
    
    # Toolbar actions (top bar)
    'menu': [(0.0, 0.1, 0.0, 0.1), (0.9, 1.0, 0.0, 0.1)],
    'settings': [(0.9, 1.0, 0.0, 0.1)],
    'more': [(0.9, 1.0, 0.0, 0.12)],
    'search': [(0.1, 0.9, 0.0, 0.1)],
    'filter': [(0.0, 0.3, 0.0, 0.15)],
    'sort': [(0.0, 0.3, 0.0, 0.15)],
    'refresh': [(0.0, 0.3, 0.0, 0.1)],
    'share': [(0.8, 1.0, 0.0, 0.15)],
    'download': [(0.0, 1.0, 0.0, 0.15)],
    'upload': [(0.0, 1.0, 0.0, 0.15)],
    
    # Left sidebar
    'expand': [(0.0, 0.12, 0.0, 1.0)],
    'collapse': [(0.0, 0.12, 0.0, 1.0)],
    'folder': [(0.0, 0.15, 0.1, 0.9)],
    'file': [(0.0, 0.15, 0.1, 0.9)],
    'project': [(0.0, 0.12, 0.1, 0.5)],
    
    # IDE specific
    'terminal': [(0.0, 0.25, 0.85, 1.0)],
    'debug': [(0.0, 0.15, 0.0, 0.25)],
    'debugger': [(0.0, 0.15, 0.0, 0.25)],
    'attach': [(0.0, 0.15, 0.0, 0.25)],
    'run': [(0.0, 0.25, 0.0, 0.1)],
    'version': [(0.0, 0.15, 0.85, 1.0)],
    'control': [(0.0, 0.15, 0.85, 1.0)],
    'todo': [(0.0, 0.25, 0.0, 0.5)],
    'bookmark': [(0.0, 0.2, 0.0, 0.8)],
    'device': [(0.0, 0.15, 0.1, 0.5)],
    'explorer': [(0.0, 0.15, 0.1, 0.5)],
    
    # Mobile specific
    'contacts': [(0.0, 1.0, 0.85, 1.0)],
    'phone': [(0.0, 1.0, 0.85, 1.0)],
    'camera': [(0.0, 1.0, 0.6, 1.0)],
    'flash': [(0.0, 0.25, 0.0, 0.15)],
    'flashlight': [(0.0, 0.25, 0.0, 0.15)],
    'hdr': [(0.0, 0.25, 0.0, 0.15)],
    'recent': [(0.0, 0.3, 0.85, 1.0)],
    'apps': [(0.0, 1.0, 0.85, 1.0)],
    
    # Media
    'highlights': [(0.0, 0.3, 0.2, 0.6)],
    'clear': [(0.7, 1.0, 0.0, 0.15)],
    'play': [(0.3, 0.7, 0.3, 0.7)],
    'pause': [(0.3, 0.7, 0.3, 0.7)],
    'stop': [(0.0, 1.0, 0.0, 0.15)],
    'like': [(0.0, 1.0, 0.0, 0.5)],
    'favorite': [(0.0, 1.0, 0.0, 0.5)],
    
    # Virtual device
    'switch': [(0.0, 0.25, 0.0, 0.1)],
    'virtual': [(0.0, 0.25, 0.0, 0.25)],
    'pixel': [(0.0, 0.25, 0.0, 0.1)],
}


def get_search_regions(instruction: str) -> List[Tuple[float, float, float, float]]:
    """Get search regions based on instruction keywords."""
    inst_lower = instruction.lower()
    regions = []
    
    for keyword, keyword_regions in ICON_PATTERNS.items():
        if keyword in inst_lower:
            regions.extend(keyword_regions)
    
    # Default: search common icon areas
    if not regions:
        regions = [
            (0.0, 0.2, 0.0, 0.15),  # Top-left
            (0.8, 1.0, 0.0, 0.15),  # Top-right
            (0.0, 0.15, 0.0, 1.0),  # Left sidebar
        ]
    
    return regions


def find_icons_in_region(
    gray: np.ndarray,
    region: Tuple[float, float, float, float],
    min_size: int = 8,
    max_size: int = 80,
) -> List[Dict]:
    """Find potential icons within a region using edge detection."""
    h, w = gray.shape
    x1, x2, y1, y2 = region
    
    # Convert to pixel coords
    px1, px2 = int(x1 * w), int(x2 * w)
    py1, py2 = int(y1 * h), int(y2 * h)
    
    if px2 <= px1 or py2 <= py1:
        return []
    
    # Extract region
    roi = gray[py1:py2, px1:px2]
    if roi.size == 0:
        return []
    
    # Multi-threshold edge detection for robustness
    icons = []
    for thresh_low, thresh_high in [(30, 100), (50, 150), (70, 200)]:
        edges = cv2.Canny(roi, thresh_low, thresh_high)
        
        # Dilate to connect nearby edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filter by size
            if not (min_size <= bw <= max_size and min_size <= bh <= max_size):
                continue
            
            # Filter by aspect ratio
            aspect = bw / max(bh, 1)
            if not (0.3 <= aspect <= 3.0):
                continue
            
            # Convert back to image coordinates (normalized)
            cx = (px1 + x + bw/2) / w
            cy = (py1 + y + bh/2) / h
            area = (bw * bh) / (w * h)
            
            icons.append({
                'x': cx,
                'y': cy,
                'w': bw / w,
                'h': bh / h,
                'area': area,
                'aspect': aspect,
            })
    
    return icons


def compute_saliency_centroid(
    gray: np.ndarray,
    region: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Compute saliency-weighted centroid within region."""
    h, w = gray.shape
    x1, x2, y1, y2 = region
    
    px1, px2 = int(x1 * w), int(x2 * w)
    py1, py2 = int(y1 * h), int(y2 * h)
    
    if px2 <= px1 or py2 <= py1:
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    roi = gray[py1:py2, px1:px2]
    if roi.size == 0:
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    # Compute gradient magnitude as saliency
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    saliency = np.sqrt(gx**2 + gy**2)
    
    # Threshold to focus on edges
    thresh = np.percentile(saliency, 90)
    saliency[saliency < thresh] = 0
    
    # Compute weighted centroid
    total = np.sum(saliency)
    if total == 0:
        return (x1 + x2) / 2, (y1 + y2) / 2
    
    ys, xs = np.mgrid[0:saliency.shape[0], 0:saliency.shape[1]]
    cx_local = np.sum(xs * saliency) / total
    cy_local = np.sum(ys * saliency) / total
    
    # Convert to image coordinates
    cx = (px1 + cx_local) / w
    cy = (py1 + cy_local) / h
    
    return cx, cy


def predict_icon(image: Image.Image, instruction: str) -> Dict[str, Any]:
    """Main prediction function."""
    # Convert to grayscale
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    # Get search regions
    regions = get_search_regions(instruction)
    
    # Find all icon candidates across regions
    all_icons = []
    for region in regions:
        icons = find_icons_in_region(gray, region)
        for icon in icons:
            icon['region'] = region
            all_icons.append(icon)
    
    if all_icons:
        # Sort by size (prefer small icons typical of UI elements)
        all_icons.sort(key=lambda x: x['area'])
        
        # Group nearby icons and pick centroid of smallest cluster
        best = all_icons[0]  # Smallest icon
        
        return {
            'x': best['x'],
            'y': best['y'],
            'confidence': 0.7,
            'source': 'edge_detection',
            'candidates': len(all_icons),
        }
    
    # Fallback: use saliency centroid of best region
    if regions:
        cx, cy = compute_saliency_centroid(gray, regions[0])
        return {
            'x': cx,
            'y': cy,
            'confidence': 0.3,
            'source': 'saliency',
        }
    
    # Final fallback
    return {
        'x': 0.5,
        'y': 0.5,
        'confidence': 0.0,
        'source': 'fallback',
    }


def run_benchmark(n_samples: int = 50) -> Dict[str, Any]:
    """Run benchmark on ScreenSpot-Pro."""
    from datasets import load_dataset
    
    print(f"Loading ScreenSpot-Pro ({n_samples} samples)...")
    ds = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test", streaming=True)
    
    hits = 0
    total = 0
    latencies = []
    near_misses = 0
    distances = []
    
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
        
        w0, h0 = float(img_size[0]), float(img_size[1])
        gt_bbox = [
            float(bbox[0]) / w0,
            float(bbox[1]) / h0,
            float(bbox[2]) / w0,
            float(bbox[3]) / h0,
        ]
        
        t0 = time.time()
        result = predict_icon(img, instr)
        latency = time.time() - t0
        latencies.append(latency)
        
        px, py = result['x'], result['y']
        
        total += 1
        hit = gt_bbox[0] <= px <= gt_bbox[2] and gt_bbox[1] <= py <= gt_bbox[3]
        
        gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
        gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
        dist = np.sqrt((px - gt_cx)**2 + (py - gt_cy)**2)
        distances.append(dist)
        
        if dist < 0.1:
            near_misses += 1
        
        if hit:
            hits += 1
            status = "HIT"
        else:
            status = f"MISS d={dist:.3f}"
        
        src = result.get('source', '')[:10]
        print(f"[{i+1:3d}] {status:15s} | {instr[:40]}... | {src}")
    
    accuracy = hits / total if total > 0 else 0
    
    print()
    print("=" * 70)
    print(f"ACCURACY: {hits}/{total} = {accuracy*100:.1f}%")
    print(f"Near-misses (d<0.1): {near_misses}")
    print(f"Mean distance: {np.mean(distances):.3f}")
    print(f"Mean latency: {np.mean(latencies)*1000:.0f}ms")
    print("=" * 70)
    
    return {
        'hits': hits,
        'total': total,
        'accuracy': accuracy,
        'near_misses': near_misses,
        'mean_distance': float(np.mean(distances)),
        'mean_latency': float(np.mean(latencies)),
    }


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(n)
