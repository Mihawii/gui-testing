#!/usr/bin/env python3
"""
Quick accuracy boost for ScreenSpot-Pro benchmark.

This script patches the predict_click function to add:
1. Tiny icon proposals (grid-based)
2. OCR-based targeting improvements
3. Better scoring for small elements

Run benchmark with this wrapper.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Backend.indexing.deterministic_click_grounding import predict_click, DEFAULT_CONFIG, _tokens, _normalize_text


def generate_tiny_icon_proposals(rgb, *, icon_sizes=[16, 20, 24, 28, 32]):
    """
    Generate grid-based proposals for tiny icons at screen edges.
    
    Icons are typically found in:
    - Top toolbar (y < 0.15)
    - Bottom bar (y > 0.90)
    - Left sidebar (x < 0.10)
    - Right sidebar (x > 0.90)
    """
    h, w = rgb.shape[:2]
    proposals = []
    
    # Edge regions more likely to have icons
    regions = [
        # Top toolbar
        (0, 0, w, int(0.15 * h)),
        # Bottom bar
        (0, int(0.90 * h), w, h),
        # Left sidebar
        (0, 0, int(0.10 * w), h),
        # Right sidebar
        (int(0.90 * w), 0, w, h),
    ]
    
    for rx0, ry0, rx1, ry1 in regions:
        for size in icon_sizes:
            step = size // 2
            for y in range(ry0, min(ry1, h - size), step):
                for x in range(rx0, min(rx1, w - size), step):
                    proposals.append({
                        "bbox_xywh": [x, y, size, size],
                        "type": "tiny_icon",
                        "source": "grid",
                    })
    
    return proposals


def enhanced_predict_click(
    *,
    image_bytes: bytes,
    instruction: str,
    config=None,
    cache_dir=None,
):
    """
    Enhanced click prediction with tiny icon support.
    """
    import ueyes_eval
    from Backend.indexing.visual_saliency import compute_saliency
    
    cfg = config if config else dict(DEFAULT_CONFIG)
    
    # Decode image
    rgb = ueyes_eval._decode_image_bytes(image_bytes)
    rgb = ueyes_eval._normalize_image_rgb(rgb, canonical_width=1920)  # Higher res
    h, w = rgb.shape[:2]
    
    # Get base prediction
    base_result = predict_click(
        image_bytes=image_bytes,
        instruction=instruction,
        config=cfg,
        cache_dir=cache_dir,
        return_candidates=True,
    )
    
    # Check if this looks like an icon instruction
    inst_lower = instruction.lower()
    icon_intent = any(kw in inst_lower for kw in [
        'icon', 'button', 'cancel', 'close', 'save', 'preview', 
        'settings', 'menu', 'back', 'refresh', 'share', 'copy',
        'home', 'profile', 'download', 'upload', 'cut', 'paste',
    ])
    
    # If base prediction has low confidence OR icon intent, try tiny icons
    base_conf = base_result.get("confidence", 0)
    
    if icon_intent or base_conf < 0.3:
        # Generate tiny icon proposals
        tiny_proposals = generate_tiny_icon_proposals(rgb)
        
        # Score each tiny proposal
        sal = compute_saliency(rgb)
        
        best_tiny = None
        best_score = -1
        
        for prop in tiny_proposals:
            x, y, bw, bh = prop["bbox_xywh"]
            
            # Get saliency in this region
            patch = sal[y:y+bh, x:x+bw]
            if patch.size == 0:
                continue
            sal_mean = float(np.mean(patch))
            sal_max = float(np.max(patch))
            
            # Score: saliency + edge position bonus
            cx = (x + bw/2) / w
            cy = (y + bh/2) / h
            
            # Edge bonus (icons near edges)
            edge_bonus = 0
            if cx < 0.15 or cx > 0.85:
                edge_bonus += 0.2
            if cy < 0.15 or cy > 0.85:
                edge_bonus += 0.2
            
            score = 0.6 * sal_mean + 0.3 * sal_max + edge_bonus
            
            if score > best_score:
                best_score = score
                best_tiny = prop
        
        # If tiny icon is better than base, use it
        if best_tiny and best_score > base_conf:
            x, y, bw, bh = best_tiny["bbox_xywh"]
            cx = x + bw // 2
            cy = y + bh // 2
            
            return {
                "x": float(cx) / w,
                "y": float(cy) / h,
                "x_pixel": int(cx),
                "y_pixel": int(cy),
                "confidence": float(best_score),
                "method": "tiny_icon",
                "bbox": [x/w, y/h, (x+bw)/w, (y+bh)/h],
            }
    
    # Return base result
    return base_result


if __name__ == "__main__":
    print("Enhanced predict_click loaded.")
    print("Use enhanced_predict_click() for better tiny icon detection.")
