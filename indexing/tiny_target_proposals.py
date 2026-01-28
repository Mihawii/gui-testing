"""
Tiny Target Proposal Generation.

This module addresses the critical issue identified in ScreenSpot-Pro benchmarking:
- 84% of targets are <0.1% of screen (tiny icons)
- Current proposal generation misses these tiny targets
- Oracle ceiling is only 6% with current proposals

This module adds specialized proposal generation for:
1. Toolbar icons (top/bottom edges)
2. Sidebar icons (left/right edges)
3. Small buttons in dense UI regions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2
from Backend.common.math_utils import clamp01


def generate_edge_icon_proposals(
    rgb: np.ndarray,
    *,
    icon_sizes: Tuple[int, ...] = (16, 20, 24, 28, 32, 40, 48),
    edge_depth: float = 0.12,  # Search 12% from each edge
    step_ratio: float = 0.5,  # Overlap ratio between proposals
    min_contrast: float = 0.05,  # Minimum local contrast to consider
) -> List[Dict[str, Any]]:
    """
    Generate grid-based proposals for tiny icons at screen edges.
    
    Icons in professional UIs are typically found in:
    - Top toolbar (y < 12% of height)
    - Bottom status bar (y > 88% of height)  
    - Left sidebar (x < 12% of width)
    - Right sidebar/panel (x > 88% of width)
    
    Args:
        rgb: RGB image array
        icon_sizes: Icon dimensions to generate proposals for
        edge_depth: How far from edge to search (0.12 = 12%)
        step_ratio: Step size as ratio of icon size (smaller = more overlap)
        min_contrast: Minimum contrast to keep proposal
    
    Returns:
        List of icon proposal dicts with bbox and metadata
    """
    h, w = rgb.shape[:2]
    proposals = []
    
    # Define edge regions
    top_y = int(edge_depth * h)
    bottom_y = int((1 - edge_depth) * h)
    left_x = int(edge_depth * w)
    right_x = int((1 - edge_depth) * w)
    
    # Convert to grayscale for contrast computation
    cv2 = maybe_cv2()
    if cv2 is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = np.mean(rgb.astype(np.float32), axis=2) / 255.0
    
    # Edge regions to search: (x_start, y_start, x_end, y_end, region_name)
    regions = [
        (0, 0, w, top_y, "top_toolbar"),           # Top toolbar
        (0, bottom_y, w, h, "bottom_bar"),         # Bottom bar
        (0, top_y, left_x, bottom_y, "left_sidebar"),   # Left sidebar
        (right_x, top_y, w, bottom_y, "right_sidebar"), # Right sidebar
    ]
    
    for size in icon_sizes:
        step = max(int(size * step_ratio), 4)
        
        for rx0, ry0, rx1, ry1, region_name in regions:
            # Skip if region is too small for this icon size
            if rx1 - rx0 < size or ry1 - ry0 < size:
                continue
            
            for y in range(ry0, ry1 - size + 1, step):
                for x in range(rx0, rx1 - size + 1, step):
                    # Compute local contrast
                    patch = gray[y:y+size, x:x+size]
                    if patch.size == 0:
                        continue
                    
                    contrast = float(np.std(patch))
                    if contrast < min_contrast:
                        continue
                    
                    # Add proposal
                    proposals.append({
                        "bbox": [x, y, x + size, y + size],
                        "bbox_xywh": [x, y, size, size],
                        "type": "tiny_icon",
                        "source": "edge_grid",
                        "region": region_name,
                        "size": size,
                        "contrast": float(contrast),
                    })
    
    return proposals


def generate_high_contrast_proposals(
    rgb: np.ndarray,
    *,
    min_size: int = 12,
    max_size: int = 64,
    contrast_threshold: float = 0.15,
    max_proposals: int = 200,
) -> List[Dict[str, Any]]:
    """
    Generate proposals based on high-contrast regions.
    
    Icons typically have sharp edges and high local contrast.
    This method finds such regions without relying on connected components.
    
    Args:
        rgb: RGB image array
        min_size: Minimum icon dimension
        max_size: Maximum icon dimension
        contrast_threshold: Minimum edge response threshold
        max_proposals: Maximum number of proposals to return
    
    Returns:
        List of high-contrast proposal dicts
    """
    h, w = rgb.shape[:2]
    cv2 = maybe_cv2()
    
    if cv2 is None:
        # Fallback: return edge-based proposals only
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute edges with lower threshold to catch subtle icons
    edges = cv2.Canny(gray, 30, 100)
    
    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Filter by size
        if cw < min_size or ch < min_size:
            continue
        if cw > max_size or ch > max_size:
            continue
        
        # Compute aspect ratio (prefer square-ish icons)
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
        if aspect > 2.5:
            continue
        
        # Compute edge density
        region = edges[y:y+ch, x:x+cw]
        edge_density = float(np.mean(region > 0))
        
        if edge_density < 0.03:
            continue
        
        # Compute contrast
        gray_region = gray[y:y+ch, x:x+cw]
        contrast = float(np.std(gray_region.astype(np.float32) / 255.0))
        
        if contrast < contrast_threshold:
            continue
        
        proposals.append({
            "bbox": [x, y, x + cw, y + ch],
            "bbox_xywh": [x, y, cw, ch],
            "type": "high_contrast",
            "source": "contour",
            "size": max(cw, ch),
            "aspect_ratio": float(aspect),
            "edge_density": float(edge_density),
            "contrast": float(contrast),
        })
    
    # Sort by contrast and return top proposals
    proposals.sort(key=lambda p: p.get("contrast", 0), reverse=True)
    return proposals[:max_proposals]


def generate_corner_anchors(
    rgb: np.ndarray,
    *,
    corner_size: int = 80,
    icon_sizes: Tuple[int, ...] = (24, 32, 48),
) -> List[Dict[str, Any]]:
    """
    Generate proposals anchored at screen corners.
    
    Many UI icons are positioned at corners:
    - Top-left: Back button, menu icon
    - Top-right: Close, settings, share
    - Bottom-left/right: Navigation icons
    
    Args:
        rgb: RGB image array
        corner_size: Size of corner region to search
        icon_sizes: Icon dimensions to generate
    
    Returns:
        List of corner-anchored proposals
    """
    h, w = rgb.shape[:2]
    proposals = []
    
    # Corner regions
    corners = [
        (0, 0, "top_left"),
        (w - corner_size, 0, "top_right"),
        (0, h - corner_size, "bottom_left"),
        (w - corner_size, h - corner_size, "bottom_right"),
    ]
    
    for cx, cy, corner_name in corners:
        # Ensure valid bounds
        cx = max(0, min(cx, w - corner_size))
        cy = max(0, min(cy, h - corner_size))
        
        for size in icon_sizes:
            step = size // 2
            for y in range(cy, cy + corner_size - size, step):
                for x in range(cx, cx + corner_size - size, step):
                    proposals.append({
                        "bbox": [x, y, x + size, y + size],
                        "bbox_xywh": [x, y, size, size],
                        "type": "corner_icon",
                        "source": "corner_grid",
                        "corner": corner_name,
                        "size": size,
                    })
    
    return proposals


def generate_tiny_target_proposals(
    rgb: np.ndarray,
    *,
    instruction: str = "",
    include_edge_icons: bool = True,
    include_high_contrast: bool = True,
    include_corners: bool = True,
    max_total_proposals: int = 500,
) -> List[Dict[str, Any]]:
    """
    Main entry point for tiny target proposal generation.
    
    Combines multiple proposal strategies to maximize coverage
    of tiny UI elements that standard proposal generation misses.
    
    Args:
        rgb: RGB image array
        instruction: User instruction (used to prioritize regions)
        include_edge_icons: Whether to include edge-based proposals
        include_high_contrast: Whether to include contrast-based proposals
        include_corners: Whether to include corner-anchored proposals
        max_total_proposals: Maximum total proposals to return
    
    Returns:
        Combined list of proposals from all strategies
    """
    all_proposals = []
    
    # 1. Edge-based icon proposals (most important for toolbars)
    if include_edge_icons:
        edge_props = generate_edge_icon_proposals(rgb)
        all_proposals.extend(edge_props)
    
    # 2. High-contrast contour proposals
    if include_high_contrast:
        contrast_props = generate_high_contrast_proposals(rgb)
        all_proposals.extend(contrast_props)
    
    # 3. Corner-anchored proposals
    if include_corners:
        corner_props = generate_corner_anchors(rgb)
        all_proposals.extend(corner_props)
    
    # Prioritize based on instruction if provided
    if instruction:
        inst_lower = instruction.lower()
        
        # Boost toolbar icons if instruction mentions common actions
        toolbar_keywords = ["cancel", "close", "save", "back", "menu", "settings", 
                          "refresh", "share", "search", "home", "profile"]
        is_toolbar_intent = any(kw in inst_lower for kw in toolbar_keywords)
        
        if is_toolbar_intent:
            # Sort toolbar/edge proposals first
            def priority(p):
                region = p.get("region", "")
                if region in ("top_toolbar", "bottom_bar"):
                    return 2
                if p.get("type") == "corner_icon":
                    return 1
                return 0
            
            all_proposals.sort(key=priority, reverse=True)
    
    # Limit total proposals
    if len(all_proposals) > max_total_proposals:
        all_proposals = all_proposals[:max_total_proposals]
    
    return all_proposals
