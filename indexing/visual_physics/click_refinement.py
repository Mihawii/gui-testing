"""
Click target refinement using visual physics.

This module provides deterministic anchoring of predicted click locations
to actual visual boundaries, preventing model hallucination.

The key insight is that valid click targets have physical properties:
    - They have visible edges (buttons have borders)
    - They have consistent color regions
    - They are at specific sizes (icons ~24-48px, buttons ~100-200px wide)

By enforcing these physical constraints, we ensure predictions land on
real, interactable elements rather than arbitrary locations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2
from Backend.common.math_utils import clamp01, normalize_01


@dataclass
class RefinementConfig:
    """Configuration for click refinement."""
    
    # Search radius
    max_search_radius: int = 50  # Max pixels to search from prediction
    
    # Element size constraints  
    min_element_size: int = 16  # Minimum valid target (tiny icons)
    max_element_size: int = 500  # Maximum valid target
    
    # Edge detection
    edge_threshold: float = 0.3  # Minimum edge strength
    
    # Scoring weights
    edge_weight: float = 0.4
    color_uniformity_weight: float = 0.3
    size_weight: float = 0.2
    center_bias_weight: float = 0.1


@dataclass
class RefinedTarget:
    """Result of click target refinement."""
    
    # Location (pixels)
    x: int
    y: int
    
    # Bounding box of detected element
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
    # Confidence and scoring
    confidence: float
    edge_score: float
    uniformity_score: float
    
    # Original prediction for reference
    original_x: int
    original_y: int
    shift_distance: float
    
    # Is this a valid target?
    is_valid: bool


def refine_click_target(
    image_rgb: np.ndarray,
    predicted_x: int,
    predicted_y: int,
    *,
    config: Optional[RefinementConfig] = None,
    saliency_map: Optional[np.ndarray] = None,
) -> RefinedTarget:
    """
    Refine a predicted click location to snap to actual UI element.
    
    This function takes a model's click prediction and adjusts it to
    land on a valid, visible element by:
    
    1. Detecting edges in a local region
    2. Finding enclosed regions (potential elements)
    3. Scoring candidates by edge strength, color uniformity, size
    4. Returning the best match with confidence score
    
    Args:
        image_rgb: RGB image of shape (H, W, 3)
        predicted_x: Predicted X coordinate
        predicted_y: Predicted Y coordinate
        config: Refinement configuration
        saliency_map: Optional saliency for weighting candidates
    
    Returns:
        RefinedTarget with adjusted location and confidence.
    
    Example:
        >>> refined = refine_click_target(image, pred_x, pred_y)
        >>> if refined.is_valid:
        ...     click(refined.x, refined.y)
    """
    if config is None:
        config = RefinementConfig()
    
    h, w = image_rgb.shape[:2]
    cv2 = maybe_cv2()
    
    # Clamp prediction to image bounds
    predicted_x = max(0, min(predicted_x, w - 1))
    predicted_y = max(0, min(predicted_y, h - 1))
    
    # Define search region
    r = config.max_search_radius
    x1 = max(0, predicted_x - r)
    y1 = max(0, predicted_y - r)
    x2 = min(w, predicted_x + r)
    y2 = min(h, predicted_y + r)
    
    # Extract local region
    region = image_rgb[y1:y2, x1:x2]
    region_h, region_w = region.shape[:2]
    
    if region_h < 10 or region_w < 10:
        # Region too small, return original
        return RefinedTarget(
            x=predicted_x, y=predicted_y,
            bbox=(predicted_x - 8, predicted_y - 8, 16, 16),
            confidence=0.5, edge_score=0.0, uniformity_score=0.0,
            original_x=predicted_x, original_y=predicted_y,
            shift_distance=0.0, is_valid=True,
        )
    
    # Convert to grayscale
    if cv2 is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    else:
        region_f = region.astype(np.float32)
        gray = (0.299 * region_f[:, :, 0] + 
                0.587 * region_f[:, :, 1] + 
                0.114 * region_f[:, :, 2]).astype(np.uint8)
    
    # Detect edges
    if cv2 is not None:
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
    else:
        # Simple gradient-based edge detection
        gy, gx = np.gradient(gray.astype(np.float32))
        edges = np.sqrt(gx**2 + gy**2)
        edges = normalize_01(edges)
        edges = (edges > 0.3).astype(np.float32)
    
    # Find potential element bounding boxes
    candidates = _find_element_candidates(
        edges=edges,
        gray=gray,
        config=config,
        cv2=cv2,
    )
    
    if not candidates:
        # No candidates found - return slightly adjusted prediction
        return RefinedTarget(
            x=predicted_x, y=predicted_y,
            bbox=(predicted_x - 16, predicted_y - 16, 32, 32),
            confidence=0.3, edge_score=0.0, uniformity_score=0.0,
            original_x=predicted_x, original_y=predicted_y,
            shift_distance=0.0, is_valid=False,
        )
    
    # Score candidates
    pred_local_x = predicted_x - x1
    pred_local_y = predicted_y - y1
    
    scored = []
    for candidate in candidates:
        cx, cy, cw, ch = candidate
        
        # Edge score: how strong are edges around this region
        edge_region = edges[cy:cy+ch, cx:cx+cw]
        edge_score = float(np.mean(edge_region)) if edge_region.size > 0 else 0.0
        
        # Uniformity score: how uniform is the color inside
        color_region = region[cy:cy+ch, cx:cx+cw]
        uniformity_score = _compute_uniformity(color_region)
        
        # Size score: prefer reasonable UI element sizes
        area = cw * ch
        if config.min_element_size**2 <= area <= config.max_element_size**2:
            size_score = 1.0
        else:
            size_score = 0.3
        
        # Center bias: prefer candidates containing the prediction
        center_x = cx + cw / 2
        center_y = cy + ch / 2
        dist = np.sqrt((center_x - pred_local_x)**2 + (center_y - pred_local_y)**2)
        center_score = 1.0 / (1.0 + dist / 20.0)
        
        # Total score
        total = (
            config.edge_weight * edge_score +
            config.color_uniformity_weight * uniformity_score +
            config.size_weight * size_score +
            config.center_bias_weight * center_score
        )
        
        # Check if prediction is inside candidate
        inside = (cx <= pred_local_x < cx + cw) and (cy <= pred_local_y < cy + ch)
        if inside:
            total *= 1.2  # Boost candidates containing prediction
        
        scored.append((total, candidate, edge_score, uniformity_score))
    
    # Select best candidate
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_candidate, edge_score, uniformity_score = scored[0]
    cx, cy, cw, ch = best_candidate
    
    # Convert back to global coordinates
    global_x = x1 + cx + cw // 2
    global_y = y1 + cy + ch // 2
    global_bbox = (x1 + cx, y1 + cy, cw, ch)
    
    shift_distance = np.sqrt((global_x - predicted_x)**2 + (global_y - predicted_y)**2)
    
    return RefinedTarget(
        x=global_x,
        y=global_y,
        bbox=global_bbox,
        confidence=clamp01(best_score),
        edge_score=edge_score,
        uniformity_score=uniformity_score,
        original_x=predicted_x,
        original_y=predicted_y,
        shift_distance=float(shift_distance),
        is_valid=True,
    )


def _find_element_candidates(
    edges: np.ndarray,
    gray: np.ndarray,
    config: RefinementConfig,
    cv2: Any,
) -> List[Tuple[int, int, int, int]]:
    """Find potential UI element bounding boxes from edges."""
    candidates = []
    
    if cv2 is not None:
        # Use contour detection
        edges_u8 = (edges * 255).astype(np.uint8)
        contours_result = cv2.findContours(edges_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if config.min_element_size**2 <= area <= config.max_element_size**2:
                candidates.append((x, y, w, h))
    else:
        # Simple fallback: grid-based region detection
        h, w = edges.shape
        grid_size = 24  # Check 24x24 regions
        
        for gy in range(0, h - grid_size, grid_size // 2):
            for gx in range(0, w - grid_size, grid_size // 2):
                region = edges[gy:gy+grid_size, gx:gx+grid_size]
                if np.mean(region) > config.edge_threshold:
                    candidates.append((gx, gy, grid_size, grid_size))
    
    return candidates


def _compute_uniformity(region: np.ndarray) -> float:
    """Compute color uniformity of a region (higher = more uniform)."""
    if region.size == 0:
        return 0.0
    
    # Compute standard deviation of each channel
    if region.ndim == 3:
        stds = [np.std(region[:, :, c]) for c in range(region.shape[2])]
        mean_std = np.mean(stds)
    else:
        mean_std = np.std(region)
    
    # Convert to uniformity (lower std = higher uniformity)
    return float(1.0 / (1.0 + mean_std / 30.0))


def snap_to_nearest_edge(
    image_rgb: np.ndarray,
    x: int,
    y: int,
    *,
    max_distance: int = 30,
) -> Tuple[int, int, float]:
    """
    Snap a point to the nearest visible edge.
    
    This is useful for ensuring clicks land on button borders
    or other visible element boundaries.
    
    Args:
        image_rgb: RGB image
        x, y: Current point
        max_distance: Maximum distance to search
    
    Returns:
        Tuple of (new_x, new_y, edge_strength)
    """
    h, w = image_rgb.shape[:2]
    cv2 = maybe_cv2()
    
    # Clamp to bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    # Define search region
    r = max_distance
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)
    
    region = image_rgb[y1:y2, x1:x2]
    
    # Compute edges
    if cv2 is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    else:
        region_f = region.astype(np.float32)
        gray = (0.299 * region_f[:, :, 0] + 
                0.587 * region_f[:, :, 1] + 
                0.114 * region_f[:, :, 2])
        gy, gx = np.gradient(gray)
        edges = np.sqrt(gx**2 + gy**2)
        edges = normalize_01(edges)
    
    # Find strongest edge point
    if edges.size == 0:
        return x, y, 0.0
    
    max_idx = np.argmax(edges)
    local_y, local_x = np.unravel_index(max_idx, edges.shape)
    
    new_x = x1 + local_x
    new_y = y1 + local_y
    edge_strength = float(edges[local_y, local_x])
    
    return int(new_x), int(new_y), edge_strength
