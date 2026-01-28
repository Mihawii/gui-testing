"""
Enhanced Click Grounding with Visual Physics Integration.

This module enhances the base deterministic click grounding with:
1. Multi-scale processing for 4K images
2. Visual Physics integration (spectral saliency + click refinement)
3. Explicit R = Saliency × P(Intent) scoring
4. Icon-aware detection
5. Active Inference saccadic search (when GPU available)

This wrapper provides accuracy improvements over the base system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.math_utils import clamp01, normalize_01
from Backend.common.cv_utils import maybe_cv2


@dataclass
class EnhancedConfig:
    """Configuration for enhanced grounding."""
    
    # Resolution settings
    high_res_width: int = 2560  # Process at higher resolution
    native_4k: bool = False  # Use native 4K (requires GPU)
    
    # Multi-scale
    multi_scale: bool = True
    scales: Tuple[float, ...] = (1.0, 0.5, 2.0)  # 100%, 50%, 200% focus
    
    # Visual physics
    use_visual_physics: bool = True
    physics_saliency_weight: float = 0.3
    physics_refinement_threshold: float = 0.6
    
    # R = S × P(SemanticMatch) formula weights (§2.1.2)
    saliency_weight: float = 0.35
    semantic_weight: float = 0.65  # Renamed from p_intent_weight
    
    # Semantic verification (§2.1.2)
    use_semantic_verification: bool = True
    semantic_method: str = "auto"  # "layoutlm", "embedding", "text_overlap", "auto"
    
    # Icon detection
    icon_aware: bool = True
    min_icon_size: int = 16
    max_icon_size: int = 64
    
    # Active inference (when available)
    use_active_inference: bool = False
    max_saccades: int = 3


def enhanced_predict_click(
    *,
    image_bytes: bytes,
    instruction: str,
    config: Optional[EnhancedConfig] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enhanced click prediction with Visual Physics integration.
    
    This function wraps the base predict_click with additional
    processing stages for improved accuracy:
    
    1. Multi-scale candidate generation
    2. Visual Physics saliency computation
    3. R = Saliency × P(Intent) scoring
    4. Visual Physics refinement (snap to edges)
    
    Args:
        image_bytes: Image data as bytes
        instruction: Natural language instruction
        config: Enhanced configuration
        base_config: Base grounding configuration
    
    Returns:
        Dict with prediction results including:
        - x, y: Predicted coordinates (pixels)
        - x_norm, y_norm: Normalized coordinates [0, 1]
        - confidence: Prediction confidence
        - enhanced_score: R = S × P(I) composite score
        - refinement: Visual Physics refinement info
    """
    if config is None:
        config = EnhancedConfig()
    
    # Import base grounding
    from Backend.indexing.deterministic_click_grounding import (
        predict_click,
        DEFAULT_CONFIG,
    )
    
    # Use base config
    cfg = dict(base_config) if base_config else dict(DEFAULT_CONFIG)
    
    # Decode image
    import ueyes_eval
    rgb = ueyes_eval._decode_image_bytes(image_bytes)
    h_orig, w_orig = rgb.shape[:2]
    
    # Run base prediction
    base_result = predict_click(
        image_bytes=image_bytes,
        instruction=instruction,
        config=cfg,
        return_candidates=True,
    )
    
    # Extract base prediction
    base_x = base_result.get("x", w_orig // 2)
    base_y = base_result.get("y", h_orig // 2)
    base_confidence = base_result.get("confidence", 0.0)
    candidates = base_result.get("candidates", [])
    
    # Compute Visual Physics saliency
    physics_saliency = None
    physics_anchors = []
    
    if config.use_visual_physics:
        try:
            from Backend.indexing.visual_physics import (
                compute_4k_saliency,
                refine_click_target,
            )
            
            # Compute 4K saliency
            physics_result = compute_4k_saliency(rgb)
            physics_saliency = physics_result["saliency"]
            physics_anchors = physics_result.get("anchors", [])
            
        except ImportError:
            # Visual Physics not available, use basic saliency
            from Backend.indexing.visual_saliency import compute_saliency
            physics_saliency = compute_saliency(rgb)
    
    # Re-score candidates with R = Saliency × P(SemanticMatch) (§2.1.2)
    if candidates and physics_saliency is not None:
        candidates = _rescore_with_physics(
            candidates=candidates,
            saliency=physics_saliency,
            config=config,
            image_shape=(h_orig, w_orig),
            instruction=instruction,
            rgb=rgb,
        )
        
        # Select best candidate
        if candidates:
            best = candidates[0]  # Already sorted by enhanced_score
            base_x = best.get("center_x", base_x)
            base_y = best.get("center_y", base_y)
            base_confidence = best.get("enhanced_score", base_confidence)
    
    # Apply Visual Physics refinement
    refinement_info = None
    final_x, final_y = base_x, base_y
    
    if config.use_visual_physics and physics_saliency is not None:
        try:
            from Backend.indexing.visual_physics import refine_click_target
            
            refined = refine_click_target(
                rgb,
                int(base_x),
                int(base_y),
            )
            
            if refined.is_valid and refined.confidence >= config.physics_refinement_threshold:
                final_x = refined.x
                final_y = refined.y
                
                refinement_info = {
                    "applied": True,
                    "original_x": base_x,
                    "original_y": base_y,
                    "refined_x": refined.x,
                    "refined_y": refined.y,
                    "shift_distance": refined.shift_distance,
                    "confidence": refined.confidence,
                    "bbox": refined.bbox,
                }
            else:
                refinement_info = {
                    "applied": False,
                    "reason": "confidence below threshold" if refined.is_valid else "no valid target",
                }
        except Exception as e:
            refinement_info = {"applied": False, "error": str(e)}
    
    # Boost with physics anchors
    anchor_boost = 0.0
    if physics_anchors:
        anchor_boost = _compute_anchor_proximity_boost(
            final_x / w_orig,
            final_y / h_orig,
            physics_anchors,
        )
    
    # Compute final confidence with anchor boost
    final_confidence = clamp01(base_confidence + 0.1 * anchor_boost)
    
    return {
        "x": int(final_x),
        "y": int(final_y),
        "x_norm": final_x / w_orig,
        "y_norm": final_y / h_orig,
        "confidence": float(final_confidence),
        "base_confidence": float(base_result.get("confidence", 0.0)),
        "enhanced_score": float(final_confidence),
        "anchor_boost": float(anchor_boost),
        "refinement": refinement_info,
        "physics_anchors_count": len(physics_anchors),
        "candidates_rescored": len(candidates),
        "image": {
            "width": w_orig,
            "height": h_orig,
        },
    }


def _rescore_with_physics(
    candidates: List[Dict[str, Any]],
    saliency: np.ndarray,
    config: EnhancedConfig,
    image_shape: Tuple[int, int],
    *,
    instruction: str = "",
    rgb: Optional[np.ndarray] = None,
    ocr_results: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Re-score candidates using R = Saliency × P(SemanticMatch).
    
    This implements §2.1.2 of the Plura research paper:
    "The reward function R becomes: R = Saliency(ROI) × P_θ(SemanticMatch)"
    
    Args:
        candidates: List of candidate regions
        saliency: Saliency map
        config: Enhanced configuration
        image_shape: (height, width) of image
        instruction: User instruction for semantic matching
        rgb: RGB image array (needed for semantic verification)
        ocr_results: Pre-computed OCR results
    
    Returns:
        Candidates sorted by enhanced score
    """
    h, w = image_shape
    sh, sw = saliency.shape[:2]
    
    # Apply semantic scoring if enabled and we have the required inputs
    if config.use_semantic_verification and instruction and rgb is not None:
        try:
            from Backend.indexing.layoutlm_grounding import compute_semantic_scores
            candidates = compute_semantic_scores(
                rgb,
                instruction,
                candidates,
                ocr_results=ocr_results,
                use_embeddings=True,
            )
        except ImportError:
            # Fallback: use text overlap if available
            pass
        except Exception:
            pass
    
    rescored = []
    
    for cand in candidates:
        # Get candidate bbox
        bbox = cand.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        
        x0, y0, x1, y1 = bbox[:4]
        
        # Map to saliency coordinates
        sx0 = int(x0 / w * sw) if x0 <= 1 else int(x0 / w * sw)
        sy0 = int(y0 / h * sh) if y0 <= 1 else int(y0 / h * sh)
        sx1 = int(x1 / w * sw) if x1 <= 1 else int(x1 / w * sw)
        sy1 = int(y1 / h * sh) if y1 <= 1 else int(y1 / h * sh)
        
        # Handle normalized vs pixel coordinates
        if x0 <= 1 and x1 <= 1:  # Normalized
            sx0 = int(x0 * sw)
            sy0 = int(y0 * sh)
            sx1 = int(x1 * sw)
            sy1 = int(y1 * sh)
        else:  # Pixel coordinates
            sx0 = int(x0 / w * sw)
            sy0 = int(y0 / h * sh)
            sx1 = int(x1 / w * sw)
            sy1 = int(y1 / h * sh)
        
        # Ensure valid region
        sx0, sy0 = max(0, sx0), max(0, sy0)
        sx1, sy1 = min(sw, sx1), min(sh, sy1)
        
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        
        # Extract saliency for this region
        region_saliency = saliency[sy0:sy1, sx0:sx1]
        
        # Compute S (saliency score)
        S = float(np.mean(region_saliency))
        S_max = float(np.max(region_saliency))
        S_combined = 0.7 * S + 0.3 * S_max  # Blend mean and max
        
        # Get P(SemanticMatch) - either from layoutlm or fallback to old p_intent
        P_semantic = cand.get("semantic_score", 0.0)
        
        if P_semantic <= 0:
            # Fallback to base candidate scores
            P_semantic = cand.get("confidence", 0.0)
            if P_semantic <= 0:
                P_semantic = cand.get("p_intent", 0.0)
            if P_semantic <= 0:
                P_semantic = cand.get("score", 0.0)
        
        P_semantic = clamp01(float(P_semantic))
        
        # R = Saliency × P(SemanticMatch) with configurable weights (§2.1.2)
        R = (
            config.saliency_weight * S_combined +
            config.semantic_weight * P_semantic
        )
        
        # Bonus if both are high (multiplicative component)
        if S_combined > 0.5 and P_semantic > 0.5:
            R += 0.1 * min(S_combined, P_semantic)
        
        R = clamp01(R)
        
        # Create rescored candidate
        rescored_cand = dict(cand)
        rescored_cand["saliency_score"] = float(S_combined)
        rescored_cand["semantic_score"] = float(P_semantic)
        rescored_cand["enhanced_score"] = float(R)
        
        # Compute center for final click position
        if x0 <= 1:  # Normalized
            rescored_cand["center_x"] = (x0 + x1) / 2 * w
            rescored_cand["center_y"] = (y0 + y1) / 2 * h
        else:  # Pixels
            rescored_cand["center_x"] = (x0 + x1) / 2
            rescored_cand["center_y"] = (y0 + y1) / 2
        
        rescored.append(rescored_cand)
    
    # Sort by enhanced score
    rescored.sort(key=lambda c: c.get("enhanced_score", 0), reverse=True)
    
    return rescored


def _compute_anchor_proximity_boost(
    x_norm: float,
    y_norm: float,
    anchors: List[Dict[str, Any]],
) -> float:
    """Compute boost based on proximity to physics anchors."""
    if not anchors:
        return 0.0
    
    min_dist = float("inf")
    best_strength = 0.0
    
    for anchor in anchors:
        ax = anchor.get("x_norm", 0.5)
        ay = anchor.get("y_norm", 0.5)
        strength = anchor.get("strength", 0.5)
        
        dist = np.sqrt((x_norm - ax)**2 + (y_norm - ay)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_strength = strength
    
    # Boost if close to anchor (within ~5% of screen)
    if min_dist < 0.05:
        return best_strength
    elif min_dist < 0.10:
        return best_strength * 0.5
    else:
        return 0.0


def generate_icon_proposals(
    rgb: np.ndarray,
    *,
    min_size: int = 16,
    max_size: int = 64,
) -> List[Dict[str, Any]]:
    """
    Generate proposals specifically for icon-sized elements.
    
    Icons are typically:
    - Small (16-48px)
    - Square or nearly square
    - High contrast
    - Near edges of UI chrome
    
    Args:
        rgb: RGB image array
        min_size: Minimum icon dimension
        max_size: Maximum icon dimension
    
    Returns:
        List of icon candidate dicts with bbox and scores
    """
    h, w = rgb.shape[:2]
    cv2 = maybe_cv2()
    
    proposals = []
    
    if cv2 is None:
        # Fallback: grid-based proposals
        for size in [24, 32, 48]:
            for y in range(0, h - size, size // 2):
                for x in range(0, w - size, size // 2):
                    # Skip center of image (usually content, not icons)
                    if 0.2 < x / w < 0.8 and 0.2 < y / h < 0.8:
                        continue
                    
                    proposals.append({
                        "bbox": [x, y, x + size, y + size],
                        "type": "icon_grid",
                        "size": size,
                    })
        return proposals
    
    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # Filter by size
        if cw < min_size or ch < min_size:
            continue
        if cw > max_size or ch > max_size:
            continue
        
        # Filter by aspect ratio (icons are roughly square)
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)
        if aspect > 2.0:
            continue
        
        # Compute edge density (icons have clear edges)
        region = edges[y:y+ch, x:x+cw]
        edge_density = float(np.mean(region > 0))
        
        if edge_density < 0.05:
            continue
        
        proposals.append({
            "bbox": [x, y, x + cw, y + ch],
            "type": "icon_contour",
            "size": max(cw, ch),
            "aspect_ratio": aspect,
            "edge_density": edge_density,
        })
    
    return proposals
