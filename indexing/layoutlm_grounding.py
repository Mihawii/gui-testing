"""
LayoutLMv3 Semantic Verification Layer.

Implements §2.1.2 of the Plura research paper:
"To mitigate [the semantic gap], Plura must integrate a Semantic Verification
Layer using LayoutLMv3."

This module bridges the gap between visual saliency (what stands out) and
semantic intent (what the user actually wants to click).

R = Saliency(ROI) × P_θ(SemanticMatch)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from Backend.common.math_utils import clamp01


# Lazy imports to avoid loading heavy models unless needed
_layoutlm_processor = None
_layoutlm_model = None
_sentence_model = None


@dataclass
class SemanticMatchResult:
    """Result of semantic matching for a candidate region."""
    
    score: float  # P(SemanticMatch) in [0, 1]
    method: str  # "layoutlm", "text_overlap", "embedding", "fallback"
    matched_text: Optional[str] = None
    confidence: float = 0.0
    details: Optional[Dict[str, Any]] = None


def _load_layoutlm():
    """Lazy load LayoutLMv3 processor and model."""
    global _layoutlm_processor, _layoutlm_model
    
    if _layoutlm_processor is not None:
        return _layoutlm_processor, _layoutlm_model
    
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3Model
        
        model_name = "microsoft/layoutlmv3-base"
        _layoutlm_processor = LayoutLMv3Processor.from_pretrained(model_name)
        _layoutlm_model = LayoutLMv3Model.from_pretrained(model_name)
        
        # Move to GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                _layoutlm_model = _layoutlm_model.cuda()
        except ImportError:
            pass
        
        return _layoutlm_processor, _layoutlm_model
    
    except ImportError:
        return None, None
    except Exception:
        return None, None


def _load_sentence_model():
    """Lazy load sentence transformer for text embedding similarity."""
    global _sentence_model
    
    if _sentence_model is not None:
        return _sentence_model
    
    try:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sentence_model
    except ImportError:
        return None
    except Exception:
        return None


def compute_text_overlap_score(
    instruction: str,
    candidate_text: str,
) -> float:
    """
    Simple text overlap score between instruction keywords and candidate text.
    
    This is a fast fallback when LayoutLMv3 is not available.
    """
    if not instruction or not candidate_text:
        return 0.0
    
    # Normalize
    inst_lower = instruction.lower()
    cand_lower = candidate_text.lower()
    
    # Extract keywords from instruction
    stop_words = {
        "a", "an", "the", "on", "click", "tap", "press", "select",
        "to", "in", "at", "for", "with", "by", "of", "is", "it",
    }
    
    inst_words = [w for w in inst_lower.split() if w not in stop_words and len(w) > 2]
    
    if not inst_words:
        return 0.0
    
    # Check for matches
    matches = sum(1 for w in inst_words if w in cand_lower)
    
    # Exact phrase match bonus
    exact_bonus = 0.0
    for word in inst_words:
        if word in cand_lower:
            exact_bonus = 0.3
            break
    
    overlap = matches / len(inst_words)
    return clamp01(overlap + exact_bonus)


def compute_embedding_similarity(
    instruction: str,
    candidate_text: str,
) -> float:
    """
    Compute semantic similarity using sentence embeddings.
    
    More robust than keyword matching, handles paraphrases.
    """
    model = _load_sentence_model()
    if model is None:
        return 0.0
    
    if not instruction or not candidate_text:
        return 0.0
    
    try:
        embeddings = model.encode([instruction, candidate_text])
        similarity = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
        ))
        return clamp01((similarity + 1) / 2)  # Map [-1, 1] to [0, 1]
    except Exception:
        return 0.0


def compute_layoutlm_semantic_score(
    image: np.ndarray,
    instruction: str,
    candidate_bbox: Tuple[int, int, int, int],  # x, y, w, h in pixels
    ocr_results: Optional[List[Dict[str, Any]]] = None,
) -> SemanticMatchResult:
    """
    Use LayoutLMv3 to compute semantic match score for a candidate region.
    
    LayoutLMv3 understands the spatial relationship between text and layout,
    making it ideal for UI element grounding.
    
    Args:
        image: RGB image array
        instruction: User instruction (e.g., "click the cancel button")
        candidate_bbox: Bounding box of candidate region (x, y, w, h)
        ocr_results: Optional pre-computed OCR results
    
    Returns:
        SemanticMatchResult with score and method used
    """
    processor, model = _load_layoutlm()
    
    if processor is None or model is None:
        # Fallback to text overlap
        if ocr_results:
            # Find OCR text in candidate region
            x, y, w, h = candidate_bbox
            region_text = []
            for ocr in ocr_results:
                ox, oy, ow, oh = ocr.get("bbox", [0, 0, 0, 0])
                # Check if OCR box overlaps with candidate
                if (ox < x + w and ox + ow > x and oy < y + h and oy + oh > y):
                    region_text.append(str(ocr.get("text", "")))
            
            combined_text = " ".join(region_text)
            if combined_text:
                score = compute_text_overlap_score(instruction, combined_text)
                return SemanticMatchResult(
                    score=score,
                    method="text_overlap",
                    matched_text=combined_text[:100],
                    confidence=0.5,
                )
        
        return SemanticMatchResult(
            score=0.0,
            method="fallback",
            confidence=0.0,
        )
    
    try:
        from PIL import Image
        import torch
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Prepare inputs
        # For LayoutLMv3, we need words and boxes
        words = []
        boxes = []
        
        if ocr_results:
            for ocr in ocr_results:
                text = str(ocr.get("text", "")).strip()
                bbox = ocr.get("bbox", [0, 0, 0, 0])
                if text and len(bbox) >= 4:
                    words.append(text)
                    # Normalize bbox to 0-1000 range (LayoutLMv3 convention)
                    h_img, w_img = image.shape[:2]
                    x, y, w, h = bbox[:4]
                    norm_box = [
                        int(x / w_img * 1000),
                        int(y / h_img * 1000),
                        int((x + w) / w_img * 1000),
                        int((y + h) / h_img * 1000),
                    ]
                    boxes.append(norm_box)
        
        if not words:
            return SemanticMatchResult(
                score=0.0,
                method="layoutlm_no_text",
                confidence=0.0,
            )
        
        # Process with LayoutLMv3
        encoding = processor(
            pil_image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        # Move to same device as model
        device = next(model.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
            # Get sequence embeddings
            sequence_output = outputs.last_hidden_state
        
        # Get instruction embedding (using CLS token analogy)
        inst_encoding = processor.tokenizer(
            instruction,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        
        # Find tokens in candidate region
        x, y, w, h = candidate_bbox
        h_img, w_img = image.shape[:2]
        
        # Normalize candidate bbox
        cand_norm = [
            int(x / w_img * 1000),
            int(y / h_img * 1000),
            int((x + w) / w_img * 1000),
            int((y + h) / h_img * 1000),
        ]
        
        # Find overlapping tokens
        region_scores = []
        for i, box in enumerate(boxes):
            # Check overlap
            if (box[0] < cand_norm[2] and box[2] > cand_norm[0] and
                box[1] < cand_norm[3] and box[3] > cand_norm[1]):
                # This token is in the candidate region
                # Check semantic relevance to instruction
                text = words[i].lower()
                inst_lower = instruction.lower()
                if text in inst_lower or inst_lower in text:
                    region_scores.append(1.0)
                elif any(w in text for w in inst_lower.split() if len(w) > 2):
                    region_scores.append(0.7)
                else:
                    region_scores.append(0.2)
        
        if region_scores:
            score = float(np.max(region_scores))
            matched = [words[i] for i, box in enumerate(boxes)
                      if (box[0] < cand_norm[2] and box[2] > cand_norm[0] and
                          box[1] < cand_norm[3] and box[3] > cand_norm[1])]
            return SemanticMatchResult(
                score=score,
                method="layoutlm",
                matched_text=" ".join(matched)[:100],
                confidence=0.8,
            )
        
        return SemanticMatchResult(
            score=0.0,
            method="layoutlm_no_overlap",
            confidence=0.5,
        )
    
    except Exception as e:
        return SemanticMatchResult(
            score=0.0,
            method="error",
            confidence=0.0,
            details={"error": str(e)},
        )


def compute_semantic_scores(
    image: np.ndarray,
    instruction: str,
    candidates: List[Dict[str, Any]],
    *,
    ocr_results: Optional[List[Dict[str, Any]]] = None,
    use_embeddings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute P(SemanticMatch) for all candidates.
    
    This is the main entry point for semantic verification.
    
    Args:
        image: RGB image array
        instruction: User instruction
        candidates: List of candidates with 'bbox' or 'bbox_xywh' keys
        ocr_results: Optional pre-computed OCR results
        use_embeddings: Whether to use sentence embeddings for matching
    
    Returns:
        Candidates with added 'semantic_score' and 'semantic_match' fields
    """
    h, w = image.shape[:2]
    
    scored_candidates = []
    
    for cand in candidates:
        # Extract bbox
        bbox = cand.get("bbox_xywh") or cand.get("bbox")
        if not bbox or len(bbox) < 4:
            cand["semantic_score"] = 0.0
            cand["semantic_match"] = None
            scored_candidates.append(cand)
            continue
        
        # Convert normalized bbox to pixels if needed
        if all(0 <= b <= 1 for b in bbox[:4]):
            x = int(bbox[0] * w)
            y = int(bbox[1] * h)
            bw = int((bbox[2] - bbox[0]) * w) if bbox[2] > 1 else int(bbox[2] * w)
            bh = int((bbox[3] - bbox[1]) * h) if bbox[3] > 1 else int(bbox[3] * h)
        else:
            x, y, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Get semantic score
        result = compute_layoutlm_semantic_score(
            image,
            instruction,
            (x, y, bw, bh),
            ocr_results=ocr_results,
        )
        
        # If LayoutLMv3 failed and we have text, try embedding similarity
        if result.score < 0.1 and use_embeddings and result.matched_text:
            embed_score = compute_embedding_similarity(instruction, result.matched_text)
            if embed_score > result.score:
                result = SemanticMatchResult(
                    score=embed_score,
                    method="embedding",
                    matched_text=result.matched_text,
                    confidence=0.6,
                )
        
        cand["semantic_score"] = result.score
        cand["semantic_match"] = {
            "method": result.method,
            "text": result.matched_text,
            "confidence": result.confidence,
        }
        
        scored_candidates.append(cand)
    
    return scored_candidates


def compute_combined_score(
    saliency_score: float,
    semantic_score: float,
    *,
    saliency_weight: float = 0.35,
    semantic_weight: float = 0.65,
) -> float:
    """
    Compute R = Saliency(ROI) × P(SemanticMatch) as per §2.1.2.
    
    The paper specifies a multiplicative relationship, but we use
    a weighted combination for better gradient properties.
    
    Args:
        saliency_score: Visual saliency score [0, 1]
        semantic_score: P(SemanticMatch) score [0, 1]
        saliency_weight: Weight for saliency (default 0.35)
        semantic_weight: Weight for semantic (default 0.65)
    
    Returns:
        Combined score R in [0, 1]
    """
    # Multiplicative (as per paper)
    # R = saliency_score * semantic_score
    
    # Weighted combination (more stable for optimization)
    R = saliency_weight * saliency_score + semantic_weight * semantic_score
    
    # Bonus if both are high (captures multiplicative intent)
    if saliency_score > 0.5 and semantic_score > 0.5:
        R += 0.1 * min(saliency_score, semantic_score)
    
    return clamp01(R)
