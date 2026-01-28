"""
Active Inference Controller for optimal visual search.

This module implements the Active Inference framework for GUI understanding,
replacing heuristic "zoom and crop" strategies with mathematically optimal
saccadic eye movements that minimize Expected Free Energy.

Key Concepts:
    - Expected Free Energy (EFE): Combines epistemic (information-seeking)
      and pragmatic (goal-fulfilling) values to guide attention
    - Saccades: Rapid eye movements to high-value regions
    - Belief updating: Bayesian updating of target location beliefs

Architecture:
    Prior P(s): Visual physics saliency map
    Likelihood P(o|s): Mamba feature similarity to target
    EFE G(π): Expected information gain + goal proximity
    Action: Move attention window to argmin G(π)

Status: SKELETON - Full implementation requires GPU for real-time processing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.math_utils import normalize_01, clamp01


@dataclass
class SaccadeResult:
    """Result of a single saccade (attention shift)."""
    
    # Location
    center_x: float  # Normalized [0, 1]
    center_y: float  # Normalized [0, 1]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    
    # Values
    efe_value: float  # Expected Free Energy (lower = better)
    epistemic_value: float  # Information gain
    pragmatic_value: float  # Goal alignment
    
    # Confidence
    confidence: float  # Posterior probability
    iteration: int  # Which saccade number


@dataclass
class ActiveInferenceConfig:
    """Configuration for Active Inference controller."""
    
    # Saccade settings
    max_saccades: int = 5  # Maximum attention shifts
    patch_size: Tuple[int, int] = (512, 512)  # High-res attention window
    
    # EFE weights
    epistemic_weight: float = 0.4  # Weight for information seeking
    pragmatic_weight: float = 0.6  # Weight for goal fulfillment
    
    # Convergence
    confidence_threshold: float = 0.85  # Stop when confident
    efe_improvement_threshold: float = 0.05  # Stop if EFE not improving
    
    # Prior settings
    prior_temperature: float = 1.0  # Sharpness of saliency prior


class ActiveInferenceController:
    """
    Active Inference controller for optimal GUI element search.
    
    This controller replaces simple grid-based or heuristic search
    strategies with a principled approach based on Active Inference,
    a theory of brain function that models perception as inference.
    
    The key insight is that visual search should minimize Expected
    Free Energy (EFE), which combines:
    
    1. Epistemic value: Regions that reduce uncertainty about target location
    2. Pragmatic value: Regions likely to contain the target
    
    This produces human-like saccadic eye movements that efficiently
    locate small UI elements in large 4K dashboards.
    
    Example:
        >>> controller = ActiveInferenceController()
        >>> # Set the target we're looking for
        >>> controller.set_target("Save button", target_embedding=embed)
        >>> # Set the prior from visual saliency
        >>> controller.set_prior(saliency_map)
        >>> # Run inference
        >>> result = controller.search(image_features)
        >>> print(f"Target at {result.bbox} with confidence {result.confidence}")
    
    Note:
        Full GPU acceleration requires mamba-ssm and CUDA.
        CPU fallback uses simplified heuristics.
    """
    
    def __init__(self, config: Optional[ActiveInferenceConfig] = None):
        """Initialize the Active Inference controller."""
        self.config = config or ActiveInferenceConfig()
        
        # State
        self._prior: Optional[np.ndarray] = None  # P(s) from visual physics
        self._belief: Optional[np.ndarray] = None  # Q(s) current belief
        self._target_embedding: Optional[np.ndarray] = None
        self._target_text: Optional[str] = None
        
        # History
        self._saccade_history: List[SaccadeResult] = []
    
    def set_target(
        self, 
        target_text: str,
        target_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set the target element to search for.
        
        Args:
            target_text: Text description of target (e.g., "Settings icon")
            target_embedding: Optional pre-computed embedding vector
        """
        self._target_text = target_text
        self._target_embedding = target_embedding
        self._saccade_history = []
        
        # Reset belief to uniform if no prior
        if self._prior is not None:
            self._belief = self._prior.copy()
        else:
            self._belief = None
    
    def set_prior(self, saliency_map: np.ndarray) -> None:
        """
        Set the prior probability distribution P(s) from visual saliency.
        
        The prior encodes where the target is likely to be based on
        visual features alone (without considering the target query).
        
        Args:
            saliency_map: 2D array of saliency values, will be normalized.
        """
        # Normalize to valid probability distribution
        prior = saliency_map.astype(np.float32)
        prior = np.maximum(prior, 1e-6)  # Ensure no zeros
        prior = prior / prior.sum()  # Normalize
        
        # Apply temperature
        if self.config.prior_temperature != 1.0:
            log_prior = np.log(prior + 1e-10)
            log_prior = log_prior / self.config.prior_temperature
            prior = np.exp(log_prior - np.max(log_prior))
            prior = prior / prior.sum()
        
        self._prior = prior
        
        # Initialize belief from prior
        if self._belief is None:
            self._belief = prior.copy()
    
    def compute_efe(
        self,
        features: np.ndarray,
        location: Tuple[int, int],
        *,
        image_shape: Tuple[int, int],
    ) -> Tuple[float, float, float]:
        """
        Compute Expected Free Energy at a given location.
        
        EFE = Epistemic Value + Pragmatic Value
        
        Lower EFE = Better location to look at
        
        Args:
            features: Feature map from Vision Mamba
            location: (x, y) center location to evaluate
            image_shape: (height, width) of original image
        
        Returns:
            Tuple of (efe, epistemic_value, pragmatic_value)
        """
        h, w = image_shape
        x, y = location
        
        # Normalize location
        x_norm = x / max(w, 1)
        y_norm = y / max(h, 1)
        
        if self._belief is None:
            # No prior information - uniform uncertainty
            return 0.0, 0.0, 0.0
        
        # Sample belief at location
        bh, bw = self._belief.shape
        bx = int(clamp01(x_norm) * (bw - 1))
        by = int(clamp01(y_norm) * (bh - 1))
        
        belief_value = float(self._belief[by, bx])
        
        # Epistemic value: How much would looking here reduce uncertainty?
        # Approximated as entropy reduction potential
        window = self._get_belief_window(bx, by, size=5)
        local_entropy = self._entropy(window)
        epistemic = local_entropy  # Higher entropy = more to learn
        
        # Pragmatic value: How likely is the target here?
        # Based on prior × belief update
        pragmatic = belief_value
        
        if self._target_embedding is not None and features is not None:
            # Add feature similarity if available
            fh, fw = features.shape[:2]
            fx = int(clamp01(x_norm) * (fw - 1))
            fy = int(clamp01(y_norm) * (fh - 1))
            local_features = features[fy, fx]
            similarity = self._cosine_similarity(local_features, self._target_embedding)
            pragmatic = 0.5 * pragmatic + 0.5 * similarity
        
        # EFE combines both (negative because we minimize EFE)
        efe = -(
            self.config.epistemic_weight * epistemic +
            self.config.pragmatic_weight * pragmatic
        )
        
        return efe, epistemic, pragmatic
    
    def search(
        self,
        image_rgb: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> SaccadeResult:
        """
        Run Active Inference search to locate target element.
        
        Performs a sequence of saccades (attention shifts), each time:
        1. Computing EFE across the image
        2. Moving attention to location with lowest EFE
        3. Updating beliefs based on observations
        4. Stopping when confident or max saccades reached
        
        Args:
            image_rgb: Full resolution image
            features: Optional feature map from Vision Mamba encoder
        
        Returns:
            SaccadeResult with final target location estimate.
        """
        h, w = image_rgb.shape[:2]
        
        # Initialize belief from prior if not set
        if self._belief is None:
            self._belief = np.ones((h // 16, w // 16), dtype=np.float32)
            self._belief = self._belief / self._belief.sum()
        
        best_result: Optional[SaccadeResult] = None
        
        for i in range(self.config.max_saccades):
            # Find location with minimum EFE
            efe_map = np.zeros_like(self._belief)
            bh, bw = self._belief.shape
            
            for by in range(bh):
                for bx in range(bw):
                    x = int(bx / bw * w)
                    y = int(by / bh * h)
                    efe, _, _ = self.compute_efe(features, (x, y), image_shape=(h, w))
                    efe_map[by, bx] = efe
            
            # Find minimum EFE location
            min_idx = np.argmin(efe_map)
            min_by, min_bx = np.unravel_index(min_idx, efe_map.shape)
            
            # Convert to image coordinates
            center_x = (min_bx + 0.5) / bw
            center_y = (min_by + 0.5) / bh
            
            # Compute patch bbox
            pw, ph = self.config.patch_size
            x1 = int(center_x * w - pw / 2)
            y1 = int(center_y * h - ph / 2)
            x1 = max(0, min(x1, w - pw))
            y1 = max(0, min(y1, h - ph))
            
            # Get values at this location
            efe_value = float(efe_map[min_by, min_bx])
            _, epistemic, pragmatic = self.compute_efe(
                features, 
                (int(center_x * w), int(center_y * h)),
                image_shape=(h, w)
            )
            
            # Compute confidence from belief
            confidence = float(self._belief[min_by, min_bx]) / float(self._belief.max() + 1e-6)
            
            result = SaccadeResult(
                center_x=center_x,
                center_y=center_y,
                bbox=(x1, y1, pw, ph),
                efe_value=efe_value,
                epistemic_value=epistemic,
                pragmatic_value=pragmatic,
                confidence=clamp01(confidence),
                iteration=i,
            )
            
            self._saccade_history.append(result)
            best_result = result
            
            # Update belief (sharpen around current location)
            self._update_belief(min_bx, min_by)
            
            # Check convergence
            if confidence >= self.config.confidence_threshold:
                break
        
        return best_result or SaccadeResult(
            center_x=0.5, center_y=0.5,
            bbox=(w // 4, h // 4, w // 2, h // 2),
            efe_value=0.0, epistemic_value=0.0, pragmatic_value=0.0,
            confidence=0.0, iteration=0,
        )
    
    def _get_belief_window(self, bx: int, by: int, size: int = 5) -> np.ndarray:
        """Extract a window from belief map centered at (bx, by)."""
        if self._belief is None:
            return np.array([1.0])
        
        bh, bw = self._belief.shape
        half = size // 2
        
        x1 = max(0, bx - half)
        x2 = min(bw, bx + half + 1)
        y1 = max(0, by - half)
        y2 = min(bh, by + half + 1)
        
        return self._belief[y1:y2, x1:x2]
    
    def _entropy(self, p: np.ndarray) -> float:
        """Compute entropy of a probability distribution."""
        p = p.flatten().astype(np.float32)
        p = p / (p.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        
        return clamp01((dot / (norm_a * norm_b) + 1) / 2)
    
    def _update_belief(self, bx: int, by: int, *, sharpening: float = 0.3) -> None:
        """Update belief distribution after observing a location."""
        if self._belief is None:
            return
        
        bh, bw = self._belief.shape
        
        # Create Gaussian centered at observed location
        yy, xx = np.mgrid[0:bh, 0:bw]
        sigma = max(bh, bw) * 0.15
        gaussian = np.exp(-((xx - bx)**2 + (yy - by)**2) / (2 * sigma**2))
        gaussian = gaussian.astype(np.float32)
        
        # Sharpen belief around observed location
        self._belief = (1 - sharpening) * self._belief + sharpening * gaussian
        self._belief = self._belief / (self._belief.sum() + 1e-10)
    
    @property
    def saccade_history(self) -> List[SaccadeResult]:
        """Get the history of saccades performed during search."""
        return self._saccade_history.copy()
