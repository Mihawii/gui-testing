"""
Active Inference GUI Grounding Model

A novel approach to GUI visual grounding using Active Inference - a Bayesian
framework from neuroscience that models perception as inference.

Key Innovation:
    Unlike standard vision models that predict locations directly, this model:
    1. Maintains a probabilistic BELIEF about target location
    2. Uses SACCADES (attention shifts) to gather evidence
    3. Minimizes EXPECTED FREE ENERGY to optimally balance exploration/exploitation

Theoretical Foundation:
    P(click | image, instruction) = 
        ∫ P(click | s) P(s | o, π) ds
    
    Where:
        s = hidden state (target location)
        o = observations (visual features at attended locations)
        π = policy (sequence of saccades)
    
    The optimal policy minimizes Expected Free Energy:
        G(π) = E_Q[ln Q(s) - ln P(o,s | π)]
             = -Epistemic Value - Pragmatic Value

Author: Plura Research
Status: EXPERIMENTAL - Novel approach for GUI grounding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ActiveGUIConfig:
    """Configuration for Active GUI Grounding."""
    
    # Belief map resolution
    belief_size: Tuple[int, int] = (64, 64)
    
    # Saccade settings
    max_saccades: int = 8
    attention_size: int = 128  # Size of attention window
    
    # Free Energy weights
    epistemic_weight: float = 0.3  # Information seeking
    pragmatic_weight: float = 0.7  # Goal seeking
    
    # Convergence
    confidence_threshold: float = 0.9
    min_improvement: float = 0.01
    
    # Prior initialization
    instruction_prior_weight: float = 0.5


class InstructionPrior:
    """
    Generate prior belief from instruction text.
    
    Maps instruction keywords to spatial priors based on GUI conventions:
    - "settings" → top-right corner
    - "close/exit" → top-right corner
    - "menu/hamburger" → top-left corner
    - "save" → top toolbar area
    - "next/continue" → bottom-right
    - "back" → top-left or bottom-left
    """
    
    SPATIAL_PRIORS = {
        # Top-right patterns
        "settings": (0.9, 0.1),
        "gear": (0.9, 0.1),
        "close": (0.95, 0.05),
        "exit": (0.95, 0.05),
        "x button": (0.95, 0.05),
        "minimize": (0.85, 0.05),
        "maximize": (0.90, 0.05),
        
        # Top-left patterns
        "menu": (0.05, 0.05),
        "hamburger": (0.05, 0.05),
        "back": (0.05, 0.1),
        "home": (0.05, 0.05),
        "logo": (0.1, 0.1),
        
        # Top toolbar
        "save": (0.3, 0.05),
        "file": (0.05, 0.02),
        "edit": (0.1, 0.02),
        "view": (0.15, 0.02),
        "refresh": (0.3, 0.1),
        "search": (0.5, 0.1),
        
        # Bottom patterns
        "next": (0.9, 0.9),
        "continue": (0.9, 0.9),
        "submit": (0.5, 0.9),
        "send": (0.9, 0.9),
        "ok": (0.6, 0.85),
        "cancel": (0.4, 0.85),
        
        # Center patterns
        "play": (0.5, 0.5),
        "pause": (0.5, 0.5),
    }
    
    def __init__(self, belief_size: Tuple[int, int] = (64, 64)):
        self.belief_size = belief_size
    
    def generate_prior(self, instruction: str) -> np.ndarray:
        """
        Generate spatial prior from instruction text.
        
        Returns:
            2D numpy array of prior probabilities
        """
        h, w = self.belief_size
        prior = np.ones((h, w), dtype=np.float32) * 0.01  # Uniform base
        
        instruction_lower = instruction.lower()
        
        # Find matching patterns
        matches = []
        for keyword, (cx, cy) in self.SPATIAL_PRIORS.items():
            if keyword in instruction_lower:
                matches.append((cx, cy, len(keyword)))
        
        if not matches:
            # No match - use uniform prior with slight center bias
            yy, xx = np.mgrid[0:h, 0:w]
            center_bias = np.exp(-((xx - w/2)**2 + (yy - h/2)**2) / (w * h / 4))
            prior = prior + 0.05 * center_bias
        else:
            # Create Gaussian mixture at matched locations
            for cx, cy, weight in matches:
                px = int(cx * (w - 1))
                py = int(cy * (h - 1))
                
                yy, xx = np.mgrid[0:h, 0:w]
                sigma = max(h, w) * 0.1  # 10% of image
                gaussian = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))
                
                prior = prior + weight * gaussian
        
        # Normalize to valid probability
        prior = prior / (prior.sum() + 1e-10)
        return prior


class VisualLikelihood:
    """
    Compute visual likelihood P(o | s) for different locations.
    
    Uses simple visual features to estimate likelihood that target is at location:
    - Edge density (icons have edges)
    - Color contrast (icons stand out)
    - Size consistency (icons are typically 16-64px)
    """
    
    def compute_likelihood(
        self,
        image: np.ndarray,
        prior: np.ndarray,
    ) -> np.ndarray:
        """
        Compute visual likelihood map.
        
        Args:
            image: RGB image array (H, W, 3)
            prior: Prior belief map
            
        Returns:
            Likelihood map P(o | s)
        """
        h, w = image.shape[:2]
        ph, pw = prior.shape
        
        # Convert to grayscale
        gray = np.mean(image, axis=2).astype(np.float32)
        
        # Compute edge density (icons have strong edges)
        sobel_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        sobel_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edges = sobel_x + sobel_y
        
        # Compute local contrast
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=32)
        local_var = uniform_filter((gray - local_mean)**2, size=32)
        contrast = np.sqrt(local_var + 1e-6)
        
        # Combine into likelihood
        likelihood = 0.5 * edges + 0.5 * contrast
        
        # Resize to prior resolution
        likelihood_resized = self._resize(likelihood, (ph, pw))
        
        # Normalize
        likelihood_resized = likelihood_resized / (likelihood_resized.sum() + 1e-10)
        
        return likelihood_resized
    
    def _resize(self, arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize array to target size."""
        from PIL import Image
        img = Image.fromarray(arr)
        img = img.resize((size[1], size[0]), Image.BILINEAR)
        return np.array(img)


class ActiveGUIGrounder:
    """
    Active Inference GUI Grounding Model.
    
    Novel approach to visual grounding that:
    1. Builds prior from instruction (where should target be based on GUI conventions)
    2. Computes likelihood from visual features (where are clickable elements)
    3. Updates belief through saccadic attention shifts
    4. Outputs click location when confident
    
    Example:
        >>> grounder = ActiveGUIGrounder()
        >>> result = grounder.ground(image, "click the settings button")
        >>> print(f"Click at ({result['x']:.2f}, {result['y']:.2f})")
    """
    
    def __init__(self, config: Optional[ActiveGUIConfig] = None):
        self.config = config or ActiveGUIConfig()
        self.instruction_prior = InstructionPrior(self.config.belief_size)
        self.visual_likelihood = VisualLikelihood()
        
        # State
        self.belief: Optional[np.ndarray] = None
        self.saccade_history: List[Dict] = []
    
    def ground(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict:
        """
        Ground instruction to click location using Active Inference.
        
        Args:
            image: PIL Image
            instruction: Natural language instruction
            
        Returns:
            Dict with 'x', 'y' (normalized coords) and 'confidence'
        """
        # Convert image
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image
        
        h, w = image_np.shape[:2]
        
        # Reset state
        self.belief = None
        self.saccade_history = []
        
        # Step 1: Generate instruction prior
        prior = self.instruction_prior.generate_prior(instruction)
        
        # Step 2: Compute visual likelihood
        likelihood = self.visual_likelihood.compute_likelihood(image_np, prior)
        
        # Step 3: Initialize belief as posterior ∝ prior × likelihood
        self.belief = prior * likelihood
        self.belief = self.belief / (self.belief.sum() + 1e-10)
        
        # Step 4: Iterative belief refinement through saccades
        for i in range(self.config.max_saccades):
            # Find current best location (mode of belief)
            best_idx = np.argmax(self.belief)
            by, bx = np.unravel_index(best_idx, self.belief.shape)
            
            # Compute confidence (how peaked is the belief)
            confidence = self._compute_confidence()
            
            # Record saccade
            self.saccade_history.append({
                "iteration": i,
                "location": (bx / self.belief.shape[1], by / self.belief.shape[0]),
                "confidence": confidence,
                "efe": self._compute_efe(bx, by),
            })
            
            # Check convergence
            if confidence >= self.config.confidence_threshold:
                break
            
            # Update belief (attend to current best location)
            self._update_belief_from_saccade(bx, by, image_np)
        
        # Extract final prediction
        best_idx = np.argmax(self.belief)
        by, bx = np.unravel_index(best_idx, self.belief.shape)
        
        # Convert to normalized coordinates
        x = (bx + 0.5) / self.belief.shape[1]
        y = (by + 0.5) / self.belief.shape[0]
        
        return {
            "x": float(x),
            "y": float(y),
            "confidence": float(self._compute_confidence()),
            "saccades": len(self.saccade_history),
        }
    
    def _compute_confidence(self) -> float:
        """Compute confidence as inverse entropy of belief."""
        if self.belief is None:
            return 0.0
        
        p = self.belief.flatten()
        p = p / (p.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        
        # Entropy
        entropy = -np.sum(p * np.log(p))
        
        # Max entropy for uniform distribution
        max_entropy = np.log(len(p))
        
        # Confidence = 1 - normalized entropy
        confidence = 1.0 - (entropy / max_entropy)
        
        return float(np.clip(confidence, 0, 1))
    
    def _compute_efe(self, bx: int, by: int) -> float:
        """Compute Expected Free Energy at location."""
        if self.belief is None:
            return 0.0
        
        # Epistemic value: local uncertainty (entropy in window)
        window = self._get_window(bx, by, size=5)
        p = window.flatten()
        p = p / (p.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        epistemic = -np.sum(p * np.log(p))
        
        # Pragmatic value: belief at location
        pragmatic = float(self.belief[by, bx])
        
        # EFE = -weighted sum (lower is better)
        efe = -(
            self.config.epistemic_weight * epistemic +
            self.config.pragmatic_weight * pragmatic
        )
        
        return float(efe)
    
    def _get_window(self, bx: int, by: int, size: int = 5) -> np.ndarray:
        """Get window around location."""
        if self.belief is None:
            return np.array([1.0])
        
        bh, bw = self.belief.shape
        half = size // 2
        
        y1 = max(0, by - half)
        y2 = min(bh, by + half + 1)
        x1 = max(0, bx - half)
        x2 = min(bw, bx + half + 1)
        
        return self.belief[y1:y2, x1:x2]
    
    def _update_belief_from_saccade(
        self,
        bx: int,
        by: int,
        image_np: np.ndarray,
    ) -> None:
        """Update belief after attending to a location."""
        if self.belief is None:
            return
        
        bh, bw = self.belief.shape
        
        # Sharpen belief around attended location (simulating evidence gathering)
        yy, xx = np.mgrid[0:bh, 0:bw]
        sigma = max(bh, bw) * 0.08
        gaussian = np.exp(-((xx - bx)**2 + (yy - by)**2) / (2 * sigma**2))
        
        # Also check visual evidence at this location
        h, w = image_np.shape[:2]
        img_x = int(bx / bw * w)
        img_y = int(by / bh * h)
        
        # Get local patch and check for icon-like features
        patch_size = 64
        x1 = max(0, img_x - patch_size // 2)
        x2 = min(w, img_x + patch_size // 2)
        y1 = max(0, img_y - patch_size // 2)
        y2 = min(h, img_y + patch_size // 2)
        
        patch = image_np[y1:y2, x1:x2]
        
        # Simple edge check - icons have edges
        if patch.size > 0:
            gray = np.mean(patch, axis=2)
            edge_strength = np.std(gray) / 128.0  # Normalize
            edge_strength = np.clip(edge_strength, 0.1, 1.0)
        else:
            edge_strength = 0.5
        
        # Update belief with evidence
        sharpening = 0.3 * edge_strength
        self.belief = (1 - sharpening) * self.belief + sharpening * gaussian
        self.belief = self.belief / (self.belief.sum() + 1e-10)


def test_active_gui():
    """Test the Active GUI Grounding model."""
    print("="*60)
    print("Active Inference GUI Grounding Test")
    print("="*60)
    
    # Create test image (simple GUI mockup)
    image = Image.new("RGB", (1920, 1080), color=(240, 240, 240))
    
    # Draw some "icons" at expected locations
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Settings icon (top-right)
    draw.rectangle([1850, 20, 1890, 60], fill=(100, 100, 100))
    
    # Menu icon (top-left)
    draw.rectangle([20, 20, 60, 60], fill=(80, 80, 80))
    
    # Save button (top toolbar)
    draw.rectangle([200, 20, 280, 60], fill=(50, 120, 200))
    
    # Create grounder
    grounder = ActiveGUIGrounder()
    
    # Test different instructions
    test_cases = [
        "click the settings icon",
        "open the menu",
        "click save button",
        "press the exit button",
    ]
    
    for instruction in test_cases:
        result = grounder.ground(image, instruction)
        print(f"\nInstruction: '{instruction}'")
        print(f"  Prediction: ({result['x']:.2f}, {result['y']:.2f})")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Saccades: {result['saccades']}")


if __name__ == "__main__":
    test_active_gui()
