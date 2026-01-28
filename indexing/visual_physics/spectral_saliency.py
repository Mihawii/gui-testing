"""
Enhanced Spectral Residual Saliency for 4K processing.

This module extends the standard spectral residual algorithm to work
efficiently on 4K images while preserving small GUI element details.

The key insight is that GUI elements (icons, buttons, text) create
distinctive frequency signatures that survive at high resolution,
unlike natural images where downsampling loses less critical detail.

Algorithm:
    1. Multi-scale processing (full res + downsampled)
    2. Log-amplitude spectrum computation
    3. Spectral residual extraction
    4. Gaussian smoothing with adaptive sigma
    5. Scale fusion for final saliency map
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Backend.common.cv_utils import maybe_cv2
from Backend.common.math_utils import normalize_01


@dataclass
class SaliencyConfig:
    """Configuration for 4K saliency computation."""
    
    # Multi-scale settings
    scales: Tuple[float, ...] = (1.0, 0.5, 0.25)  # Scale factors
    scale_weights: Tuple[float, ...] = (0.5, 0.3, 0.2)  # Weight per scale
    
    # Spectral residual settings
    avg_filter_size: int = 3  # Size of averaging filter
    gaussian_sigma: float = 8.0  # Smoothing sigma
    
    # GUI-specific enhancements  
    edge_boost: float = 0.2  # Boost edges for UI elements
    small_element_boost: float = 0.3  # Boost for icon-sized regions
    small_element_max_size: int = 64  # Max size considered "small"


def compute_4k_saliency(
    image_rgb: np.ndarray,
    *,
    config: Optional[SaliencyConfig] = None,
    return_multiscale: bool = False,
) -> Dict[str, Any]:
    """
    Compute saliency map optimized for 4K GUI images.
    
    Unlike standard saliency which struggles with fine details,
    this uses multi-scale processing to preserve small UI elements
    while maintaining computational efficiency.
    
    Args:
        image_rgb: RGB image of shape (H, W, 3). Can be up to 4K.
        config: Configuration options.
        return_multiscale: If True, return per-scale saliency maps.
    
    Returns:
        Dict containing:
            - saliency: Final fused saliency map (H, W)
            - anchors: Detected visual anchor points
            - scales: Per-scale results if return_multiscale=True
    
    Example:
        >>> result = compute_4k_saliency(image_4k)
        >>> saliency = result["saliency"]  # Shape matches input
    """
    if config is None:
        config = SaliencyConfig()
    
    h, w = image_rgb.shape[:2]
    cv2 = maybe_cv2()
    
    # Convert to grayscale
    if cv2 is not None:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        image_f = image_rgb.astype(np.float32)
        gray = (0.299 * image_f[:, :, 0] + 
                0.587 * image_f[:, :, 1] + 
                0.114 * image_f[:, :, 2]).astype(np.float32)
    
    # Multi-scale saliency computation
    scale_saliencies = []
    
    for scale in config.scales:
        if scale == 1.0:
            scaled_gray = gray
        else:
            new_h = int(h * scale)
            new_w = int(w * scale)
            if cv2 is not None:
                scaled_gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                # Simple numpy downscale
                scaled_gray = _numpy_resize(gray, (new_h, new_w))
        
        # Compute spectral residual at this scale
        sal = _spectral_residual(scaled_gray, config)
        
        # Upscale back to original resolution
        if scale != 1.0:
            if cv2 is not None:
                sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                sal = _numpy_resize(sal, (h, w))
        
        scale_saliencies.append(sal.astype(np.float32))
    
    # Fuse scales
    fused = np.zeros((h, w), dtype=np.float32)
    weights = config.scale_weights[:len(scale_saliencies)]
    weight_sum = sum(weights)
    
    for sal, weight in zip(scale_saliencies, weights):
        fused += (weight / weight_sum) * sal
    
    # Apply GUI-specific enhancements
    if config.edge_boost > 0:
        edges = _compute_edges(gray, cv2)
        fused = fused + config.edge_boost * edges
    
    # Normalize final result
    fused = normalize_01(fused)
    
    # Detect anchor points
    anchors = detect_visual_anchors(fused, min_distance=20, threshold=0.5)
    
    result = {
        "saliency": fused,
        "anchors": anchors,
        "resolution": (h, w),
    }
    
    if return_multiscale:
        result["scales"] = [
            {"scale": s, "saliency": sal}
            for s, sal in zip(config.scales, scale_saliencies)
        ]
    
    return result


def _spectral_residual(gray: np.ndarray, config: SaliencyConfig) -> np.ndarray:
    """Compute spectral residual saliency on grayscale image."""
    cv2 = maybe_cv2()
    
    # FFT
    fft = np.fft.fft2(gray)
    amp = np.abs(fft) + 1e-8
    log_amp = np.log(amp).astype(np.float32)
    phase = np.angle(fft).astype(np.float32)
    
    # Smooth log amplitude to get average
    k = config.avg_filter_size
    if cv2 is not None:
        avg = cv2.blur(log_amp, (k, k))
    else:
        avg = _numpy_blur(log_amp, k)
    
    # Spectral residual = log amplitude - smoothed log amplitude
    spectral_residual = log_amp - avg
    
    # Reconstruct with residual amplitude and original phase
    residual_amp = np.exp(spectral_residual)
    fft_residual = residual_amp * np.exp(1j * phase)
    
    # Inverse FFT to get saliency
    saliency = np.abs(np.fft.ifft2(fft_residual)) ** 2
    saliency = saliency.astype(np.float32)
    
    # Gaussian smoothing
    sigma = config.gaussian_sigma
    if cv2 is not None and sigma > 0:
        saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=sigma)
    elif sigma > 0:
        saliency = _numpy_blur(saliency, int(sigma * 3))
    
    return normalize_01(saliency)


def _compute_edges(gray: np.ndarray, cv2: Any) -> np.ndarray:
    """Compute edge magnitude map."""
    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
    else:
        # Simple numpy gradient
        gy, gx = np.gradient(gray)
        mag = np.sqrt(gx**2 + gy**2)
    
    return normalize_01(mag.astype(np.float32))


def _numpy_resize(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Simple numpy-based image resize (fallback when cv2 unavailable)."""
    h, w = arr.shape[:2]
    new_h, new_w = size
    
    # Simple nearest-neighbor resize
    row_indices = (np.arange(new_h) * h / new_h).astype(int)
    col_indices = (np.arange(new_w) * w / new_w).astype(int)
    
    return arr[row_indices[:, np.newaxis], col_indices]


def _numpy_blur(arr: np.ndarray, ksize: int) -> np.ndarray:
    """Simple box blur fallback."""
    if ksize < 1:
        return arr
    
    k = max(1, ksize)
    padded = np.pad(arr, ((k//2, k//2), (k//2, k//2)), mode="edge")
    
    # Simple box filter
    result = np.zeros_like(arr)
    for i in range(k):
        for j in range(k):
            result += padded[i:i+arr.shape[0], j:j+arr.shape[1]]
    
    return (result / (k * k)).astype(np.float32)


def detect_visual_anchors(
    saliency: np.ndarray,
    *,
    min_distance: int = 20,
    threshold: float = 0.5,
    max_anchors: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detect visual anchor points from saliency map.
    
    These are local maxima in the saliency map that likely
    correspond to important UI elements.
    
    Args:
        saliency: Normalized saliency map (H, W)
        min_distance: Minimum distance between anchors
        threshold: Minimum saliency value for an anchor
        max_anchors: Maximum number of anchors to return
    
    Returns:
        List of anchor dicts with x, y, strength, bbox
    """
    h, w = saliency.shape
    
    # Find local maxima
    anchors = []
    cv2 = maybe_cv2()
    
    if cv2 is not None:
        # Use dilation to find local maxima
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance * 2 + 1, min_distance * 2 + 1))
        dilated = cv2.dilate(saliency, kernel)
        local_max = (saliency == dilated) & (saliency >= threshold)
        
        # Get coordinates
        ys, xs = np.where(local_max)
        strengths = saliency[ys, xs]
        
        # Sort by strength
        order = np.argsort(strengths)[::-1]
        
        for idx in order[:max_anchors]:
            x, y = int(xs[idx]), int(ys[idx])
            strength = float(saliency[y, x])
            
            anchors.append({
                "id": f"anchor_{len(anchors)}",
                "x": x,
                "y": y,
                "x_norm": x / w,
                "y_norm": y / h,
                "strength": strength,
                "bbox": [max(0, x - 16), max(0, y - 16), 32, 32],
            })
    else:
        # Simple fallback: grid-based peak detection
        grid_h = max(1, h // min_distance)
        grid_w = max(1, w // min_distance)
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                y1 = gy * min_distance
                y2 = min((gy + 1) * min_distance, h)
                x1 = gx * min_distance
                x2 = min((gx + 1) * min_distance, w)
                
                patch = saliency[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                
                max_val = float(np.max(patch))
                if max_val >= threshold:
                    local_y, local_x = np.unravel_index(np.argmax(patch), patch.shape)
                    x = x1 + local_x
                    y = y1 + local_y
                    
                    anchors.append({
                        "id": f"anchor_{len(anchors)}",
                        "x": int(x),
                        "y": int(y),
                        "x_norm": x / w,
                        "y_norm": y / h,
                        "strength": max_val,
                        "bbox": [max(0, x - 16), max(0, y - 16), 32, 32],
                    })
        
        # Sort by strength and limit
        anchors.sort(key=lambda a: a["strength"], reverse=True)
        anchors = anchors[:max_anchors]
    
    return anchors
