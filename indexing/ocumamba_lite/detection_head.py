"""
Icon-Aware Detection Head for OcuMamba-Lite.

Anchor-free detection head optimized for tiny UI targets (8-64 pixels).
Unlike standard object detectors that use large anchors (32x32+), this
head is specifically designed for small GUI elements.

Key features:
- Anchor-free design for precise localization
- Multi-scale prediction with scale-aware loss
- Centerness prediction for improved detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DetectionHeadConfig:
    """Configuration for detection head."""
    
    hidden_dim: int = 384
    num_conv_layers: int = 4
    conv_dim: int = 256
    
    # For single-point prediction
    predict_centerness: bool = True


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class IconDetectionHead(nn.Module):
    """
    Detection head for GUI icon localization.
    
    Predicts:
    - Click point (x, y) in normalized coordinates
    - Confidence score
    - Optional: centerness for improved localization
    """
    
    def __init__(self, config: Optional[DetectionHeadConfig] = None):
        super().__init__()
        self.config = config or DetectionHeadConfig()
        
        # Shared conv tower
        self.conv_tower = nn.ModuleList()
        in_channels = self.config.hidden_dim
        for i in range(self.config.num_conv_layers):
            self.conv_tower.append(ConvBlock(in_channels, self.config.conv_dim))
            in_channels = self.config.conv_dim
        
        # Prediction heads
        self.xy_head = nn.Conv1d(self.config.conv_dim, 2, kernel_size=1)  # x, y
        self.conf_head = nn.Conv1d(self.config.conv_dim, 1, kernel_size=1)  # confidence
        
        if self.config.predict_centerness:
            self.center_head = nn.Conv1d(self.config.conv_dim, 1, kernel_size=1)
        
        # Global pooling for single-point prediction
        self.use_global_pool = True
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_all_scales: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict click location from fused features.
        
        Args:
            features: Dict with multi-scale features ("scale_4", "scale_8", "scale_16")
            return_all_scales: If True, return predictions at all scales
            
        Returns:
            Dict with:
                - xy: (B, 2) predicted click points [x, y] in [0, 1]
                - confidence: (B, 1) prediction confidence
                - centerness: (B, 1) centerness score (if enabled)
        """
        # Aggregate multi-scale features
        all_features = []
        for scale in ["scale_4", "scale_8", "scale_16"]:
            if scale in features:
                all_features.append(features[scale])
        
        if not all_features:
            # Fallback to direct features
            if "features" in features:
                all_features = [features["features"]]
            else:
                raise ValueError("No features found in input")
        
        # Average features across scales
        # Each is (B, N, D)
        avg_features = torch.stack(all_features, dim=0).mean(dim=0)
        
        # Process through conv tower
        x = avg_features.transpose(1, 2)  # (B, D, N)
        
        for conv in self.conv_tower:
            x = conv(x)
        
        # Predict at each position
        xy_pred = self.xy_head(x)  # (B, 2, N)
        conf_pred = self.conf_head(x)  # (B, 1, N)
        
        if self.config.predict_centerness:
            center_pred = self.center_head(x)  # (B, 1, N)
        
        # Aggregate predictions
        if self.use_global_pool:
            # Weight by confidence
            weights = F.softmax(conf_pred.squeeze(1), dim=-1)  # (B, N)
            
            # Weighted average of xy predictions
            xy = (xy_pred * weights.unsqueeze(1)).sum(dim=-1)  # (B, 2)
            xy = torch.sigmoid(xy)  # Normalize to [0, 1]
            
            # Max confidence
            conf = conf_pred.max(dim=-1)[0]  # (B, 1)
            conf = torch.sigmoid(conf)
            
            result = {
                "xy": xy,
                "confidence": conf,
            }
            
            if self.config.predict_centerness:
                center = (center_pred.squeeze(1) * weights).sum(dim=-1, keepdim=True)
                center = torch.sigmoid(center)
                result["centerness"] = center
            
            return result
        else:
            # Return per-position predictions
            return {
                "xy_map": xy_pred.transpose(1, 2),  # (B, N, 2)
                "conf_map": conf_pred.transpose(1, 2),  # (B, N, 1)
            }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            predictions: Output from forward()
            targets: Dict with:
                - xy: (B, 2) ground truth click points
                - mask: (B,) valid samples mask
                
        Returns:
            Dict with loss components
        """
        pred_xy = predictions["xy"]
        target_xy = targets["xy"]
        
        # L1 loss for coordinates
        xy_loss = F.l1_loss(pred_xy, target_xy, reduction='mean')
        
        # Scale-aware loss: penalize more for larger deviations
        dist = torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1))
        scale_loss = (dist ** 2).mean()  # Quadratic penalty
        
        # Confidence loss: should be high when prediction is good
        if "confidence" in predictions:
            conf = predictions["confidence"].squeeze(-1)
            # Clamp to valid range to avoid BCE errors
            conf = torch.clamp(conf, min=1e-7, max=1-1e-7)
            # Handle NaN values
            if torch.isnan(conf).any():
                conf = torch.nan_to_num(conf, nan=0.5)
            # Target confidence based on distance (closer = higher)
            target_conf = torch.exp(-10 * dist)  # Exponential decay
            target_conf = torch.clamp(target_conf, min=1e-7, max=1-1e-7)
            conf_loss = F.binary_cross_entropy(conf, target_conf.detach())
        else:
            conf_loss = torch.tensor(0.0, device=pred_xy.device)
        
        total_loss = xy_loss + 0.5 * scale_loss + 0.1 * conf_loss
        
        return {
            "loss": total_loss,
            "xy_loss": xy_loss,
            "scale_loss": scale_loss,
            "conf_loss": conf_loss,
        }


def create_detection_head(
    hidden_dim: int = 384,
    **kwargs,
) -> IconDetectionHead:
    """Create detection head with given settings."""
    config = DetectionHeadConfig(hidden_dim=hidden_dim, **kwargs)
    return IconDetectionHead(config)
