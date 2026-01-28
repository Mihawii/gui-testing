"""
Multi-Scale Fusion Module for OcuMamba-Lite.

Fuses instruction embeddings with visual features at multiple scales.
This enables instruction-conditioned attention where the model "looks for"
specific UI elements based on the instruction.

Key features:
- Cross-attention between instruction and visual features
- Feature Pyramid Network (FPN) style fusion
- Scale-aware processing for tiny icon detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for multi-scale fusion."""
    
    hidden_dim: int = 384
    num_scales: int = 3
    num_heads: int = 6
    dropout: float = 0.1
    
    # FPN settings
    fpn_dim: int = 256


class CrossAttention(nn.Module):
    """Cross-attention for instruction-visual fusion."""
    
    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_value_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from query to key/value.
        
        Args:
            query: (B, Nq, D) query features (visual)
            key_value: (B, Nkv, D) key/value features (instruction)
            key_value_mask: (B, Nkv) attention mask
            
        Returns:
            (B, Nq, D) attended features
        """
        B, Nq, D = query.shape
        Nkv = key_value.shape[1]
        
        # Project
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if key_value_mask is not None:
            attn = attn.masked_fill(
                ~key_value_mask.unsqueeze(1).unsqueeze(2).bool(),
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        
        return self.norm(query + out)


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion of instruction and visual features.
    
    Creates instruction-conditioned visual features at multiple scales
    for detecting UI elements of different sizes.
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()
        
        # Cross-attention for each scale
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_scales)
        ])
        
        # FPN lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(self.config.hidden_dim, self.config.fpn_dim, kernel_size=1)
            for _ in range(self.config.num_scales)
        ])
        
        # FPN output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv1d(self.config.fpn_dim, self.config.fpn_dim, kernel_size=3, padding=1)
            for _ in range(self.config.num_scales)
        ])
        
        # Final projection back to hidden dim
        self.final_proj = nn.Linear(self.config.fpn_dim, self.config.hidden_dim)
    
    def forward(
        self,
        visual_features: Dict[str, torch.Tensor],
        instruction_features: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse instruction with multi-scale visual features.
        
        Args:
            visual_features: Dict with "scale_4", "scale_8", "scale_16" features
            instruction_features: (B, L, D) instruction token embeddings
            instruction_mask: (B, L) mask for instruction tokens
            
        Returns:
            Dict with fused features at each scale
        """
        scales = ["scale_4", "scale_8", "scale_16"][:self.config.num_scales]
        
        # Apply cross-attention at each scale
        attended = {}
        for i, scale in enumerate(scales):
            if scale in visual_features:
                vis_feat = visual_features[scale]
                attended[scale] = self.cross_attns[i](
                    vis_feat, instruction_features, instruction_mask
                )
        
        # FPN: top-down pathway
        # Start from coarsest scale
        fpn_features = {}
        prev_feat = None
        
        for i, scale in enumerate(reversed(scales)):
            if scale not in attended:
                continue
            
            # Lateral connection
            feat = attended[scale].transpose(1, 2)  # (B, D, N)
            lateral = self.lateral_convs[len(scales) - 1 - i](feat)
            
            # Add upsampled previous feature
            if prev_feat is not None:
                if lateral.shape[2] != prev_feat.shape[2]:
                    prev_feat = F.interpolate(
                        prev_feat, size=lateral.shape[2], mode='nearest'
                    )
                lateral = lateral + prev_feat
            
            # Output conv
            out = self.output_convs[len(scales) - 1 - i](lateral)
            fpn_features[scale] = out.transpose(1, 2)  # (B, N, D)
            prev_feat = out
        
        # Project to hidden dim
        fused = {}
        for scale, feat in fpn_features.items():
            fused[scale] = self.final_proj(feat)
        
        return fused
    
    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim


def create_fusion_module(
    hidden_dim: int = 384,
    num_scales: int = 3,
) -> MultiScaleFusion:
    """Create fusion module with given settings."""
    config = FusionConfig(
        hidden_dim=hidden_dim,
        num_scales=num_scales,
    )
    return MultiScaleFusion(config)
