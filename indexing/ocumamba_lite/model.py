"""
OcuMamba-Lite: Main Model Class.

A lightweight (~500M param) GUI grounding model that combines:
- Mamba-2 Visual Encoder for linear O(N) high-res processing
- Instruction Encoder for natural language understanding
- Multi-Scale Fusion for instruction-conditioned attention
- Icon-Aware Detection Head for tiny target localization

Usage:
    model = OcuMambaLite.from_config("small")
    result = model.predict(image, instruction)
    # result = {"x": 0.5, "y": 0.3, "confidence": 0.85}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

from Backend.indexing.ocumamba_lite.mamba_visual_encoder import (
    MambaVisualEncoder, MambaEncoderConfig, create_mamba_encoder
)
from Backend.indexing.ocumamba_lite.instruction_encoder import (
    InstructionEncoder, InstructionEncoderConfig, SimpleTokenizer, create_instruction_encoder
)
from Backend.indexing.ocumamba_lite.multiscale_fusion import (
    MultiScaleFusion, FusionConfig, create_fusion_module
)
from Backend.indexing.ocumamba_lite.detection_head import (
    IconDetectionHead, DetectionHeadConfig, create_detection_head
)


@dataclass
class OcuMambaLiteConfig:
    """Configuration for complete OcuMamba-Lite model."""
    
    # Model size
    size: str = "small"  # "tiny", "small", "base"
    
    # Component dimensions (auto-set based on size)
    hidden_dim: int = 384
    
    # Visual encoder
    image_size: int = 1024
    patch_size: int = 16
    num_visual_layers: int = 12
    
    # Instruction encoder
    max_instruction_length: int = 64
    num_instruction_layers: int = 4
    
    # Fusion
    num_scales: int = 3
    
    # Training
    dropout: float = 0.1


class OcuMambaLite(nn.Module):
    """
    OcuMamba-Lite: Lightweight GUI Grounding Model.
    
    Predicts click coordinates for natural language instructions on GUI screenshots.
    Optimized for tiny UI elements (10-50 pixels) that standard detectors miss.
    """
    
    # Size presets
    SIZE_CONFIGS = {
        "tiny": {
            "hidden_dim": 192,
            "num_visual_layers": 6,
            "num_instruction_layers": 2,
        },
        "small": {
            "hidden_dim": 384,
            "num_visual_layers": 12,
            "num_instruction_layers": 4,
        },
        "base": {
            "hidden_dim": 512,
            "num_visual_layers": 16,
            "num_instruction_layers": 6,
        },
    }
    
    def __init__(self, config: Optional[OcuMambaLiteConfig] = None):
        super().__init__()
        self.config = config or OcuMambaLiteConfig()
        
        # Apply size preset
        if self.config.size in self.SIZE_CONFIGS:
            preset = self.SIZE_CONFIGS[self.config.size]
            for key, value in preset.items():
                if not hasattr(self.config, key) or getattr(self.config, key) == OcuMambaLiteConfig.__dataclass_fields__[key].default:
                    setattr(self.config, key, value)
        
        hidden_dim = self.config.hidden_dim
        
        # Build components
        self.visual_encoder = create_mamba_encoder(
            size=self.config.size,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            hidden_dim=hidden_dim,
            num_layers=self.config.num_visual_layers,
        )
        
        self.instruction_encoder, self.tokenizer = create_instruction_encoder(
            size=self.config.size,
            hidden_dim=hidden_dim,
            num_layers=self.config.num_instruction_layers,
            max_length=self.config.max_instruction_length,
        )
        
        self.fusion = create_fusion_module(
            hidden_dim=hidden_dim,
            num_scales=self.config.num_scales,
        )
        
        self.detection_head = create_detection_head(
            hidden_dim=hidden_dim,
        )
        
        # Image preprocessing
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image, numpy array, or torch tensor
            
        Returns:
            (1, 3, H, W) normalized tensor
        """
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = image.convert("RGB")
            image = image.resize((self.config.image_size, self.config.image_size))
            image = np.array(image)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).float()
            if image.max() > 1.0:
                image = image / 255.0
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Resize if needed
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            image = F.interpolate(
                image,
                size=(self.config.image_size, self.config.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize
        image = (image - self.pixel_mean.to(image.device)) / self.pixel_std.to(image.device)
        
        return image
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            pixel_values: (B, 3, H, W) normalized images
            input_ids: (B, L) instruction token IDs
            attention_mask: (B, L) instruction attention mask
            
        Returns:
            Dict with predictions
        """
        # Encode visual features
        visual_out = self.visual_encoder(pixel_values, return_multiscale=True)
        
        # Encode instruction
        instr_out = self.instruction_encoder(input_ids, attention_mask)
        
        # Fuse instruction with visual features
        fused = self.fusion(
            visual_out,
            instr_out["last_hidden_state"],
            attention_mask,
        )
        
        # Predict click location
        predictions = self.detection_head(fused)
        
        return predictions
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        instruction: str,
        return_confidence: bool = True,
    ) -> Dict[str, float]:
        """
        Predict click location for instruction on image.
        
        Args:
            image: Input image (PIL, numpy, or tensor)
            instruction: Natural language instruction
            return_confidence: Whether to return confidence score
            
        Returns:
            Dict with "x", "y" (in [0, 1]) and optionally "confidence"
        """
        self.eval()
        
        # Get device
        device = next(self.parameters()).device
        
        # Preprocess image
        pixel_values = self.preprocess_image(image).to(device)
        
        # Tokenize instruction
        tokens = self.tokenizer([instruction], padding=True, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Forward pass
        predictions = self.forward(pixel_values, input_ids, attention_mask)
        
        # Extract results
        xy = predictions["xy"][0].cpu().numpy()
        result = {
            "x": float(xy[0]),
            "y": float(xy[1]),
        }
        
        if return_confidence and "confidence" in predictions:
            result["confidence"] = float(predictions["confidence"][0].cpu().numpy())
        
        return result
    
    def compute_loss(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        target_xy: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            pixel_values: (B, 3, H, W) images
            input_ids: (B, L) instruction tokens
            target_xy: (B, 2) ground truth click points
            attention_mask: (B, L) instruction mask
            
        Returns:
            Dict with loss components
        """
        predictions = self.forward(pixel_values, input_ids, attention_mask)
        
        targets = {"xy": target_xy}
        loss_dict = self.detection_head.compute_loss(predictions, targets)
        
        return loss_dict
    
    @classmethod
    def from_config(cls, size: str = "small", **kwargs) -> "OcuMambaLite":
        """Create model from size preset."""
        config = OcuMambaLiteConfig(size=size, **kwargs)
        return cls(config)
    
    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        import json
        from pathlib import Path
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), save_dir / "model.pt")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "OcuMambaLite":
        """Load model from checkpoint."""
        import json
        from pathlib import Path
        
        save_dir = Path(path)
        
        # Load config
        with open(save_dir / "config.json") as f:
            config_dict = json.load(f)
        
        config = OcuMambaLiteConfig(**config_dict)
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(save_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model


def test_model():
    """Quick test of model forward pass."""
    print("Testing OcuMamba-Lite...")
    
    # Create model
    model = OcuMambaLite.from_config("tiny")
    print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
    
    # Create dummy inputs
    B = 2
    image = torch.randn(B, 3, 256, 256)  # Small for testing
    instruction = ["click the save button", "open file menu"]
    
    # Tokenize
    tokens = model.tokenizer(instruction, padding=True, return_tensors="pt")
    
    # Forward pass
    predictions = model(
        pixel_values=image,
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )
    
    print(f"Predictions shape: xy={predictions['xy'].shape}, conf={predictions['confidence'].shape}")
    print(f"Sample prediction: x={predictions['xy'][0, 0]:.3f}, y={predictions['xy'][0, 1]:.3f}")
    
    # Test loss
    target_xy = torch.rand(B, 2)
    loss_dict = model.compute_loss(
        pixel_values=image,
        input_ids=tokens["input_ids"],
        target_xy=target_xy,
        attention_mask=tokens["attention_mask"],
    )
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    
    print("Test passed!")
    return model


if __name__ == "__main__":
    test_model()
