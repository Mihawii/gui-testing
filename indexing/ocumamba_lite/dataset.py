"""
Dataset classes for OcuMamba-Lite training.

Includes:
- ScreenSpotProDataset: Load ScreenSpot-Pro benchmark data
- SyntheticIconDataset: Generate synthetic training data
- GUIGroundingDataset: Combined dataset for training
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


@dataclass
class GUIGroundingSample:
    """A single GUI grounding sample."""
    image: torch.Tensor  # (3, H, W)
    instruction: str
    target_x: float  # Normalized [0, 1]
    target_y: float  # Normalized [0, 1]
    metadata: Dict[str, Any]


class ScreenSpotProDataset(Dataset):
    """
    Dataset for ScreenSpot-Pro benchmark.
    
    Expected structure:
    data_dir/
        images/
            sample_001.png
            ...
        annotations.json  # or test_data.json
    
    Annotation format:
    [
        {
            "img_filename": "sample_001.png",
            "instruction": "Click the close button",
            "bbox": [x1, y1, x2, y2]  # or "point": [x, y]
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        image_size: int = 1024,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Load annotations
        self.samples = self._load_annotations()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from JSON file."""
        # Try different annotation file names
        for name in ["annotations.json", "test_data.json", "data.json"]:
            ann_file = self.data_dir / name
            if ann_file.exists():
                with open(ann_file) as f:
                    return json.load(f)
        
        # Try loading from parquet
        parquet_file = self.data_dir / "test_data.parquet"
        if parquet_file.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_file)
                return df.to_dict('records')
            except ImportError:
                pass
        
        raise FileNotFoundError(f"No annotations found in {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / "images" / sample.get("img_filename", sample.get("image", ""))
        if not img_path.exists():
            # Try direct path
            img_path = self.data_dir / sample.get("img_filename", sample.get("image", ""))
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Get target coordinates
        if "bbox" in sample:
            # Use center of bbox
            x1, y1, x2, y2 = sample["bbox"]
            target_x = (x1 + x2) / 2 / orig_w
            target_y = (y1 + y2) / 2 / orig_h
        elif "point" in sample:
            px, py = sample["point"]
            target_x = px / orig_w
            target_y = py / orig_h
        else:
            # Fallback to center
            target_x, target_y = 0.5, 0.5
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        
        # Apply transform if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            "pixel_values": image_tensor,
            "instruction": sample.get("instruction", sample.get("text", "")),
            "target_xy": torch.tensor([target_x, target_y], dtype=torch.float32),
            "metadata": {
                "filename": sample.get("img_filename", ""),
                "original_size": (orig_w, orig_h),
            }
        }


class SyntheticIconDataset(Dataset):
    """
    Synthetic dataset for GUI grounding training.
    
    Generates images with synthetic icons/buttons and corresponding
    instructions to train the model's localization ability.
    """
    
    # Icon types and their visual characteristics
    ICON_TYPES = [
        ("close", "X", "close button", "×"),
        ("minimize", "_", "minimize button", "−"),
        ("maximize", "□", "maximize button", "⬜"),
        ("menu", "≡", "menu button", "☰"),
        ("search", "🔍", "search button", "⌕"),
        ("settings", "⚙", "settings button", "⚙"),
        ("back", "←", "back button", "◀"),
        ("forward", "→", "forward button", "▶"),
        ("refresh", "↻", "refresh button", "⟳"),
        ("home", "🏠", "home button", "⌂"),
        ("save", "💾", "save button", "▢"),
        ("print", "🖨", "print button", "⎙"),
        ("undo", "↶", "undo button", "↩"),
        ("redo", "↷", "redo button", "↪"),
        ("zoom_in", "+", "zoom in button", "⊕"),
        ("zoom_out", "-", "zoom out button", "⊖"),
    ]
    
    # Instruction templates
    INSTRUCTION_TEMPLATES = [
        "Click the {name}",
        "Press the {name}",
        "Select the {name}",
        "Tap on the {name}",
        "Click on {name}",
        "Find and click the {name}",
    ]
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 1024,
        icon_size_range: Tuple[int, int] = (16, 64),
        num_distractors: int = 10,
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.icon_size_range = icon_size_range
        self.num_distractors = num_distractors
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate a synthetic GUI image
        image = self._generate_gui_image()
        
        # Select target icon type
        target_type = random.choice(self.ICON_TYPES)
        icon_name, icon_char, icon_desc, icon_symbol = target_type
        
        # Place target icon
        icon_size = random.randint(*self.icon_size_range)
        target_x = random.randint(icon_size, self.image_size - icon_size)
        target_y = random.randint(icon_size, self.image_size - icon_size)
        
        # Draw target icon
        self._draw_icon(image, target_x, target_y, icon_size, icon_char, is_target=True)
        
        # Place distractor icons
        for _ in range(self.num_distractors):
            dist_type = random.choice(self.ICON_TYPES)
            dist_size = random.randint(*self.icon_size_range)
            dist_x = random.randint(dist_size, self.image_size - dist_size)
            dist_y = random.randint(dist_size, self.image_size - dist_size)
            
            # Avoid overlap with target
            if abs(dist_x - target_x) > icon_size and abs(dist_y - target_y) > icon_size:
                self._draw_icon(image, dist_x, dist_y, dist_size, dist_type[1], is_target=False)
        
        # Generate instruction
        template = random.choice(self.INSTRUCTION_TEMPLATES)
        instruction = template.format(name=icon_desc)
        
        # Normalize coordinates
        norm_x = target_x / self.image_size
        norm_y = target_y / self.image_size
        
        # Convert to tensor
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0
        
        return {
            "pixel_values": image_tensor,
            "instruction": instruction,
            "target_xy": torch.tensor([norm_x, norm_y], dtype=torch.float32),
            "metadata": {
                "icon_type": icon_name,
                "icon_size": icon_size,
            }
        }
    
    def _generate_gui_image(self) -> Image.Image:
        """Generate a synthetic GUI background."""
        # Create base image with gradient background
        img = Image.new("RGB", (self.image_size, self.image_size))
        pixels = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Random background color scheme
        bg_type = random.choice(["light", "dark", "gradient"])
        
        if bg_type == "light":
            base_color = np.array([240, 240, 245])
            pixels[:] = base_color
        elif bg_type == "dark":
            base_color = np.array([30, 30, 35])
            pixels[:] = base_color
        else:
            # Gradient
            for y in range(self.image_size):
                t = y / self.image_size
                color = (1 - t) * np.array([220, 220, 230]) + t * np.array([180, 180, 200])
                pixels[y, :] = color.astype(np.uint8)
        
        # Add some random rectangles (windows, panels)
        for _ in range(random.randint(2, 5)):
            x1 = random.randint(0, self.image_size - 200)
            y1 = random.randint(0, self.image_size - 200)
            x2 = x1 + random.randint(100, 400)
            y2 = y1 + random.randint(100, 400)
            
            panel_color = np.array([
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255),
            ])
            pixels[y1:y2, x1:x2] = panel_color
        
        return Image.fromarray(pixels)
    
    def _draw_icon(
        self,
        image: Image.Image,
        x: int,
        y: int,
        size: int,
        char: str,
        is_target: bool = False,
    ) -> None:
        """Draw an icon on the image."""
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(image)
        
        # Icon background
        half_size = size // 2
        bbox = (x - half_size, y - half_size, x + half_size, y + half_size)
        
        # Random icon style
        bg_color = random.choice([
            (100, 100, 100),
            (80, 80, 80),
            (150, 150, 150),
            (200, 200, 200),
        ])
        
        draw.ellipse(bbox, fill=bg_color)
        
        # Draw character
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=size//2)
        except:
            font = ImageFont.load_default()
        
        text_color = (255, 255, 255) if sum(bg_color) < 400 else (0, 0, 0)
        draw.text((x, y), char, fill=text_color, font=font, anchor="mm")


class GUIGroundingDataset(Dataset):
    """
    Combined dataset for GUI grounding training.
    
    Supports mixing real and synthetic data with configurable ratios.
    """
    
    def __init__(
        self,
        real_dataset: Optional[Dataset] = None,
        synthetic_dataset: Optional[Dataset] = None,
        synthetic_ratio: float = 0.5,
    ):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_ratio = synthetic_ratio
        
        self.real_len = len(real_dataset) if real_dataset else 0
        self.synthetic_len = len(synthetic_dataset) if synthetic_dataset else 0
        
    def __len__(self) -> int:
        return self.real_len + self.synthetic_len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if random.random() < self.synthetic_ratio and self.synthetic_dataset:
            synth_idx = random.randint(0, self.synthetic_len - 1)
            return self.synthetic_dataset[synth_idx]
        elif self.real_dataset:
            real_idx = idx % self.real_len
            return self.real_dataset[real_idx]
        elif self.synthetic_dataset:
            synth_idx = idx % self.synthetic_len
            return self.synthetic_dataset[synth_idx]
        else:
            raise ValueError("No dataset available")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    target_xy = torch.stack([item["target_xy"] for item in batch])
    instructions = [item["instruction"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "instructions": instructions,
        "target_xy": target_xy,
    }


def create_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 1024,
    synthetic_samples: int = 10000,
    synthetic_ratio: float = 0.5,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Path to ScreenSpot-Pro data (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size to resize to
        synthetic_samples: Number of synthetic samples
        synthetic_ratio: Ratio of synthetic samples in training
        
    Returns:
        (train_loader, val_loader) tuple
    """
    from torch.utils.data import DataLoader
    
    # Create synthetic dataset
    synthetic_dataset = SyntheticIconDataset(
        num_samples=synthetic_samples,
        image_size=image_size,
        seed=42,
    )
    
    # Create real dataset if data_dir provided
    real_dataset = None
    if data_dir and os.path.exists(data_dir):
        try:
            real_dataset = ScreenSpotProDataset(
                data_dir=data_dir,
                image_size=image_size,
            )
        except FileNotFoundError:
            print(f"Warning: No annotations found in {data_dir}, using synthetic only")
    
    # Combine datasets
    train_dataset = GUIGroundingDataset(
        real_dataset=real_dataset,
        synthetic_dataset=synthetic_dataset,
        synthetic_ratio=synthetic_ratio if real_dataset else 1.0,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Validation loader (real data only if available)
    val_loader = None
    if real_dataset and len(real_dataset) > 0:
        val_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test synthetic dataset
    print("Testing SyntheticIconDataset...")
    dataset = SyntheticIconDataset(num_samples=10, image_size=512)
    sample = dataset[0]
    print(f"  Image shape: {sample['pixel_values'].shape}")
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Target: ({sample['target_xy'][0]:.3f}, {sample['target_xy'][1]:.3f})")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, _ = create_dataloaders(
        batch_size=4,
        num_workers=0,
        image_size=512,
        synthetic_samples=100,
    )
    batch = next(iter(train_loader))
    print(f"  Batch pixel_values: {batch['pixel_values'].shape}")
    print(f"  Batch target_xy: {batch['target_xy'].shape}")
    print(f"  Instructions: {batch['instructions'][:2]}")
    print("\nDataset tests passed!")
