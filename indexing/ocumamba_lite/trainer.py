"""
Training script for OcuMamba-Lite.

Supports:
- Mixed synthetic + real data training
- Gradient accumulation for larger effective batch sizes
- Learning rate scheduling with warmup
- Checkpointing and logging
- Multi-GPU training via DDP (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from .model import OcuMambaLite
from .dataset import create_dataloaders, SyntheticIconDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train OcuMamba-Lite")
    
    # Model
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=["tiny", "small", "base"],
                        help="Model size preset")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Input image size")
    
    # Data
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to ScreenSpot-Pro data")
    parser.add_argument("--synthetic-samples", type=int, default=50000,
                        help="Number of synthetic training samples")
    parser.add_argument("--synthetic-ratio", type=float, default=0.7,
                        help="Ratio of synthetic samples in training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps")
    
    # System
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    return parser.parse_args()


class Trainer:
    """Trainer for OcuMamba-Lite."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print(f"Creating OcuMamba-Lite ({args.model_size})...")
        self.model = OcuMambaLite.from_config(
            args.model_size,
            image_size=args.image_size,
        ).to(self.device)
        print(f"Model parameters: {self.model.num_parameters() / 1e6:.1f}M")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        
        # Resume if specified
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return step / max(1, self.args.warmup_steps)
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Main training loop."""
        print(f"\nCreating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            data_dir=self.args.data_dir,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            image_size=self.args.image_size,
            synthetic_samples=self.args.synthetic_samples,
            synthetic_ratio=self.args.synthetic_ratio,
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        
        print(f"\nStarting training...")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size} x {self.args.grad_accum} = {self.args.batch_size * self.args.grad_accum}")
        print(f"  Learning rate: {self.args.lr}")
        
        for epoch in range(self.epoch, self.args.epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch(train_loader)
            
            print(f"\nEpoch {epoch + 1}/{self.args.epochs} - Loss: {epoch_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self._validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt")
            
            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}.pt")
        
        # Save final model
        self._save_checkpoint("final.pt")
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            pixel_values = batch["pixel_values"].to(self.device)
            target_xy = batch["target_xy"].to(self.device)
            instructions = batch["instructions"]
            
            # Tokenize instructions
            tokens = self.model.tokenizer(instructions, padding=True, return_tensors="pt")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Forward pass
            predictions = self.model(
                pixel_values=pixel_values,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            
            # Compute loss
            targets = {"xy": target_xy}
            loss_dict = self.model.compute_loss(
                pixel_values=pixel_values,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                targets=targets,
            )
            
            loss = loss_dict["loss"] / self.args.grad_accum
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.args.grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.args.grad_accum
            num_batches += 1
            
            # Logging
            if batch_idx % self.args.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                pred_xy = predictions["xy"][0].detach().cpu()
                tgt_xy = target_xy[0].cpu()
                dist = torch.sqrt(((pred_xy - tgt_xy) ** 2).sum()).item()
                
                print(f"  Step {self.global_step} | "
                      f"Loss: {loss.item() * self.args.grad_accum:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Dist: {dist:.4f}")
            
            # Checkpointing
            if self.global_step > 0 and self.global_step % self.args.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}.pt")
        
        return total_loss / max(1, num_batches)
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate on real data."""
        self.model.eval()
        total_loss = 0.0
        total_dist = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                target_xy = batch["target_xy"].to(self.device)
                instructions = batch["instructions"]
                
                tokens = self.model.tokenizer(instructions, padding=True, return_tensors="pt")
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                predictions = self.model(
                    pixel_values=pixel_values,
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                
                targets = {"xy": target_xy}
                loss_dict = self.model.compute_loss(
                    pixel_values=pixel_values,
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    targets=targets,
                )
                
                total_loss += loss_dict["loss"].item() * len(batch["instructions"])
                
                # Compute distance
                pred_xy = predictions["xy"]
                dist = torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1)).mean()
                total_dist += dist.item() * len(batch["instructions"])
                num_samples += len(batch["instructions"])
        
        avg_loss = total_loss / max(1, num_samples)
        avg_dist = total_dist / max(1, num_samples)
        print(f"  Average distance: {avg_dist:.4f}")
        
        return avg_loss
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "args": vars(self.args),
        }
        
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        print(f"  Resumed from epoch {self.epoch}, step {self.global_step}")


def train_ocumamba_lite(
    model_size: str = "tiny",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    data_dir: Optional[str] = None,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    """
    Convenience function to train OcuMamba-Lite.
    
    Args:
        model_size: "tiny", "small", or "base"
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        data_dir: Path to ScreenSpot-Pro data (optional)
        device: Device to use
        checkpoint_dir: Where to save checkpoints
    """
    class Args:
        pass
    
    args = Args()
    args.model_size = model_size
    args.image_size = 1024
    args.data_dir = data_dir
    args.synthetic_samples = 50000
    args.synthetic_ratio = 0.7 if data_dir else 1.0
    args.epochs = epochs
    args.batch_size = batch_size
    args.grad_accum = 4
    args.lr = lr
    args.weight_decay = 0.01
    args.warmup_steps = 500
    args.num_workers = 4
    args.device = device
    args.checkpoint_dir = checkpoint_dir
    args.log_interval = 50
    args.save_interval = 1000
    args.resume = None
    
    trainer = Trainer(args)
    trainer.train()
    
    return trainer.model


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
