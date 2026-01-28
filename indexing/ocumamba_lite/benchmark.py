"""
Benchmark script for OcuMamba-Lite on ScreenSpot-Pro.

Evaluates a trained model on the ScreenSpot-Pro test set and reports:
- Accuracy (within threshold)
- Mean distance error
- Near-miss rate
- Inference latency
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from PIL import Image


def load_screenspot_pro(data_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load ScreenSpot-Pro test data."""
    data_path = Path(data_path)
    
    # Try parquet first
    parquet_file = data_path / "test_data.parquet"
    if parquet_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            samples = df.to_dict('records')
        except ImportError:
            samples = []
    else:
        # Try JSON
        for name in ["test_data.json", "annotations.json", "data.json"]:
            json_file = data_path / name
            if json_file.exists():
                with open(json_file) as f:
                    samples = json.load(f)
                break
        else:
            raise FileNotFoundError(f"No data found in {data_path}")
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def evaluate_sample(
    model,
    image: Image.Image,
    instruction: str,
    target_x: float,
    target_y: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    # Preprocess image
    image_resized = image.resize((1024, 1024), Image.BILINEAR)
    image_tensor = torch.from_numpy(
        np.array(image_resized).transpose(2, 0, 1)
    ).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Tokenize instruction
    tokens = model.tokenizer([instruction], padding=True, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(
            pixel_values=image_tensor,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
    latency = time.time() - start_time
    
    # Get prediction
    pred_x = predictions["xy"][0, 0].item()
    pred_y = predictions["xy"][0, 1].item()
    confidence = predictions["confidence"][0, 0].item()
    
    # Compute distance
    distance = np.sqrt((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2)
    
    return {
        "pred_x": pred_x,
        "pred_y": pred_y,
        "target_x": target_x,
        "target_y": target_y,
        "distance": distance,
        "confidence": confidence,
        "latency": latency,
    }


def run_benchmark(
    model_path: str,
    data_path: str,
    threshold: float = 0.05,
    near_miss_threshold: float = 0.1,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run benchmark on ScreenSpot-Pro.
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to ScreenSpot-Pro data
        threshold: Distance threshold for correct prediction
        near_miss_threshold: Distance threshold for near-miss
        max_samples: Max samples to evaluate
        device: Device to use
        
    Returns:
        Dict with benchmark results
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    from .model import OcuMambaLite
    
    model = OcuMambaLite.from_config("tiny")
    model.load(model_path)
    model = model.to(device)
    model.eval()
    print(f"Model parameters: {model.num_parameters() / 1e6:.1f}M")
    
    # Load data
    samples = load_screenspot_pro(data_path, max_samples)
    data_dir = Path(data_path)
    
    # Evaluate
    results = []
    correct = 0
    near_misses = 0
    
    print(f"\nEvaluating {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        # Load image
        img_filename = sample.get("img_filename", sample.get("image", ""))
        img_path = data_dir / "images" / img_filename
        if not img_path.exists():
            img_path = data_dir / img_filename
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_filename}: {e}")
            continue
        
        orig_w, orig_h = image.size
        
        # Get target
        if "bbox" in sample:
            x1, y1, x2, y2 = sample["bbox"]
            target_x = (x1 + x2) / 2 / orig_w
            target_y = (y1 + y2) / 2 / orig_h
        elif "point" in sample:
            px, py = sample["point"]
            target_x = px / orig_w
            target_y = py / orig_h
        else:
            continue
        
        instruction = sample.get("instruction", sample.get("text", ""))
        
        # Evaluate
        result = evaluate_sample(
            model, image, instruction,
            target_x, target_y, device
        )
        result["filename"] = img_filename
        result["instruction"] = instruction
        results.append(result)
        
        # Check accuracy
        if result["distance"] < threshold:
            correct += 1
        elif result["distance"] < near_miss_threshold:
            near_misses += 1
        
        # Progress
        if (i + 1) % 10 == 0:
            acc = correct / (i + 1) * 100
            print(f"  [{i+1}/{len(samples)}] Accuracy: {acc:.1f}%")
    
    # Compute stats
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    near_miss_rate = near_misses / total * 100 if total > 0 else 0
    mean_dist = np.mean([r["distance"] for r in results])
    mean_latency = np.mean([r["latency"] for r in results])
    
    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Accuracy (d < {threshold}): {accuracy:.1f}% ({correct}/{total})")
    print(f"Near-misses (d < {near_miss_threshold}): {near_miss_rate:.1f}% ({near_misses}/{total})")
    print(f"Mean distance: {mean_dist:.4f}")
    print(f"Mean latency: {mean_latency * 1000:.1f}ms")
    print("=" * 50)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "near_misses": near_misses,
        "total": total,
        "mean_distance": mean_dist,
        "mean_latency": mean_latency,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark OcuMamba-Lite")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to ScreenSpot-Pro data")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Distance threshold for correct")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = run_benchmark(
        model_path=args.model,
        data_path=args.data,
        threshold=args.threshold,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    if args.output:
        # Save detailed results (without full results list for brevity)
        output = {k: v for k, v in results.items() if k != "results"}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
