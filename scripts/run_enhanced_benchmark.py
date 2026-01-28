#!/usr/bin/env python3
"""
Enhanced Plura Grounding Benchmark.

This script runs the improved Plura grounding system with:
1. LayoutLMv3 semantic verification (§2.1.2)
2. Tiny target proposals (edge icons, high-contrast, corners)
3. R = Saliency × P(SemanticMatch) scoring

Usage:
    python scripts/run_enhanced_benchmark.py --limit 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def hit_bbox(x: float, y: float, bbox: List[float]) -> bool:
    """Check if point (x, y) is inside bbox [x0, y0, x1, y1]."""
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def dist_to_center(x: float, y: float, bbox: List[float]) -> float:
    """Compute distance from point to bbox center."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return float(np.sqrt((x - cx)**2 + (y - cy)**2))


def run_benchmark(
    samples: List[Dict[str, Any]],
    *,
    use_semantic: bool = True,
    use_tiny_proposals: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run enhanced benchmark on samples.
    
    Args:
        samples: List of sample dicts with 'img', 'instruction', 'bbox'
        use_semantic: Enable LayoutLMv3 semantic verification
        use_tiny_proposals: Enable tiny target proposals
        verbose: Print per-sample results
    
    Returns:
        Results dict with accuracy metrics
    """
    from Backend.indexing.enhanced_grounding import (
        enhanced_predict_click,
        EnhancedConfig,
    )
    from Backend.indexing.tiny_target_proposals import generate_tiny_target_proposals
    from Backend.indexing.visual_saliency import compute_saliency
    from Backend.indexing.layoutlm_grounding import (
        compute_semantic_scores,
        compute_combined_score,
    )
    from Backend.common.math_utils import clamp01
    
    import ueyes_eval
    import io
    from PIL import Image
    
    # Configure
    config = EnhancedConfig(
        use_semantic_verification=use_semantic,
        use_visual_physics=True,
        saliency_weight=0.35,
        semantic_weight=0.65,
    )
    
    # Results
    hits = 0
    total = 0
    distances = []
    latencies = []
    methods_used = {"base": 0, "tiny_proposals": 0, "semantic": 0}
    
    for i, sample in enumerate(samples):
        img = sample["img"]
        instruction = sample["instruction"]
        bbox = sample["bbox"]
        
        if not bbox or not img:
            continue
        
        total += 1
        t0 = time.time()
        
        try:
            # Encode image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            
            # Decode for processing
            rgb = ueyes_eval._decode_image_bytes(img_bytes)
            h, w = rgb.shape[:2]
            
            # Method 1: Try enhanced grounding with semantic verification
            result = enhanced_predict_click(
                image_bytes=img_bytes,
                instruction=instruction,
                config=config,
            )
            
            pred_x = result.get("x_norm", 0.5)
            pred_y = result.get("y_norm", 0.5)
            confidence = result.get("confidence", 0)
            
            # Method 2: If low confidence, try tiny target proposals
            if use_tiny_proposals and confidence < 0.4:
                tiny_props = generate_tiny_target_proposals(
                    rgb,
                    instruction=instruction,
                )
                
                if tiny_props:
                    # Compute saliency
                    sal = compute_saliency(rgb)
                    
                    # Score tiny proposals with semantic matching
                    if use_semantic:
                        tiny_props = compute_semantic_scores(
                            rgb, instruction, tiny_props,
                            use_embeddings=True,
                        )
                    
                    # Find best tiny proposal
                    best_tiny = None
                    best_score = -1
                    
                    for prop in tiny_props:
                        # Get bbox in pixels
                        pb = prop.get("bbox", [0, 0, 0, 0])
                        if pb[2] <= pb[0] or pb[3] <= pb[1]:
                            continue
                        
                        # Get saliency
                        sx0 = int(pb[0] / w * sal.shape[1])
                        sy0 = int(pb[1] / h * sal.shape[0])
                        sx1 = int(pb[2] / w * sal.shape[1])
                        sy1 = int(pb[3] / h * sal.shape[0])
                        
                        if sx1 <= sx0 or sy1 <= sy0:
                            continue
                        
                        region_sal = sal[sy0:sy1, sx0:sx1]
                        sal_score = float(np.mean(region_sal)) if region_sal.size > 0 else 0
                        
                        # Get semantic score
                        sem_score = prop.get("semantic_score", 0)
                        
                        # Combined score
                        score = compute_combined_score(sal_score, sem_score)
                        
                        if score > best_score:
                            best_score = score
                            best_tiny = prop
                    
                    # Use tiny proposal if better than base
                    if best_tiny and best_score > confidence:
                        pb = best_tiny.get("bbox", [0, 0, 0, 0])
                        pred_x = (pb[0] + pb[2]) / 2 / w
                        pred_y = (pb[1] + pb[3]) / 2 / h
                        confidence = best_score
                        methods_used["tiny_proposals"] += 1
                    else:
                        methods_used["base"] += 1
                else:
                    methods_used["base"] += 1
            else:
                if use_semantic:
                    methods_used["semantic"] += 1
                else:
                    methods_used["base"] += 1
            
            latency = time.time() - t0
            latencies.append(latency)
            
            # Check hit
            is_hit = hit_bbox(pred_x, pred_y, bbox)
            dist = dist_to_center(pred_x, pred_y, bbox)
            distances.append(dist)
            
            if is_hit:
                hits += 1
                status = "✓ HIT"
            else:
                status = f"✗ MISS (dist={dist:.3f})"
            
            if verbose:
                print(f"[{i+1}/{len(samples)}] {status} | {instruction[:40]}...")
        
        except Exception as e:
            if verbose:
                print(f"[{i+1}/{len(samples)}] ERROR: {e}")
    
    # Compute summary
    accuracy = hits / total if total > 0 else 0
    mean_dist = float(np.mean(distances)) if distances else 1.0
    mean_latency = float(np.mean(latencies)) if latencies else 0
    
    return {
        "total": total,
        "hits": hits,
        "accuracy": accuracy,
        "mean_distance": mean_dist,
        "mean_latency": mean_latency,
        "methods_used": methods_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Run enhanced Plura benchmark")
    parser.add_argument("--limit", type=int, default=20, help="Limit samples")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic")
    parser.add_argument("--no-tiny", action="store_true", help="Disable tiny proposals")
    args = parser.parse_args()
    
    print("Loading ScreenSpot-Pro samples...")
    
    # Load samples from HuggingFace
    from datasets import load_dataset
    import json as json_lib
    
    ds = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
    
    # Get samples with shuffling
    import random
    rng = random.Random(42)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:args.limit]
    
    samples = []
    for idx in indices:
        example = ds[idx]
        
        img = example.get("image")
        instruction = str(example.get("instruction") or "")
        bbox = example.get("bbox")
        img_size = example.get("img_size")
        
        # Parse bbox
        if isinstance(bbox, str):
            try:
                bbox = json_lib.loads(bbox)
            except:
                bbox = None
        
        if isinstance(img_size, str):
            try:
                img_size = json_lib.loads(img_size)
            except:
                img_size = None
        
        # Normalize bbox
        if bbox and img_size and len(bbox) == 4 and len(img_size) == 2:
            w0, h0 = float(img_size[0]), float(img_size[1])
            if w0 > 0 and h0 > 0:
                x0, y0, x1, y1 = [float(v) for v in bbox]
                bbox = [x0/w0, y0/h0, x1/w0, y1/h0]
            else:
                bbox = None
        else:
            bbox = None
        
        if img and bbox and instruction:
            samples.append({
                "img": img,
                "instruction": instruction,
                "bbox": bbox,
            })
    
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No valid samples found!")
        return 1
    
    print("\nRunning enhanced benchmark...")
    print(f"  Semantic verification: {not args.no_semantic}")
    print(f"  Tiny target proposals: {not args.no_tiny}")
    print()
    
    results = run_benchmark(
        samples,
        use_semantic=not args.no_semantic,
        use_tiny_proposals=not args.no_tiny,
        verbose=args.verbose,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("ENHANCED PLURA BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total samples:    {results['total']}")
    print(f"Hits:             {results['hits']}")
    print(f"Accuracy:         {results['accuracy']*100:.1f}%")
    print(f"Mean distance:    {results['mean_distance']:.4f}")
    print(f"Mean latency:     {results['mean_latency']:.2f}s")
    print()
    print("Methods used:")
    for method, count in results["methods_used"].items():
        print(f"  {method}: {count}")
    print("=" * 60)
    
    # Save results
    output_path = Path("enhanced_benchmark_results.json")
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
