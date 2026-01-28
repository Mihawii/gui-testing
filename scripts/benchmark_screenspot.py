#!/usr/bin/env python3
"""
ScreenSpot-Pro Benchmark Runner for Plura.

Runs the Plura click grounding system against ScreenSpot-Pro
and measures accuracy improvements.

Usage:
    python scripts/benchmark_screenspot.py --data-dir /path/to/screenspot_pro --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add Backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Backend.indexing.deterministic_click_grounding import predict_click, DEFAULT_CONFIG


@dataclass
class BenchmarkResult:
    """Result for a single benchmark sample."""
    
    index: int
    file_name: str
    instruction: str
    target_bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1 normalized
    
    # Prediction
    pred_x: float  # Normalized
    pred_y: float  # Normalized
    pred_confidence: float
    
    # Metrics
    hit: bool
    distance_to_center: float
    iou: float
    
    # Timing
    latency_seconds: float
    
    # Error if any
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""
    
    total: int = 0
    hits: int = 0
    errors: int = 0
    
    accuracy: float = 0.0
    mean_distance: float = 0.0
    mean_iou: float = 0.0
    mean_latency: float = 0.0
    
    # By category
    by_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_screenspot_pro(data_dir: Path, *, limit: int = 0) -> List[Dict[str, Any]]:
    """Load ScreenSpot-Pro benchmark data."""
    # Try common file patterns
    patterns = [
        "screenspot_pro.jsonl",
        "screenspot_pro.json",
        "*.jsonl",
        "images/*",
    ]
    
    samples = []
    
    # Try JSONL format
    jsonl_files = list(data_dir.glob("*.jsonl"))
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    
    # Try JSON format
    if not samples:
        json_files = list(data_dir.glob("*.json"))
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
    
    # Try CSV format
    if not samples:
        csv_files = list(data_dir.glob("*.csv"))
        for csv_file in csv_files:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append(row)
    
    if limit > 0:
        samples = samples[:limit]
    
    return samples


def compute_hit(
    pred_x: float,
    pred_y: float,
    bbox: Tuple[float, float, float, float],
) -> bool:
    """Check if prediction is inside target bbox."""
    x0, y0, x1, y1 = bbox
    return x0 <= pred_x <= x1 and y0 <= pred_y <= y1


def compute_distance(
    pred_x: float,
    pred_y: float,
    bbox: Tuple[float, float, float, float],
) -> float:
    """Compute distance from prediction to bbox center."""
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return float(np.sqrt((pred_x - cx)**2 + (pred_y - cy)**2))


def compute_iou(
    pred_x: float,
    pred_y: float,
    bbox: Tuple[float, float, float, float],
    *,
    pred_radius: float = 0.01,  # 1% of screen
) -> float:
    """Compute IoU between prediction point (as small box) and target bbox."""
    x0, y0, x1, y1 = bbox
    
    # Create prediction bbox (small circle approximated as square)
    px0 = pred_x - pred_radius
    py0 = pred_y - pred_radius
    px1 = pred_x + pred_radius
    py1 = pred_y + pred_radius
    
    # Intersection
    ix0 = max(x0, px0)
    iy0 = max(y0, py0)
    ix1 = min(x1, px1)
    iy1 = min(y1, py1)
    
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    
    # Union
    bbox_area = (x1 - x0) * (y1 - y0)
    pred_area = (px1 - px0) * (py1 - py0)
    union_area = bbox_area + pred_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return float(inter_area / union_area)


def run_benchmark(
    samples: List[Dict[str, Any]],
    data_dir: Path,
    *,
    config: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
    """Run benchmark on all samples."""
    results = []
    summary = BenchmarkSummary()
    
    distances = []
    ious = []
    latencies = []
    
    type_stats: Dict[str, Dict[str, List]] = {}
    source_stats: Dict[str, Dict[str, List]] = {}
    
    for i, sample in enumerate(samples):
        # Extract sample info
        file_name = sample.get("file_name") or sample.get("image") or sample.get("img") or ""
        instruction = sample.get("instruction") or sample.get("query") or ""
        
        # Extract bbox (handle multiple formats)
        bbox = None
        if "bbox" in sample:
            bbox = sample["bbox"]
        elif all(k in sample for k in ["bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"]):
            bbox = (
                float(sample["bbox_x0"]),
                float(sample["bbox_y0"]),
                float(sample["bbox_x1"]),
                float(sample["bbox_y1"]),
            )
        
        if bbox is None:
            continue
        
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = tuple(float(x) for x in bbox)
        else:
            continue
        
        data_type = sample.get("data_type") or sample.get("type") or "unknown"
        data_source = sample.get("data_source") or sample.get("source") or "unknown"
        
        # Initialize stats
        if data_type not in type_stats:
            type_stats[data_type] = {"hits": [], "distances": [], "latencies": []}
        if data_source not in source_stats:
            source_stats[data_source] = {"hits": [], "distances": [], "latencies": []}
        
        # Load image
        image_path = data_dir / file_name
        if not image_path.exists():
            # Try finding in subdirectories
            matches = list(data_dir.rglob(Path(file_name).name))
            if matches:
                image_path = matches[0]
            else:
                results.append(BenchmarkResult(
                    index=i,
                    file_name=file_name,
                    instruction=instruction,
                    target_bbox=bbox,
                    pred_x=0.0,
                    pred_y=0.0,
                    pred_confidence=0.0,
                    hit=False,
                    distance_to_center=1.0,
                    iou=0.0,
                    latency_seconds=0.0,
                    error=f"Image not found: {file_name}",
                ))
                summary.errors += 1
                continue
        
        try:
            image_bytes = image_path.read_bytes()
            
            t0 = time.time()
            prediction = predict_click(
                image_bytes=image_bytes,
                instruction=instruction,
                config=config,
                return_candidates=False,
            )
            latency = time.time() - t0
            
            # Extract prediction coordinates
            pred_x = prediction.get("x_norm") or prediction.get("x", 0) 
            pred_y = prediction.get("y_norm") or prediction.get("y", 0)
            confidence = prediction.get("confidence", 0.0)
            
            # Handle pixel coordinates
            if pred_x > 1.0:
                w = prediction.get("image", {}).get("width", 1920)
                pred_x = pred_x / w
            if pred_y > 1.0:
                h = prediction.get("image", {}).get("height", 1080)
                pred_y = pred_y / h
            
            hit = compute_hit(pred_x, pred_y, bbox)
            distance = compute_distance(pred_x, pred_y, bbox)
            iou = compute_iou(pred_x, pred_y, bbox)
            
            result = BenchmarkResult(
                index=i,
                file_name=file_name,
                instruction=instruction,
                target_bbox=bbox,
                pred_x=pred_x,
                pred_y=pred_y,
                pred_confidence=confidence,
                hit=hit,
                distance_to_center=distance,
                iou=iou,
                latency_seconds=latency,
            )
            results.append(result)
            
            if hit:
                summary.hits += 1
            
            distances.append(distance)
            ious.append(iou)
            latencies.append(latency)
            
            type_stats[data_type]["hits"].append(1 if hit else 0)
            type_stats[data_type]["distances"].append(distance)
            type_stats[data_type]["latencies"].append(latency)
            
            source_stats[data_source]["hits"].append(1 if hit else 0)
            source_stats[data_source]["distances"].append(distance)
            source_stats[data_source]["latencies"].append(latency)
            
            if verbose:
                status = "✓" if hit else "✗"
                print(f"[{i+1}/{len(samples)}] {status} {file_name[:30]:30s} dist={distance:.3f}")
        
        except Exception as e:
            results.append(BenchmarkResult(
                index=i,
                file_name=file_name,
                instruction=instruction,
                target_bbox=bbox,
                pred_x=0.0,
                pred_y=0.0,
                pred_confidence=0.0,
                hit=False,
                distance_to_center=1.0,
                iou=0.0,
                latency_seconds=0.0,
                error=str(e),
            ))
            summary.errors += 1
            if verbose:
                print(f"[{i+1}/{len(samples)}] ERROR: {e}")
        
        summary.total += 1
    
    # Compute summary stats
    if summary.total > 0:
        summary.accuracy = summary.hits / summary.total
    if distances:
        summary.mean_distance = float(np.mean(distances))
    if ious:
        summary.mean_iou = float(np.mean(ious))
    if latencies:
        summary.mean_latency = float(np.mean(latencies))
    
    # Aggregate by-type stats
    for dtype, stats in type_stats.items():
        if stats["hits"]:
            summary.by_type[dtype] = {
                "accuracy": float(np.mean(stats["hits"])),
                "count": len(stats["hits"]),
                "mean_distance": float(np.mean(stats["distances"])),
            }
    
    for source, stats in source_stats.items():
        if stats["hits"]:
            summary.by_source[source] = {
                "accuracy": float(np.mean(stats["hits"])),
                "count": len(stats["hits"]),
                "mean_distance": float(np.mean(stats["distances"])),
            }
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Run ScreenSpot-Pro benchmark")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to ScreenSpot-Pro data")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of samples")
    parser.add_argument("--output", type=str, default="", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample results")
    parser.add_argument("--config", type=str, default="", help="Config file path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Load config
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if config_path.exists():
            with open(config_path) as f:
                loaded = json.load(f)
                config.update(loaded)
    
    print(f"Loading samples from {data_dir}...")
    samples = load_screenspot_pro(data_dir, limit=args.limit)
    print(f"Loaded {len(samples)} samples")
    
    if not samples:
        print("No samples found!")
        return 1
    
    print("\nRunning benchmark...")
    results, summary = run_benchmark(
        samples,
        data_dir,
        config=config,
        verbose=args.verbose,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total samples:    {summary.total}")
    print(f"Hits:             {summary.hits}")
    print(f"Errors:           {summary.errors}")
    print(f"Accuracy:         {summary.accuracy * 100:.1f}%")
    print(f"Mean distance:    {summary.mean_distance:.4f}")
    print(f"Mean IoU:         {summary.mean_iou:.4f}")
    print(f"Mean latency:     {summary.mean_latency:.2f}s")
    
    if summary.by_type:
        print("\nBy Type:")
        for dtype, stats in sorted(summary.by_type.items()):
            print(f"  {dtype:15s}: {stats['accuracy']*100:5.1f}% ({stats['count']} samples)")
    
    if summary.by_source:
        print("\nBy Source:")
        for source, stats in sorted(summary.by_source.items()):
            print(f"  {source:15s}: {stats['accuracy']*100:5.1f}% ({stats['count']} samples)")
    
    # Save output
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_data = {
            "summary": {
                "total": summary.total,
                "hits": summary.hits,
                "errors": summary.errors,
                "accuracy": summary.accuracy,
                "mean_distance": summary.mean_distance,
                "mean_iou": summary.mean_iou,
                "mean_latency": summary.mean_latency,
                "by_type": summary.by_type,
                "by_source": summary.by_source,
            },
            "results": [
                {
                    "index": r.index,
                    "file_name": r.file_name,
                    "instruction": r.instruction,
                    "hit": r.hit,
                    "distance": r.distance_to_center,
                    "iou": r.iou,
                    "latency": r.latency_seconds,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
