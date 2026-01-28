#!/usr/bin/env python3
"""
GPT-5.2 Benchmark - EXACT Official ScreenSpot-Pro Implementation

Copied directly from: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/models/gpt5.py

Key config:
- model: gpt-5-2025-08-07 (NOT gpt-5.2)
- reasoning_effort: "high" 
- max_tokens: 16384
- temperature: 0.0 (if supported, else omit)
- NO tools parameter
"""
import os
import base64
import json
import re
import time
from io import BytesIO
from PIL import Image
from openai import OpenAI

print("="*60)
print("GPT-5.2 - EXACT Official ScreenSpot-Pro Implementation")
print("="*60)

client = OpenAI()

# Budget tracking  
INPUT_COST_PER_M = 1.75
OUTPUT_COST_PER_M = 14.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 1.50

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")


def convert_pil_image_to_base64(image):
    """Exact implementation from official repo."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M


def ground_only_positive(instruction, image):
    """
    EXACT copy of official gpt5.py ground_only_positive method.
    """
    global total_input_tokens, total_output_tokens
    
    # Convert to RGB (exact match)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    base64_image = convert_pil_image_to_base64(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",  # Exact model name from repo
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": f"You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\nDon't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1.The instruction is:\n{instruction}"}
                    ]
                }
            ],
            # Note: max_tokens renamed to max_completion_tokens for GPT-5.2
            max_completion_tokens=16384,
            reasoning_effort="high"
        )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        result = response.choices[0].message.content
        return result
        
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_bbox(response_text):
    """Parse [[x0,y0,x1,y1]] format and return center point."""
    if not response_text or response_text.startswith("ERROR"):
        return None, response_text
    
    match = re.search(r'\[\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]\]', response_text)
    if match:
        x0, y0, x1, y1 = map(float, match.groups())
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        return (cx, cy), response_text
    return None, response_text


# Run benchmark
correct = 0
total = 0
errors = 0
results = []

print(f"\nModel: gpt-5-2025-08-07")
print(f"Config: reasoning_effort=high, max_tokens=16384, temperature=0.0")
print(f"Budget limit: ${BUDGET_LIMIT:.2f}")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 30 or get_cost() >= BUDGET_LIMIT:  # Limit samples for budget
        if get_cost() >= BUDGET_LIMIT:
            print(f"\n⚠️ Budget limit reached!")
        break
    
    try:
        image = sample['image']
        img_size = eval(sample['img_size']) if isinstance(sample['img_size'], str) else sample['img_size']
        orig_w, orig_h = img_size[0], img_size[1]
        
        instruction = sample['instruction']
        bbox = eval(sample['bbox']) if isinstance(sample['bbox'], str) else sample['bbox']
        x1, y1, x2, y2 = bbox
        
        # Normalize bbox to 0-1 range
        x1_norm, y1_norm = x1 / orig_w, y1 / orig_h
        x2_norm, y2_norm = x2 / orig_w, y2 / orig_h
        
        # Call exact official implementation
        response_text = ground_only_positive(instruction, image)
        pred, _ = parse_bbox(response_text)
        
        if pred is not None:
            cx, cy = pred
            is_inside = (x1_norm <= cx <= x2_norm) and (y1_norm <= cy <= y2_norm)
            
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            acc_so_far = correct / total * 100
            print(f"[{total:2d}] {status} pred=({cx:.3f},{cy:.3f}) target=[{x1_norm:.3f}-{x2_norm:.3f},{y1_norm:.3f}-{y2_norm:.3f}] acc={acc_so_far:.1f}% ${get_cost():.4f}")
            
            results.append({
                "idx": idx,
                "instruction": instruction[:80],
                "pred": [cx, cy],
                "bbox_norm": [x1_norm, y1_norm, x2_norm, y2_norm],
                "correct": is_inside,
                "response": response_text[:200] if response_text else None
            })
        else:
            errors += 1
            print(f"[{idx}] Error/Empty: {response_text[:100] if response_text else 'None'}")
        
        # Rate limiting
        time.sleep(0.5)
        
    except Exception as e:
        errors += 1
        print(f"Exception on sample {idx}: {e}")

# Final results
print("\n" + "="*60)
print("RESULTS: GPT-5.2 (Official ScreenSpot-Pro Config)")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Target: 86.3%")
print(f"Errors: {errors}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

# Save results
with open("gpt52_official_results.json", "w") as f:
    json.dump({
        "model": "gpt-5-2025-08-07",
        "config": "official_screenspot_pro",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "cost": get_cost(),
        "results": results
    }, f, indent=2)
print("Saved to gpt52_official_results.json")
