#!/usr/bin/env python3
"""
GPT-5.2 / O1 Benchmark - CORRECT Configuration

Based on user analysis of the 86.3% result, the key requirements are:
1. Use a REASONING model (o1-preview, o1-mini, or gpt-5.2-thinking)
2. Native resolution with detail="high" (no downscaling)
3. "Constrained output, unconstrained thought" prompting
4. Point-in-box metric (center point falls inside target)

This script tries multiple models to find which gives best results.
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
print("GUI Grounding - Correct Configuration Test")
print("="*60)

client = OpenAI()

# Budget tracking  
INPUT_COST_PER_M = 15.00  # o1 pricing
OUTPUT_COST_PER_M = 60.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 0.50

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")


def convert_pil_image_to_base64(image):
    """Convert image to base64 - NO RESIZING for native resolution."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=False)
    return base64.b64encode(buffered.getvalue()).decode()


def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M


def predict_with_reasoning_model(instruction, image, img_size, model="o1-preview"):
    """
    Use reasoning model with correct configuration:
    - Native resolution image
    - detail="high" to prevent downscaling
    - Constrained output format
    """
    global total_input_tokens, total_output_tokens
    
    base64_image = convert_pil_image_to_base64(image)
    orig_w, orig_h = img_size
    
    # "Constrained output, unconstrained thought" approach
    # The model can think step-by-step internally but outputs only coordinates
    prompt = f"""You are finding a UI element in a {orig_w}x{orig_h} screenshot.

Task: {instruction}

Think step-by-step about the pixel coordinates:
1. Where is this element relative to screen corners?
2. Calculate the approximate x,y position as fraction of screen size
3. Output ONLY the bounding box as [[x0, y0, x1, y1]] with coordinates normalized 0-1

Output format: [[x0, y0, x1, y1]]"""

    try:
        # Try different model configurations
        if model.startswith("o1"):
            # O1 models use different API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # CRITICAL: native resolution
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_completion_tokens=4096
            )
        else:
            # GPT-5 style with reasoning_effort
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at finding UI elements. Think carefully about coordinates."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # CRITICAL: native resolution
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_completion_tokens=16384,
                reasoning_effort="high"
            )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_bbox(response_text):
    """Parse [[x0,y0,x1,y1]] and return center point."""
    if not response_text or response_text.startswith("ERROR"):
        return None, response_text
    
    match = re.search(r'\[\[([\d.]+),?\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)\]\]', response_text)
    if match:
        x0, y0, x1, y1 = map(float, match.groups())
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        return (cx, cy), response_text
    return None, response_text


def check_hit(pred, bbox_norm):
    """Point-in-box check (the actual ScreenSpot metric)."""
    cx, cy = pred
    x1, y1, x2, y2 = bbox_norm
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


# Test with different models
MODELS_TO_TRY = [
    "gpt-5-2025-08-07",  # With reasoning_effort
    # "o1-preview",      # Uncomment if you have access
    # "o1-mini",         # Uncomment if you have access
]

for model in MODELS_TO_TRY:
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"Config: detail='high' (native res), reasoning enabled")
    print(f"{'='*60}")
    
    correct = 0
    total = 0
    errors = 0
    
    for idx, sample in enumerate(dataset):
        if idx >= 10 or get_cost() >= BUDGET_LIMIT:
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
            
            # Normalize bbox to 0-1
            bbox_norm = [x1/orig_w, y1/orig_h, x2/orig_w, y2/orig_h]
            
            response = predict_with_reasoning_model(instruction, image, (orig_w, orig_h), model)
            pred, _ = parse_bbox(response)
            
            if pred is not None:
                is_hit = check_hit(pred, bbox_norm)
                if is_hit:
                    correct += 1
                total += 1
                
                status = "✓" if is_hit else "✗"
                acc = correct / total * 100
                print(f"[{total:2d}] {status} pred=({pred[0]:.3f},{pred[1]:.3f}) target=[{bbox_norm[0]:.3f}-{bbox_norm[2]:.3f},{bbox_norm[1]:.3f}-{bbox_norm[3]:.3f}] acc={acc:.1f}% ${get_cost():.4f}")
            else:
                errors += 1
                print(f"[{idx}] Error: {response[:80] if response else 'None'}")
            
            time.sleep(0.5)
            
        except Exception as e:
            errors += 1
            print(f"Exception: {e}")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{model} Result: {accuracy:.1f}% ({correct}/{total})")

print("\n" + "="*60)
print("TEST COMPLETE")
print(f"Total cost: ${get_cost():.4f}")
print("="*60)
