#!/usr/bin/env python3
"""
GPT-5.2 Benchmark on ScreenSpot-Pro - OFFICIAL FORMAT
Based on: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/models/gpt5.py

Key settings:
- Model: gpt-5-2025-08-07
- Image FIRST, then text prompt
- Normalized coordinates (0-1)
- temperature=0.0
- max_tokens=16384
- reasoning_effort="high"
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
print("GPT-5.2 - ScreenSpot-Pro (Official Format)")
print("="*60)

client = OpenAI()

# Budget
INPUT_COST_PER_M = 1.75
OUTPUT_COST_PER_M = 14.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 2.00

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")

# Official prompts from ScreenSpot-Pro repo
SYSTEM_PROMPT = "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."

USER_PROMPT_TEMPLATE = """You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.
Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1.
The instruction is:
{instruction}"""


def encode_image(image):
    """Convert to RGB PNG and base64 encode."""
    image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M


def predict(image, instruction):
    """Official GPT-5.2 prediction format."""
    global total_input_tokens, total_output_tokens
    
    b64 = encode_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        # Image FIRST (official order)
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT_TEMPLATE.format(instruction=instruction),
                        },
                    ],
                },
            ],
            max_completion_tokens=8000,
            reasoning_effort="high",
        )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        result = response.choices[0].message.content
        if not result:
            return None, "Empty response"
            
        result = result.strip()
        
        # Parse [[x0,y0,x1,y1]] format
        match = re.search(r'\[\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]\]', result)
        if match:
            x0, y0, x1, y1 = map(float, match.groups())
            # Return center point (normalized)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            return (cx, cy), result
        return None, result
        
    except Exception as e:
        return None, str(e)


# Run benchmark
correct = 0
total = 0
errors = 0
results = []

print(f"\nBudget: ${BUDGET_LIMIT:.2f}")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 100 or get_cost() >= BUDGET_LIMIT:
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
        x1_norm, y1_norm = x1 / orig_w, y1 / orig_h
        x2_norm, y2_norm = x2 / orig_w, y2 / orig_h
        
        pred, response = predict(image, instruction)
        
        if pred is not None:
            cx, cy = pred
            # Check if center is inside normalized bbox
            is_inside = (x1_norm <= cx <= x2_norm) and (y1_norm <= cy <= y2_norm)
            
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            print(f"[{total:2d}] {status} pred=({cx:.3f},{cy:.3f}) target=[{x1_norm:.3f},{y1_norm:.3f},{x2_norm:.3f},{y2_norm:.3f}] ${get_cost():.4f}")
            
            results.append({
                "idx": idx,
                "instruction": instruction[:80],
                "pred": [cx, cy],
                "bbox_norm": [x1_norm, y1_norm, x2_norm, y2_norm],
                "correct": is_inside,
                "response": response[:200] if response else ""
            })
        else:
            errors += 1
            if errors <= 5:
                print(f"[{idx}] Error: {response[:80] if response else 'unknown'}")
        
        time.sleep(0.3)
        
    except Exception as e:
        errors += 1
        print(f"Exception: {e}")

# Results
print("\n" + "="*60)
print("RESULTS: GPT-5.2 (Official Format)")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Errors: {errors}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

with open("gpt52_official_results.json", "w") as f:
    json.dump({
        "model": "gpt-5-2025-08-07",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "cost": get_cost(),
        "results": results
    }, f, indent=2)
print("Saved to gpt52_official_results.json")
