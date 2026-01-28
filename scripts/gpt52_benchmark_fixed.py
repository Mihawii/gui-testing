#!/usr/bin/env python3
"""
GPT-5.2 Benchmark on ScreenSpot-Pro - Fixed Version
Uses reasoning_effort=high and properly formatted images
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
print("GPT-5.2 - ScreenSpot-Pro Benchmark (Fixed)")
print("="*60)

client = OpenAI()

# Budget 
INPUT_COST_PER_M = 1.75
OUTPUT_COST_PER_M = 14.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 2.50

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")

def encode_image(image):
    # Convert to RGB, resize to smaller size for reliability
    image = image.convert('RGB')
    max_size = 768  # Smaller for reliability
    if max(image.width, image.height) > max_size:
        scale = max_size / max(image.width, image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Use PNG not JPEG
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), image.width, image.height

def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M

def predict(image, instruction, orig_w, orig_h):
    global total_input_tokens, total_output_tokens
    
    b64, w, h = encode_image(image)
    
    # Scale factors for converting back
    scale_x = orig_w / w
    scale_y = orig_h / h
    
    prompt = f"""GUI click task. Original resolution: {orig_w}x{orig_h}, displayed at: {w}x{h}.
Instruction: "{instruction}"

Find the exact center of the UI element to click. Think carefully about what element matches the instruction.

Return ONLY JSON: {{"x": <int>, "y": <int>}} in ORIGINAL resolution ({orig_w}x{orig_h})."""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}}
                ]
            }],
            reasoning_effort="high",
            max_completion_tokens=200
        )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        result = response.choices[0].message.content
        if not result:
            return None, None, "Empty response"
            
        result = result.strip()
        match = re.search(r'\{[^}]+\}', result)
        if match:
            coords = json.loads(match.group())
            return int(coords.get("x", 0)), int(coords.get("y", 0)), result
        return None, None, result
    except Exception as e:
        return None, None, str(e)

# Run
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
        
        pred_x, pred_y, reasoning = predict(image, instruction, orig_w, orig_h)
        
        if pred_x is not None:
            is_inside = (x1 <= pred_x <= x2) and (y1 <= pred_y <= y2)
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            print(f"[{total:2d}] {status} ({pred_x},{pred_y}) target=[{x1},{y1},{x2},{y2}] ${get_cost():.4f}")
            
            results.append({
                "idx": idx,
                "instruction": instruction[:100],
                "pred": [pred_x, pred_y],
                "bbox": bbox,
                "correct": is_inside,
                "reasoning": reasoning[:200] if reasoning else ""
            })
        else:
            errors += 1
            if errors <= 5:
                print(f"[{idx}] Error: {reasoning[:80]}")
        
        time.sleep(0.3)
    except Exception as e:
        errors += 1
        print(f"Exception: {e}")

# Results
print("\n" + "="*60)
print("RESULTS: GPT-5.2")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Errors: {errors}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

with open("gpt52_final_results.json", "w") as f:
    json.dump({
        "model": "gpt-5.2",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "cost": get_cost(),
        "results": results
    }, f, indent=2)
print("Saved to gpt52_final_results.json")
