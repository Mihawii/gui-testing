#!/usr/bin/env python3
"""
GPT-4o Benchmark on ScreenSpot-Pro (GPT-5.2 vision not working)
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
print("GPT-4o - ScreenSpot-Pro Benchmark")
print("="*60)

client = OpenAI()

# Budget
INPUT_COST_PER_M = 2.50  # GPT-4o pricing
OUTPUT_COST_PER_M = 10.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 2.50  # Use remaining budget

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")

def encode_image(image):
    image = image.convert('RGB')
    max_size = 1024
    if max(image.width, image.height) > max_size:
        scale = max_size / max(image.width, image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), image.width, image.height

def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M

def predict(image, instruction, orig_w, orig_h):
    global total_input_tokens, total_output_tokens
    
    b64, w, h = encode_image(image)
    
    prompt = f"""GUI grounding: Find the click location.
Original: {orig_w}x{orig_h}, Current: {w}x{h}
Instruction: {instruction}

Return ONLY JSON: {{"x": <int>, "y": <int>}} scaled to original resolution."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
                ]
            }],
            max_tokens=150
        )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        result = response.choices[0].message.content.strip()
        match = re.search(r'\{[^}]+\}', result)
        if match:
            coords = json.loads(match.group())
            return coords.get("x", 0), coords.get("y", 0), result
        return None, None, result
    except Exception as e:
        return None, None, str(e)

# Run
correct = 0
total = 0
results = []

print(f"\nBudget: ${BUDGET_LIMIT:.2f}")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 100 or get_cost() >= BUDGET_LIMIT:
        if get_cost() >= BUDGET_LIMIT:
            print(f"\n⚠️ Budget limit!")
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
            print(f"[{total:2d}] {status} ({pred_x},{pred_y}) bbox=[{x1},{y1},{x2},{y2}] ${get_cost():.4f}")
            
            results.append({
                "instruction": instruction,
                "pred": [pred_x, pred_y],
                "bbox": bbox,
                "correct": is_inside,
                "reasoning": reasoning
            })
        
        time.sleep(0.2)
    except Exception as e:
        print(f"Error: {e}")

# Results
print("\n" + "="*60)
print("RESULTS: GPT-4o")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

with open("gpt4o_results.json", "w") as f:
    json.dump({"model": "gpt-4o", "accuracy": accuracy, "correct": correct, "total": total, "cost": get_cost(), "results": results}, f, indent=2)
print("Saved to gpt4o_results.json")
