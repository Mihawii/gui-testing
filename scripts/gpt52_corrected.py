#!/usr/bin/env python3
"""
GPT-5.2 Benchmark on ScreenSpot-Pro - CORRECTED CONFIG v2

Key fixes based on research:
1. reasoning_effort='xhigh' (maximum, not 'high')  
2. tools=[{'type': 'code_interpreter'}] REQUIRED for 86% accuracy
3. Using image_url (base64) since image_file not supported in chat API
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
print("GPT-5.2 - ScreenSpot-Pro (CORRECTED v2: xhigh + code_interpreter)")
print("="*60)

client = OpenAI()

# Budget
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

# Official prompts
SYSTEM_PROMPT = "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."

USER_PROMPT_TEMPLATE = """You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.
Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1.
The instruction is:
{instruction}"""


def encode_image(image):
    """Convert to RGB PNG and base64 encode (no resizing for full resolution)."""
    image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M


def predict(image, instruction):
    """GPT-5.2 with CORRECTED config: xhigh + code_interpreter."""
    global total_input_tokens, total_output_tokens
    
    b64 = encode_image(image)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                        {"type": "text", "text": USER_PROMPT_TEMPLATE.format(instruction=instruction)},
                    ],
                },
            ],
            tools=[{"type": "code_interpreter"}],  # CRITICAL for 86%
            reasoning_effort="xhigh",  # CRITICAL: Maximum reasoning
            max_completion_tokens=16000,
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
print("Config: reasoning_effort=xhigh, tools=[code_interpreter], detail=high")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 50 or get_cost() >= BUDGET_LIMIT:
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
            is_inside = (x1_norm <= cx <= x2_norm) and (y1_norm <= cy <= y2_norm)
            
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            acc_so_far = correct / total * 100
            print(f"[{total:2d}] {status} pred=({cx:.3f},{cy:.3f}) bbox=[{x1_norm:.3f}-{x2_norm:.3f},{y1_norm:.3f}-{y2_norm:.3f}] acc={acc_so_far:.1f}% ${get_cost():.4f}")
            
            results.append({
                "idx": idx,
                "instruction": instruction[:80],
                "pred": [cx, cy],
                "bbox_norm": [x1_norm, y1_norm, x2_norm, y2_norm],
                "correct": is_inside,
            })
        else:
            errors += 1
            if errors <= 3:
                print(f"[{idx}] Error: {response[:100] if response else 'unknown'}")
        
        time.sleep(0.5)
        
    except Exception as e:
        errors += 1
        print(f"Exception: {e}")

# Results
print("\n" + "="*60)
print("RESULTS: GPT-5.2 (xhigh + code_interpreter)")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Expected: 86.3%")
print(f"Errors: {errors}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

with open("gpt52_corrected_results.json", "w") as f:
    json.dump({
        "model": "gpt-5.2",
        "config": "xhigh + code_interpreter + detail_high",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "cost": get_cost(),
        "results": results
    }, f, indent=2)
print("Saved to gpt52_corrected_results.json")
