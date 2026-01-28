#!/usr/bin/env python3
"""
GPT-5.2 Benchmark with Chain-of-Thought (CoT) Reasoning - FIXED

KEY FIXES:
1. REMOVED "Don't output any analysis" constraint (enables CoT)
2. INCREASED max_completion_tokens to 16384 (model needs room for reasoning + output)
3. Proper handling of truncated responses (finish_reason='length')

Based on debug findings:
- Sample 1 worked: 3776 reasoning tokens + output
- Samples 2-3 failed: 4096 tokens ALL used for reasoning, no output
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
print("GPT-5.2 with Chain-of-Thought (CoT) - FIXED")
print("="*60)

client = OpenAI()

# Budget tracking  
INPUT_COST_PER_M = 1.75
OUTPUT_COST_PER_M = 14.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 1.00

# Load dataset
print("Loading ScreenSpot-Pro...")
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded {len(dataset)} samples")


# CoT-ENABLED PROMPTS
SYSTEM_PROMPT = """You are an expert in visual grounding on graphical user interfaces. 
You have excellent spatial reasoning abilities and can precisely locate UI elements.
Think step by step to calculate accurate coordinates."""

USER_PROMPT_TEMPLATE = """Find the bounding box of a UI element in this screenshot.

Instruction: {instruction}

Steps:
1. Describe where the element is relative to screen landmarks
2. Estimate proportional position (e.g., "1/3 from left")
3. Calculate normalized coordinates (0 to 1)
4. Output bounding box as [[x0, y0, x1, y1]]

Image: {width}x{height} pixels. Coordinates normalized 0-1."""


def convert_pil_image_to_base64(image):
    """Convert image to base64 - NO RESIZING."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_cost():
    return (total_input_tokens / 1e6) * INPUT_COST_PER_M + (total_output_tokens / 1e6) * OUTPUT_COST_PER_M


def predict_with_cot(instruction, image, img_size):
    """GPT-5.2 with CoT reasoning - FIXED max_completion_tokens."""
    global total_input_tokens, total_output_tokens
    
    base64_image = convert_pil_image_to_base64(image)
    orig_w, orig_h = img_size
    prompt = USER_PROMPT_TEMPLATE.format(instruction=instruction, width=orig_w, height=orig_h)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            max_completion_tokens=16384,  # CRITICAL: increased from 4096
            reasoning_effort="high"
        )
        
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens
        
        # Check for truncation
        if response.choices[0].finish_reason == 'length':
            # Even if truncated, try to parse what we got
            content = response.choices[0].message.content or ""
            return content if content else "TRUNCATED"
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"ERROR: {str(e)}"


def parse_bbox(response_text):
    """Parse [[x0,y0,x1,y1]] format and return center point."""
    if not response_text or response_text.startswith("ERROR") or response_text == "TRUNCATED":
        return None, response_text
    
    match = re.search(r'\[\[([\d.]+),?\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)\]\]', response_text)
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
truncated = 0
results = []

print(f"\nModel: gpt-5-2025-08-07 with Chain-of-Thought")
print(f"Config: reasoning_effort=high, max_completion_tokens=16384 (fixed!)")
print(f"Key Fix: CoT enabled + sufficient token budget")
print(f"Budget limit: ${BUDGET_LIMIT:.2f}")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 20 or get_cost() >= BUDGET_LIMIT:
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
        
        response_text = predict_with_cot(instruction, image, (orig_w, orig_h))
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
                "correct": is_inside
            })
        else:
            if response_text == "TRUNCATED":
                truncated += 1
                print(f"[{idx}] TRUNCATED (model ran out of tokens)")
            else:
                errors += 1
                print(f"[{idx}] Error/Parse fail: {response_text[:80] if response_text else 'None'}")
        
        time.sleep(0.5)
        
    except Exception as e:
        errors += 1
        print(f"Exception on sample {idx}: {e}")

# Final results
print("\n" + "="*60)
print("RESULTS: GPT-5.2 with Chain-of-Thought (FIXED)")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Target: 86.3%")
print(f"Errors: {errors}")
print(f"Truncated: {truncated}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_cost():.4f}")

with open("gpt52_cot_fixed_results.json", "w") as f:
    json.dump({
        "model": "gpt-5-2025-08-07",
        "config": "cot_fixed_16k_tokens",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "truncated": truncated,
        "cost": get_cost(),
        "results": results
    }, f, indent=2)
print("Saved to gpt52_cot_fixed_results.json")
