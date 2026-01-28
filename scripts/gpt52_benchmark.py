#!/usr/bin/env python3
"""
GPT-5.2 Thinking Benchmark on ScreenSpot-Pro
Budget: $0.30 for 50 samples

Pricing:
- Input: $1.75 / 1M tokens
- Output: $14.00 / 1M tokens
"""
import os
import base64
import json
import re
import time
import pandas as pd
import ast
from io import BytesIO
from PIL import Image
import openai

print("="*60)
print("GPT-5.2 Thinking - ScreenSpot-Pro Benchmark")
print("="*60)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)

client = openai.OpenAI(api_key=api_key)

# Model - GPT-5.2 Thinking
MODEL = "gpt-5.2-thinking"  # May need to adjust based on actual API name
print(f"Using model: {MODEL}")

# Budget tracking
INPUT_COST_PER_M = 1.75  # $1.75 per million input tokens
OUTPUT_COST_PER_M = 14.00  # $14 per million output tokens
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 0.30  # $0.30 limit for Phase 1

# Load test data
test_path = "/workspace/.hf_home/hub/datasets--TIGER-Lab--ScreenSpot-Pro/snapshots/710340f16b943d995d9422f52cfe3476444e6964/test/data-00000-of-00001.parquet"
df = pd.read_parquet(test_path).head(50)
print(f"Test samples: {len(df)}")

def encode_image(image):
    """Convert PIL Image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_current_cost():
    """Calculate current spend."""
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return input_cost + output_cost

def get_click_location(image, instruction, img_size):
    """Use GPT-5.2 Thinking to predict click location."""
    global total_input_tokens, total_output_tokens
    
    orig_w, orig_h = img_size
    
    # Resize to manage token costs (1024px max)
    if max(image.width, image.height) > 1024:
        scale = 1024 / max(image.width, image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    base64_image = encode_image(image)
    
    prompt = f"""You are a GUI automation assistant. Given a screenshot and an instruction, return the EXACT pixel coordinates where to click.

Original image resolution: {orig_w}x{orig_h}
Current image resolution: {image.width}x{image.height}

Instruction: {instruction}

Think step by step:
1. Identify what element needs to be clicked
2. Locate it in the screenshot
3. Calculate the center coordinates

Respond with JSON: {{"x": <pixel_x>, "y": <pixel_y>, "element": "<description>"}}
Scale coordinates to ORIGINAL resolution ({orig_w}x{orig_h}).
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
        )
        
        # Track tokens
        usage = response.usage
        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            coords = json.loads(json_match.group())
            x = coords.get("x", 0)
            y = coords.get("y", 0)
            return x, y, result
        else:
            return None, None, result
            
    except Exception as e:
        print(f"API Error: {e}")
        return None, None, str(e)

# Run benchmark
correct = 0
total = 0
errors = 0
results = []

print("\n" + "="*60)
print(f"Budget: ${BUDGET_LIMIT:.2f} | Starting benchmark...")
print("="*60)

for idx, row in df.iterrows():
    # Check budget
    current_cost = get_current_cost()
    if current_cost >= BUDGET_LIMIT:
        print(f"\n⚠️ Budget limit reached: ${current_cost:.4f}")
        break
    
    try:
        img_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        img_size = ast.literal_eval(row["img_size"])
        orig_w, orig_h = img_size[0], img_size[1]
        
        instruction = row["instruction"]
        bbox = ast.literal_eval(row["bbox"])
        x1, y1, x2, y2 = bbox
        
        pred_x, pred_y, reasoning = get_click_location(image, instruction, img_size)
        
        if pred_x is not None:
            is_inside = (x1 <= pred_x <= x2) and (y1 <= pred_y <= y2)
            
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            current_cost = get_current_cost()
            print(f"[{total:2d}] {status} pred=({pred_x:.0f},{pred_y:.0f}) bbox=[{x1},{y1},{x2},{y2}] cost=${current_cost:.4f}")
            
            results.append({
                "idx": idx,
                "instruction": instruction,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "bbox": bbox,
                "correct": is_inside,
                "reasoning": reasoning
            })
        else:
            errors += 1
            
        # Rate limiting
        time.sleep(0.3)
        
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"Error: {e}")

# Final stats
print("\n" + "="*60)
print(f"RESULTS: {MODEL}")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
final_cost = get_current_cost()
print(f"Point-in-Box Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Errors: {errors}")
print(f"Total Input Tokens: {total_input_tokens:,}")
print(f"Total Output Tokens: {total_output_tokens:,}")
print(f"Total Cost: ${final_cost:.4f}")
print("="*60)

# Save results with reasoning for distillation
import json
with open("/workspace/gpt52_results.json", "w") as f:
    json.dump({
        "model": MODEL,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "cost": final_cost,
        "results": results
    }, f, indent=2)
print(f"\nResults saved to /workspace/gpt52_results.json")
