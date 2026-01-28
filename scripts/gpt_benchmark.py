#!/usr/bin/env python3
"""
Benchmark GPT-4V / GPT-5 on ScreenSpot-Pro

Requires: OPENAI_API_KEY environment variable
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
print("GPT-4V/5 GUI Grounding Benchmark")
print("="*60)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: Set OPENAI_API_KEY environment variable")
    exit(1)

client = openai.OpenAI(api_key=api_key)

# Model to test
MODEL = "gpt-4o"  # Change to "gpt-5" if available
print(f"Using model: {MODEL}")

# Load test data
test_path = "/workspace/.hf_home/hub/datasets--TIGER-Lab--ScreenSpot-Pro/snapshots/710340f16b943d995d9422f52cfe3476444e6964/test/data-00000-of-00001.parquet"
df = pd.read_parquet(test_path).head(50)  # Limit to 50 due to API cost
print(f"Test samples: {len(df)}")

def encode_image(image):
    """Convert PIL Image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_click_location(image, instruction, img_size):
    """Use GPT-4V to predict click location."""
    orig_w, orig_h = img_size
    
    # Resize for API (max 2048px)
    if max(image.width, image.height) > 2048:
        scale = 2048 / max(image.width, image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    base64_image = encode_image(image)
    
    prompt = f"""You are a GUI automation assistant. Given a screenshot and an instruction, return the EXACT pixel coordinates where to click.

Instruction: {instruction}

Respond ONLY with JSON in this exact format:
{{"x": <pixel_x>, "y": <pixel_y>}}

The image is {image.width}x{image.height} pixels. Give coordinates within this resolution.
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
            max_tokens=100,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            coords = json.loads(json_match.group())
            x = coords.get("x", 0)
            y = coords.get("y", 0)
            
            # Scale back to original resolution
            scale_x = orig_w / image.width
            scale_y = orig_h / image.height
            
            return x * scale_x, y * scale_y
        else:
            return None, None
            
    except Exception as e:
        print(f"API Error: {e}")
        return None, None

# Run benchmark
correct = 0
total = 0
errors = 0

print("\n" + "="*60)

for idx, row in df.iterrows():
    try:
        img_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        img_size = ast.literal_eval(row["img_size"])
        orig_w, orig_h = img_size[0], img_size[1]
        
        instruction = row["instruction"]
        bbox = ast.literal_eval(row["bbox"])
        x1, y1, x2, y2 = bbox
        
        pred_x, pred_y = get_click_location(image, instruction, img_size)
        
        if pred_x is not None:
            is_inside = (x1 <= pred_x <= x2) and (y1 <= pred_y <= y2)
            
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            if total <= 10 or total % 10 == 0:
                print(f"[{total:2d}] {status} pred=({pred_x:.0f},{pred_y:.0f}) bbox=[{x1},{y1},{x2},{y2}]")
        else:
            errors += 1
            
        # Rate limiting
        time.sleep(0.5)
        
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"Error: {e}")

print("\n" + "="*60)
print(f"RESULTS: {MODEL}")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Point-in-Box Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Errors: {errors}")
print("="*60)
