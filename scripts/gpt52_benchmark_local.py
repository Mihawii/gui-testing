#!/usr/bin/env python3
"""
GPT-5.2 Thinking Benchmark on ScreenSpot-Pro (Local Version)
Budget: $0.30 for ~50 samples
"""
import os
import base64
import json
import re
import time
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

# Try different model names
MODEL_OPTIONS = ["gpt-5.2-thinking", "gpt-5.2", "o1", "gpt-4o"]
MODEL = None

# Test which model is available
print("Testing available models...")
for model_name in MODEL_OPTIONS:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10
        )
        MODEL = model_name
        print(f"✓ Using model: {MODEL}")
        break
    except Exception as e:
        print(f"✗ {model_name}: {str(e)[:50]}...")

if not MODEL:
    print("No compatible model found!")
    exit(1)

# Budget tracking
INPUT_COST_PER_M = 1.75 if "5.2" in MODEL else 2.50  # GPT-5.2 vs GPT-4o pricing
OUTPUT_COST_PER_M = 14.00 if "5.2" in MODEL else 10.00
total_input_tokens = 0
total_output_tokens = 0
BUDGET_LIMIT = 0.30

# Load test data from HuggingFace
print("\nLoading ScreenSpot-Pro dataset...")
try:
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
    print(f"Loaded {len(dataset)} test samples")
except Exception as e:
    print(f"Dataset load error: {e}")
    print("Installing datasets...")
    os.system("pip install datasets -q")
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")

def encode_image(image):
    """Convert PIL Image to base64."""
    buffered = BytesIO()
    # Resize to reduce token cost
    if max(image.width, image.height) > 1024:
        scale = 1024 / max(image.width, image.height)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image.save(buffered, format="PNG", optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8"), image.width, image.height

def get_current_cost():
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    return input_cost + output_cost

def get_click_location(image, instruction, orig_w, orig_h):
    global total_input_tokens, total_output_tokens
    
    base64_image, img_w, img_h = encode_image(image)
    
    prompt = f"""You are a GUI automation assistant. Given a screenshot and an instruction, return the EXACT pixel coordinates where to click.

Original image: {orig_w}x{orig_h}, Current: {img_w}x{img_h}
Instruction: {instruction}

Think step by step, then respond with JSON: {{"x": <pixel_x>, "y": <pixel_y>, "element": "<what you clicked>"}}
Scale coordinates to original resolution ({orig_w}x{orig_h}).
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }],
            max_tokens=500,
        )
        
        usage = response.usage
        total_input_tokens += usage.prompt_tokens
        total_output_tokens += usage.completion_tokens
        
        result = response.choices[0].message.content.strip()
        
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            coords = json.loads(json_match.group())
            return coords.get("x", 0), coords.get("y", 0), result
        return None, None, result
            
    except Exception as e:
        return None, None, str(e)

# Run benchmark
correct = 0
total = 0
errors = 0
results = []

print("\n" + "="*60)
print(f"Budget: ${BUDGET_LIMIT:.2f} | Starting benchmark...")
print("="*60)

for idx, sample in enumerate(dataset):
    if idx >= 50:  # Limit to 50 samples
        break
        
    current_cost = get_current_cost()
    if current_cost >= BUDGET_LIMIT:
        print(f"\n⚠️ Budget reached: ${current_cost:.4f}")
        break
    
    try:
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image["bytes"])).convert("RGB")
        
        img_size = eval(sample["img_size"]) if isinstance(sample["img_size"], str) else sample["img_size"]
        orig_w, orig_h = img_size[0], img_size[1]
        
        instruction = sample["instruction"]
        bbox = eval(sample["bbox"]) if isinstance(sample["bbox"], str) else sample["bbox"]
        x1, y1, x2, y2 = bbox
        
        pred_x, pred_y, reasoning = get_click_location(image, instruction, orig_w, orig_h)
        
        if pred_x is not None:
            is_inside = (x1 <= pred_x <= x2) and (y1 <= pred_y <= y2)
            if is_inside:
                correct += 1
            total += 1
            
            status = "✓" if is_inside else "✗"
            print(f"[{total:2d}] {status} pred=({pred_x:.0f},{pred_y:.0f}) bbox=[{x1},{y1},{x2},{y2}] ${get_current_cost():.4f}")
            
            results.append({
                "instruction": instruction,
                "pred": [pred_x, pred_y],
                "bbox": bbox,
                "correct": is_inside,
                "reasoning": reasoning
            })
        else:
            errors += 1
            
        time.sleep(0.3)
        
    except Exception as e:
        errors += 1
        print(f"Error: {e}")

# Final stats
print("\n" + "="*60)
print(f"RESULTS: {MODEL}")
print("="*60)
accuracy = correct / total * 100 if total > 0 else 0
print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
print(f"Errors: {errors}")
print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
print(f"Cost: ${get_current_cost():.4f}")
print("="*60)

# Save results
with open("gpt52_results.json", "w") as f:
    json.dump({"model": MODEL, "accuracy": accuracy, "cost": get_current_cost(), "results": results}, f, indent=2)
print("Results saved to gpt52_results.json")
