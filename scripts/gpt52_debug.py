#!/usr/bin/env python3
"""
GPT-5.2 Benchmark with Chain-of-Thought (CoT) - DEBUG VERSION

Adds verbose output to diagnose why responses return None.
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
print("GPT-5.2 CoT DEBUG - Checking Response Structure")
print("="*60)

client = OpenAI()

# Load ONE sample for deep debugging
from datasets import load_dataset
dataset = load_dataset("TIGER-Lab/ScreenSpot-Pro", split="test")
print(f"Loaded dataset")


SYSTEM_PROMPT = """You are an expert in visual grounding on graphical user interfaces. 
Think step by step to locate UI elements precisely."""

USER_PROMPT_TEMPLATE = """Find the bounding box of a UI element in this screenshot.

Instruction: {instruction}

Steps:
1. Describe where the element is relative to screen landmarks
2. Estimate proportional position
3. Calculate normalized coordinates (0 to 1)
4. Output the final bounding box as [[x0, y0, x1, y1]]

Image resolution: {width}x{height} pixels."""


def convert_pil_image_to_base64(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# Run on 3 samples with full debug
for idx in range(3):
    sample = dataset[idx]
    image = sample['image']
    img_size = eval(sample['img_size']) if isinstance(sample['img_size'], str) else sample['img_size']
    orig_w, orig_h = img_size[0], img_size[1]
    instruction = sample['instruction']
    
    print(f"\n{'='*60}")
    print(f"SAMPLE {idx}: {instruction[:60]}...")
    print(f"Image size: {orig_w}x{orig_h}")
    print(f"{'='*60}")
    
    base64_image = convert_pil_image_to_base64(image)
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
            max_completion_tokens=4096,
            reasoning_effort="high"
        )
        
        print("\n--- RAW RESPONSE OBJECT ---")
        print(f"Response type: {type(response)}")
        print(f"Choices count: {len(response.choices)}")
        
        if response.choices:
            choice = response.choices[0]
            print(f"\nChoice finish_reason: {choice.finish_reason}")
            print(f"Message type: {type(choice.message)}")
            print(f"Message content type: {type(choice.message.content)}")
            print(f"Message content repr: {repr(choice.message.content)}")
            print(f"Message content (first 500): {choice.message.content[:500] if choice.message.content else 'NONE'}")
            
            # Check for refusal
            if hasattr(choice.message, 'refusal'):
                print(f"Refusal: {choice.message.refusal}")
            
            # Check for tool calls
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                print(f"Tool calls: {choice.message.tool_calls}")
            
            # Try to find reasoning or thinking output
            if hasattr(choice.message, 'reasoning'):
                print(f"Reasoning: {choice.message.reasoning}")
            if hasattr(choice.message, 'thinking'):
                print(f"Thinking: {choice.message.thinking}")
        
        print(f"\nUsage: {response.usage}")
        
        # Try parsing
        content = response.choices[0].message.content if response.choices else None
        if content:
            match = re.search(r'\[\[([\d.]+),?\s*([\d.]+),?\s*([\d.]+),?\s*([\d.]+)\]\]', content)
            if match:
                x0, y0, x1, y1 = map(float, match.groups())
                print(f"\n✓ PARSED: [[{x0}, {y0}, {x1}, {y1}]]")
            else:
                print(f"\n✗ Could not parse bbox from content")
        else:
            print(f"\n✗ Content is None/empty")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    time.sleep(1)

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)
