# GUI Visual Grounding Research - Codex Context File

## 🎯 Core Objective

Build a **tiny, efficient model** that can accurately locate UI elements in high-resolution screenshots (4K) given natural language instructions like "Click the search button" — achieving competitive accuracy against trillion-parameter models like GPT-5.2 (86.3% on ScreenSpot-Pro benchmark).

**The Holy Grail**: A model small enough to run on-device (11.5M parameters) that can ground GUI elements with high precision on professional software interfaces (Blender, Photoshop, Android Studio, CAD tools) where targets are often **tiny icons (40-50px) on 4K screens**.

---

## 📊 Benchmark: ScreenSpot-Pro

- **Dataset**: 1,581 samples from 23 professional applications
- **Challenge**: Tiny UI targets (some <50px) on high-resolution screens (3840x2160)
- **Metric**: Point-in-Box accuracy (predicted center must fall inside ground truth bbox)
- **SOTA**: GPT-5.2 Thinking @ 86.3%, UI-TARS @ 84.1%
- **Our Current**: 0% (broken) → Goal: 30%+ with 11.5M params

---

## 🏗 Architecture: OcuMamba-Lite

A novel architecture combining:

1. **Mamba Visual Encoder** - State-space model for efficient image processing
2. **Instruction Encoder** - Sentence transformer for text understanding  
3. **Multiscale Fusion** - Combines features at multiple resolutions
4. **Detection Head** - Predicts (x, y) coordinates

**Key Files**:
```
indexing/ocumamba_lite/
├── model.py              # Main model class
├── mamba_visual_encoder.py
├── instruction_encoder.py
├── multiscale_fusion.py
├── detection_head.py
├── dataset.py            # BROKEN (see flaws below)
├── dataset_fixed.py      # FIXED version
├── trainer.py
└── benchmark.py
```

---

## 🐛 IDENTIFIED KILLER FLAWS (Critical)

### Flaw 1: Resolution Trap
**File**: `dataset.py` line 125
```python
image.resize((self.image_size, self.image_size))  # Usually 256x256
```
**Problem**: A 40px icon on 4K (3840px) → resized to 256px = **2.6 pixels**. Literally invisible.
**Fix Applied**: `dataset_fixed.py` uses 512x512 crops from full-resolution images.

### Flaw 2: Aspect Ratio Squash  
**File**: `dataset.py`
**Problem**: 16:9 screens forced to 1:1 square. Round buttons become ovals.
**Fix Applied**: `AspectPreservingDataset` with padding.

### Flaw 3: Late Fusion (NOT YET FIXED)
**File**: `multiscale_fusion.py`
**Problem**: Text instruction is concatenated AFTER visual processing. By then, tiny icons may be discarded as noise because the model didn't know they were important.
**Required Fix**: Early fusion — inject text into the FIRST Mamba layer.

---

## 🔬 GPT-5.2 Benchmark Investigation

We tried to reproduce the 86.3% accuracy. Results:

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Basic GPT-4o | 0% | No spatial reasoning |
| GPT-5.2 + reasoning_effort="high" | 0% | Parameters may be ignored |
| GPT-5.2 + detail="high" | 0% | Close predictions but miss |
| GPT-5.2 + CoT prompting | 0% | Model reasons but still fails |

**Root Cause Hypothesis**: The 86.3% requires a **specific "Thinking" model variant** (possibly `gpt-5.2-preview-thinking` or `o1-preview`) that we don't have API access to. Standard API ignores `reasoning_effort`.

---

## 🧪 Active Inference Experiment

Alternative approach using predictive processing:
- **File**: `indexing/mamba/active_inference.py`
- **Concept**: Generate belief about element location, then refine through iterative prediction error minimization
- **Status**: Implemented but untested on GPU

---

## 📁 Project Structure

```
Backend/
├── indexing/
│   ├── ocumamba_lite/     # Main model
│   ├── mamba/             # Active inference variant
│   └── visual_physics/    # Click refinement utilities
├── scripts/
│   ├── gpt52_cot_fixed.py       # GPT benchmark with CoT
│   ├── gpt52_correct_config.py  # detail="high" tests
│   └── benchmark_screenspot.py
├── cache/                 # Cached predictions
└── docs/research_report.md
```

---

## 🔍 DIAGNOSTIC QUESTIONS FOR CODE REVIEW

### Architecture Questions

1. **Multiscale Fusion**: Does the current fusion mechanism preserve spatial precision for tiny targets? Or does pooling/averaging destroy sub-pixel information?

2. **Detection Head**: Is the coordinate regression head architecture optimal? Should we use:
   - Direct (x,y) regression?
   - Heatmap prediction + argmax?
   - Anchor-based detection?

3. **Mamba SSM**: Is the state-space model appropriate for 2D image understanding, or would a modified architecture (2D-Mamba, Vision Mamba) be better?

4. **Early Fusion Implementation**: How exactly should we inject text conditioning into the first visual layer? Cross-attention? FiLM conditioning? Concatenation?

### Dataset/Training Questions

5. **Crop Strategy**: Current `dataset_fixed.py` uses 80% target-centric crops. Is this ratio optimal? Should we include more background crops for negative mining?

6. **Scale Invariance**: Do we need multi-scale training (different crop sizes) to handle the huge variance in icon sizes (16px to 200px)?

7. **Loss Function**: Is MSE on (x,y) coordinates the right loss? Should we use:
   - Smooth L1?
   - IoU-based losses?
   - Focal loss for hard examples?

8. **Data Augmentation**: What augmentations are safe for coordinate regression without corrupting ground truth labels?

### Numerical/Bug Questions

9. **Coordinate Normalization**: Are we consistent about 0-1 normalization everywhere? Check for off-by-one errors at image boundaries.

10. **Tensor Shapes**: Are channel dimensions (NCHW vs NHWC) consistent throughout the pipeline?

11. **Gradient Flow**: Does the gradient flow properly from the detection head back through multiscale fusion to the visual encoder?

12. **Memory Leaks**: In training loops, are we properly detaching tensors and clearing GPU memory?

### Integration Questions

13. **HuggingFace Dataset Loading**: Are we loading images at full resolution? Check if `load_dataset` applies any automatic resizing.

14. **Batch Collation**: Does `collate_fn` handle variable-size metadata correctly?

15. **Evaluation Metric**: Is our point-in-box check mathematically identical to the official ScreenSpot-Pro evaluation?

### Performance Questions

16. **Inference Speed**: What is the actual FPS on target hardware? Is Mamba giving us the expected efficiency gains?

17. **Parameter Count**: Is the 11.5M figure accurate? Profile each component.

18. **Bottlenecks**: Where is the computational bottleneck — visual encoder, fusion, or detection head?

---

## 🎯 Success Criteria

1. **Minimum Viable**: 15% accuracy on ScreenSpot-Pro (currently 0%)
2. **Competitive**: 30%+ accuracy with <20M parameters
3. **Stretch Goal**: Match UI-TARS (84%) with model distillation

---

## 💡 Specific Areas to Audit

Please carefully review:

1. `indexing/ocumamba_lite/model.py` - Main forward pass logic
2. `indexing/ocumamba_lite/multiscale_fusion.py` - Feature combination
3. `indexing/ocumamba_lite/detection_head.py` - Coordinate prediction
4. `indexing/ocumamba_lite/dataset_fixed.py` - Data loading
5. `indexing/mamba/vision_mamba.py` - SSM implementation

Look for:
- Shape mismatches
- Normalization inconsistencies  
- Missing gradient paths
- Suboptimal architecture choices
- Bugs in coordinate handling

---

## 📚 References

- ScreenSpot-Pro Paper: https://arxiv.org/abs/2504.07981
- Official Repo: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
- Our Repo: https://github.com/Mihawii/gui-testing
- Mamba: https://arxiv.org/abs/2312.00752
