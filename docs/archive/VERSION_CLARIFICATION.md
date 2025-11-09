# ROCm Version Clarification

**Date**: November 8, 2025

## Question Asked
"I thought we were using ROCm 6.2.4 not 5.3"

## Answer
**You are 100% correct!** We ARE using ROCm 6.2.4.

The confusion comes from how we reference the regression timeline.

## What You're Using RIGHT NOW

| Component | Version | Status |
|-----------|---------|--------|
| System ROCm | **6.2.4** (build 6.2.41134) | ✅ Current |
| PyTorch | 2.5.1+rocm**6.2** | ✅ Current |
| PyTorch's bundled ROCm | 6.2.41133 | ✅ Matches! |
| GPU | RX 5600 XT (gfx1010) | Hardware |
| OS | Ubuntu 24.04.3 LTS | System |

## What "ROCm 5.3+" Means

When we say "ROCm 5.3+ regression", we mean:

**"A bug that was introduced in ROCm 5.3 and still exists in all newer versions"**

### Timeline of the Bug

```
ROCm 5.2 (2022)           ✅ Works perfectly
         |
         | (October 2023: Bug introduced)
         |
ROCm 5.3 (Oct 2023)       ❌ Bug appears
ROCm 5.4                  ❌ Bug still present
ROCm 5.5                  ❌ Bug still present
ROCm 5.6                  ❌ Bug still present
ROCm 5.7                  ❌ Bug still present
ROCm 6.0                  ❌ Bug still present
ROCm 6.1                  ❌ Bug still present
ROCm 6.2.4  ⬅️ YOU ARE HERE  ❌ Bug still present
ROCm 6.3 (future)         ❌ Bug likely still present
```

### What "5.3+" Means

The "+" symbol means **"and all versions after"**

So "ROCm 5.3+" includes:
- ROCm 5.3 ✓
- ROCm 5.4 ✓
- ROCm 5.5 ✓
- ROCm 5.6 ✓
- ROCm 5.7 ✓
- ROCm 6.0 ✓
- ROCm 6.1 ✓
- **ROCm 6.2.4** ✓ (your current version)
- ROCm 6.3 and beyond ✓

## The Regression Explained

### What Happened in ROCm 5.3
1. AMD changed memory access code for gfx1030 (RDNA2)
2. This change broke compatibility with gfx1010 (RDNA1) when spoofed as gfx1030
3. The bug was reported ([ROCm Issue #2527](https://github.com/ROCm/ROCm/issues/2527))
4. AMD acknowledged it in November 2023
5. **It has never been fixed**

### Why It Affects ROCm 6.2.4
The bug introduced in 5.3 was never removed or fixed. Each new version (5.4, 5.5, 6.0, 6.1, 6.2.4, etc.) inherited the broken code.

## Why We Use ROCm 6.2.4

Even though ROCm 6.2.4 has the Conv2d bug, we MUST use it because:

1. **Version Matching is Critical**: PyTorch 2.5.1 bundles ROCm 6.2.41133
2. **ABI Compatibility**: ROCm 6.2.4 (6.2.41134) is close enough to match
3. **Non-CNN Operations Work**: Transformers, RNNs, linear layers all work fine
4. **Can't Downgrade**: ROCm 5.2 (last working version) is no longer available

## What Works vs What Doesn't

### ✅ What Works on ROCm 6.2.4
- Non-convolutional neural networks
- Transformers (BERT, GPT, T5)
- RNNs, LSTMs, GRUs
- Linear layers (fully connected)
- Matrix operations
- Batch normalization
- Pooling operations
- Activation functions

### ❌ What Doesn't Work on ROCm 6.2.4
- Conv2d operations (hangs during kernel compilation)
- Conv1d operations (likely same issue)
- Conv3d operations (likely same issue)
- All CNN models (ResNet, VGG, EfficientNet, YOLO, etc.)
- Computer vision tasks
- Image classification, object detection, segmentation

## Summary

### In Simple Terms

**Question**: "Are we using ROCm 6.2.4 or 5.3?"
**Answer**: ROCm **6.2.4**

**Question**: "Then why do you mention 5.3?"
**Answer**: That's when the bug started. It's still in 6.2.4.

**Analogy**: 
- A crack appeared in a building's foundation in 2023 (ROCm 5.3)
- You're living in the building in 2024 (ROCm 6.2.4)
- The crack from 2023 is still there
- We reference "the 2023 crack" even though we're in 2024

### Technical Terms

- **Current version**: ROCm 6.2.4 ✅
- **Bug introduced**: ROCm 5.3 (October 2023)
- **Bug status**: Still present in 6.2.4 ❌
- **Fix status**: None (waiting for AMD)

## References

- [ROCm Issue #2527](https://github.com/ROCm/ROCm/issues/2527) - Official bug report
- Opened: October 5, 2023
- Status: Under Investigation (2+ years, no fix)
- Affects: All RDNA1 GPUs (RX 5000 series)

## Bottom Line

You are using ROCm 6.2.4, which is correct. When we say "ROCm 5.3+ regression", we're describing a bug that:
1. Started in version 5.3
2. Has never been fixed
3. Still exists in your current version (6.2.4)

The README has been updated with a clarification box to make this clear.

---

*Documentation clarified: November 8, 2025*
