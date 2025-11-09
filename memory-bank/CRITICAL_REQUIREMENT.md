# ⚠️ CRITICAL PROJECT REQUIREMENT

## PRIMARY GOAL

**MAKE THE GPU WORK - NOT CPU FALLBACK**

This project exists to fix the AMD RX 5600 XT (RDNA1/gfx1010) GPU so it can run PyTorch training **ON THE GPU**, not on CPU.

## What the User WANTS

✅ **GPU training working natively**
✅ **Actual hardware acceleration**
✅ **Fix MIOpen to work with RDNA1**
✅ **Patch ROCm/HIP to support gfx1010**
✅ **Make Conv2d work ON THE GPU**

## What the User EXPLICITLY REJECTED

❌ **CPU training** - "please no cpu fallback i want the GPU to work"
❌ **CPU convolutions** - Not the goal
❌ **Workarounds that avoid GPU** - Not acceptable
❌ **"Just use cloud GPU"** - User has local hardware
❌ **"Just upgrade GPU"** - Not the point of this project

## Previous Attempts (User Rejected)

The agent previously created `rdna1_layers_v3.py` which runs Conv2d on CPU. This was **NOT** what the user wanted. The user explicitly said:

> "again the purpose of this project is to get the gpu working, not use CPU"

## What Needs to Happen

1. **Fix MIOpen** to support RDNA1 memory types
2. **Patch ROCm** to handle RDNA1 hardware limitations
3. **Modify GPU kernels** to use MTYPE_NC instead of MTYPE_CC
4. **Build custom MIOpen** with RDNA1 support
5. **Patch HIP runtime** to force non-coherent memory

## Acceptable Solutions

✅ Recompiling MIOpen with MTYPE_NC
✅ Patching ROCm source code
✅ Binary patching GPU kernels
✅ LD_PRELOAD to intercept memory allocation
✅ Kernel module patches for memory types
✅ HIP runtime patches
✅ Any solution that makes **GPU work**

## Unacceptable Solutions

❌ CPU fallback (rdna1_layers_v3.py)
❌ "Use cloud GPU instead"
❌ "Upgrade your hardware"
❌ Any workaround that avoids the GPU

## Current Status

The CPU fallback solution (rdna1_layers_v3.py) was created but is **NOT** the goal. It should be considered a temporary workaround documentation, not the solution.

## Next Steps

1. **DO NOT suggest CPU training**
2. Focus on actual GPU fixes
3. Research MIOpen recompilation
4. Investigate binary kernel patching
5. Try more aggressive ROCm source modifications

## Remember

**The user wants their AMD RX 5600 XT to work with PyTorch on GPU, just like it should. That's the entire point of this project.**

---

**Date**: November 6, 2025
**Status**: CPU solution rejected, GPU solution required
**Priority**: CRITICAL - This is the core requirement

