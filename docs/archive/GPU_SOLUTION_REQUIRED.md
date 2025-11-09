# ⚠️ PROJECT STATUS: GPU SOLUTION REQUIRED

## User Requirement Clarification

The user has **explicitly rejected** the CPU training solution created in `pytorch_extensions/rdna1_layers_v3.py`.

### User's Statement

> "again the purpose of this project is to get the gpu working, not use CPU, make that into memorybank, never ever try to solve problem by going to CPU; keep that in memory"

## What This Means

### ❌ REJECTED Solutions
- `rdna1_layers_v3.py` (CPU training) - **NOT ACCEPTABLE**
- `rdna1_layers.py` (CPU conv, GPU other) - **NOT ACCEPTABLE**
- Any CPU fallback approach - **NOT ACCEPTABLE**

### ✅ REQUIRED Solution
- **GPU must work natively** with PyTorch
- Conv2d must run **ON THE GPU**
- Full ROCm/MIOpen support for RDNA1
- Actual hardware acceleration

## Why CPU Solution Was Created (Context)

The agent explored 8 different approaches:
1. Environment variables → Failed
2. LD_PRELOAD → Failed  
3. Memory formats → Failed
4. ROCm source build → Failed (LLVM conflicts)
5. Docker ROCm 5.7 → Failed
6. Kernel parameter → Partial (MIOpen overrides)
7. Python override → Partial (forward only)
8. **CPU training → User rejected**

The CPU solution was created as a last resort after all GPU-based approaches failed. However, **the user does not want a CPU workaround**.

## What We Know

### Root Cause
- RDNA1 (gfx1010) lacks fine-grained SVM
- MIOpen's pre-compiled kernels hardcode MTYPE_CC requests
- Hardware cannot handle cache-coherent memory
- Error: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

### Attempted GPU Fixes
1. **mtype_local=1 kernel parameter** - Applied but MIOpen overrides it
2. **ROCm 6.2 source build** - LLVM 16 vs 20 bitcode conflict
3. **LD_PRELOAD memory intercept** - Breaks HIP initialization
4. **Environment variables** - Applied too late in stack

### Why They Failed
- MIOpen's GPU kernels have **hardcoded** MTYPE_CC in compiled binaries
- Can't change without recompiling MIOpen
- Kernel parameter only sets driver default, libraries override it
- Control hierarchy: GPU kernel code > library > runtime > driver

## What Needs to Be Done

### Option A: Recompile MIOpen with MTYPE_NC
**Difficulty**: High
**Time**: 8-12 hours
**Success Rate**: 60-70%
**Approach**:
1. Download MIOpen source
2. Find MTYPE_CC references in kernel code
3. Change to MTYPE_NC
4. Recompile for gfx1010
5. Replace system MIOpen

### Option B: Binary Patch MIOpen Kernels
**Difficulty**: Very High
**Time**: 16-24 hours
**Success Rate**: 40-50%
**Approach**:
1. Extract compiled GPU kernels from MIOpen
2. Disassemble with ROCm tools
3. Find MTYPE_CC instructions
4. Replace with MTYPE_NC
5. Reassemble and test

### Option C: Custom HIP Memory Allocator
**Difficulty**: High
**Time**: 8-16 hours
**Success Rate**: 50-60%
**Approach**:
1. Create LD_PRELOAD library that works
2. Intercept ALL memory allocation calls
3. Force MTYPE_NC for all allocations
4. Test with PyTorch

### Option D: Patch ROCm/HIP Runtime
**Difficulty**: Very High
**Time**: 12-20 hours
**Success Rate**: 40-50%
**Approach**:
1. Modify ROCm source to default MTYPE_NC for gfx1010
2. Rebuild entire ROCm stack
3. Install custom ROCm
4. Test

## Recommendation

The most promising approach is **Option A: Recompile MIOpen**.

### Why:
1. Direct fix at the source of the problem
2. MIOpen is the layer requesting MTYPE_CC
3. Source code is available
4. ROCm documentation shows how to build
5. Can target specific gfx1010 architecture

### Steps:
1. Clone MIOpen repository
2. Search for MTYPE or memory coherency settings
3. Modify kernel code to use NC (non-coherent)
4. Build with: `cmake -DMIOPEN_GPU_SYNC=Off -DMIOPEN_USE_MLIR=Off`
5. Target gfx1010 specifically
6. Install to custom location
7. Test with PyTorch

## Status

- **CPU Solution**: ❌ Created but REJECTED by user
- **GPU Solution**: ⏳ Still needed
- **Next Action**: Attempt MIOpen recompilation with MTYPE_NC

## Critical Notes

1. **DO NOT suggest CPU training again**
2. **Focus only on GPU solutions**
3. **User wants actual hardware acceleration**
4. **This is about fixing RDNA1 support, not working around it**

---

**Updated**: November 6, 2025
**Status**: CPU solution rejected, GPU solution in progress
**Priority**: CRITICAL

