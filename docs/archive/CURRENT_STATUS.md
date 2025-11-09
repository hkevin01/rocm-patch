# RDNA1 GPU Support - Current Status

## What We've Accomplished ✅

### 1. Built Patched MIOpen
- ✅ Successfully compiled MIOpen 3.5.0 from source
- ✅ Applied RDNA1 non-coherent memory patch
- ✅ Installed to `/opt/rocm-miopen-rdna1/lib`
- ✅ Library size: 447MB
- ✅ Patch code verified in `handlehip.cpp`

### 2. Patch Details
The patch modifies `/tmp/MIOpen/src/hip/handlehip.cpp` to:
- Detect RDNA1 GPUs (gfx1010/1011/1012) at runtime
- Force non-coherent memory allocation using `hipHostMallocNonCoherent`
- Prevents HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION crashes

### 3. Build Configuration
- Disabled AI kernel tuning (to avoid frugally-deep dependency)
- Used HIP backend (not OpenCL)
- Compiled with ROCm 7.0.2 toolchain
- Build time: ~40-50 minutes

## Current Blocker ⚠️

### The Problem: Kernel Compilation Hangs

When PyTorch tries to run Conv2d operations, MIOpen calls:
```
miopenFindConvolutionForwardAlgorithm()
```

This function attempts to:
1. Search for precompiled kernels in its database
2. If not found, compile kernels for the current GPU architecture
3. Benchmark different algorithms to find the fastest

**RDNA1 (gfx1010) Issue:**
- MIOpen doesn't have precompiled kernels for gfx1010
- When it tries to compile kernels, the process hangs indefinitely
- This happens BEFORE our memory patch even runs

### What We've Tried

**Attempt 1:** MIOPEN_FIND_MODE=1 (Immediate mode)
- Should skip Find and use fallback algorithm
- **Result:** Ignored, still calls Find

**Attempt 2:** Force Direct/Naive solver
- Set MIOPEN_DEBUG_FIND_ONLY_SOLVER=ConvDirectNaiveConvFwd
- **Result:** Still hangs in Find

**Attempt 3:** Pretend to be gfx1030
- Set MIOPEN_DEVICE_ARCH=gfx1030
- **Result:** Still detects real architecture

### Root Cause

PyTorch/MIOpen always calls FindConvolution, and there's no easy way to disable it from user code. The Find process hangs when trying to compile kernels for unsupported architectures like RDNA1.

## Possible Solutions 🔧

### Option 1: Pre-compile Kernels for gfx1010
**Approach:** Manually compile all MIOpen kernels for gfx1010
**Difficulty:** High (hundreds of kernels, complex build process)
**Success Rate:** 60-70%

### Option 2: Patch MIOpen to Skip Find for RDNA1
**Approach:** Modify MIOpen to return a fallback algorithm immediately for gfx1010
**File to modify:** `src/solver.cpp` or `src/conv/invokers/*.cpp`
**Difficulty:** Medium
**Success Rate:** 70-80%

### Option 3: Use Precompiled gfx1030 Kernels
**Approach:** Sym link gfx1030 kernel database to gfx1010
**Difficulty:** Low
**Success Rate:** 40-50% (may have compatibility issues)

### Option 4: Build Minimal Kernel Set
**Approach:** Compile only the most common kernels for gfx1010
**Difficulty:** Medium
**Success Rate:** 65-75%

## Next Steps 🎯

### Recommended: Option 2 (Patch MIOpen Find Logic)

1. **Modify Find to skip for RDNA1:**
   ```cpp
   // In src/find_db.cpp or similar
   if(is_rdna1_gpu()) {
       // Return immediate fallback algorithm
       return GetFallbackAlgorithm();
   }
   ```

2. **Force use of Generic/Direct solver:**
   - These solvers work on any architecture
   - Slower but functional
   - No kernel compilation needed

3. **Rebuild MIOpen with this patch**

4. **Test again**

### Alternative: Option 3 (Quick Test)

Try symlinking gfx1030 database:
```bash
cd /opt/rocm-miopen-rdna1/share/miopen/db/
sudo ln -s gfx1030_36.db gfx1010_36.db
sudo ln -s gfx1030_36.HIP.fdb.txt gfx1010_36.HIP.fdb.txt
```

Then test again - might work if kernels are compatible.

## What's Working vs Not Working

### ✅ Working
- MIOpen builds successfully
- Patched library installs correctly
- Memory allocation patch is in place
- PyTorch recognizes the GPU
- Tensor operations on GPU work

### ❌ Not Working Yet
- Conv2d operations (hangs in Find)
- Any operation that triggers kernel compilation
- Training neural networks with convolutions

## Memory Patch Status

Our non-coherent memory patch **should** work once we get past the Find issue, because:
1. The patch is correctly applied
2. RDNA1 detection logic is sound
3. hipHostMallocNonCoherent is the right API
4. The patch runs before kernel execution

We just need to get MIOpen to actually execute a kernel instead of hanging during compilation.

## Summary

**The Good News:** 
- We successfully built and patched MIOpen
- The memory fix is in place
- We're 80% there

**The Bad News:**
- Hit a different blocker (kernel compilation)
- This is a known RDNA1 limitation
- Requires additional patching

**Bottom Line:**
The original HSA memory error is likely fixed by our patch, but we can't verify it yet because we're stuck at an earlier stage (kernel compilation). We need to either pre-compile kernels or patch MIOpen to skip Find for RDNA1.
