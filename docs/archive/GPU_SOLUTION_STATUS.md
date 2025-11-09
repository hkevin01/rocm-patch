# GPU Solution Status - MIOpen RDNA1 Patch

## ✅ Current Progress

### What We've Done

1. ✅ **Identified the root cause**:
   - MIOpen uses `hipMalloc`/`hipHostMalloc`
   - These default to cache-coherent memory (MTYPE_CC)
   - RDNA1 hardware can't handle it

2. ✅ **Found the solution**:
   - HIP has `hipHostMallocNonCoherent` flag (0x80000000)
   - Can force non-coherent memory at allocation time
   - Conv2d will still run **ON THE GPU**

3. ✅ **Located MIOpen allocation code**:
   - File: `src/hip/handlehip.cpp`
   - Function: `allocator::default_allocator()`
   - Line 106: `hipMalloc(&ptr, sz)`
   - Line 112: `hipHostMalloc(&ptr, sz)`

4. ✅ **Created the patch**:
   - Detects RDNA1 GPUs at runtime (gfx1010/1011/1012)
   - Uses `hipExtMallocWithFlags` with `hipHostMallocNonCoherent`
   - Falls back to `hipHostMalloc` with NC flag
   - **No changes to GPU kernel execution**

5. ✅ **Applied patch to source**:
   - Location: `/tmp/MIOpen/src/hip/handlehip.cpp`
   - Verified: `diff` shows 35 lines added
   - Backup created: `handlehip.cpp.orig`

6. ✅ **Created build script**:
   - Location: `scripts/build_miopen_rdna1.sh`
   - CMake configuration ready
   - Will install to `/opt/rocm-miopen-rdna1/`

7. ✅ **Resolved dependency issues**:
   - Disabled AI kernel tuning (requires frugally-deep)
   - Updated build script with -DMIOPEN_ENABLE_AI_KERNEL_TUNING=Off
   - Build script updated with -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=Off

8. ✅ **Built MIOpen Successfully**:
   - Build completed in ~45 minutes
   - Library size: 447MB
   - Installed to: `/opt/rocm-miopen-rdna1/lib`
   - Log: /tmp/miopen_build_final.log

9. ✅ **Installed Patched MIOpen**:
   - Installation successful
   - Library accessible at `/opt/rocm-miopen-rdna1/lib/libMIOpen.so`
   - Patch verified in source code

10. ⚠️ **Testing Blocked by Kernel Compilation**:
   - MIOpen hangs during `miopenFindConvolutionForwardAlgorithm()`
   - Trying to compile kernels for unsupported gfx1010 architecture
   - This happens BEFORE our memory patch can run
   - See `CURRENT_STATUS.md` for details and next steps

## 📋 Next Steps

### Step 1: Build Patched MIOpen (~45-75 minutes)

```bash
cd /home/kevin/Projects/rocm-patch
./scripts/build_miopen_rdna1.sh
```

This will:
- Configure CMake with HIP backend
- Compile MIOpen from source
- Create patched library

**Note**: This uses your CPU to compile, not GPU. It's a one-time build.

### Step 2: Install Patched MIOpen (~2 minutes)

```bash
cd /tmp/MIOpen/build_rdna1
sudo make install
```

Installs to `/opt/rocm-miopen-rdna1/`

### Step 3: Test with PyTorch (~1 minute)

```bash
export LD_LIBRARY_PATH=/opt/rocm-miopen-rdna1/lib:$LD_LIBRARY_PATH
python3 -c "
import torch
import torch.nn as nn

print('Testing patched MIOpen with RDNA1...')
model = nn.Conv2d(1, 32, 3).cuda()
x = torch.randn(1, 1, 28, 28).cuda()
y = model(x)
print('✓ Forward pass works!')

loss = y.sum()
loss.backward()
print('✓ Backward pass works!')
print('🎉 GPU training works!')
"
```

### Step 4: Verify GPU Usage

```bash
# Should show GPU activity during Conv2d
rocm-smi --showuse
```

## 🎯 Expected Results

### If Patch Works ✅

```
Testing patched MIOpen with RDNA1...
RDNA1 GPU detected (gfx1010), forcing non-coherent memory
✓ Forward pass works!
✓ Backward pass works!
🎉 GPU training works!
```

- Conv2d runs **ON THE GPU**
- No crashes
- Full speed (not 10x slower like CPU)
- Standard PyTorch code works

### If Patch Fails ❌

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
Aborted (core dumped)
```

Then we need to:
- Check if hipHostMallocNonCoherent actually works on RDNA1
- Try binary patching GPU kernels instead
- Investigate deeper HIP/HSA runtime changes

## 💡 Why This Should Work

### Theory

1. **HIP API** supports non-coherent flag explicitly
2. **ROCm 7.0.2** has the hipHostMallocNonCoherent definition
3. **MIOpen** uses HIP allocation functions we can patch
4. **RDNA1** hardware supports non-coherent memory (MTYPE_NC)
5. **Kernel parameter** `mtype_local=1` proved NC memory works

### Evidence

```bash
# Flag exists in HIP API
$ grep "hipHostMallocNonCoherent" /opt/rocm/include/hip/hip_runtime_api.h
#define hipHostMallocNonCoherent  0x80000000

# Kernel parameter works (driver level)
$ cat /sys/module/amdgpu/parameters/mtype_local
1

# MIOpen uses these functions
$ grep "hipHostMalloc" /tmp/MIOpen/src/hip/handlehip.cpp
const auto status_host = hipHostMalloc(&ptr, sz, flags);
```

### Success Factors

✅ We're patching at the **right level** (MIOpen allocation)
✅ We're using the **right API** (hipHostMallocNonCoherent)
✅ We have **runtime detection** (automatic for RDNA1)
✅ We have **fallbacks** (multiple allocation paths)
✅ **GPU kernels unchanged** (still run on GPU)

## 📊 Comparison

| Approach | GPU Used | Speed | Status |
|----------|----------|-------|--------|
| **Environment variables** | ❌ | - | Failed |
| **LD_PRELOAD** | ❌ | - | Failed |
| **Kernel parameter** | ⚠️ | 1x | Partial (MIOpen overrides) |
| **CPU fallback (v3)** | ❌ | 0.1x | User rejected |
| **MIOpen patch** | ✅ | 1x | ⏳ Testing |

## ⏱️ Time Investment

- Build time: 45-75 minutes (one-time)
- Install time: 2 minutes
- Test time: 1 minute
- **Total**: ~1 hour

## 🎲 Success Probability

Based on:
- ✅ HIP API support for NC memory
- ✅ Kernel parameter proves NC works
- ✅ Patching at correct level (MIOpen)
- ✅ Runtime RDNA1 detection
- ⚠️ Unknown: Does hipHostMallocNonCoherent work for device memory?

**Estimated success rate: 60-70%**

## 🚀 Ready to Build?

The patch is ready. To start building:

```bash
cd /home/kevin/Projects/rocm-patch
./scripts/build_miopen_rdna1.sh
```

**Note**: This will take 45-75 minutes. You can monitor progress or let it run in the background.

---

**Date**: November 6, 2025
**Approach**: MIOpen source patch with non-coherent memory
**Goal**: GPU working natively (NOT CPU fallback)
**Status**: ⏳ Ready to build
**Files**:
- Patch: `/tmp/MIOpen/src/hip/handlehip.cpp`
- Script: `scripts/build_miopen_rdna1.sh`
- Docs: `MIOPEN_RDNA1_PATCH.md`

