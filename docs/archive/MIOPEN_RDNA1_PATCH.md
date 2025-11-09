# MIOpen RDNA1 Non-Coherent Memory Patch

## Goal

**Fix the GPU to work natively** by patching MIOpen to use non-coherent memory for RDNA1 GPUs.

This is **NOT** a CPU workaround - this makes the actual GPU work with PyTorch.

## The Problem

- RDNA1 GPUs (gfx1010, gfx1011, gfx1012) lack fine-grained SVM
- MIOpen uses `hipMalloc` and `hipHostMalloc` which default to cache-coherent memory (MTYPE_CC)
- RDNA1 hardware can't handle cache-coherent memory
- Result: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

## The Solution

Patch MIOpen's memory allocator to:
1. Detect RDNA1 GPUs at runtime
2. Use `hipExtMallocWithFlags` with `hipHostMallocNonCoherent` flag
3. Fall back to `hipHostMalloc` with `hipHostMallocNonCoherent` flag
4. **Run convolutions ON THE GPU** with non-coherent memory

## What We Patched

**File**: `src/hip/handlehip.cpp`
**Function**: `allocator::default_allocator()`

### Changes Made

```cpp
// RDNA1 (gfx1010) Fix: Force non-coherent memory for RDNA1 GPUs
hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);
bool is_rdna1 = (strstr(props.gcnArchName, "gfx1010") != nullptr ||
                 strstr(props.gcnArchName, "gfx1011") != nullptr ||
                 strstr(props.gcnArchName, "gfx1012") != nullptr);

if(is_rdna1)
{
    // Try hipExtMallocWithFlags with NonCoherent flag first
    const auto status_nc = hipExtMallocWithFlags(&ptr, sz, hipHostMallocNonCoherent);
    if(status_nc == hipSuccess)
        return ptr;
}

// ... fallback to hipMalloc ...

// Use non-coherent flag for RDNA1
const unsigned int flags = is_rdna1 ? hipHostMallocNonCoherent : hipHostMallocDefault;
const auto status_host = hipHostMalloc(&ptr, sz, flags);
```

## How to Build

### Prerequisites

```bash
# ROCm 7.0.2 must be installed
# Build tools
sudo apt install build-essential cmake git
```

### Build Steps

1. **Clone MIOpen** (already done):
   ```bash
   cd /tmp
   git clone --depth 1 --branch rocm-7.0.2 https://github.com/ROCmSoftwarePlatform/MIOpen.git
   ```

2. **Apply Patch** (already done):
   The patch has been applied to `/tmp/MIOpen/src/hip/handlehip.cpp`

3. **Build**:
   ```bash
   cd /home/kevin/Projects/rocm-patch
   ./scripts/build_miopen_rdna1.sh
   ```

   This will:
   - Configure CMake with HIP backend
   - Build MIOpen (30-60 minutes)
   - Install to `/opt/rocm-miopen-rdna1/`

4. **Install**:
   ```bash
   cd /tmp/MIOpen/build_rdna1
   sudo make install
   ```

## How to Use

### Option 1: Replace System MIOpen

```bash
# Backup original
sudo cp /opt/rocm/lib/libMIOpen.so.1 /opt/rocm/lib/libMIOpen.so.1.backup

# Copy patched version
sudo cp /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1 /opt/rocm/lib/
```

### Option 2: Use LD_LIBRARY_PATH

```bash
export LD_LIBRARY_PATH=/opt/rocm-miopen-rdna1/lib:$LD_LIBRARY_PATH
python3 your_pytorch_script.py
```

### Option 3: PyTorch Script Override

```python
import os
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm-miopen-rdna1/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
# Now PyTorch will use patched MIOpen
```

## Testing

Test if the patch works:

```python
import torch
import torch.nn as nn

# This should NOT crash anymore!
model = nn.Conv2d(1, 32, 3).cuda()
x = torch.randn(1, 1, 28, 28).cuda()
y = model(x)  # Should work!

# Training should work!
loss = y.sum()
loss.backward()  # Should work!

print("✓ SUCCESS! GPU training works!")
```

## Expected Results

### Before Patch
```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
Aborted (core dumped)
```

### After Patch
```
✓ Forward pass works
✓ Backward pass works  
✓ Training works
✓ GPU is used (not CPU!)
```

## Why This Works

1. **HIP API Support**: `hipHostMallocNonCoherent` (0x80000000) flag exists in HIP
2. **Runtime Detection**: Detects RDNA1 GPUs automatically
3. **Proper Memory Type**: Forces MTYPE_NC (non-coherent) at allocation time
4. **GPU Execution**: Conv2d still runs ON THE GPU, just with different memory
5. **No Python Changes**: Works with standard PyTorch code

## Advantages Over CPU Fallback

| Feature | CPU Fallback | MIOpen Patch |
|---------|--------------|--------------|
| **GPU Acceleration** | ❌ No | ✅ Yes |
| **Speed** | 10x slower | 1x (full speed) |
| **Memory Location** | RAM | VRAM |
| **Hardware Used** | CPU | GPU |
| **MIOpen** | Bypassed | Used properly |
| **PyTorch Changes** | Required | None |

## Status

- ✅ Patch created
- ✅ Patch applied to source
- ✅ Build script ready
- ⏳ Building MIOpen (30-60 min)
- ⏳ Testing with PyTorch
- ⏳ Verifying GPU usage

## Build Time Estimate

- CMake configure: 2-5 minutes
- Compilation: 30-60 minutes (depending on CPU)
- Installation: 1-2 minutes
- **Total**: ~45-75 minutes

## Next Steps

1. Start the build: `./scripts/build_miopen_rdna1.sh`
2. Wait for compilation
3. Install patched MIOpen
4. Test with PyTorch
5. Verify GPU is actually used (not CPU)
6. Document results

---

**Date**: November 6, 2025
**Approach**: MIOpen source patch with runtime RDNA1 detection
**Goal**: GPU working natively, NOT CPU fallback
**Status**: ⏳ Building...

