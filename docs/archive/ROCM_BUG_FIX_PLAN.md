# ROCm 6.2.4 Bug Fix Plan for RDNA1 (gfx1010)

**Date**: November 8, 2025  
**Target**: Fix Conv2d hang/memory aperture violation for RX 5600 XT  
**Approach**: Patch ROCr-Runtime to force coarse-grained memory for gfx1010

## Problem Analysis

### Root Cause (from ROCm #2527 investigation)
1. **What changed in ROCm 5.3**: Memory access code for gfx1030 was modified
2. **Why it breaks gfx1010**: When gfx1010 is spoofed as gfx1030 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`), it inherits gfx1030's memory model
3. **The issue**: gfx1030 (RDNA2) has fine-grained SVM support, gfx1010 (RDNA1) doesn't
4. **The symptom**: Memory aperture violations or kernel compilation hangs

### Why Our MIOpen Patches Didn't Work
- MIOpen patches operate at too high a level
- Memory allocation happens in HSA runtime (ROCr)
- By the time MIOpen tries to allocate, HSA has already decided on memory type

## Fix Strategy

### Approach 1: Patch ROCr-Runtime Memory Region ⭐ RECOMMENDED

**File**: `ROCR-Runtime/src/core/runtime/amd_memory_region.cpp`

**What to patch**:
1. Detect if GPU is actually gfx1010 despite HSA_OVERRIDE
2. Force coarse-grained memory for gfx1010
3. Avoid fine-grained allocations that cause violations

### Approach 2: Patch HIP Runtime

**File**: HIP source (need to clone ROCm/HIP)

**What to patch**:
1. Override hipMalloc for gfx1010
2. Force specific memory flags
3. More invasive, affects all HIP applications

### Approach 3: Create LD_PRELOAD Shim

**Pros**: No recompilation needed  
**Cons**: May not catch all paths, fragile

## Implementation Plan

### Phase 1: Identify True GPU Architecture ✅

We need to read the actual GPU device ID, not the spoofed version.

**Location**: `ROCR-Runtime/src/core/runtime/amd_gpu_agent.cpp`

**Code to add**:
```cpp
// Get true device ID before any overrides
bool GpuAgent::IsTrueRDNA1() const {
  // Device IDs for RDNA1 (RX 5000 series)
  uint16_t true_device_id = properties_.DeviceId;
  
  // RX 5600 XT: 0x731F
  // RX 5700 XT: 0x731F (same family)
  // RX 5500 XT: 0x7340
  
  return (true_device_id == 0x731F || 
          true_device_id == 0x7340 ||
          true_device_id == 0x7341);
}
```

### Phase 2: Force Coarse-Grained for RDNA1 ✅

**Location**: `ROCR-Runtime/src/core/runtime/amd_memory_region.cpp`

**In `MemoryRegion::AllocateImpl()`** (around line 177):

```cpp
hsa_status_t MemoryRegion::AllocateImpl(size_t& size, AllocateFlags alloc_flags,
                                        void** address, int agent_node_id) const {
  // RDNA1 FIX: Force coarse-grained memory for gfx1010
  // Check if owner GPU is true RDNA1 (not just spoofed)
  HsaMemFlags kmt_alloc_flags = mem_flag_;
  
  // Get owner agent
  const AMD::GpuAgent* gpu_owner = static_cast<const AMD::GpuAgent*>(owner());
  
  // Force coarse-grained if true RDNA1
  if (gpu_owner && gpu_owner->IsTrueRDNA1()) {
    // Force coarse-grained memory
    kmt_alloc_flags.ui32.CoarseGrain = 1;
    kmt_alloc_flags.ui32.ExtendedCoherent = 0;
    
    fprintf(stderr, "[RDNA1 FIX] Forcing coarse-grained memory for gfx1010\n");
  }
  
  // Rest of existing allocation code...
}
```

### Phase 3: Skip Fine-Grained Checks for RDNA1 ✅

**Location**: Same file, `GetAccessInfo()` (around line 537)

```cpp
// Return never allowed if memory is fine grained
// link type is not xGMI i.e. link is PCIe

// RDNA1 FIX: Always allow for true RDNA1, even if fine-grained
const AMD::GpuAgent* requesting_gpu = static_cast<const AMD::GpuAgent*>(&agent);
if (requesting_gpu && requesting_gpu->IsTrueRDNA1()) {
  return HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT;
}

return HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
```

## Build Instructions

### 1. Clone and Patch ROCr-Runtime

```bash
cd /tmp
git clone --branch rocm-6.2.4 https://github.com/ROCm/ROCR-Runtime.git
cd ROCR-Runtime

# Apply patches (see next section)
# patch -p1 < /path/to/rdna1-fix.patch
```

### 2. Build ROCr

```bash
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm-rdna1-rocr \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(nproc)
sudo make install
```

### 3. Replace System Library

```bash
# Backup original
sudo cp /opt/rocm/lib/libhsa-runtime64.so.1 \
        /opt/rocm/lib/libhsa-runtime64.so.1.backup

# Install patched version
sudo cp /opt/rocm-rdna1-rocr/lib/libhsa-runtime64.so.1 \
        /opt/rocm/lib/libhsa-runtime64.so.1
```

### 4. Test

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 << EOF
import torch
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
print(f"✅ SUCCESS! Conv2d worked: {y.shape}")
