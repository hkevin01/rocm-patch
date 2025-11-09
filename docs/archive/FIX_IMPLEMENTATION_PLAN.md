# RDNA1 Fix Implementation Plan

## Problem Summary
- **Hardware**: RX 5600 XT (Navi 10, gfx1010, device ID 0x731F)
- **Override**: HSA_OVERRIDE_GFX_VERSION=10.3.0 (spoof as gfx1030/RDNA2)
- **Issue**: gfx1030 expects fine-grained SVM, but gfx1010 only has coarse-grained
- **Result**: Memory model mismatch → system crashes

## Why Previous Attempts Failed

### ❌ Attempt 1: Patching HSA Runtime Directly
- Modified `amd_memory_region.cpp` to force coarse-grained
- **Problem**: Memory model already initialized by kernel driver
- **Result**: Accessing invalid memory → crash

### ❌ Attempt 2: LD_PRELOAD Shim
- Intercepted memory allocation calls
- **Problem**: Can't safely override HSA internal structures
- **Result**: ABI mismatch → crash

### ❌ Attempt 3: Type Casting in Runtime
- Used static_cast to GpuAgent
- **Problem**: Cast without proper type checking + RTTI disabled
- **Result**: Invalid pointer access → crash

## Root Cause Analysis

The memory model is set in this order:
1. **Kernel Driver** (amdgpu.ko): Detects 0x731F → sets coarse-grained
2. **KFD**: Reads hardware caps → reports coarse-grained (flags=0)
3. **HSA Runtime**: Applies HSA_OVERRIDE → thinks it's gfx1030
4. **Memory Allocation**: Expects fine-grained but hardware has coarse
5. **💥 CRASH**: Trying to use unsupported memory mode

## Solution Options

### Option 1: Don't Use HSA_OVERRIDE ⭕ Not Viable
**Approach**: Use gfx1010 natively without spoofing
**Pros**: No memory model mismatch
**Cons**: MIOpen doesn't have gfx1010 kernels → Conv2d fails
**Verdict**: Can't use AI frameworks without gfx1030 kernels

### Option 2: Build ROCm with gfx1010 Support ⭕ Time-Consuming
**Approach**: Compile entire ROCm stack with gfx1010 kernels
**Pros**: Proper native support
**Cons**: 
- Takes days to compile
- Need to rebuild for every ROCm update
- Might still have compatibility issues
**Verdict**: Technically correct but impractical

### Option 3: Kernel Module Patch ✅ BEST OPTION
**Approach**: Patch amdgpu/KFD to allow fine-grained mode for 0x731F
**Pros**:
- Fixes at the source
- Applied before HSA initialization
- Persistent across reboots (with DKMS)
**Cons**:
- Requires kernel module rebuild
- Need to maintain for kernel updates
**Verdict**: Most reliable solution

### Option 4: MIOpen-Only Solution 🟡 SAFEST
**Approach**: Only patch MIOpen algorithm selection, don't spoof architecture
**Pros**:
- Doesn't touch memory model
- Safe, no crashes
**Cons**:
- Limited to MIOpen operations
- Other frameworks still broken
**Verdict**: Good fallback if Option 3 fails

## Recommended Implementation: Kernel Module Patch

### Phase 1: Identify Patch Location
Find where KFD sets memory capabilities based on device ID

**Files to check**:
- `drivers/gpu/drm/amd/amdkfd/kfd_topology.c`
- `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c`
- `drivers/gpu/drm/amd/amdkfd/kfd_device.c`

### Phase 2: Create Patch
Add special case for device ID 0x731F:
```c
// In kfd_topology.c or similar
if (device_id == 0x731F && override_gfx_version == 0x10030) {
    // Allow fine-grained mode for gfx1010 when spoofed as gfx1030
    mem_flags |= HSA_MEM_FLAGS_HOT_PLUGGABLE;
    mem_flags |= HSA_MEM_FLAGS_FINE_GRAINED;
}
```

### Phase 3: Build & Install
```bash
# Get kernel sources
apt-get source linux-image-$(uname -r)

# Or use DKMS for amdgpu
cd /usr/src/amdgpu-6.8.5
# Apply patch
# Rebuild module
sudo dkms build -m amdgpu -v 6.8.5
sudo dkms install -m amdgpu -v 6.8.5
```

### Phase 4: Test
```bash
# Reboot with new module
sudo reboot

# Test Conv2d
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 test_conv2d.py
```

## Alternative: Hybrid Approach

If kernel patching is too complex:
1. ✅ Use MIOpen patches for Conv2d (already working in some cases)
2. ✅ Add environment variable to disable fine-grained checks
3. ✅ Build custom PyTorch with gfx1010 support

## Next Steps

- [ ] Locate amdgpu DKMS source
- [ ] Find KFD memory capability code
- [ ] Create targeted patch for 0x731F
- [ ] Build test module
- [ ] Test in safe mode
- [ ] Deploy if successful

## References
- Device 0x731F: RX 5600 XT (Navi 10, RDNA1, gfx1010)
- KFD flags=0: Coarse-grained memory
- KFD flags=1: Fine-grained memory
- HSA_OVERRIDE_GFX_VERSION: Spoofs ISA version but not hardware caps
