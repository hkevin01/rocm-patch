# Hardware Architecture Analysis

## The Core Problem

### Actual Hardware
- **GPU**: AMD Radeon RX 5600 XT
- **Codename**: Navi 10
- **Architecture**: RDNA1
- **GFX Target**: gfx1010
- **PCI Device ID**: 0x731F
- **Capabilities**: Coarse-grained memory only, NO fine-grained SVM

### What ROCm Reports (with override)
- **GFX Target**: gfx1030 (spoofed via HSA_OVERRIDE_GFX_VERSION)
- **Architecture**: RDNA2 (falsely assumed)
- **Expected Capabilities**: Fine-grained SVM support

### The Mismatch
```
Hardware Reality:  gfx1010 (RDNA1) → Coarse-grained memory only
ROCm Assumption:   gfx1030 (RDNA2) → Fine-grained SVM supported
Result:            MEMORY MODEL VIOLATION → System Crash
```

## Why We Need the Override

Without `HSA_OVERRIDE_GFX_VERSION=10.3.0`:
- MIOpen kernel database doesn't have gfx1010 kernels
- Conv2d operations fail with "No suitable kernel found"
- Many AI frameworks don't support RDNA1

With `HSA_OVERRIDE_GFX_VERSION=10.3.0`:
- MIOpen uses gfx1030 kernels (which are similar enough)
- BUT ROCm's memory allocator tries to use fine-grained memory
- gfx1010 hardware doesn't support this → Crash

## The Fix Strategy

### What Doesn't Work
❌ **Patching HSA runtime memory allocation** - Causes system crashes
❌ **LD_PRELOAD shim** - Causes system crashes
❌ **Modifying memory flags at runtime** - Causes system crashes

### Why These Crash
The memory model is initialized early in the GPU driver stack:
1. Kernel driver (amdgpu) sets up memory apertures
2. KFD (Kernel Fusion Driver) configures memory access
3. HSA runtime uses these configurations
4. Trying to change memory model after initialization = CRASH

### What We Need
✅ **Kernel-level patch** - Modify amdgpu/KFD to force coarse-grained for 0x731F
✅ **Early override** - Detect real hardware before memory model initialization
✅ **MIOpen-only patches** - Fix at algorithm selection level, not memory level

## Detection Method

The key is to detect the REAL hardware (0x731F) before any memory allocation:

```cpp
// This works - reading hardware register
uint32_t device_id = properties_.DeviceId;  // Returns 0x731F (real hardware)

// This doesn't work - already using spoofed architecture
uint32_t isa_version = GetIsaVersion();  // Returns gfx1030 (spoofed)
```

## Next Steps

### Option 1: Kernel Module Patch (Most Reliable)
Patch `amdgpu.ko` to force coarse-grained memory for device ID 0x731F

### Option 2: MIOpen-Only Solution (Safest)
Only patch MIOpen kernel selection, don't touch memory model

### Option 3: Custom ROCm Build
Build ROCm from source with proper gfx1010 support

## References
- Device ID 0x731F: Navi 10 (RX 5600 XT / RX 5700 XT)
- RDNA1 Architecture: Coarse-grained memory only
- RDNA2 Architecture: Fine-grained SVM supported
- ROCm Issue #2527: gfx1030 memory model breaks gfx1010 hardware
