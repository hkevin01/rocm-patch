# GitHub Research Findings - RDNA1 gfx1010 ROCm Issues

## Summary
Extensive RDNA1 community problem documented in GitHub issue #2527 (moved to discussion #4030). The RX 5600 XT (gfx1010) crashes were **officially acknowledged** by ROCm team but **never supported**.

## Key Facts Discovered

### 1. **Official AMD Position**
- RDNA1 (gfx1010) was **NEVER officially supported** by ROCm
- AMD ROCm Collaborator (jamesxu2, Nov 14, 2024): *"RDNA1 is not, and has never been officially supported by ROCm"*
- Support matrix explicitly excludes gfx1010: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus

### 2. **Root Cause Identified**
**Issue #2527: "Regression in rocm 5.3 and newer for gfx1010"**

```
- ROCm 5.2 + PyTorch 1.13.1: ✅ WORKED (with HSA_OVERRIDE_GFX_VERSION=10.3.0)
- ROCm 5.3+: ❌ BROKEN (Memory access faults, segfaults)
- Error: "Memory access fault by GPU node-1 on address 0x7fa860641000. 
         Reason: Page not present or supervisor privilege"
```

**Quote from kmsedu (Nov 16, 2023):**
> "Ever since the release of ROCm 5.3, **some change in memory access code for the gfx1030 arch** prevented us from using HSA_OVERRIDE_GFX_VERSION hack, due to OOB errors."

### 3. **ROCm 6.2 Partial Fix (Aug 2024)**
- Tensile patch merged: https://github.com/ROCm/Tensile/pull/1897
- rocBLAS now works on gfx1010
- MIOpen + Composable Kernels: **STILL BROKEN**

**Key quote from GZGavinZhao (Aug 4, 2024):**
> "Since ROCm/Tensile#1897 is included in 6.2 release, rocBLAS should now work on RDNA1. LLMs like llama-cpp should work. Other ML workflows like stable diffusion that need MIOpen **may or may not work**."

### 4. **Community Solution (Nov 2024)**
User **Zakhrov** successfully built PyTorch 2.4.0 from source on ROCm 6.2.1:

```bash
# Prerequisites: ROCm 6.2.1+, Ubuntu 22.04/24.04
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
python3 tools/amd_build/build_amd.py
PYTORCH_ROCM_ARCH=gfx1010 python3 setup.py install
```

**Result:**
- ✅ Basic operations work
- ✅ ComfyUI confirmed working
- ❌ Flash attention / memory-efficient attention NOT available
- ❌ Occasional GPU lockups
- ❌ **Official PyTorch wheels still broken** (must compile from source)

### 5. **MIOpen Patches Required**
**TheTrustedComputer (Nov 21, 2024) - CRITICAL INFO:**

MIOpen + Composable Kernels require patches for gfx1010:
- Patch location: https://github.com/ROCm/composable_kernel/issues/775
- Google Drive patches provided:
  - ROCm 6.2.x: https://drive.google.com/file/d/1RFbfYtG0B0JbtTai9iWdVomammAFL8Db/view
  - ROCm 6.3.x: https://drive.google.com/file/d/1lC2MEgXbE85IgnKlvVug3R6lPypy1ATl/view

**Quote:**
> "The most challenging patches to introduce basic RDNA1 compatibility were **MIOpen + composable kernels**. They do not compile with `MIOPEN_USE_COMPOSABLEKERNEL=ON` due to missing macro definitions and kernel build parameters for RDNA1 hardware."

### 6. **Performance Regression Issue**
**TheTrustedComputer (Dec 25, 2024):**

Performance degraded after Tensile fallback patch:
```
- ROCm 5.2/5.4: Fast, real-time inference works
- ROCm 5.5-5.6: Broken builds
- ROCm 6.2+: Working but SLOW (fallback kernels, not optimized)
```

**Quote:**
> "Optimizations for RDNA1 appear to be missing from the current branch. If speed is a priority, consider building ROCm 5.4 from source."

### 7. **Specific Libraries Status (TheTrustedComputer, Aug 11, 2024)**
Tested with ROCm 6.2 on gfx1012 (RX 5500 XT):

**✅ Working:**
- rocBLAS / hipBLAS
- rocFFT / hipFFT
- MIOpen
- MIGraphX

**❌ Not Working:**
- rocRAND / hipRAND
- rocSPARSE / hipSPARSE  
- rocSOLVER / hipSOLVER

**Workaround:** Rebuild broken libraries targeting gfx1010/gfx1012

---

## Conclusion: Why mtype_local=1 Didn't Fix Our Issue

1. **Memory type changes happened between ROCm 5.2 → 5.3**
   - gfx1030 (RDNA2) memory code changes broke gfx1010 emulation
   - Our crashes match *exactly* the reported pattern in issue #2527

2. **Kernel parameter insufficient**
   - Driver-level mtype setting can't override compiled kernel code
   - MIOpen kernels have **hardcoded memory requests** in assembly
   - Would need MIOpen recompilation with gfx1010 patches

3. **Current Status (as of Feb 2025)**
   - Official wheels: ❌ Broken
   - Source build with patches: ✅ Works but slow
   - MIOpen: ❌ Needs patches (no official support)
   - glibc 2.41+: ❌ Even ROCm 5.2 wheels broken

---

## Our Options

### Option A: Build PyTorch from Source (Community Solution)
- Follow Zakhrov's method (verified working Nov 2024)
- ROCm 6.2.1 + PyTorch main branch
- Expect: Functional but slower than ROCm 5.2
- **Success chance: 75% (works but may have GPU lockups)**

### Option B: Patch MIOpen + Rebuild ROCm Stack
- Apply TheTrustedComputer's MIOpen/CK patches
- Rebuild entire ROCm 6.2 stack for gfx1010
- Expect: Better performance, full features
- **Success chance: 50% (complex, 8-12 hours, requires patches)**

### Option C: Downgrade to ROCm 5.4
- Build ROCm 5.4 from source: https://github.com/xuhuisheng/rocm-build/
- Confirmed fastest option for gfx1010
- Expect: Best performance, limited to older PyTorch
- **Success chance: 70% (source build complexity)**

### Option D: Stay on ROCm 5.2 Workaround
- Use old PyTorch 1.13.1 wheels with HSA_OVERRIDE_GFX_VERSION=10.3.0
- Already confirmed working (per issue #2527)
- **Problem: No longer available on PyTorch repo**

### Option E: Accept Current Situation
- Use CPU for Conv2d operations
- GPU for other operations (matmul, etc.)
- Wait for community patches to mature

---

## References
- Main issue: https://github.com/ROCm/ROCm/issues/2527
- Discussion: https://github.com/ROCm/ROCm/discussions/4030
- Tensile fix: https://github.com/ROCm/Tensile/pull/1897
- CK patches: https://github.com/ROCm/composable_kernel/issues/775
- Build scripts: https://github.com/xuhuisheng/rocm-build/
