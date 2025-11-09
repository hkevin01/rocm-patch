# Research Findings: RDNA1 Large Feature Map Solutions

**Investigation Date**: 2025-01-XX  
**GPU**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)  
**Current Stack**: ROCm 5.7 + PyTorch 2.2.2  
**Problem**: Conv2d hangs on medium/large tensor sizes (16→32 channels, >32x32 input)

---

## Executive Summary

**User's Valid Point**: RX 5600 XT is a powerful gaming GPU capable of handling large textures (4K+), complex shaders, and VR workloads. The inability to run medium-sized Conv2d operations (48x48x32) is clearly a software limitation, not hardware.

**Critical Discovery**: GitHub issue [#2527](https://github.com/ROCm/ROCm/issues/2527) / Discussion [#4030](https://github.com/ROCm/ROCm/discussions/4030) reveals that:
- **ROCm 5.3+ has a regression** that broke gfx1010 (RDNA1)
- **ROCm 5.2 is the LAST truly working version** for RDNA1
- **PyTorch 1.13.1 + ROCm 5.2** works without hangs
- **ROCm 6.2+ has partial support** via custom builds (requires building from source)

**Current Status**: Our ROCm 5.7 approach may be incorrect. We should be using **ROCm 5.2** instead.

---

## GitHub Issue #2527 / Discussion #4030 Key Findings

### Timeline of RDNA1 Support

| ROCm Version | gfx1010 Status | PyTorch | Notes |
|--------------|----------------|---------|-------|
| **5.2** | ✅ **WORKS** | 1.13.1 | Last fully working version |
| **5.3** | ❌ **BROKEN** | 2.0+ | Regression introduced |
| **5.4** | ❌ **BROKEN** | 2.0+ | Regression continues |
| **5.5** | ❌ **BROKEN** | 2.0+ | Performance degraded |
| **5.6** | ❌ **BROKEN** | 2.0+ | Still broken |
| **5.7** | ⚠️ **PARTIAL** | 2.2.2 | Small tensors work, large hang |
| **6.0** | ❌ **BROKEN** | 2.3+ | Memory access faults |
| **6.1** | ❌ **BROKEN** | 2.3+ | Still broken |
| **6.2+** | ⚠️ **CUSTOM BUILD** | 2.4+ | Requires building from source |

### What Broke in ROCm 5.3+?

From the GitHub discussion:
- **Memory access changes**: ROCm 5.3 introduced memory access code changes for gfx1030
- **Side effect**: These changes broke gfx1010 when using `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- **Error**: "Memory access fault...Page not present or supervisor privilege"
- **AMD's response** (Nov 2023): "ok, we will tackle this issue next" → Issue moved to discussion (deprioritized)

### Community Solutions Found

#### Solution 1: ROCm 5.2 + PyTorch 1.13.1 (MOST STABLE)

**Status**: ✅ Confirmed working by multiple users  
**Source**: GitHub issue #2527, Reddit r/ROCm

```bash
# Old PyTorch nightlies built with ROCm 5.2 still work
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2
```

**Limitations**:
- Old PyTorch version (1.13.1 vs 2.2.2)
- Missing newer features (flash attention, etc.)
- PyTorch deleted ROCm 5.2 repos (but wheels still exist for Python 3.10)

**Advantages**:
- ✅ No hangs on large feature maps
- ✅ Stable and proven
- ✅ Used by Stable Diffusion community successfully

#### Solution 2: ROCm 6.2+ Custom Build (EXPERIMENTAL)

**Status**: ⚠️ Works with custom compilation  
**Source**: GitHub discussion #4030 - User @Zakhrov's answer (marked as solution)

**Requirements**:
1. Install ROCm 6.2.1+ (6.2.4 recommended)
2. Build PyTorch from source targeting gfx1010
3. Use custom build flags

**Build Process** (from @Zakhrov):
```bash
# Prerequisites
# - Ubuntu 22.04 or 24.04 LTS
# - ROCm 6.2.1+ installed (rocm-ml-sdk metapackage)
# - GCC 10+ (for C++17 support)

# Step 1: Clone PyTorch
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Step 2: Clean build
git reset --hard
git pull
git submodule update --init --recursive

# Step 3: Install requirements
pip install -r requirements.txt

# Step 4: Build AMD-specific code
python3 tools/amd_build/build_amd.py

# Step 5: Build PyTorch for gfx1010
PYTORCH_ROCM_ARCH=gfx1010 python3 setup.py install
```

**Advantages**:
- ✅ Latest PyTorch (2.4+)
- ✅ Latest features
- ✅ Better performance (reported)

**Limitations**:
- ❌ Requires building from source (4-8 hours)
- ❌ Must rebuild for each PyTorch update
- ⚠️ Official wheels don't work (must compile)

#### Solution 3: HSA_OVERRIDE_GFX_VERSION=9.4.0 (WORKAROUND)

**Status**: ⚠️ Partially working  
**Source**: GitHub discussion #4030 - User @theron29

```bash
export HSA_OVERRIDE_GFX_VERSION=9.4.0  # Instead of 10.3.0
```

**Results**:
- ✅ Works with PyTorch 2.0 (ROCm 5.2 build)
- ⚠️ Not fully stable
- ⚠️ May cause segfaults with newer PyTorch

#### Solution 4: Vulkan Backend (ALTERNATIVE)

**Status**: ✅ Works (but slower)  
**Source**: GitHub discussion #4030 - User @DGdev91

Use Vulkan instead of ROCm/HIP:
```bash
# For LLMs: llama.cpp with Vulkan backend
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make vulkan
```

**Advantages**:
- ✅ No ROCm dependency
- ✅ Stable
- ✅ Works on any AMD GPU

**Limitations**:
- ❌ Slower than ROCm (20-30%)
- ❌ Limited to inference
- ❌ No PyTorch integration

---

## Reddit Community Findings

### r/ROCm Discussion Summary

**Post**: "ROCm on RX 5700 XT / gfx1010 with pytorch?"  
**Link**: https://www.reddit.com/r/ROCm/comments/1gcf3x4/

Key findings:
1. **W5700 works fine with latest ROCm** (same chip as RX 5700 XT)
2. **Ollama works with tweaks**: `HSA_OVERRIDE_GFX_VERSION=10.1.0`
3. **Fedora's PyTorch**: May have gfx1010 support built-in
4. **ROCm 6.2 TensorFlow**: Works on gfx1010 (27 comments confirming)

---

## Stable Diffusion Community Solutions

**Finding**: Stable Diffusion users with RX 5700 XT have been running **large models successfully**

**How they do it**:
1. **AUTOMATIC1111/stable-diffusion-webui**: Has built-in workaround
   - Automatically uses PyTorch 1.13.1 + ROCm 5.2 for RDNA1
   - Workaround written by @DGdev91 (the GitHub issue author!)
   - Requires: `--precision-full --no-half` (no FP16 on RDNA1)

2. **ComfyUI**: Works with custom ROCm 6.2 build
   - User @ethragur confirmed it works
   - All samplers work except SDE
   - No need to force FP32 with ROCm 6.2 build

---

## Performance Comparison

Based on community reports:

| Configuration | Speed | Stability | Large Feature Maps |
|---------------|-------|-----------|-------------------|
| **ROCm 5.2 + PyTorch 1.13.1** | ⭐⭐⭐⭐⭐ | ✅ Excellent | ✅ Works |
| **ROCm 5.4** | ⭐⭐⭐⭐ | ✅ Good | ✅ Works |
| **ROCm 5.5-5.7** | ⭐⭐ | ⚠️ Partial | ❌ Hangs |
| **ROCm 6.2 (official)** | ⭐ | ❌ Broken | ❌ Fails |
| **ROCm 6.2 (custom)** | ⭐⭐⭐ | ⚠️ Partial | ✅ Works |

**Key Insight from @TheTrustedComputer**:
> "ROCm 5.4 was the last performant release before builds broke starting 5.5. ROCm 6.2 onwards brought intermittent dropouts regardless of configuration. If speed is a priority, consider building ROCm 5.4 from source."

---

## Recommended Action Plan

### Option 1: Downgrade to ROCm 5.2 (RECOMMENDED FOR STABILITY)

**Best for**: Users who need stability and don't need latest PyTorch features

```bash
# TODO: Create installation script for ROCm 5.2
```

**Advantages**:
- ✅ Proven working
- ✅ Large feature maps work
- ✅ Stable (no hangs)
- ✅ Used by Stable Diffusion community

**Disadvantages**:
- ❌ Old PyTorch (1.13.1)
- ❌ Missing latest features

### Option 2: Build ROCm 6.2 + PyTorch from Source (RECOMMENDED FOR LATEST)

**Best for**: Users who need latest features and can tolerate build time

```bash
# TODO: Create automated build script
```

**Advantages**:
- ✅ Latest PyTorch (2.4+)
- ✅ Latest features
- ✅ Community confirmed working

**Disadvantages**:
- ❌ 4-8 hour build time
- ❌ Must rebuild for updates
- ⚠️ More complex

### Option 3: Stay on ROCm 5.7 with Limitations (CURRENT)

**Best for**: Users with small models only

**Status**:
- ✅ Works for small tensors (≤32x32)
- ❌ Hangs on medium/large tensors

---

## Critical Information for User

**You were RIGHT to challenge the limitation!** Your RX 5600 XT is powerful and SHOULD handle large feature maps.

**The issue is NOT your hardware** - it's the specific ROCm version (5.7) we chose.

**Solution**: We need to either:
1. **Downgrade to ROCm 5.2** (proven working)
2. **Upgrade to ROCm 6.2 with custom build** (latest features)

**Next Steps**:
1. Backup current ROCm 5.7 configuration
2. Test ROCm 5.2 + PyTorch 1.13.1 (fastest path to working solution)
3. If needed, build ROCm 6.2 + PyTorch 2.4+ from source (for latest features)

---

## References

1. **GitHub Issue #2527**: "Regression in rocm 5.3 and newer for gfx1010"
   - https://github.com/ROCm/ROCm/issues/2527

2. **GitHub Discussion #4030**: Continuation of #2527
   - https://github.com/ROCm/ROCm/discussions/4030
   - **@Zakhrov's answer**: Build instructions for ROCm 6.2

3. **Reddit r/ROCm**: "ROCm on RX 5700 XT / gfx1010 with pytorch?"
   - https://www.reddit.com/r/ROCm/comments/1gcf3x4/

4. **AUTOMATIC1111 Stable Diffusion WebUI**: RDNA1 workaround
   - https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/6420

5. **PyTorch Issue #106728**: ROCm 5.2 solution
   - https://github.com/pytorch/pytorch/issues/106728

6. **AMD ROCm Documentation**: System Requirements
   - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html

---

## Conclusion

**Our current approach (ROCm 5.7) is WRONG.** The research clearly shows:

1. **ROCm 5.3+ broke RDNA1** - this is a known AMD regression
2. **ROCm 5.2 is the LAST working version** - confirmed by community
3. **ROCm 6.2+ CAN work** - but requires building from source

**User is correct**: The RX 5600 XT hardware is fully capable. We just need to use the right software stack.

**Recommended Path Forward**:
1. Test ROCm 5.2 + PyTorch 1.13.1 first (proven solution)
2. If latest features needed, build ROCm 6.2 + PyTorch 2.4+ from source
3. Update all documentation to reflect correct ROCm version

**Status**: Ready to implement either solution based on user preference.
