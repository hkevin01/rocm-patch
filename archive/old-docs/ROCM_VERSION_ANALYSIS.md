# ROCm Version Analysis for RDNA1 (gfx1010) on Ubuntu 24.04
**Date**: November 9, 2025

## Problem Summary
- **Goal**: Fix Conv2d hangs on large feature maps (>32x32, power-of-2 channels)
- **Hardware**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)
- **OS**: Ubuntu 24.04.3 LTS (kernel 6.8)
- **Current**: ROCm 5.7 + PyTorch 2.2.2+rocm5.7

## Test Results
### ROCm 5.7 (Currently Installed)
✅ **Small convolutions work** (3→16, 32x32)
❌ **Medium/large convolutions hang** (16→32, ≥48x48)

## ROCm Version Compatibility Research

### ROCm 5.2 (Community Recommended for gfx1010)
- **Source**: luaartist/Rocm_Project recommends ROCm 5.2 for gfx1010
- **Ubuntu Compatibility**: Built for Ubuntu 20.04/22.04
- **Ubuntu 24.04 Issue**: ❌ **INCOMPATIBLE**
  - Requires `libtinfo5` (not available in Ubuntu 24.04)
  - Requires `libncurses5` (not available in Ubuntu 24.04)
  - Requires old `libstdc++-5-dev` or `libstdc++-7-dev` (not available)
- **Verdict**: Cannot install on Ubuntu 24.04 without major hacks

### ROCm 5.7 (Currently Installed)
- **Ubuntu Compatibility**: ✅ Installed and working
- **RDNA1 Support**: Partial (small convs work, large convs hang)
- **Known Issues**:
  - MIOpen GEMM hangs on specific conv configurations
  - MIOPEN_FIND_ENFORCE=3 causes 33+ minute hangs (fixed by removal)
- **Verdict**: Workable for small models, fails for larger models

### ROCm 6.x/7.x (Latest)
- **Ubuntu Compatibility**: ✅ Official support for Ubuntu 24.04.3
- **RDNA1 Support**: According to AMD compatibility matrix:
  - gfx1030 is listed with footnote [19]
  - "For ROCm 7.0.x - AMD Radeon PRO W6800 (gfx1030) only supports Ubuntu 24.04.3"
  - **WARNING**: gfx1010 is NOT explicitly listed (only gfx1030)
- **Community Reports**: ROCm 6.x may break RDNA1 (fine-grained memory assumptions)
- **Risk**: High - may not work at all with gfx1010

## Root Cause Analysis
The hang appears to be a **MIOpen/Tensile library issue** in ROCm 5.7:
1. Specific GEMM kernel selection for power-of-2 channels
2. rocBLAS/Tensile fallback logic triggers problematic code paths
3. Tensile PRs #1862/#1897 addressed gfx1010 issues but may not be in ROCm 5.7 packaging

## Options Analysis

### Option A: Stay with ROCm 5.7 + Application-Level Workarounds
**Pros**:
- Already installed and working for small convs
- No reinstall/reboot required
- PyTorch 2.2.2 verified working

**Cons**:
- Large convs will not work natively
- Requires model modifications

**Workarounds**:
1. Tile large feature maps into smaller chunks (e.g., 32x32 patches)
2. Use non-power-of-2 channels (e.g., 15→31 instead of 16→32)
3. Use adaptive pooling to reduce sizes before problematic convs
4. Restrict model architectures to small feature maps

### Option B: Try ROCm 6.0 (Moderate Risk)
**Pros**:
- Official Ubuntu 24.04 support
- May have Tensile fixes
- More recent PyTorch versions available

**Cons**:
- gfx1010 support unclear (only gfx1030 explicitly listed)
- Community warns ROCm 6.x breaks RDNA1
- May lose ALL functionality (not just large convs)

**Risk Assessment**: 🟡 MEDIUM-HIGH risk of complete failure

### Option C: Downgrade to Ubuntu 22.04 + ROCm 5.2
**Pros**:
- Community-validated solution for gfx1010
- ROCm 5.2 compatibility confirmed

**Cons**:
- Requires OS reinstall
- Downgrading Ubuntu is complex
- Time-consuming (2-4 hours)

**Risk Assessment**: 🟢 LOW risk, but HIGH effort

### Option D: Hardware Upgrade to RDNA2/RDNA3
**Pros**:
- Full ROCm support
- All convolutions work
- Future-proof

**Cons**:
- Costs $300-$600
- Not a software solution

## Recommendations

### Immediate Action (Next 30 minutes):
**Try application-level workarounds with ROCm 5.7**
1. Modify models to use tiling or smaller feature maps
2. Test with non-power-of-2 channels
3. Document working configurations

### Short-term (If workarounds insufficient):
**Test ROCm 6.0 (with backup plan)**
1. Create full system backup
2. Document rollback procedure
3. Install ROCm 6.0 and test
4. If fails, rollback to ROCm 5.7

### Long-term (Production solution):
- **If ROCm 6.0 works**: Update documentation, done!
- **If ROCm 6.0 fails**: Either:
  - Accept ROCm 5.7 limitations and use workarounds
  - Downgrade to Ubuntu 22.04 + ROCm 5.2 (requires OS reinstall)
  - Upgrade hardware to RDNA2+ GPU

## Decision Matrix

| Solution | Effort | Risk | Success Probability | Time to Test |
|----------|--------|------|-------------------|--------------|
| Workarounds (ROCm 5.7) | Low | Low | 90% | 30 min |
| ROCm 6.0 | Medium | Medium-High | 40% | 1 hour |
| Ubuntu 22.04 + ROCm 5.2 | High | Low | 85% | 3-4 hours |
| Hardware Upgrade | High | None | 100% | N/A (cost) |

## Next Steps
**Recommended path**: Start with Option A (workarounds), then evaluate Option B (ROCm 6.0) if needed.

