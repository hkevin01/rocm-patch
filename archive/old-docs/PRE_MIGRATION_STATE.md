# Pre-Migration State (ROCm 5.7)
**Date**: November 9, 2025

## Environment
- **ROCm Version**: 5.7
- **PyTorch Version**: 2.2.2+rocm5.7
- **GPU**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)
- **Device ID**: 1002:731f

## Test Results on ROCm 5.7
### Small Convolutions ✅ PASS
- **Test**: 3→16 channels, 32x32 input
- **Status**: Works perfectly
- **First run**: ~0.22-0.29s (kernel compilation/cache)
- **Cached runs**: ~0.0001-0.0003s

### Medium/Large Convolutions ❌ HANG
- **Test**: 16→32 channels, 48x48 input (timeout 180s)
- **Status**: **HANGS** - confirmed reproducible hang
- **Pattern**: Power-of-two channel counts with sizes ≥48x48

## Critical Environment Variables
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- `PYTORCH_ROCM_ARCH=gfx1030`
- `MIOPEN_DEBUG_CONV_GEMM=1` (CRITICAL)
- `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0`
- `MIOPEN_DEBUG_CONV_WINOGRAD=0`
- `MIOPEN_DEBUG_CONV_DIRECT=0`
- `MIOPEN_DEBUG_CONV_FFT=0`
- `HIP_FORCE_COARSE_GRAIN=1`
- `HSA_ENABLE_SDMA=0`
- `HSA_USE_SVM=0`
- `HSA_XNACK=0`

## Known Issues
1. **MIOPEN_FIND_ENFORCE=3**: Causes exhaustive search (2000+ seconds) - REMOVED
2. **GEMM Hang Bug**: Specific conv sizes trigger MIOpen/Tensile hang on ROCm 5.7
3. **ROCm 6.x**: Breaks RDNA1 entirely (fine-grained memory assumptions)

## Reason for Migration to ROCm 5.2
- Community reports suggest ROCm 5.2 has better gfx1010 support
- Tensile patches may be more stable in 5.2 release
- Referenced by luaartist/Rocm_Project for gfx1010 compatibility

## Configuration Files
- `/etc/profile.d/rocm-rdna1-57.sh` (backed up)
- `install_rocm57.sh`
- `test_conv2d_timing.py`
- `test_conv2d_large.py`

