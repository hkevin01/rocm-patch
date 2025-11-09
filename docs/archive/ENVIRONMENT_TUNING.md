# 🎯 Option 3: Environment Tuning (Safe Solution)

**Status**: ✅ **RECOMMENDED** - Safe, no code changes  
**Date**: November 8, 2025

---

## 🎓 Why This Approach Works

After testing kernel, runtime, and library patches (all caused crashes), we've learned:

> **You cannot fake hardware capabilities. Work WITH your hardware, not against it.**

This approach:
- ✅ Uses hardware as-is (coarse-grained memory only)
- ✅ Configures software to work within limitations
- ✅ No code changes = no crashes
- ✅ Fully reversible
- ⚠️ May be slower, but STABLE

---

## 🔧 Configuration

### Basic Environment Setup

Create a file: `~/rocm_rdna1_env.sh`

```bash
#!/bin/bash
# ROCm RDNA1 (gfx1010) Environment Configuration
# For AMD Radeon RX 5600 XT / RX 5700 XT

# Enable gfx1030 kernels (for PyTorch compatibility)
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# MIOpen: Disable implicit GEMM convolutions (problematic on RDNA1)
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0

# MIOpen: Use database only (don't search for new algorithms)
export MIOPEN_FIND_ENFORCE=3

# MIOpen: Disable Winograd convolutions (may cause issues)
export MIOPEN_DEBUG_CONV_WINOGRAD=0

# MIOpen: Disable direct convolutions (use GEMM fallback)
export MIOPEN_DEBUG_CONV_DIRECT=0

# MIOpen: Enable verbose logging (for debugging)
export MIOPEN_LOG_LEVEL=4

# ROCm: Disable fine-grained memory warnings
export HSA_ENABLE_SDMA=0

# PyTorch: Use compatible convolution algorithms
export PYTORCH_ROCM_ARCH=gfx1030

echo "✅ ROCm RDNA1 environment configured"
echo "   - HSA Override: gfx1030"
echo "   - MIOpen: Using GEMM fallback algorithms"
echo "   - Memory: Coarse-grained only"
