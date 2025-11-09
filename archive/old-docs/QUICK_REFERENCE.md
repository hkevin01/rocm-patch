# Quick Reference Card

## 📋 Required Versions

| Software | Version | Install Command |
|----------|---------|----------------|
| ROCm | 5.7 | [See INSTALL.md](INSTALL.md) |
| PyTorch | 2.2.2+rocm5.7 | `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7` |
| Python | 3.8+ | `python3 --version` |
| Ubuntu | 20.04/22.04/24.04 | `lsb_release -a` |

## 🚀 Quick Commands

```bash
# Verify setup
./verify_setup.sh

# Test Conv2d timing
python3 test_conv2d_timing.py

# Check environment variables
source /etc/profile.d/rocm-rdna1-57.sh
env | grep -E "HSA_OVERRIDE|MIOPEN|HIP_FORCE|PYTORCH_ROCM"

# Check PyTorch version
python3 -c "import torch; print(torch.__version__)"

# Quick Conv2d test
python3 -c "import torch; x = torch.randn(1,3,32,32).cuda(); conv = torch.nn.Conv2d(3,16,3,padding=1).cuda(); print(conv(x).shape)"
```

## 🔧 Critical Environment Variables

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0      # Spoof gfx1010 → gfx1030
MIOPEN_DEBUG_CONV_GEMM=1             # ✅ ENABLE GEMM
HIP_FORCE_COARSE_GRAIN=1             # ✅ FORCE COARSE-GRAINED
MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0    # ❌ DISABLE
MIOPEN_DEBUG_CONV_WINOGRAD=0         # ❌ DISABLE
MIOPEN_DEBUG_CONV_DIRECT=0           # ❌ DISABLE
```

**Location**: `/etc/profile.d/rocm-rdna1-57.sh` (auto-loads)

## 🎮 Supported GPUs

- AMD Radeon RX 5600 XT (Device ID: 731F)
- AMD Radeon RX 5700 XT (Device ID: 731E)
- AMD Radeon RX 5700 (Device IDs: 7310, 7312)

All RDNA1 (Navi 10) architecture.

## ⏱️ Expected Behavior

### First Run (Per Unique Conv2d Config)
- **Time**: 30-60 seconds
- **Reason**: MIOpen compiling kernels
- **Status**: ✅ NORMAL
- **What happens**: Kernels cached to `~/.config/miopen/`

### Subsequent Runs
- **Time**: <0.01 seconds
- **Reason**: Using cached kernels
- **Status**: ✅ NORMAL
- **Speedup**: ~1000x faster

### Debug First Run
```bash
export MIOPEN_LOG_LEVEL=7
python3 test_conv2d_timing.py
# You'll see: "Starting find for miopenConvolutionFwdAlgoGEMM"
```

## ❌ Troubleshooting

### Conv2d hangs?
1. Open **NEW terminal** (important!)
2. Check: `echo $MIOPEN_DEBUG_CONV_GEMM` (should be `1`)
3. If not set: `source /etc/profile.d/rocm-rdna1-57.sh`

### Wrong PyTorch version?
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/rocm5.7
```

### First run taking forever?
- ✅ **NORMAL** if 30-60 seconds (kernel compilation)
- ❌ **PROBLEM** if >5 minutes (check environment variables)

### CUDA not available?
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
# Should be: True
# If False, check ROCm installation
```

## 🔍 Verification Checklist

Run `./verify_setup.sh` - should show all ✅:

- ✅ Configuration file exists
- ✅ HSA_OVERRIDE_GFX_VERSION=10.3.0
- ✅ MIOPEN_DEBUG_CONV_GEMM=1
- ✅ HIP_FORCE_COARSE_GRAIN=1
- ✅ RDNA1 GPU detected
- ✅ PyTorch 2.2.2+rocm5.7
- ✅ CUDA available

## 🐛 AMP (Automatic Mixed Precision)

Add this bypass at the top of your script:

```python
import torch
torch.cuda.amp.common.amp_definitely_not_available = lambda: False

# Now use AMP normally
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
```

## 📚 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `README_ROCM57.md` | ROCm 5.7 specific guide |
| `SOLUTION_ROCM57.md` | Solution summary |
| `PROJECT_STRUCTURE.md` | File organization |
| `QUICK_REFERENCE.md` | This file |

## 🔗 Links

- **This Repo**: [hkevin01/rocm-patch](https://github.com/hkevin01/rocm-patch)
- **ROCm Issue**: [ROCm/ROCm#2527](https://github.com/ROCm/ROCm/issues/2527)
- **ROCm 5.7 Docs**: https://rocm.docs.amd.com/
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/

## 📞 Support

1. Check `README.md` - comprehensive guide
2. Run `./verify_setup.sh` - automated checks
3. Check `docs/archive/` - investigation history
4. Open GitHub issue - if problem persists

---

**Status**: ✅ WORKING (as of November 8, 2025)
**Verified On**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)
