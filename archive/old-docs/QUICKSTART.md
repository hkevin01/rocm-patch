# Quick Start Guide - ROCm Conv2d Fix

**Problem**: Conv2d hangs on AMD RDNA1 GPUs (RX 5600/5700 XT) at 44×44+ pixels  
**Solution**: Proper version matching + IMPLICIT_GEMM algorithm

---

## TL;DR: Copy-Paste Install

```bash
# 1. Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev

# 2. Create project directory and virtual environment
mkdir -p ~/rocm-patch && cd ~/rocm-patch
python3.10 -m venv venv-py310-rocm52
source venv-py310-rocm52/bin/activate

# 3. Install PyTorch 1.13.1+rocm5.2 (EXACT VERSION MATCH!)
pip install --upgrade pip
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install "numpy<2"

# 4. Configure environment (add to ~/.bashrc)
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export ROCM_PATH=/opt/rocm-5.2.0
export LD_LIBRARY_PATH=/opt/rocm-5.2.0/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm-5.2.0/bin:$PATH

# 5. Test
python -c "import torch; print(f'✅ PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)}')"
```

---

## Critical Configuration

| Component | EXACT Version | Why |
|-----------|---------------|-----|
| ROCm | 5.2.0 | Best RDNA1 support |
| PyTorch | 1.13.1+rocm5.2 | Must match ROCm binary |
| Python | 3.10.x | PyTorch 1.13.1 requirement |
| NumPy | <2.0 | Binary compatibility |
| Algorithm | IMPLICIT_GEMM | Avoids RDNA1 bug |

---

## The Magic Environment Variable

```bash
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
```

**What it does**: Forces MIOpen to use stable matrix multiplication algorithm instead of buggy direct convolution on RDNA1.

**Without it**: Conv2d hangs on tensors >42×42 pixels  
**With it**: All sizes work perfectly ✅

---

## Quick Test

```python
import torch
conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
x = torch.randn(1, 3, 44, 44).cuda()  # Previously hung here!
y = conv(x)  # ✅ Now works!
print(f"✅ Success! Output: {y.shape}")
```

---

## Troubleshooting One-Liners

```bash
# Verify versions match
python -c "import torch; print(f'PyTorch: {torch.__version__}, HIP: {torch.version.hip}')"
# Should show: PyTorch: 1.13.1+rocm5.2, HIP: 5.2.21151-afdc89f8

# Check environment variable
echo $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM  # Should output: 1

# Check ROCm path
readlink -f /opt/rocm  # Should show: /opt/rocm-5.2.0

# Fix NumPy if needed
pip install "numpy<2"
```

---

## Why This Works

**Root Causes Found**:
1. ❌ ROCm 5.7/6.x dropped RDNA1 optimizations
2. ❌ Version mismatch causes memory errors
3. ❌ Default algorithm has RDNA1-specific bug at 42×42 boundary
4. ❌ Python 3.12 incompatible with PyTorch 1.13.1 binaries

**Solution**:
1. ✅ ROCm 5.2.0 has full RDNA1 support
2. ✅ PyTorch 1.13.1+rocm5.2 matches runtime exactly
3. ✅ IMPLICIT_GEMM bypasses buggy algorithm
4. ✅ Python 3.10 venv for binary compatibility

---

## Performance

| Size | Time | Status |
|------|------|--------|
| 32×32 | 2.083s (first) / 0.3s (cached) | ✅ |
| 44×44 | 0.278s | ✅ (previously hung!) |
| 224×224 | 0.180s | ✅ |

---

**Full Documentation**: See [README.md](./README.md) for complete technical details, mathematical explanations, and troubleshooting.

**Status**: ✅ Production-ready for RDNA1 GPUs (RX 5600/5700 XT)
