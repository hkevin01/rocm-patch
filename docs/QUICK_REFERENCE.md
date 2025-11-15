# ROCm Quick Reference Card

**Last Updated**: November 15, 2025  
**For**: AMD RDNA1 GPUs (RX 5600 XT, RX 5700 series)

---

## 🚀 Essential Setup (Copy-Paste Ready)

### Minimal Training Script Template

```python
#!/usr/bin/env python3
"""
ROCm-compatible PyTorch training script
"""

import multiprocessing as mp

# ⚠️ CRITICAL: Must be FIRST, before any torch imports!
mp.set_start_method('spawn', force=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# Set ROCm environment
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# Enable MIOpen bypass (optional, for stability)
import sys
sys.path.insert(0, '/path/to/rocm-patch/src/patches/miopen_bypass')
from conv2d_fallback import enable_miopen_bypass
enable_miopen_bypass()

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,                    # ✅ Works!
    multiprocessing_context='spawn',
    persistent_workers=True,
    pin_memory=True
)

# Train
model = YourModel().cuda()
for epoch in range(epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        # ... training loop ...
```

---

## 📋 ROCm Checklist

### Before Training

```markdown
✅ ROCm 5.2.0 installed
✅ PyTorch 1.13.1+rocm5.2 installed
✅ Python 3.10 virtual environment
✅ MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1 set
✅ HSA_OVERRIDE_GFX_VERSION=10.3.0 set
✅ mp.set_start_method('spawn') BEFORE torch import
✅ DataLoader uses num_workers=4, persistent_workers=True
```

### Verify Setup

```bash
# Check versions
python -c "import torch; print(torch.__version__)"  # Should be 1.13.1+rocm5.2
ls -la /opt/rocm  # Should point to rocm-5.2.0

# Check GPU
rocm-smi
# or
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Check environment
echo $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM  # Should be 1
echo $HSA_OVERRIDE_GFX_VERSION         # Should be 10.3.0

# Check multiprocessing
python -c "import multiprocessing as mp; mp.set_start_method('spawn', force=True); print(mp.get_start_method())"  # Should be 'spawn'
```

---

## 🔧 Common Issues & Fixes

### Issue 1: Conv2d Hangs (>42×42)

**Symptom**: Training hangs on forward pass, no error

**Fix**:
```bash
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
```

### Issue 2: DataLoader Workers Hang

**Symptom**: Training hangs after first epoch, worker timeout

**Fix**:
```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Before torch import!

DataLoader(..., persistent_workers=True)  # Keep workers alive
```

### Issue 3: Memory Access Violation

**Symptom**: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

**Fix**: Version mismatch, reinstall correct versions:
```bash
pip uninstall torch torchvision
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --extra-index-url https://download.pytorch.org/whl/rocm5.2
```

### Issue 4: NumPy 2.x Warning

**Symptom**: `module compiled with NumPy 1.x cannot run with NumPy 2.x`

**Fix**:
```bash
pip install "numpy<2"
```

### Issue 5: "context has already been set"

**Symptom**: `RuntimeError: context has already been set`

**Fix**: Use `force=True`
```python
mp.set_start_method('spawn', force=True)
```

---

## 📊 Optimal Configuration

### Hardware: RX 5600 XT (6GB VRAM)

| Setting | Value | Why |
|---------|-------|-----|
| **Batch Size** | 4-8 | Fits in 6GB VRAM |
| **num_workers** | 4 | Tested optimal |
| **persistent_workers** | True | 2x faster |
| **pin_memory** | True | Faster GPU transfer |
| **Image Size** | 640 | Standard for YOLO |
| **Mixed Precision** | False | PyTorch 1.13.1 limited support |

### Expected Performance (YOLOv8n)

```
Batch size: 4
Image size: 640×640
Workers: 4 (persistent)

Speed: 4.7 iterations/second
GPU Utilization: 98%
VRAM Usage: 3.2GB / 6.4GB
Temperature: 73-83°C
Training time: ~10 days for 50 epochs
```

---

## 🔑 Environment Variables

### Essential

```bash
# Must have
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1  # Use IMPLICIT_GEMM algorithm
export HSA_OVERRIDE_GFX_VERSION=10.3.0    # GPU architecture override

# Add to ~/.bashrc for persistence
echo 'export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1' >> ~/.bashrc
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
```

### Optional (for debugging)

```bash
# MIOpen logging (verbose)
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=4

# ROCm profiling
export HSA_ENABLE_SDMA=0  # Disable SDMA (can help stability)
```

---

## 🐍 Python Code Snippets

### Setup Multiprocessing

```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

### Verify Setup

```python
import multiprocessing as mp
import torch

print(f"Multiprocessing: {mp.get_start_method()}")  # Should be 'spawn'
print(f"PyTorch: {torch.__version__}")              # Should be 1.13.1+rocm5.2
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"GPU: {torch.cuda.get_device_name(0)}")     # Should be RX 5600 XT
```

### Create Optimal DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    multiprocessing_context='spawn',
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2
)
```

### Enable MIOpen Bypass

```python
import sys
sys.path.insert(0, '/path/to/rocm-patch/src/patches/miopen_bypass')
from conv2d_fallback import enable_miopen_bypass

enable_miopen_bypass()  # Auto strategy (recommended)
```

### Use ROCm Compatibility Utils

```python
from src.utils.rocm_compat import setup_rocm_multiprocessing, patch_dataloader

setup_rocm_multiprocessing()  # Before torch import
import torch
patch_dataloader()  # After torch import
```

---

## 📚 Documentation Links

### Core Guides

- **[Main README](../README.md)** - Complete project overview
- **[DataLoader Guide](ROCM_DATALOADER_MULTIPROCESSING.md)** - Multiprocessing setup
- **[MIOpen Bypass](../src/patches/miopen_bypass/README_GPU_ONLY.md)** - Conv2d fixes

### Technical Docs

- **[Solution Summary](SOLUTION_SUMMARY.md)** - Quick technical reference
- **[MIOpen Bypass Solution](MIOPEN_BYPASS_SOLUTION.md)** - Detailed Conv2d guide
- **[GPU-Only Solution](GPU_ONLY_SOLUTION.md)** - GPU unfold+matmul approach

### Utility Code

- **[ROCm Compat Utils](../src/utils/rocm_compat.py)** - Compatibility module
- **[Conv2d Fallback](../src/patches/miopen_bypass/conv2d_fallback.py)** - Bypass implementation

---

## 🎯 Training Flow Checklist

### 1. Environment Setup

```bash
□ Install ROCm 5.2.0
□ Create Python 3.10 venv
□ Install PyTorch 1.13.1+rocm5.2
□ Set MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
□ Set HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### 2. Script Setup

```python
□ mp.set_start_method('spawn') BEFORE torch import
□ Import torch and other libraries
□ Enable MIOpen bypass (optional)
□ Create DataLoader with num_workers=4, persistent_workers=True
```

### 3. First Run Test

```bash
□ Run with batch_size=1, 1 epoch to verify setup
□ Check GPU utilization (should be >90%)
□ Monitor temperature (should be <85°C)
□ Verify no hangs or crashes
```

### 4. Full Training

```bash
□ Adjust batch_size for GPU memory
□ Set appropriate num_epochs
□ Enable logging/checkpointing
□ Monitor training metrics
```

---

## 💡 Pro Tips

### Performance

- ✅ Use `persistent_workers=True` for ~2x speedup
- ✅ Set `num_workers=4` for optimal performance
- ✅ Enable `pin_memory=True` if enough RAM
- ✅ Use `prefetch_factor=2` (default, works well)

### Stability

- ✅ Always set multiprocessing to 'spawn' before torch import
- ✅ Use MIOpen bypass for complex models (YOLOv8, etc.)
- ✅ Monitor GPU temperature (keep <85°C)
- ✅ Save checkpoints frequently

### Debugging

- ✅ Start with `num_workers=0` to isolate data loading issues
- ✅ Use `verbose=True` in bypass config to see what's happening
- ✅ Check `rocm-smi` for GPU utilization
- ✅ Monitor system logs: `dmesg | grep -i amd`

---

## �� Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| **ROCm** | 5.2.0 | ✅ Only version fully supporting RDNA1 |
| **PyTorch** | 1.13.1+rocm5.2 | ✅ Must match ROCm version exactly |
| **Python** | 3.10.x | ✅ PyTorch 1.13.1 max support |
| **NumPy** | <2.0 (1.26.4) | ✅ PyTorch 1.13.1 binary compatibility |
| **CUDA** | 11.4 equivalent | ℹ️ ROCm 5.2 maps to CUDA 11.4 |

---

## 🎓 Key Learnings

### Critical Discoveries

1. **Multiprocessing Context**
   - ROCm requires 'spawn', not 'fork'
   - Must be set BEFORE importing torch
   - `force=True` ensures it works

2. **DataLoader Configuration**
   - `num_workers=4` works perfectly
   - `persistent_workers=True` essential (~2x faster)
   - `multiprocessing_context='spawn'` must be explicit

3. **Conv2d Issues**
   - MIOpen hangs on sizes >42×42
   - IMPLICIT_GEMM algorithm works
   - GPU unfold+matmul bypasses MIOpen completely

4. **Version Lock**
   - ROCm 5.2 + PyTorch 1.13.1+rocm5.2 is the sweet spot
   - Newer versions drop RDNA1 support
   - Version mismatch causes memory errors

---

**Status**: ✅ **Production Ready**  
**Source**: Tested on robust-thermal-image-object-detection project  
**Hardware**: AMD Radeon RX 5600 XT (gfx1010)  
**Performance**: 98% GPU utilization, 4.7 it/s, stable 10-day training
