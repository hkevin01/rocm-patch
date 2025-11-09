# Migration Plan: ROCm 5.7 → ROCm 5.2 for RDNA1

**Date**: November 8, 2025  
**Critical Discovery**: ROCm 5.3+ has a regression that breaks gfx1010 (RDNA1)  
**Solution**: Downgrade to ROCm 5.2 (last working version)

---

## Why ROCm 5.2?

### Research Findings

1. **ROCm 5.3+ introduced regression for gfx1010** ([GitHub Issue](https://github.com/pytorch/pytorch/issues/111621))
2. **Multiple users confirmed ROCm 5.2 works** where 5.3+ fails
3. **PyTorch 2.2.2 is compatible with ROCm 5.2**
4. **Large feature maps (>32x32) should work on 5.2**

### Evidence from Community

- Reddit user: "ROCm 5.2 works perfectly on RX 5700 XT"
- GitHub: "Regression between 5.2 and 5.3 for gfx1010"
- luaartist repo: Specifically recommends ROCm 5.2 for gfx1010

---

## Current State (ROCm 5.7)

**What Works**:
- ✅ Small Conv2d (≤32x32)
- ✅ 3→16 channels at any size
- ✅ Basic operations

**What Doesn't Work**:
- ❌ 16→32 channels with >32x32 input (hangs)
- ❌ 32→64 channels with >48x48 input (hangs)
- ❌ Large ResNet models
- ❌ Standard ImageNet models

---

## Expected State (ROCm 5.2)

**Should Work**:
- ✅ All Conv2d operations (any size)
- ✅ Large feature maps (224x224+)
- ✅ ResNet50, YOLO, etc.
- ✅ Standard ImageNet models
- ✅ Full deep learning workflow

---

## Migration Steps

### Step 1: Backup Current Setup

```bash
# Save current configuration
cp /etc/profile.d/rocm-rdna1-57.sh ~/rocm-57-backup.sh

# Save environment
env | grep -E "ROCM|MIOPEN|HIP|HSA" > ~/env-backup.txt

# Note PyTorch version
python3 -c "import torch; print(torch.__version__)" > ~/pytorch-version.txt
```

### Step 2: Uninstall ROCm 5.7

```bash
# Remove ROCm 5.7 packages
sudo apt remove --purge -y rocm* hip* miopen* rocblas* rocfft* rocrand* \
    rocsolver* hipblas* hipfft* hipsparse* rccl*

# Clean up
sudo apt autoremove -y
sudo apt autoclean

# Remove configuration
sudo rm -f /etc/profile.d/rocm-rdna1-57.sh
```

### Step 3: Install ROCm 5.2

```bash
# Run the installation script
./install_rocm52_rdna1.sh
```

Or manually:

```bash
# Add ROCm 5.2 repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install
sudo apt update
sudo apt install -y rocm-dev rocm-libs rocminfo rocm-smi hip-base \
    hip-runtime-amd hip-dev miopen-hip rocblas

# Add user to groups
sudo usermod -a -G render,video $USER
```

### Step 4: Install PyTorch 2.2.2 for ROCm 5.2

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch 2.2.2 with ROCm 5.2
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2
```

### Step 5: Configure Environment

```bash
# Create new configuration
sudo tee /etc/profile.d/rocm-rdna1-52.sh > /dev/null << 'CONFIG'
#!/bin/bash
# ROCm 5.2 Configuration for RDNA1 (gfx1010)

if lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    # ROCm paths
    export ROCM_PATH=/opt/rocm-5.2.0
    export PATH=$PATH:/opt/rocm-5.2.0/bin:/opt/rocm-5.2.0/hip/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-5.2.0/lib:/opt/rocm-5.2.0/hip/lib
    
    # Architecture
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030
    
    # MIOpen - GEMM only (safer)
    export MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    export MIOPEN_DEBUG_CONV_WINOGRAD=0
    export MIOPEN_DEBUG_CONV_DIRECT=0
    export MIOPEN_DEBUG_CONV_FFT=0
    
    # Let MIOpen use default find mode
    export MIOPEN_FIND_MODE=NORMAL
    export MIOPEN_DISABLE_CACHE=0
    
    # Memory
    export HIP_FORCE_COARSE_GRAIN=1
    export HSA_ENABLE_SDMA=0
    
    # Logging
    export MIOPEN_LOG_LEVEL=3
    export HIP_VISIBLE_DEVICES=0
fi
CONFIG

sudo chmod +x /etc/profile.d/rocm-rdna1-52.sh
```

### Step 6: Reboot

```bash
sudo reboot
```

### Step 7: Verify Installation

After reboot:

```bash
# Check ROCm version
rocminfo | grep "Name:" | head -1
/opt/rocm/bin/hipcc --version

# Check PyTorch
python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
