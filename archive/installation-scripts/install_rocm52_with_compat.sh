#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ROCm 5.2 Installation with Compatibility Layer                ║"
echo "║   For Ubuntu 24.04 + RDNA1 (gfx1010)                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for RDNA1 GPU
if ! lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    echo "❌ Error: No RDNA1 GPU detected!"
    echo "   Supported: RX 5600 XT, RX 5700 XT, RX 5700"
    exit 1
fi

echo "✅ RDNA1 GPU detected"
lspci -nn | grep -iE "1002:731f|1002:731e|1002:7310|1002:7312"
echo ""

# Check if compatibility libs are installed
echo "📍 Checking for compatibility libraries..."
if ! dpkg -l | grep -q libtinfo5; then
    echo "❌ libtinfo5 not found"
    echo ""
    echo "You need to install compatibility libraries first!"
    echo "Run: sudo ./install_rocm52_compat_libs.sh"
    echo ""
    exit 1
fi

if ! dpkg -l | grep -q libncurses5; then
    echo "❌ libncurses5 not found"
    echo ""
    echo "You need to install compatibility libraries first!"
    echo "Run: sudo ./install_rocm52_compat_libs.sh"
    echo ""
    exit 1
fi

echo "✅ Compatibility libraries found"
echo ""

# Confirm installation
echo "⚠️  This will:"
echo "   1. Remove any existing ROCm installation"
echo "   2. Install ROCm 5.2 for gfx1010"
echo "   3. Configure environment for RDNA1"
echo "   4. Install PyTorch 2.2.2+rocm5.2"
echo "   5. Require system reboot"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Step 1: Backup current setup
echo "📍 Step 1: Backing up current configuration..."
mkdir -p ~/rocm-backups
sudo cp /etc/profile.d/rocm-rdna1-*.sh ~/rocm-backups/ 2>/dev/null || true
env | grep -E "ROCM|MIOPEN|HIP|HSA" > ~/rocm-backups/env-backup.txt 2>/dev/null || true
python3 -c "import torch; print(torch.__version__)" > ~/rocm-backups/pytorch-version.txt 2>/dev/null || true
echo "✅ Backup complete (saved to ~/rocm-backups/)"
echo ""

# Step 2: Remove existing ROCm
echo "📍 Step 2: Removing existing ROCm installation..."
sudo apt remove --purge -y rocm* hip* miopen* rocblas* rocfft* rocrand* \
    rocsolver* hipblas* hipfft* hipsparse* rccl* 2>/dev/null || true
sudo apt autoremove -y
sudo apt autoclean
sudo rm -f /etc/profile.d/rocm-rdna1-*.sh
sudo rm -f /etc/apt/sources.list.d/rocm.list
echo "✅ Existing ROCm removed"
echo ""

# Step 3: Add ROCm 5.2 repository
echo "📍 Step 3: Adding ROCm 5.2 repository..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
echo "✅ ROCm 5.2 repository added"
echo ""

# Step 4: Install ROCm 5.2
echo "📍 Step 4: Installing ROCm 5.2 (this may take 10-15 minutes)..."
echo "   Installing core packages..."
sudo apt install -y \
    rocm-dev \
    rocm-libs \
    rocminfo \
    rocm-smi \
    hip-base \
    hip-runtime-amd \
    hip-dev \
    miopen-hip \
    rocblas \
    2>&1 | tee /tmp/rocm52-install.log

# Add user to groups
sudo usermod -a -G render,video $USER

echo "✅ ROCm 5.2 installed"
echo ""

# Step 5: Create environment configuration
echo "📍 Step 5: Creating RDNA1 environment configuration..."
sudo tee /etc/profile.d/rocm-rdna1-52.sh << 'ENVCONFIG'
# ROCm 5.2 Environment for RDNA1 (gfx1010)
# AMD Radeon RX 5600 XT / RX 5700 XT / RX 5700

# ROCm paths
export ROCM_PATH=/opt/rocm-5.2.0
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH}

# RDNA1 architecture override (report as gfx1030 for better compatibility)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030

# MIOpen: Force GEMM-only convolutions (avoid problematic algorithms)
export MIOPEN_DEBUG_CONV_GEMM=1
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_FFT=0

# MIOpen: Use NORMAL find mode (not exhaustive)
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_LOG_LEVEL=3

# HIP: Force coarse-grained memory (RDNA1 requirement)
export HIP_FORCE_COARSE_GRAIN=1

# HSA: Disable features not supported by RDNA1
export HSA_ENABLE_SDMA=0
export HSA_USE_SVM=0
export HSA_XNACK=0

# Unset problematic variables
unset HSA_FORCE_FINE_GRAIN_PCIE
unset MIOPEN_FIND_ENFORCE
unset MIOPEN_DEBUG_FIND_ONLY_SOLVER

# Known limitation: Conv2d operations limited to ≤42x42 feature maps
# See FINAL_FINDINGS.md for details
ENVCONFIG

sudo chmod +x /etc/profile.d/rocm-rdna1-52.sh
echo "✅ Environment configuration created"
echo ""

# Step 6: Install PyTorch
echo "📍 Step 6: Installing PyTorch 2.2.2 for ROCm 5.2..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.2
echo "✅ PyTorch installed"
echo ""

# Step 7: Verify installation
echo "📍 Step 7: Verifying installation..."
source /etc/profile.d/rocm-rdna1-52.sh

echo "ROCm version:"
rocminfo | grep "Name:" | head -2

echo ""
echo "GPU detected:"
rocm-smi | head -10

echo ""
echo "PyTorch version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 || echo "⚠️  PyTorch verification will work after reboot"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ✅ ROCm 5.2 Installation Complete!                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  IMPORTANT: You MUST reboot your system now!"
echo ""
echo "After reboot:"
echo "  1. Open a new terminal"
echo "  2. Run: source /etc/profile.d/rocm-rdna1-52.sh"
echo "  3. Test: python3 test_conv2d_subprocess.py"
echo ""
echo "📝 Installation log saved to: /tmp/rocm52-install.log"
echo ""
read -p "Reboot now? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Rebooting in 5 seconds..."
    sleep 5
    sudo reboot
else
    echo "Remember to reboot before using ROCm!"
fi
