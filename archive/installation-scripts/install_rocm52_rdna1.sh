#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ROCm 5.2 Installation for RDNA1 (gfx1010)                     ║"
echo "║   Fix for Conv2d hangs on large feature maps                    ║"
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

# Confirm migration
echo "⚠️  WARNING: This will:"
echo "   1. Remove ROCm 5.7"
echo "   2. Install ROCm 5.2"
echo "   3. Reinstall PyTorch 2.2.2 for ROCm 5.2"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Step 1: Backup current setup
echo "📍 Step 1: Backing up current configuration..."
sudo cp /etc/profile.d/rocm-rdna1-57.sh ~/rocm-57-backup.sh 2>/dev/null || true
sudo chown $USER:$USER ~/rocm-57-backup.sh 2>/dev/null || true
env | grep -E "ROCM|MIOPEN|HIP|HSA" > ~/env-backup.txt 2>/dev/null || true
python3 -c "import torch; print(torch.__version__)" > ~/pytorch-version.txt 2>/dev/null || true
echo "✅ Backup complete"
echo ""

# Step 2: Remove ROCm 5.7
echo "�� Step 2: Removing ROCm 5.7..."
sudo apt remove --purge -y rocm* hip* miopen* rocblas* rocfft* rocrand* \
    rocsolver* hipblas* hipfft* hipsparse* rccl* 2>/dev/null || true
sudo apt autoremove -y
sudo apt autoclean
sudo rm -f /etc/profile.d/rocm-rdna1-57.sh
sudo rm -f /etc/apt/sources.list.d/rocm.list
echo "✅ ROCm 5.7 removed"
echo ""

# Step 3: Add ROCm 5.2 repository
echo "📍 Step 3: Adding ROCm 5.2 repository..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
echo "✅ Repository added"
echo ""

# Step 4: Install ROCm 5.2
echo "📍 Step 4: Installing ROCm 5.2 (this may take 5-10 minutes)..."
sudo apt install -y \
    rocm-dev \
    rocm-libs \
    rocminfo \
    rocm-smi \
    hip-base \
    hip-runtime-amd \
    hip-dev \
    miopen-hip \
    rocblas

# Add user to groups
sudo usermod -a -G render,video $USER
echo "✅ ROCm 5.2 installed"
echo ""

# Step 5: Create configuration file
echo "📍 Step 5: Creating configuration file..."
sudo tee /etc/profile.d/rocm-rdna1-52.sh > /dev/null << 'PROFILE'
#!/bin/bash
# ROCm 5.2 Configuration for RDNA1 (gfx1010)

if lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    # ROCm 5.2 paths
    export ROCM_PATH=/opt/rocm-5.2.0
    export PATH=$PATH:/opt/rocm-5.2.0/bin:/opt/rocm-5.2.0/hip/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-5.2.0/lib:/opt/rocm-5.2.0/hip/lib

    # Architecture spoofing
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030

    # Force GEMM algorithms only
    export MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    export MIOPEN_DEBUG_CONV_WINOGRAD=0
    export MIOPEN_DEBUG_CONV_DIRECT=0
    export MIOPEN_DEBUG_CONV_FFT=0

    # MIOpen configuration
    export MIOPEN_FIND_MODE=NORMAL
    export MIOPEN_DISABLE_CACHE=0

    # Memory configuration
    export HIP_FORCE_COARSE_GRAIN=1
    export HSA_ENABLE_SDMA=0
    export HSA_USE_SVM=0
    export HSA_XNACK=0

    # Logging
    export MIOPEN_LOG_LEVEL=3
    export HIP_VISIBLE_DEVICES=0
fi
PROFILE

sudo chmod +x /etc/profile.d/rocm-rdna1-52.sh
source /etc/profile.d/rocm-rdna1-52.sh
echo "✅ Configuration created"
echo ""

# Step 6: Install PyTorch
echo "📍 Step 6: Installing PyTorch 2.2.2 for ROCm 5.2..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2
echo "✅ PyTorch installed"
echo ""

# Step 7: Verify installation
echo "📍 Step 7: Verifying installation..."
python3 << 'PYCHECK'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
PYCHECK

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ✅ ROCm 5.2 INSTALLATION COMPLETE                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔄 IMPORTANT: You MUST reboot for changes to take effect:"
echo "   sudo reboot"
echo ""
echo "🧪 After reboot, test with:"
echo "   python3 test_conv2d_large.py"
echo ""
echo "📖 See MIGRATE_TO_ROCM52.md for full testing procedure"
echo ""
