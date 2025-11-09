#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ROCm 5.7 + PyTorch 2.2.2 RDNA1 Installer                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for RDNA1 GPU
if ! lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    echo "❌ Error: No RDNA1 GPU detected!"
    echo "   Supported: RX 5600 XT, RX 5700 XT"
    exit 1
fi

echo "✅ RDNA1 GPU detected"
echo ""

# Step 1: Remove conflicting environment files
echo "📍 Step 1: Cleaning up conflicting configurations..."
sudo rm -f /etc/profile.d/rocm-rdna-fix.sh
sudo rm -f /etc/environment.d/90-rocm*.conf
sudo rm -f /etc/systemd/system.conf.d/rocm*.conf

# Step 2: Create profile.d script
echo "📍 Step 2: Creating /etc/profile.d/rocm-rdna1-57.sh..."
sudo tee /etc/profile.d/rocm-rdna1-57.sh > /dev/null << 'PROFILE'
#!/bin/bash
# ROCm 5.7 + PyTorch 2.2.2 RDNA1 Configuration

if lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    # Get gfx1030 kernels
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1030

    # Force GEMM algorithms (NO Winograd/Direct/ImplicitGEMM)
    export MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    export MIOPEN_DEBUG_CONV_WINOGRAD=0
    export MIOPEN_DEBUG_CONV_DIRECT=0
    export MIOPEN_DEBUG_CONV_FFT=0

    # Normal find mode (no FIND_ENFORCE to avoid 30+ min hangs)
    export MIOPEN_FIND_MODE=NORMAL
    export MIOPEN_DISABLE_CACHE=0

    # Coarse-grained memory only
    export HIP_FORCE_COARSE_GRAIN=1
    export HSA_ENABLE_SDMA=0
    export HSA_USE_SVM=0
    export HSA_XNACK=0

    # Debugging (set to 3 for less verbose)
    export MIOPEN_LOG_LEVEL=3
    export HIP_VISIBLE_DEVICES=0
fi
PROFILE

sudo chmod +x /etc/profile.d/rocm-rdna1-57.sh
echo "✅ Created /etc/profile.d/rocm-rdna1-57.sh"

# Step 3: Source it now
echo "📍 Step 3: Loading configuration for current session..."
source /etc/profile.d/rocm-rdna1-57.sh

# Step 4: Verify PyTorch version
echo "📍 Step 4: Checking PyTorch + ROCm versions..."
python3 << 'PYCHECK'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
PYCHECK

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ✅ INSTALLATION COMPLETE                                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔄 IMPORTANT: Restart your terminal or source the config:"
echo "   source /etc/profile.d/rocm-rdna1-57.sh"
echo ""
echo "🧪 Test with:"
echo "   python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "📊 Expected versions:"
echo "   ROCm: 5.7"
echo "   PyTorch: 2.2.2+rocm5.7"
echo ""
