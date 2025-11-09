#!/bin/bash
# ROCm 5.2 Installation Script for RDNA1 (gfx1010)
# Based on community research and luaartist/Rocm_Project
# Addresses ROCm 5.3+ regression that causes Conv2d hangs

set -e  # Exit on error

echo "============================================"
echo "ROCm 5.2 Installation for RDNA1 (gfx1010)"
echo "============================================"
echo ""
echo "This will:"
echo "  1. Backup current ROCm 5.7 configuration"
echo "  2. Remove ROCm 5.7 completely"
echo "  3. Install ROCm 5.2 (last version before regression)"
echo "  4. Install PyTorch 2.0+rocm5.2"
echo "  5. Configure environment for RDNA1"
echo ""
read -p "Proceed with installation? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Installation cancelled."
    exit 0
fi

echo ""
echo "=== Phase 1: Backup Current Configuration ==="
echo ""

# Backup current environment
if [ -f /etc/profile.d/rocm-rdna1-57.sh ]; then
    echo "Backing up ROCm 5.7 configuration..."
    sudo cp /etc/profile.d/rocm-rdna1-57.sh ~/rocm-rdna1-57.sh.backup
    echo "✅ Configuration backed up to ~/rocm-rdna1-57.sh.backup"
fi

# Document current PyTorch version
if command -v pip3 &> /dev/null; then
    echo "Documenting current PyTorch version..."
    pip3 freeze | grep torch > ~/pytorch-57-versions.txt || true
    echo "✅ PyTorch versions saved to ~/pytorch-57-versions.txt"
fi

# Save MIOpen cache
if [ -d ~/.cache/miopen ]; then
    echo "Backing up MIOpen cache..."
    tar -czf ~/miopen-cache-backup.tar.gz ~/.cache/miopen/ 2>/dev/null || true
    echo "✅ MIOpen cache backed up to ~/miopen-cache-backup.tar.gz"
fi

echo ""
echo "=== Phase 2: Remove ROCm 5.7 ==="
echo ""

echo "Removing ROCm 5.7 packages..."
sudo apt remove --purge -y rocm* hip* miopen* rocblas* rocsparse* rocfft* \
  rocrand* rocsolver* roctx* roctracer* rocprofiler* hsakmt-roct* hsa-rocr* \
  2>/dev/null || true

echo "Cleaning up residual files..."
sudo apt autoremove -y
sudo rm -rf /opt/rocm* 2>/dev/null || true
sudo rm -rf /etc/ld.so.conf.d/rocm* 2>/dev/null || true
sudo rm -rf /etc/profile.d/rocm-rdna1-57.sh 2>/dev/null || true
sudo ldconfig

echo "✅ ROCm 5.7 removed"

echo ""
echo "=== Phase 3: Install ROCm 5.2 ==="
echo ""

# Remove old ROCm repository
sudo rm -f /etc/apt/sources.list.d/rocm.list

# Add ROCm 5.2 repository
echo "Adding ROCm 5.2 repository..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
  sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
echo "Updating package list..."
sudo apt update

# Install ROCm 5.2
echo "Installing ROCm 5.2 core packages..."
sudo apt install -y \
  rocm-dev \
  rocm-libs \
  rocminfo \
  rocm-smi \
  hip-base \
  hip-runtime-amd \
  miopen-hip \
  rocblas

echo "✅ ROCm 5.2 installed"

# Verify installation
echo ""
echo "Verifying ROCm installation..."
if /opt/rocm/bin/rocminfo | grep -q gfx1010; then
    echo "✅ gfx1010 detected by ROCm"
else
    echo "⚠️ WARNING: gfx1010 not detected. Check GPU connection."
fi

echo ""
echo "=== Phase 4: Install PyTorch 2.0+rocm5.2 ==="
echo ""

# Uninstall current PyTorch
echo "Removing current PyTorch..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch 2.0 with ROCm 5.2
echo "Installing PyTorch 2.0.0+rocm5.2..."
pip3 install \
  https://download.pytorch.org/whl/nightly/rocm5.2/torch-2.0.0.dev20230209%2Brocm5.2-cp310-cp310-linux_x86_64.whl \
  https://download.pytorch.org/whl/nightly/rocm5.2/torchvision-0.15.0.dev20230209%2Brocm5.2-cp310-cp310-linux_x86_64.whl

echo "✅ PyTorch 2.0+rocm5.2 installed"

echo ""
echo "=== Phase 5: Configure Environment ==="
echo ""

# Create new environment configuration
echo "Creating environment configuration..."
sudo tee /etc/profile.d/rocm-rdna1-52.sh > /dev/null << 'ENVEOF'
#!/bin/bash
# ROCm 5.2 Environment for RDNA1 (gfx1010) GPUs
# Auto-loaded on login for all users

# Spoof gfx1030 (RDNA2) to enable ROCm support on gfx1010
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# ROCm paths
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Force coarse-grained memory (RDNA1 limitation)
export HIP_FORCE_COARSE_GRAIN=1

# MIOpen configuration (GEMM-only, no hangs)
export MIOPEN_DEBUG_CONV_GEMM=1
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_FIND_ENFORCE=NONE
export MIOPEN_USER_DB_PATH=/tmp/miopen-cache-$USER

# HIP runtime
export HIP_VISIBLE_DEVICES=0
export GPU_DEVICE_ORDINAL=0

# Disable kernel caching issues
export HSA_ENABLE_SDMA=0

# Create MIOpen cache directory
mkdir -p /tmp/miopen-cache-$USER
ENVEOF

sudo chmod +x /etc/profile.d/rocm-rdna1-52.sh
echo "✅ Environment configured in /etc/profile.d/rocm-rdna1-52.sh"

# Load environment for current session
source /etc/profile.d/rocm-rdna1-52.sh

# Add user to video/render groups
sudo usermod -aG video,render $USER

echo ""
echo "=== Phase 6: Test Installation ==="
echo ""

echo "Testing ROCm 5.2 installation..."

# Test 1: ROCm info
echo ""
echo "Test 1: ROCm info"
/opt/rocm/bin/rocminfo | grep "Name:" | head -3

# Test 2: PyTorch CUDA availability
echo ""
echo "Test 2: PyTorch CUDA availability"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Test 3: Small Conv2d (should work)
echo ""
echo "Test 3: Small Conv2d (32x32, 3→16 channels)"
python3 << 'PYEOF'
import torch
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
print(f"✅ Small Conv2d works! Output: {y.shape}")
PYEOF

# Test 4: Medium Conv2d (THIS HANGS IN ROCm 5.7, should work in 5.2)
echo ""
echo "Test 4: Medium Conv2d (64x64, 16→32 channels)"
echo "⚠️ This hangs in ROCm 5.7, let's see if ROCm 5.2 fixes it..."
timeout 30s python3 << 'PYEOF' || echo "❌ Test timed out or failed"
import torch
print("Starting test...")
x = torch.randn(1, 16, 64, 64).cuda()
conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
y = conv(x)
print(f"✅ SUCCESS! Medium Conv2d works! Output: {y.shape}")
print("🎉 ROCm 5.2 fixed the hang bug!")
PYEOF

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  - ROCm 5.2 installed (/opt/rocm)"
echo "  - PyTorch 2.0.0+rocm5.2 installed"
echo "  - Environment configured for RDNA1"
echo "  - Backup saved to ~/"
echo ""
echo "⚠️ IMPORTANT: You must log out and log back in for group changes to take effect."
echo ""
echo "To verify everything works:"
echo "  1. Log out and log back in"
echo "  2. Run: python3 test_conv2d_timing.py"
echo "  3. Test large feature maps that previously hung"
echo ""
echo "If you encounter issues, you can restore ROCm 5.7:"
echo "  1. sudo cp ~/rocm-rdna1-57.sh.backup /etc/profile.d/rocm-rdna1-57.sh"
echo "  2. Install ROCm 5.7 repository"
echo "  3. Run: ./install_rocm57.sh"
echo ""
