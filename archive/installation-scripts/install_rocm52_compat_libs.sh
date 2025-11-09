#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ROCm 5.2 Compatibility Layer for Ubuntu 24.04                ║"
echo "║   Install missing libraries from Ubuntu 22.04 repos            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check we're on Ubuntu 24.04
if ! grep -q "24.04" /etc/lsb-release 2>/dev/null; then
    echo "❌ This script is for Ubuntu 24.04 only"
    exit 1
fi

echo "✅ Ubuntu 24.04 detected"
echo ""

# Confirm with user
echo "⚠️  This script will:"
echo "   1. Add Ubuntu 22.04 (Jammy) package sources for compatibility libs"
echo "   2. Install libtinfo5, libncurses5 from Ubuntu 22.04"
echo "   3. Create python symlink (python → python3)"
echo "   4. Keep your Ubuntu 24.04 system otherwise intact"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

# Backup sources.list
echo "📍 Step 1: Backing up package sources..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
sudo cp /etc/apt/sources.list.d/ /etc/apt/sources.list.d.backup/ -r 2>/dev/null || true
echo "✅ Backup complete"
echo ""

# Add Ubuntu 22.04 sources for specific packages
echo "📍 Step 2: Adding Ubuntu 22.04 (Jammy) sources for compat libs..."
cat << 'SOURCES' | sudo tee /etc/apt/sources.list.d/jammy-compat.list
# Ubuntu 22.04 (Jammy) sources for ROCm 5.2 compatibility libraries only
# Pin priority set low to avoid upgrading system packages
deb http://archive.ubuntu.com/ubuntu/ jammy main universe
deb http://archive.ubuntu.com/ubuntu/ jammy-updates main universe
SOURCES

# Create apt pinning to prevent system packages from downgrading
echo "📍 Step 3: Creating apt pinning preferences..."
sudo mkdir -p /etc/apt/preferences.d/
cat << 'PINNING' | sudo tee /etc/apt/preferences.d/jammy-compat
# Prefer Ubuntu 24.04 (Noble) packages by default
Package: *
Pin: release n=noble
Pin-Priority: 990

# Allow only specific compat packages from Jammy (22.04)
Package: libtinfo5 libncurses5 libncurses5-dev libtinfo-dev
Pin: release n=jammy
Pin-Priority: 500

# Block everything else from Jammy
Package: *
Pin: release n=jammy
Pin-Priority: 100
PINNING

echo "✅ Pinning configured"
echo ""

# Update package lists
echo "📍 Step 4: Updating package lists..."
sudo apt update
echo "✅ Package lists updated"
echo ""

# Install compatibility libraries
echo "📍 Step 5: Installing compatibility libraries from Ubuntu 22.04..."
sudo apt install -y libtinfo5 libncurses5 libncurses5-dev libtinfo-dev
echo "✅ Compatibility libraries installed"
echo ""

# Create python symlink
echo "📍 Step 6: Creating python → python3 symlink..."
if [ ! -e /usr/bin/python ]; then
    sudo ln -s /usr/bin/python3 /usr/bin/python
    echo "✅ Symlink created: /usr/bin/python → /usr/bin/python3"
else
    echo "✅ Python symlink already exists"
fi
echo ""

# Verify installations
echo "📍 Step 7: Verifying installation..."
echo ""

echo "Checking libtinfo5:"
dpkg -l | grep libtinfo5 && echo "✅ libtinfo5 installed" || echo "❌ libtinfo5 missing"

echo "Checking libncurses5:"
dpkg -l | grep libncurses5 && echo "✅ libncurses5 installed" || echo "❌ libncurses5 missing"

echo "Checking python symlink:"
python --version && echo "✅ python command works" || echo "❌ python command missing"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   ✅ Compatibility Layer Installation Complete                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "You can now install ROCm 5.2 packages!"
echo ""
echo "Next steps:"
echo "  1. Add ROCm 5.2 repository"
echo "  2. Install ROCm 5.2 packages"
echo "  3. Reboot system"
echo ""
echo "Run: sudo ./install_rocm52_with_compat.sh"
echo ""
