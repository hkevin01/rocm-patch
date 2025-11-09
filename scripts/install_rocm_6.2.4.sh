#!/bin/bash
# Install ROCm 6.2.4 to match PyTorch 2.5.1+rocm6.2

set -e

echo "=== ROCm 6.2.4 Installation Script ==="
echo ""
echo "⚠️  This will remove existing ROCm and install 6.2.4"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Remove existing ROCm
echo "Removing existing ROCm installations..."
sudo apt-get remove --purge -y 'rocm-*' 'hip-*' 'miopen-*' || true
sudo apt-get autoremove -y

# Add ROCm repository
echo "Adding ROCm 6.2.4 repository..."
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 noble main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

echo 'Package: *' | sudo tee /etc/apt/preferences.d/rocm-pin-600
echo 'Pin: release o=repo.radeon.com' | sudo tee -a /etc/apt/preferences.d/rocm-pin-600
echo 'Pin-Priority: 600' | sudo tee -a /etc/apt/preferences.d/rocm-pin-600

# Update and install
echo "Installing ROCm 6.2.4..."
sudo apt-get update
sudo apt-get install -y rocm-hip-sdk rocm-dev miopen-hip miopen-hip-dev

# Add user to render group
echo "Adding user to render and video groups..."
sudo usermod -a -G render,video $USER

echo ""
echo "=== Installation Complete ==="
echo "ROCm 6.2.4 installed to /opt/rocm"
echo ""
echo "⚠️  You may need to log out and back in for group membership"
echo ""
echo "Verify with:"
echo "  /opt/rocm/bin/hipcc --version"
echo "  /opt/rocm/bin/rocminfo"
