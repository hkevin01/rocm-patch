#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════════"
echo "  RDNA1 MTYPE_NC Kernel Parameter Fix"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This script forces the amdgpu kernel module to use MTYPE_NC"
echo "(non-coherent memory) instead of the default MTYPE_RW/MTYPE_CC"
echo "which causes crashes on RDNA1 GPUs."
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "✓ Running as root"
else
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

# Backup existing config if it exists
if [ -f /etc/modprobe.d/amdgpu-mtype.conf ]; then
    echo "→ Backing up existing config..."
    cp /etc/modprobe.d/amdgpu-mtype.conf /etc/modprobe.d/amdgpu-mtype.conf.backup.$(date +%Y%m%d_%H%M%S)
fi

# Create new modprobe configuration
echo "→ Creating /etc/modprobe.d/amdgpu-mtype.conf..."
cat > /etc/modprobe.d/amdgpu-mtype.conf << 'MODPROBE_EOF'
# RDNA1 Memory Type Fix
# Force MTYPE_NC (non-coherent) memory to prevent crashes
# MTYPE values: 0 = MTYPE_RW (default), 1 = MTYPE_NC, 2 = MTYPE_CC
options amdgpu mtype_local=1

# Keep existing RDNA1 compatibility parameters
options amdgpu noretry=0 vm_fragment_size=9
MODPROBE_EOF

echo "✓ Configuration created:"
cat /etc/modprobe.d/amdgpu-mtype.conf
echo ""

# Update initramfs to include the new parameters
echo "→ Updating initramfs..."
update-initramfs -u -k all

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Configuration Applied Successfully!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "IMPORTANT: You must REBOOT for changes to take effect!"
echo ""
echo "After reboot, verify with:"
echo "  cat /sys/module/amdgpu/parameters/mtype_local"
echo "  (should show: 1)"
echo ""
echo "Then test with:"
echo "  python3 tests/test_conv2d_minimal.py"
echo ""
echo "════════════════════════════════════════════════════════════════"
