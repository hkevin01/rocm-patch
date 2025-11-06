#!/bin/bash
# ROCm RDNA1/2 Memory Coherency Fix - Kernel Parameters
# Addresses: Memory access fault by GPU (Page not present or supervisor privilege)
# Target: AMD RX 5600 XT (gfx1010), RX 6000 series (gfx1030)
# ROCm versions: 6.2+

set -e

echo "=== ROCm RDNA1/2 Memory Coherency Kernel Fix ==="
echo "Target: AMD Radeon RX 5600 XT / RDNA1 GPUs"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "ERROR: This script must be run as root (sudo)"
    exit 1
fi

# Backup existing modprobe configuration
MODPROBE_CONF="/etc/modprobe.d/amdgpu-fix.conf"
if [ -f "$MODPROBE_CONF" ]; then
    cp "$MODPROBE_CONF" "${MODPROBE_CONF}.backup.$(date +%s)"
    echo "Backed up existing config: ${MODPROBE_CONF}.backup.*"
fi

# Create optimized amdgpu kernel parameters
cat > "$MODPROBE_CONF" << 'MODPROBE_EOF'
# ROCm RDNA1/2 Memory Coherency Fix
# Generated: $(date)
# Issue: Memory access fault - Page not present or supervisor privilege
# Solution: Disable coherent memory features incompatible with RDNA1/2

# Core fix: Disable system memory coherency for RDNA1/2
options amdgpu noretry=0
options amdgpu vm_fragment_size=9
options amdgpu vm_update_mode=0

# Memory allocation fixes
options amdgpu gtt_size=8192
options amdgpu moverate=64

# Disable problematic features for RDNA1/2
options amdgpu mes=0
options amdgpu tmz=0

# Power and clocking (helps stability)
options amdgpu ppfeaturemask=0xffffffff
options amdgpu gpu_recovery=1

# Debug (remove after confirming fix works)
# options amdgpu debug_mask=0x600
MODPROBE_EOF

echo "✓ Created $MODPROBE_CONF"

# Update initramfs to apply on boot
echo ""
echo "Updating initramfs..."
update-initramfs -u -k all 2>&1 | tail -5

echo ""
echo "=== Kernel parameters configured ==="
echo ""
echo "Next steps:"
echo "1. Reboot system: sudo reboot"
echo "2. After reboot, verify with: cat /sys/module/amdgpu/parameters/noretry"
echo "   (should show: 0)"
echo "3. Run training test"
echo ""
echo "OR for immediate testing WITHOUT reboot:"
echo "  sudo rmmod amdgpu && sudo modprobe amdgpu noretry=0 vm_fragment_size=9"
echo ""
