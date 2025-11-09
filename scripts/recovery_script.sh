#!/bin/bash
# Recovery Script - Remove Kernel Patch and Restore Stability

set -e

echo "🔧 ROCm RDNA1 Kernel Patch Recovery"
echo "===================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run with sudo"
    exit 1
fi

echo "📍 Step 1: Backup patched version"
cd /usr/src/amdgpu-6.16.6-2238411.24.04/amd/amdkfd
if [ ! -f kfd_crat.c.patched ]; then
    cp kfd_crat.c kfd_crat.c.patched
    echo "✅ Backed up patched version to kfd_crat.c.patched"
else
    echo "ℹ️  Backup already exists"
fi

echo ""
echo "📍 Step 2: Remove patch from kfd_crat.c"
# Remove lines 1122-1130 (the RDNA1 fix block)
sed -i '1122,1130d' kfd_crat.c
echo "✅ Removed lines 1122-1130 from kfd_crat.c"

echo ""
echo "📍 Step 3: Verify patch removed"
if grep -q "RDNA1 FIX" kfd_crat.c; then
    echo "❌ ERROR: Patch still present in file!"
    exit 1
else
    echo "✅ Patch successfully removed"
fi

echo ""
echo "📍 Step 4: Remove patched modules"
dkms remove -m amdgpu -v 6.16.6-2238411.24.04 --all || true
echo "✅ Removed patched modules"

echo ""
echo "📍 Step 5: Rebuild clean module"
dkms build -m amdgpu -v 6.16.6-2238411.24.04
echo "✅ Built clean module"

echo ""
echo "📍 Step 6: Install clean module"
dkms install -m amdgpu -v 6.16.6-2238411.24.04
echo "✅ Installed clean module"

echo ""
echo "=========================================="
echo "✅ Recovery Complete!"
echo ""
echo "Next steps:"
echo "1. Run: sudo reboot"
echo "2. After reboot, try Option 3 (environment tuning)"
echo "3. See ENVIRONMENT_TUNING.md for details"
echo ""
echo "⚠️  Do NOT try kernel/runtime patches again!"
echo "=========================================="

