#!/bin/bash
set -e

# Install Patched ROCr-Runtime Script

INSTALL_PREFIX="/opt/rocm-rdna1-rocr"
ROCM_LIB="/opt/rocm/lib"
LIBRARY_NAME="libhsa-runtime64.so.1"

echo "========================================="
echo "ROCr-Runtime Installation Script"
echo "========================================="
echo ""

# Check if patched library exists
if [ ! -f "$INSTALL_PREFIX/lib/$LIBRARY_NAME" ]; then
    echo "❌ ERROR: Patched library not found at $INSTALL_PREFIX/lib/$LIBRARY_NAME"
    echo "Please run: ./scripts/build_patched_rocr.sh first"
    exit 1
fi

echo "📦 Patched library: $INSTALL_PREFIX/lib/$LIBRARY_NAME"
echo "🎯 Target location: $ROCM_LIB/$LIBRARY_NAME"
echo ""

# Backup original if not already backed up
if [ ! -f "$ROCM_LIB/${LIBRARY_NAME}.backup" ]; then
    echo "💾 Backing up original library..."
    sudo cp "$ROCM_LIB/$LIBRARY_NAME" "$ROCM_LIB/${LIBRARY_NAME}.backup"
    echo "   ✅ Backup created: $ROCM_LIB/${LIBRARY_NAME}.backup"
else
    echo "ℹ️  Backup already exists: $ROCM_LIB/${LIBRARY_NAME}.backup"
fi

echo ""
echo "📝 Installing patched library..."
sudo cp "$INSTALL_PREFIX/lib/$LIBRARY_NAME" "$ROCM_LIB/$LIBRARY_NAME"

echo ""
echo "========================================="
echo "✅ Installation complete!"
echo "========================================="
echo ""
echo "The patched ROCr-Runtime library has been installed."
echo ""
echo "To test, run:"
echo "  export HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "  python3 -c 'import torch; x=torch.randn(1,3,32,32).cuda(); print(torch.nn.Conv2d(3,16,3,padding=1).cuda()(x).shape)'"
echo ""
echo "To restore the original library:"
echo "  sudo cp $ROCM_LIB/${LIBRARY_NAME}.backup $ROCM_LIB/$LIBRARY_NAME"
echo ""
