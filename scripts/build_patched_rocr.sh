#!/bin/bash
set -e  # Exit on error

# ROCr-Runtime Patch and Build Script
# Fixes RDNA1 (gfx1010) Conv2d hangs when spoofed as gfx1030

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PATCHES_DIR="$PROJECT_ROOT/patches"
BUILD_DIR="/tmp/ROCR-Runtime"
INSTALL_PREFIX="/opt/rocm-rdna1-rocr"

echo "========================================="
echo "ROCr-Runtime RDNA1 Patch Build Script"
echo "========================================="
echo ""

# Check if we're in the right place
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ ERROR: ROCr-Runtime source not found at $BUILD_DIR"
    echo "Please run: cd /tmp && git clone --branch rocm-6.2.4 https://github.com/ROCm/ROCR-Runtime.git"
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    echo "❌ ERROR: Patches directory not found at $PATCHES_DIR"
    exit 1
fi

cd "$BUILD_DIR"
echo "📂 Working directory: $BUILD_DIR"
echo ""

# Check if patches are already applied
if git log --oneline -1 | grep -q "RDNA1"; then
    echo "⚠️  Patches appear to be already applied. Skipping patch application."
else
    echo "📝 Applying patches..."
    
    # Apply patches in order
    for patch in "$PATCHES_DIR"/000*.patch; do
        if [ -f "$patch" ]; then
            echo "  → Applying $(basename "$patch")..."
            # Use -p1 to strip the first directory level (standard for git patches)
            patch -p1 < "$patch" || {
                echo "❌ Failed to apply patch: $(basename "$patch")"
                echo "You may need to manually resolve conflicts or reset the repository."
                exit 1
            }
        fi
    done
    
    echo "✅ All patches applied successfully!"
fi

echo ""
echo "🔨 Building ROCr-Runtime..."
echo ""

# Create build directory
rm -rf build
mkdir -p build
cd build

# Configure with CMake
echo "⚙️  Configuring build..."
cmake \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    ..

# Build
echo ""
echo "🔧 Compiling (this may take a few minutes)..."
make -j$(nproc)

echo ""
echo "📦 Installing to $INSTALL_PREFIX..."
sudo make install

echo ""
echo "========================================="
echo "✅ Build completed successfully!"
echo "========================================="
echo ""
echo "Patched library location:"
echo "  $INSTALL_PREFIX/lib/libhsa-runtime64.so.1"
echo ""
echo "To install the patched library, run:"
echo "  sudo ./scripts/install_patched_rocr.sh"
echo ""
echo "Or manually:"
echo "  sudo cp /opt/rocm/lib/libhsa-runtime64.so.1 /opt/rocm/lib/libhsa-runtime64.so.1.backup"
echo "  sudo cp $INSTALL_PREFIX/lib/libhsa-runtime64.so.1 /opt/rocm/lib/libhsa-runtime64.so.1"
echo ""
