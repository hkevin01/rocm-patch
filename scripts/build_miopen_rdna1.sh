#!/bin/bash
set -e

echo "======================================================================"
echo "Building MIOpen with RDNA1 Non-Coherent Memory Patch"
echo "======================================================================"

MIOPEN_SRC="/tmp/MIOpen"
BUILD_DIR="$MIOPEN_SRC/build_rdna1"
INSTALL_PREFIX="/opt/rocm-miopen-rdna1"

# Check if source exists
if [ ! -d "$MIOPEN_SRC" ]; then
    echo "✗ MIOpen source not found at $MIOPEN_SRC"
    echo "Run git clone first!"
    exit 1
fi

# Check if patch was applied
if ! grep -q "RDNA1 (gfx1010) Fix" "$MIOPEN_SRC/src/hip/handlehip.cpp"; then
    echo "✗ Patch not applied to MIOpen source!"
    echo "Please apply the patch first"
    exit 1
fi

echo "✓ RDNA1 patch found in source"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "======================================================================"
echo "Configuring CMake..."
echo "======================================================================"

# Configure with CMake
export CXX=/opt/rocm/llvm/bin/clang++
cmake \
    -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DMIOPEN_BACKEND=HIP \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DMIOPEN_GPU_SYNC=Off \
    -DMIOPEN_USE_MLIR=Off \
    -DMIOPEN_USE_COMPOSABLEKERNEL=Off \
    -DMIOPEN_ENABLE_AI_KERNEL_TUNING=Off \
    -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=Off \
    ..

if [ $? -ne 0 ]; then
    echo "✗ CMake configuration failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Building MIOpen (this will take 30-60 minutes)..."
echo "======================================================================"

# Build with all cores
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "✗ Build failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ MIOpen built successfully!"
echo "======================================================================"
echo ""
echo "To install:"
echo "  sudo make install"
echo ""
echo "Or to install to custom location:"
echo "  make install DESTDIR=/tmp/miopen-install"
echo ""
echo "To use this MIOpen:"
echo "  export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
echo ""
echo "Build location: $BUILD_DIR"
echo "======================================================================"

