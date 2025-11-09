#!/bin/bash
# Script to rebuild patched MIOpen for RDNA1

set -e

echo "=== MIOpen RDNA1 Patch Build Script ==="
echo ""

# Check ROCm version
echo "Checking ROCm version..."
ROCM_VERSION=$(/opt/rocm/bin/hipcc --version | grep "HIP version" | cut -d: -f2 | xargs)
echo "ROCm/HIP: $ROCM_VERSION"

# Verify source exists
if [ ! -d "/tmp/MIOpen" ]; then
    echo "Error: MIOpen source not found at /tmp/MIOpen"
    exit 1
fi

# Clean build directory
echo "Cleaning build directory..."
rm -rf /tmp/MIOpen/build_rdna1/*

# Configure
echo "Configuring MIOpen..."
cd /tmp/MIOpen/build_rdna1
cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm-miopen-rdna1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DMIOPEN_BACKEND=HIP \
  -DMIOPEN_USE_MLIR=OFF \
  -DMIOPEN_USE_HIPBLASLT=OFF \
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF \
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF \
  -DMIOPEN_USE_COMPOSABLEKERNEL=OFF \
  -DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__" \
  ..

# Build
echo "Building MIOpen (this will take 30-45 minutes)..."
make -j$(nproc)

# Install
echo "Installing to /opt/rocm-miopen-rdna1..."
sudo make install

# Replace PyTorch's library
echo "Backing up and replacing PyTorch's libMIOpen.so..."
TORCH_LIB="$HOME/.local/lib/python3.12/site-packages/torch/lib"
if [ -f "$TORCH_LIB/libMIOpen.so" ]; then
    if [ ! -f "$TORCH_LIB/libMIOpen.so.original" ]; then
        cp "$TORCH_LIB/libMIOpen.so" "$TORCH_LIB/libMIOpen.so.original"
    fi
    cp /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0 "$TORCH_LIB/libMIOpen.so"
    echo "✅ PyTorch library replaced"
else
    echo "⚠️  PyTorch library not found at $TORCH_LIB"
fi

echo ""
echo "=== Build Complete ==="
echo "Patched library: /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0"
echo "PyTorch library: $TORCH_LIB/libMIOpen.so"
