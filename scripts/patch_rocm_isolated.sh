#!/bin/bash
################################################################################
# ROCm Isolated Source-Level Patcher
# Builds ROCm 6.2.x with LLVM 16 in isolated environment
################################################################################

set -e

# Configuration
WORKSPACE=~/rocm-source-patches
INSTALL_PREFIX=/opt/rocm-6.2-patched
BUILD_TYPE=Release
ROCM_VERSION="rocm-6.2.x"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check not root
if [ "$EUID" -eq 0 ]; then
    log_error "Do not run as root. Use sudo only when prompted."
    exit 1
fi

log_info "========================================="
log_info "ROCm 6.2.x Isolated Build for RDNA1/2"
log_info "========================================="
log_info ""
log_info "This will:"
log_info "  1. Use system LLVM 16 (not ROCm 7.0's LLVM 20)"
log_info "  2. Build ROCm 6.2.x from source"
log_info "  3. Install to $INSTALL_PREFIX"
log_info "  4. Keep ROCm 7.0.2 intact"
log_info ""
log_info "Time: 2-3 hours"
log_info "Disk: ~10GB"
log_info ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Temporarily unset ROCm environment to avoid conflicts
log_info "Isolating build environment from ROCm 7.0.2..."
unset ROCM_PATH
unset LD_LIBRARY_PATH
unset PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Force use of system LLVM 16
export CC=/usr/bin/clang-16
export CXX=/usr/bin/clang++-16
export LLVM_DIR=/usr/lib/llvm-16
export CMAKE_PREFIX_PATH=/usr/lib/llvm-16

log_info "Using LLVM 16:"
$CC --version | head -1
$CXX --version | head -1

mkdir -p $WORKSPACE
cd $WORKSPACE

# Clone sources if needed
if [ ! -d "ROCT-Thunk-Interface" ]; then
    log_info "Cloning ROCT-Thunk-Interface..."
    git clone --depth 1 -b $ROCM_VERSION https://github.com/ROCm/ROCT-Thunk-Interface.git
fi

if [ ! -d "ROCR-Runtime" ]; then
    log_info "Cloning ROCR-Runtime..."
    git clone --depth 1 -b $ROCM_VERSION https://github.com/ROCm/ROCR-Runtime.git
fi

# Build ROCT
log_info "Building ROCT..."
cd $WORKSPACE/ROCT-Thunk-Interface
rm -rf build
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      ..

make -j$(nproc) 2>&1 | tee build.log
sudo make install
log_info "✅ ROCT installed"

# Build ROCR with minimal dependencies
log_info "Building ROCR (minimal build, no bitcode)..."
cd $WORKSPACE/ROCR-Runtime/src
rm -rf build
mkdir build && cd build

# Configure to skip problematic bitcode builds
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DIMAGE_SUPPORT=OFF \
      -DBUILD_SHARED_LIBS=ON \
      ..

make -j$(nproc) 2>&1 | tee build.log
sudo make install
log_info "✅ ROCR installed"

log_info ""
log_info "========================================="
log_info "✅ Build Complete!"
log_info "========================================="
log_info ""
log_info "Installed to: $INSTALL_PREFIX"
log_info ""
log_info "To use:"
log_info "  export ROCM_PATH=$INSTALL_PREFIX"
log_info "  export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
log_info ""
log_info "Note: This is a minimal ROCm runtime."
log_info "HIP/MIOpen still use system ROCm 7.0.2."
log_info "The real fix requires patching MIOpen convolution kernels."
