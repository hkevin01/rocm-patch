#!/bin/bash
################################################################################
# RMCP Environment Patcher
# Creates environment configuration to fix RDNA1/2 memory issues
# This is a simpler alternative to full source patching
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

echo "================================================================================"
echo "   RMCP Environment Configuration"
echo "   Quick fix for RDNA1/2 Memory Issues"
echo "================================================================================"
echo ""

# Check if ROCm is installed
if [ ! -d "/opt/rocm" ]; then
    log_error "ROCm not found at /opt/rocm"
    exit 1
fi

log_info "Found ROCm at /opt/rocm"

# Create environment configuration
log_info "Creating RMCP environment configuration..."

sudo tee /etc/profile.d/rocm-rdna-fix.sh > /dev/null << 'ENVEOF'
# RMCP (RDNA Memory Coherency Patch) Environment Configuration
# Fixes memory coherency issues on RDNA1/2 GPUs

# Force non-coherent memory allocations
export HSA_USE_SVM=0
export HSA_XNACK=0

# Use fine-grain memory (non-coherent)
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Disable aggressive memory optimizations
export HSA_ENABLE_SDMA=0

# PyTorch ROCm-specific settings
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# ROCm path
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH

echo "✅ RMCP environment loaded (RDNA1/2 memory fix active)"
ENVEOF

log_info "✅ Environment configuration created at /etc/profile.d/rocm-rdna-fix.sh"

# Source it for current session
source /etc/profile.d/rocm-rdna-fix.sh

log_info "Testing configuration..."

# Test if GPU is detected
if command -v rocminfo &> /dev/null; then
    GPU_NAME=$(rocminfo | grep "Name:" | head -1 | awk '{print $2}')
    log_info "GPU detected: $GPU_NAME"
else
    log_warn "rocminfo not found - cannot verify GPU"
fi

# Check architecture
if command -v rocminfo &> /dev/null; then
    ARCH=$(rocminfo | grep "Name:" | grep -i "gfx" | head -1)
    if echo "$ARCH" | grep -qE "gfx101|gfx102|gfx103"; then
        log_info "✅ RDNA1/2 GPU detected - RMCP configuration applied"
    else
        log_warn "Non-RDNA1/2 GPU - RMCP may not be necessary"
    fi
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ RMCP Environment Configuration Complete${NC}"
echo "================================================================================"
echo ""
echo "Environment variables set:"
echo "  HSA_USE_SVM=0             (Disable SVM)"
echo "  HSA_XNACK=0                (Disable XNACK)"
echo "  HSA_FORCE_FINE_GRAIN_PCIE=1 (Force non-coherent)"
echo "  HSA_ENABLE_SDMA=0         (Disable SDMA)"
echo ""
echo "To apply in current shell:"
echo "  source /etc/profile.d/rocm-rdna-fix.sh"
echo ""
echo "To test:"
echo "  cd /home/kevin/Projects/rocm-patch"
echo "  python3 tests/test_real_world_workloads.py"
echo ""
echo "================================================================================"

