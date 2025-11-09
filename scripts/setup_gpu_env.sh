#!/bin/bash
# RMCP Environment Setup - Forces non-coherent memory for RDNA1/2

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_USE_SVM=0
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
export HSA_ENABLE_SDMA=0

# Force PyTorch to use safe memory allocation
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
export PYTORCH_NO_HIP_MEMORY_CACHING=0

echo "🔧 RMCP Environment Configured:"
echo "   HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "   HSA_USE_SVM=$HSA_USE_SVM"
echo "   HSA_XNACK=$HSA_XNACK"
echo "   HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE"
echo ""
echo "Run your PyTorch code now."
