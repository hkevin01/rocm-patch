#!/bin/bash
# ROCm 5.7 + PyTorch 2.2.2 Setup Verification Script

echo "======================================================================"
echo "🔍 ROCm 5.7 + PyTorch 2.2.2 Setup Verification"
echo "======================================================================"
echo ""

# Check configuration file
echo "📁 Configuration File:"
if [ -f /etc/profile.d/rocm-rdna1-57.sh ]; then
    echo "✅ /etc/profile.d/rocm-rdna1-57.sh exists"
else
    echo "❌ /etc/profile.d/rocm-rdna1-57.sh NOT FOUND"
    echo "   Run: ./install_rocm57.sh"
    exit 1
fi
echo ""

# Source configuration
source /etc/profile.d/rocm-rdna1-57.sh

# Check environment variables
echo "🔧 Environment Variables:"
echo ""

check_env() {
    local var=$1
    local expected=$2
    local value="${!var}"
    
    if [ "$value" = "$expected" ]; then
        echo "✅ $var=$value"
    else
        echo "❌ $var=$value (expected: $expected)"
    fi
}

# Critical variables
check_env "HSA_OVERRIDE_GFX_VERSION" "10.3.0"
check_env "PYTORCH_ROCM_ARCH" "gfx1030"
check_env "MIOPEN_DEBUG_CONV_GEMM" "1"
check_env "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM" "0"
check_env "MIOPEN_DEBUG_CONV_WINOGRAD" "0"
check_env "MIOPEN_DEBUG_CONV_DIRECT" "0"
check_env "HIP_FORCE_COARSE_GRAIN" "1"
check_env "HSA_ENABLE_SDMA" "0"

echo ""

# Check GPU
echo "🎮 GPU Detection:"
if lspci -nn 2>/dev/null | grep -qiE "1002:731f|1002:731e|1002:7310|1002:7312"; then
    echo "✅ RDNA1 GPU detected:"
    lspci -nn | grep -iE "1002:731f|1002:731e|1002:7310|1002:7312" | head -1
else
    echo "❌ No RDNA1 GPU detected"
fi
echo ""

# Check PyTorch
echo "🐍 PyTorch Version:"
python3 -c "
import torch
version = torch.__version__
if '2.2.2' in version and 'rocm5.7' in version:
    print(f'✅ PyTorch {version}')
else:
    print(f'❌ PyTorch {version} (expected: 2.2.2+rocm5.7)')
    print('   Install: pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/rocm5.7')
" 2>/dev/null || echo "❌ PyTorch not installed or Python error"

echo ""

# Check CUDA
echo "🚀 CUDA Availability:"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available')
    print(f'✅ Device: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available')
" 2>/dev/null || echo "❌ Python/PyTorch error"

echo ""

# Summary
echo "======================================================================"
echo "📊 Summary"
echo "======================================================================"
echo ""
echo "✅ = Working correctly"
echo "❌ = Needs attention"
echo ""
echo "Next steps:"
echo "  1. Open a NEW terminal (to load environment)"
echo "  2. Run: python3 test_conv2d_timing.py"
echo "  3. First run will take 30-60 seconds (kernel compilation)"
echo "  4. Subsequent runs will be instant (cached kernels)"
echo ""
echo "Documentation:"
echo "  - README.md           - Main documentation"
echo "  - README_ROCM57.md    - Detailed setup guide"
echo "  - SOLUTION_ROCM57.md  - Solution summary"
echo ""
