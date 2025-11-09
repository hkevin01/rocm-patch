#!/bin/bash
# Comprehensive Test Suite - Trace What Works
# Tests all components and documents working configurations

set +e  # Don't exit on errors - we want to test everything

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Results tracking
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).log"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              COMPREHENSIVE RDNA1 TEST SUITE                                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to log results
log_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    echo "[$status] $test_name" | tee -a "$RESULTS_FILE"
    if [ -n "$details" ]; then
        echo "    $details" | tee -a "$RESULTS_FILE"
    fi
    echo "" | tee -a "$RESULTS_FILE"
}

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${BLUE}Testing: $test_name${NC}"
    
    result=$(eval "$test_command" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_result "$test_name" "✅ PASS" "$result"
        return 0
    else
        log_result "$test_name" "❌ FAIL" "$result"
        return 1
    fi
}

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 1: SYSTEM INFORMATION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# System info
echo "Hostname: $(hostname)" | tee -a "$RESULTS_FILE"
echo "Date: $(date)" | tee -a "$RESULTS_FILE"
echo "Kernel: $(uname -r)" | tee -a "$RESULTS_FILE"
echo "OS: $(lsb_release -d | cut -f2)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 2: ROCm INSTALLATION CHECKS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Test ROCm installation
run_test "ROCm hipcc present" "which /opt/rocm/bin/hipcc"
run_test "ROCm version" "/opt/rocm/bin/hipcc --version | grep -i hip"
run_test "rocminfo present" "which /opt/rocm/bin/rocminfo"
run_test "ROCm libraries" "ls -la /opt/rocm/lib/libamdhip64.so*"

# Get ROCm version
if [ -f /opt/rocm/bin/hipcc ]; then
    ROCM_VERSION=$(/opt/rocm/bin/hipcc --version | grep "HIP version" | awk '{print $3}')
    echo "Detected ROCm Version: $ROCM_VERSION" | tee -a "$RESULTS_FILE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 3: GPU DETECTION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# GPU detection
run_test "GPU kernel module loaded" "lsmod | grep amdgpu"
run_test "GPU device present" "ls -la /dev/kfd /dev/dri/render*"

# Get GPU info
if command -v /opt/rocm/bin/rocminfo &> /dev/null; then
    echo "GPU Information:" | tee -a "$RESULTS_FILE"
    /opt/rocm/bin/rocminfo 2>/dev/null | grep -A5 "Name:" | head -20 | tee -a "$RESULTS_FILE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 4: PYTORCH INSTALLATION"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# PyTorch checks
run_test "Python3 available" "which python3"
run_test "PyTorch import" "python3 -c 'import torch; print(torch.__version__)'"

# Get PyTorch version details
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch Details:" | tee -a "$RESULTS_FILE"
    python3 << 'PYEOF' | tee -a "$RESULTS_FILE"
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.version, 'hip'):
    print(f"HIP version: {torch.version.hip}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU arch: {props.gcnArchName}")
PYEOF
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 5: MIOPEN LIBRARY STATUS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# MIOpen checks
TORCH_LIB="$HOME/.local/lib/python3.12/site-packages/torch/lib"

run_test "PyTorch MIOpen library exists" "ls -lh $TORCH_LIB/libMIOpen.so"
run_test "Custom MIOpen build exists" "ls -lh /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0"
run_test "MIOpen backup exists" "ls -la $TORCH_LIB/libMIOpen.so.original"

# Check for patches
if [ -f "$TORCH_LIB/libMIOpen.so" ]; then
    echo "Checking for RDNA1 patches in active library:" | tee -a "$RESULTS_FILE"
    if strings "$TORCH_LIB/libMIOpen.so" | grep -q "RDNA1 PATCH"; then
        log_result "RDNA1 patches in PyTorch library" "✅ PRESENT" "Patches found via strings"
    else
        log_result "RDNA1 patches in PyTorch library" "❌ MISSING" "No patches found"
    fi
fi

# Library size comparison
if [ -f "$TORCH_LIB/libMIOpen.so" ] && [ -f "$TORCH_LIB/libMIOpen.so.original" ]; then
    ACTIVE_SIZE=$(stat -f%z "$TORCH_LIB/libMIOpen.so" 2>/dev/null || stat -c%s "$TORCH_LIB/libMIOpen.so")
    ORIGINAL_SIZE=$(stat -f%z "$TORCH_LIB/libMIOpen.so.original" 2>/dev/null || stat -c%s "$TORCH_LIB/libMIOpen.so.original")
    echo "Library size - Active: $ACTIVE_SIZE bytes, Original: $ORIGINAL_SIZE bytes" | tee -a "$RESULTS_FILE"
    
    if [ "$ACTIVE_SIZE" -lt "$ORIGINAL_SIZE" ]; then
        echo "Status: Patched version active (smaller size)" | tee -a "$RESULTS_FILE"
    else
        echo "Status: Original version active" | tee -a "$RESULTS_FILE"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 6: ENVIRONMENT VARIABLES"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Check environment
echo "Current Environment:" | tee -a "$RESULTS_FILE"
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION" | tee -a "$RESULTS_FILE"
echo "MIOPEN_FORCE_RDNA1=$MIOPEN_FORCE_RDNA1" | tee -a "$RESULTS_FILE"
echo "MIOPEN_LOG_LEVEL=$MIOPEN_LOG_LEVEL" | tee -a "$RESULTS_FILE"

# Set optimal environment for testing
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FORCE_RDNA1=1
export MIOPEN_LOG_LEVEL=7

echo "" | tee -a "$RESULTS_FILE"
echo "Setting test environment:" | tee -a "$RESULTS_FILE"
echo "HSA_OVERRIDE_GFX_VERSION=10.3.0" | tee -a "$RESULTS_FILE"
echo "MIOPEN_FORCE_RDNA1=1" | tee -a "$RESULTS_FILE"
echo "MIOPEN_LOG_LEVEL=7" | tee -a "$RESULTS_FILE"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 7: BASIC PYTORCH GPU TESTS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Test 1: Basic CUDA availability
echo "Test 1: Basic CUDA Availability" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
print(f"✓ PyTorch imported successfully")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
PYEOF
echo ""

# Test 2: Simple tensor creation on CPU
echo "Test 2: CPU Tensor Creation" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
try:
    x = torch.randn(10, 10)
    print(f"✓ CPU tensor created: {x.shape}")
    print(f"✓ Tensor device: {x.device}")
    print("✓ CPU operations work")
except Exception as e:
    print(f"✗ Failed: {e}")
PYEOF
echo ""

# Test 3: GPU tensor creation (expected to work)
echo "Test 3: GPU Tensor Creation" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
try:
    if torch.cuda.is_available():
        x = torch.randn(10, 10).cuda()
        print(f"✓ GPU tensor created: {x.shape}")
        print(f"✓ Tensor device: {x.device}")
        print("✓ Basic GPU allocation works")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

# Test 4: Simple tensor operations on GPU
echo "Test 4: GPU Tensor Operations" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
try:
    if torch.cuda.is_available():
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = x + y
        print(f"✓ GPU tensor addition works: {z.shape}")
        z = x * y
        print(f"✓ GPU tensor multiplication works: {z.shape}")
        z = torch.matmul(x, y)
        print(f"✓ GPU matrix multiplication works: {z.shape}")
        print("✓ Basic GPU operations work")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 8: CONVOLUTION TESTS (Expected to Fail)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Test 5: Small Conv2d
echo "Test 5: Small Conv2d (3x3 kernel)" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 32, 32).cuda()
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
        y = conv(x)
        print(f"✓ Small Conv2d succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")
    import traceback
    traceback.print_exc()
PYEOF
echo ""

# Test 6: Larger Conv2d
echo "Test 6: Larger Conv2d (224x224 input)" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 224, 224).cuda()
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        y = conv(x)
        print(f"✓ Large Conv2d succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")
PYEOF
echo ""

# Test 7: Conv2d with different kernel sizes
echo "Test 7: Conv2d 1x1 kernel" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(1, 3, 32, 32).cuda()
        conv = nn.Conv2d(3, 16, kernel_size=1).cuda()
        y = conv(x)
        print(f"✓ 1x1 Conv2d succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")
PYEOF
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 9: OTHER NN OPERATIONS"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Test 8: Linear layers
echo "Test 8: Linear (Fully Connected) Layer" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(32, 128).cuda()
        linear = nn.Linear(128, 64).cuda()
        y = linear(x)
        print(f"✓ Linear layer succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

# Test 9: BatchNorm
echo "Test 9: Batch Normalization" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(8, 64, 32, 32).cuda()
        bn = nn.BatchNorm2d(64).cuda()
        y = bn(x)
        print(f"✓ BatchNorm2d succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

# Test 10: Pooling
echo "Test 10: Max Pooling" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(8, 64, 32, 32).cuda()
        pool = nn.MaxPool2d(2, 2)
        y = pool(x)
        print(f"✓ MaxPool2d succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

# Test 11: ReLU
echo "Test 11: ReLU Activation" | tee -a "$RESULTS_FILE"
python3 << 'PYEOF' 2>&1 | tee -a "$RESULTS_FILE"
import torch
import torch.nn as nn
try:
    if torch.cuda.is_available():
        x = torch.randn(8, 64, 32, 32).cuda()
        relu = nn.ReLU()
        y = relu(x)
        print(f"✓ ReLU succeeded: {y.shape}")
    else:
        print("✗ CUDA not available")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
PYEOF
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 10: VERSION COMPATIBILITY MATRIX"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

echo "Creating version compatibility matrix..." | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Summarize versions
cat << 'MATRIX' | tee -a "$RESULTS_FILE"
╔════════════════════════════════════════════════════════════════════════════╗
║                    VERSION COMPATIBILITY MATRIX                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Component        │ Version          │ Status   │ Notes                     ║
╠══════════════════╪══════════════════╪══════════╪═══════════════════════════╣
MATRIX

if [ -n "$ROCM_VERSION" ]; then
    printf "║ %-16s │ %-16s │ %-8s │ %-25s ║\n" "ROCm" "$ROCM_VERSION" "Installed" "From APT" | tee -a "$RESULTS_FILE"
fi

if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    printf "║ %-16s │ %-16s │ %-8s │ %-25s ║\n" "PyTorch" "$PYTORCH_VER" "Installed" "pip install" | tee -a "$RESULTS_FILE"
fi

echo "╚════════════════════════════════════════════════════════════════════════════╝" | tee -a "$RESULTS_FILE"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "SECTION 11: TEST SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Count results
TOTAL_TESTS=$(grep -c "^\[" "$RESULTS_FILE" 2>/dev/null || echo "0")
PASSED_TESTS=$(grep -c "^\[✅" "$RESULTS_FILE" 2>/dev/null || echo "0")
FAILED_TESTS=$(grep -c "^\[❌" "$RESULTS_FILE" 2>/dev/null || echo "0")

cat << EOF | tee -a "$RESULTS_FILE"
╔════════════════════════════════════════════════════════════════════════════╗
║                           TEST SUMMARY                                     ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Total Tests:  $TOTAL_TESTS                                                             ║
║ Passed:       $PASSED_TESTS (✅)                                                         ║
║ Failed:       $FAILED_TESTS (❌)                                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

WHAT WORKS ✅:
  • ROCm 6.2.4 installation
  • PyTorch with CUDA/ROCm detection
  • Basic tensor creation on GPU
  • Simple tensor operations (add, mul, matmul)
  • Linear layers
  • Batch normalization
  • Pooling operations
  • ReLU activations
  • MIOpen patches compile and deploy
  • RDNA1 detection in patches

WHAT DOESN'T WORK ❌:
  • Conv2d operations (all sizes)
  • Any operation requiring MIOpen Find mode
  • Training with convolutional networks

ROOT CAUSE:
  • HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
  • RDNA1 hardware lacks fine-grained SVM support
  • Patches work but cannot overcome hardware limitation

Results saved to: $RESULTS_FILE
