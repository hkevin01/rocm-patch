#!/bin/bash
################################################################################
# ROCm Patched Installation Test Suite
# Comprehensive testing for RDNA1/2 memory fixes
################################################################################

ROCM_PATH=${ROCM_PATH:-/opt/rocm-patched}
TEST_DIR=/tmp/rocm-patch-tests

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

################################################################################
# Test 1: ROCm Environment
################################################################################
test_rocm_environment() {
    log_test "Testing ROCm environment..."
    ((TESTS_RUN++))
    
    if [ -d "$ROCM_PATH" ]; then
        log_pass "ROCm path exists: $ROCM_PATH"
    else
        log_fail "ROCm path not found: $ROCM_PATH"
        return 1
    fi
    
    if [ -x "$ROCM_PATH/bin/rocminfo" ]; then
        log_pass "rocminfo binary found"
    else
        log_fail "rocminfo binary not found"
        return 1
    fi
    
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
}

################################################################################
# Test 2: ROCm Info
################################################################################
test_rocminfo() {
    log_test "Testing rocminfo..."
    ((TESTS_RUN++))
    
    if timeout 10 $ROCM_PATH/bin/rocminfo > /tmp/rocminfo.txt 2>&1; then
        log_pass "rocminfo executed successfully"
        
        # Check for RDNA GPUs
        if grep -q "gfx10" /tmp/rocminfo.txt; then
            log_info "RDNA GPU detected:"
            grep "Name:" /tmp/rocminfo.txt | head -2
        fi
    else
        log_fail "rocminfo failed or timed out"
        return 1
    fi
}

################################################################################
# Test 3: HIP Compilation
################################################################################
test_hip_compilation() {
    log_test "Testing HIP compilation..."
    ((TESTS_RUN++))
    
    mkdir -p $TEST_DIR
    
    cat > $TEST_DIR/test_hip_compile.cpp << 'HIPTEST'
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void simple_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    printf("HIP Compilation Test\n");
    
    int deviceCount;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess) {
        printf("ERROR: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    printf("Found %d HIP device(s)\n", deviceCount);
    
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("GCN Arch: %s\n", prop.gcnArchName);
        printf("Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    }
    
    return 0;
}
HIPTEST
    
    if $ROCM_PATH/bin/hipcc $TEST_DIR/test_hip_compile.cpp -o $TEST_DIR/test_hip_compile 2>&1 | tee $TEST_DIR/compile.log; then
        log_pass "HIP compilation successful"
        
        if $TEST_DIR/test_hip_compile; then
            log_pass "HIP test program executed"
        else
            log_fail "HIP test program failed to execute"
        fi
    else
        log_fail "HIP compilation failed"
        cat $TEST_DIR/compile.log
        return 1
    fi
}

################################################################################
# Test 4: HIP Memory Operations
################################################################################
test_hip_memory() {
    log_test "Testing HIP memory operations..."
    ((TESTS_RUN++))
    
    cat > $TEST_DIR/test_hip_memory.cpp << 'MEMTEST'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

__global__ void memory_test_kernel(float* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Read and write to test memory coherency
        float val = d_data[idx];
        d_data[idx] = val * 2.0f + 1.0f;
    }
}

int main() {
    const int N = 1024 * 1024; // 1M elements
    const size_t bytes = N * sizeof(float);
    
    printf("Testing memory operations with %d elements (%.2f MB)\n", 
           N, bytes / 1024.0 / 1024.0);
    
    // Allocate host memory
    float* h_data = (float*)malloc(bytes);
    if (!h_data) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    // Allocate device memory (this is where RDNA1/2 issues occur)
    float* d_data;
    CHECK_HIP(hipMalloc(&d_data, bytes));
    printf("✓ Device memory allocated\n");
    
    // Copy to device
    CHECK_HIP(hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice));
    printf("✓ Host to device copy successful\n");
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(memory_test_kernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 
                       0, 0, d_data, N);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    printf("✓ Kernel execution successful\n");
    
    // Copy back
    CHECK_HIP(hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost));
    printf("✓ Device to host copy successful\n");
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        float expected = (float)i * 2.0f + 1.0f;
        if (fabs(h_data[i] - expected) > 0.01f) {
            printf("Mismatch at %d: expected %.2f, got %.2f\n", i, expected, h_data[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("✓ Results verified correct\n");
    } else {
        printf("✗ Results verification failed\n");
        return 1;
    }
    
    // Cleanup
    CHECK_HIP(hipFree(d_data));
    free(h_data);
    
    printf("\n✅ All memory tests passed!\n");
    return 0;
}
MEMTEST
    
    if $ROCM_PATH/bin/hipcc $TEST_DIR/test_hip_memory.cpp -o $TEST_DIR/test_hip_memory 2>&1; then
        if timeout 30 $TEST_DIR/test_hip_memory 2>&1 | tee $TEST_DIR/memory_test.log; then
            log_pass "HIP memory operations successful"
        else
            log_fail "HIP memory operations failed"
            cat $TEST_DIR/memory_test.log
            return 1
        fi
    else
        log_fail "HIP memory test compilation failed"
        return 1
    fi
}

################################################################################
# Test 5: PyTorch Integration (if available)
################################################################################
test_pytorch() {
    log_test "Testing PyTorch integration..."
    ((TESTS_RUN++))
    
    if ! python3 -c "import torch" 2>/dev/null; then
        log_info "PyTorch not installed - skipping test"
        return 0
    fi
    
    cat > $TEST_DIR/test_pytorch.py << 'PYTEST'
import torch
import sys

print("PyTorch ROCm Integration Test")
print(f"PyTorch version: {torch.__version__}")

# Check ROCm availability
if not torch.cuda.is_available():
    print("ERROR: CUDA/ROCm not available to PyTorch")
    sys.exit(1)

print(f"✓ ROCm available to PyTorch")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.device_count() > 0:
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

# Test tensor operations
try:
    print("\nTesting tensor operations...")
    
    # Create tensors
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Matrix multiplication (triggers memory operations)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    print("✓ Matrix multiplication successful")
    
    # Convolution test (critical for EEG models)
    conv = torch.nn.Conv2d(16, 32, 3).cuda()
    input_tensor = torch.randn(1, 16, 64, 64).cuda()
    output = conv(input_tensor)
    torch.cuda.synchronize()
    print("✓ Convolution successful")
    
    # Memory intensive operation
    for i in range(10):
        a = torch.randn(512, 512, device='cuda')
        b = torch.randn(512, 512, device='cuda')
        c = a @ b
        torch.cuda.synchronize()
    print("✓ Repeated operations successful")
    
    print("\n✅ All PyTorch tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ PyTorch test failed: {e}")
    sys.exit(1)
PYTEST
    
    if timeout 60 python3 $TEST_DIR/test_pytorch.py 2>&1 | tee $TEST_DIR/pytorch_test.log; then
        log_pass "PyTorch integration successful"
    else
        log_fail "PyTorch integration failed"
        cat $TEST_DIR/pytorch_test.log
        return 1
    fi
}

################################################################################
# Test 6: Kernel Memory Fault Check
################################################################################
test_kernel_faults() {
    log_test "Checking for kernel memory faults..."
    ((TESTS_RUN++))
    
    # Check dmesg for memory-related errors
    if sudo dmesg | tail -100 | grep -i "amdgpu.*memory\|page fault\|page not present" > /tmp/kernel_errors.txt 2>&1; then
        if [ -s /tmp/kernel_errors.txt ]; then
            log_fail "Kernel memory errors detected:"
            cat /tmp/kernel_errors.txt
            return 1
        fi
    fi
    
    log_pass "No kernel memory faults detected"
}

################################################################################
# Test 7: Verify Patch Application
################################################################################
test_patch_verification() {
    log_test "Verifying patches are active..."
    ((TESTS_RUN++))
    
    # Check for patch messages in dmesg
    if sudo dmesg | grep -i "rdna.*patch\|rocm.*patch" > /tmp/patch_messages.txt 2>&1; then
        if [ -s /tmp/patch_messages.txt ]; then
            log_pass "Patch messages found in kernel log:"
            cat /tmp/patch_messages.txt
        else
            log_info "No patch messages in kernel log (may not be loaded yet)"
        fi
    fi
    
    # Check HIP library for patches
    if strings $ROCM_PATH/lib/libamdhip64.so 2>/dev/null | grep -i "rdna\|patch\|coherency" > /tmp/hip_strings.txt; then
        if [ -s /tmp/hip_strings.txt ]; then
            log_pass "Patch indicators found in HIP library"
        fi
    fi
}

################################################################################
# Main Test Runner
################################################################################
main() {
    echo "========================================================================"
    echo "   ROCm Patched Installation Test Suite"
    echo "========================================================================"
    echo ""
    echo "ROCm Path: $ROCM_PATH"
    echo "Test Directory: $TEST_DIR"
    echo ""
    
    mkdir -p $TEST_DIR
    
    # Run tests
    test_rocm_environment
    test_rocminfo
    test_hip_compilation
    test_hip_memory
    test_pytorch
    test_kernel_faults
    test_patch_verification
    
    # Summary
    echo ""
    echo "========================================================================"
    echo "   Test Results Summary"
    echo "========================================================================"
    echo ""
    echo "Tests Run:    $TESTS_RUN"
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
        echo ""
        echo "Your patched ROCm installation is working correctly."
        echo "RDNA1/2 memory issues should be resolved."
        exit 0
    else
        echo -e "${RED}❌ SOME TESTS FAILED${NC}"
        echo ""
        echo "Review the test output above for details."
        echo "Check logs in: $TEST_DIR"
        exit 1
    fi
}

main "$@"
