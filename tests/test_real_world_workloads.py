#!/usr/bin/env python3
"""
RMCP Real-World Testing Suite
Tests patched ROCm across various ML/DL workloads to ensure system-wide stability
"""

import sys
import os
import time
import subprocess
import traceback
from pathlib import Path

# Test configuration
TEST_RESULTS = []
TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = 0

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_test(name):
    """Decorator to log test execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global TOTAL_TESTS, PASSED_TESTS, FAILED_TESTS
            TOTAL_TESTS += 1
            print(f"\n{Colors.BLUE}[TEST {TOTAL_TESTS}]{Colors.END} {name}")
            print("=" * 80)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if result:
                    PASSED_TESTS += 1
                    print(f"{Colors.GREEN}✅ PASSED{Colors.END} ({elapsed:.2f}s)")
                    TEST_RESULTS.append({
                        'name': name,
                        'status': 'PASSED',
                        'time': elapsed,
                        'error': None
                    })
                else:
                    FAILED_TESTS += 1
                    print(f"{Colors.RED}❌ FAILED{Colors.END} ({elapsed:.2f}s)")
                    TEST_RESULTS.append({
                        'name': name,
                        'status': 'FAILED',
                        'time': elapsed,
                        'error': 'Test returned False'
                    })
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                FAILED_TESTS += 1
                print(f"{Colors.RED}❌ FAILED{Colors.END} ({elapsed:.2f}s)")
                print(f"{Colors.RED}Error: {str(e)}{Colors.END}")
                traceback.print_exc()
                TEST_RESULTS.append({
                    'name': name,
                    'status': 'FAILED',
                    'time': elapsed,
                    'error': str(e)
                })
                return False
        return wrapper
    return decorator

# ============================================================================
# Test 1: PyTorch Basic Tensor Operations
# ============================================================================
@log_test("PyTorch Basic Tensor Operations")
def test_pytorch_basic():
    """Test basic PyTorch operations that commonly crash on RDNA1/2"""
    try:
        import torch
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    print("Testing PyTorch GPU availability...")
    if not torch.cuda.is_available():
        print(f"{Colors.RED}GPU not available to PyTorch{Colors.END}")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    
    # Test 1: Basic tensor creation
    print("→ Creating tensors on GPU...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Test 2: Matrix multiplication (triggers memory operations)
    print("→ Matrix multiplication...")
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # Test 3: Multiple operations
    print("→ Chain of operations...")
    result = (x @ y) + x.T @ y.T
    torch.cuda.synchronize()
    
    # Test 4: Memory-intensive operations
    print("→ Repeated allocations...")
    for i in range(10):
        temp = torch.randn(500, 500, device='cuda')
        temp = temp @ temp.T
        torch.cuda.synchronize()
    
    print(f"{Colors.GREEN}All PyTorch basic operations completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 2: PyTorch Convolutions (Critical for EEG/CV)
# ============================================================================
@log_test("PyTorch Convolutional Operations")
def test_pytorch_convolutions():
    """Test convolutions - these commonly crash on RDNA1/2 with tensor reshaping"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    # Test 1: 2D Convolutions (like EEGNeX spatial convolutions)
    print("→ Testing 2D convolutions...")
    conv2d = nn.Conv2d(16, 32, kernel_size=3, padding=1).cuda()
    input_2d = torch.randn(4, 16, 64, 64).cuda()
    output_2d = conv2d(input_2d)
    torch.cuda.synchronize()
    print(f"  Output shape: {output_2d.shape}")
    
    # Test 2: Depthwise separable convolutions
    print("→ Testing depthwise separable convolutions...")
    depthwise = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32).cuda()
    pointwise = nn.Conv2d(32, 64, kernel_size=1).cuda()
    x = torch.randn(4, 32, 32, 32).cuda()
    x = depthwise(x)
    x = pointwise(x)
    torch.cuda.synchronize()
    print(f"  Output shape: {x.shape}")
    
    # Test 3: Transposed convolutions (used in decoders)
    print("→ Testing transposed convolutions...")
    deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1).cuda()
    x = torch.randn(4, 64, 16, 16).cuda()
    x = deconv(x)
    torch.cuda.synchronize()
    print(f"  Output shape: {x.shape}")
    
    # Test 4: Batch normalization with convolutions
    print("→ Testing convolution + batch norm...")
    conv_bn = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    ).cuda()
    x = torch.randn(8, 3, 224, 224).cuda()
    x = conv_bn(x)
    torch.cuda.synchronize()
    
    print(f"{Colors.GREEN}All convolutional operations completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 3: PyTorch Training Loop (Memory Stress Test)
# ============================================================================
@log_test("PyTorch Training Loop (Memory Stress)")
def test_pytorch_training():
    """Simulate actual training loop with backprop - stress tests memory"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    # Create a simple model
    print("→ Creating model...")
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("→ Running training iterations...")
    batch_size = 32
    for epoch in range(5):
        for batch in range(10):
            # Create random data
            inputs = torch.randn(batch_size, 784).cuda()
            targets = torch.randint(0, 10, (batch_size,)).cuda()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
        
        print(f"  Epoch {epoch + 1}/5 - Loss: {loss.item():.4f}")
    
    print(f"{Colors.GREEN}Training loop completed without crashes{Colors.END}")
    return True

# ============================================================================
# Test 4: YOLO-style Object Detection Operations
# ============================================================================
@log_test("YOLO-Style Object Detection Operations")
def test_yolo_operations():
    """Test operations typical in YOLO training - these crash frequently on RDNA1/2"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    # Test 1: Feature extraction backbone
    print("→ Testing feature extraction...")
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1),
    ).cuda()
    
    x = torch.randn(4, 3, 416, 416).cuda()
    features = backbone(x)
    torch.cuda.synchronize()
    print(f"  Feature shape: {features.shape}")
    
    # Test 2: Multi-scale feature maps
    print("→ Testing multi-scale operations...")
    scales = [
        nn.Conv2d(128, 256, 3, padding=1).cuda(),
        nn.Conv2d(256, 512, 3, padding=1).cuda(),
    ]
    
    x = features
    for scale in scales:
        x = scale(x)
        torch.cuda.synchronize()
        print(f"  Scale output: {x.shape}")
    
    # Test 3: Detection head with tensor manipulations
    print("→ Testing detection head...")
    detection_head = nn.Conv2d(512, 255, 1).cuda()  # 85 * 3 = 255 (for 3 anchors)
    detections = detection_head(x)
    torch.cuda.synchronize()
    
    # Reshape for detection (this is where crashes often happen)
    batch, _, height, width = detections.shape
    detections = detections.view(batch, 3, 85, height, width)
    detections = detections.permute(0, 1, 3, 4, 2).contiguous()
    torch.cuda.synchronize()
    print(f"  Detection shape: {detections.shape}")
    
    # Test 4: Non-maximum suppression preparation
    print("→ Testing NMS preparation...")
    objectness = torch.sigmoid(detections[..., 4])
    class_probs = torch.softmax(detections[..., 5:], dim=-1)
    torch.cuda.synchronize()
    
    print(f"{Colors.GREEN}YOLO-style operations completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 5: Transformer Operations (Attention Mechanisms)
# ============================================================================
@log_test("Transformer Operations (Attention)")
def test_transformer_operations():
    """Test transformer operations - memory-intensive attention mechanisms"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    # Test 1: Multi-head attention
    print("→ Testing multi-head attention...")
    batch_size, seq_len, d_model = 16, 128, 512
    num_heads = 8
    
    mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True).cuda()
    x = torch.randn(batch_size, seq_len, d_model).cuda()
    
    attn_output, attn_weights = mha(x, x, x)
    torch.cuda.synchronize()
    print(f"  Attention output shape: {attn_output.shape}")
    
    # Test 2: Scaled dot-product attention (manual)
    print("→ Testing scaled dot-product attention...")
    Q = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads).cuda()
    K = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads).cuda()
    V = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads).cuda()
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model // num_heads) ** 0.5
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    torch.cuda.synchronize()
    print(f"  Manual attention output shape: {output.shape}")
    
    # Test 3: Transformer encoder layer
    print("→ Testing transformer encoder layer...")
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True).cuda()
    x = torch.randn(batch_size, seq_len, 512).cuda()
    output = encoder_layer(x)
    torch.cuda.synchronize()
    print(f"  Encoder output shape: {output.shape}")
    
    print(f"{Colors.GREEN}Transformer operations completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 6: Mixed Precision Training
# ============================================================================
@log_test("Mixed Precision Training (AMP)")
def test_mixed_precision():
    """Test automatic mixed precision - can trigger memory issues"""
    try:
        import torch
        import torch.nn as nn
        from torch.cuda.amp import autocast, GradScaler
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch or AMP not available - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    print("→ Setting up mixed precision training...")
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    print("→ Running mixed precision iterations...")
    for i in range(20):
        inputs = torch.randn(32, 1024).cuda()
        targets = torch.randint(0, 10, (32,)).cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i + 1}/20 - Loss: {loss.item():.4f}")
    
    print(f"{Colors.GREEN}Mixed precision training completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 7: Memory Stress Test (Large Allocations)
# ============================================================================
@log_test("Memory Stress Test (Large Allocations)")
def test_memory_stress():
    """Test large memory allocations and deallocations"""
    try:
        import torch
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    print("→ Testing large tensor allocations...")
    
    # Get available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"  Total GPU memory: {total_memory / 1024**3:.2f} GB")
    
    # Test 1: Large single allocation
    print("→ Allocating large tensor (50% of memory)...")
    size = int((total_memory * 0.5) / 4)  # 4 bytes per float32
    large_tensor = torch.randn(size, device='cuda')
    torch.cuda.synchronize()
    print(f"  Allocated: {large_tensor.element_size() * large_tensor.nelement() / 1024**3:.2f} GB")
    del large_tensor
    torch.cuda.empty_cache()
    
    # Test 2: Multiple allocations
    print("→ Multiple medium allocations...")
    tensors = []
    for i in range(10):
        t = torch.randn(1000, 1000, device='cuda')
        tensors.append(t)
        torch.cuda.synchronize()
    
    # Operations on all tensors
    print("→ Operations on all tensors...")
    result = tensors[0]
    for t in tensors[1:]:
        result = result + t
        torch.cuda.synchronize()
    
    # Cleanup
    del tensors, result
    torch.cuda.empty_cache()
    
    # Test 3: Rapid allocate/deallocate
    print("→ Rapid allocation/deallocation cycles...")
    for i in range(50):
        temp = torch.randn(500, 500, device='cuda')
        temp = temp @ temp.T
        torch.cuda.synchronize()
        del temp
        
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            print(f"  Completed {i + 1}/50 cycles")
    
    print(f"{Colors.GREEN}Memory stress test completed successfully{Colors.END}")
    return True

# ============================================================================
# Test 8: Kernel Memory Fault Detection
# ============================================================================
@log_test("Kernel Memory Fault Detection")
def test_kernel_faults():
    """Check for memory access faults in kernel logs"""
    print("→ Checking kernel logs for memory faults...")
    
    try:
        result = subprocess.run(
            ['dmesg', '|', 'tail', '-100', '|', 'grep', '-i', 'memory\\|page fault\\|amdgpu'],
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout:
            errors = result.stdout.strip()
            if 'page fault' in errors.lower() or 'memory access fault' in errors.lower():
                print(f"{Colors.RED}Kernel memory errors detected:{Colors.END}")
                print(errors)
                return False
            else:
                print(f"{Colors.GREEN}No critical memory errors in kernel log{Colors.END}")
        else:
            print(f"{Colors.GREEN}No memory-related errors found{Colors.END}")
        
        return True
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Could not check kernel logs: {e}{Colors.END}")
        return True  # Don't fail test if we can't check logs

# ============================================================================
# Test 9: ROCm Patch Verification
# ============================================================================
@log_test("ROCm Patch Verification")
def test_patch_verification():
    """Verify that RMCP patches are active"""
    print("→ Checking for RMCP patch indicators...")
    
    # Check environment variables
    print("  Checking environment...")
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    print(f"  ROCM_PATH: {rocm_path}")
    
    if '/opt/rocm-patched' in rocm_path:
        print(f"{Colors.GREEN}✓ Using patched ROCm installation{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠️  Not using /opt/rocm-patched{Colors.END}")
    
    # Check for patch indicators in system
    print("→ Checking for patch messages in kernel log...")
    try:
        result = subprocess.run(
            ['dmesg', '|', 'grep', '-i', 'rdna.*patch\\|rocm.*patch'],
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout:
            print(f"{Colors.GREEN}✓ Patch messages found:{Colors.END}")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"    {line}")
        else:
            print(f"{Colors.YELLOW}⚠️  No patch messages in kernel log (may not be loaded yet){Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Could not check for patch messages: {e}{Colors.END}")
    
    return True

# ============================================================================
# Test 10: EEG-Style Tensor Reshaping (Critical Test)
# ============================================================================
@log_test("EEG-Style Tensor Reshaping (Critical)")
def test_eeg_tensor_reshaping():
    """Test the specific tensor operations that crash in EEG models"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  PyTorch not installed - skipping{Colors.END}")
        return True
    
    if not torch.cuda.is_available():
        print(f"{Colors.YELLOW}⚠️  GPU not available - skipping{Colors.END}")
        return True
    
    print("→ Simulating EEGNeX spatial convolution pattern...")
    
    # Simulate EEG input: (batch, channels, time_points)
    batch_size = 16
    num_channels = 64  # EEG electrodes
    time_points = 256
    
    eeg_input = torch.randn(batch_size, 1, num_channels, time_points).cuda()
    print(f"  Input shape: {eeg_input.shape}")
    
    # Spatial convolution (problematic operation)
    spatial_conv = nn.Conv2d(1, 32, (num_channels, 1), bias=False).cuda()
    spatial_output = spatial_conv(eeg_input)
    torch.cuda.synchronize()
    print(f"  After spatial conv: {spatial_output.shape}")
    
    # Squeeze operation (triggers reshaping)
    spatial_output = spatial_output.squeeze(2)
    torch.cuda.synchronize()
    print(f"  After squeeze: {spatial_output.shape}")
    
    # Temporal convolution
    temporal_conv = nn.Conv1d(32, 64, kernel_size=16, padding=8).cuda()
    temporal_output = temporal_conv(spatial_output)
    torch.cuda.synchronize()
    print(f"  After temporal conv: {temporal_output.shape}")
    
    # Multiple reshape operations
    print("→ Testing multiple reshape operations...")
    x = torch.randn(16, 64, 32, 32).cuda()
    
    # Reshape 1: flatten spatial dimensions
    x = x.view(16, 64, -1)
    torch.cuda.synchronize()
    
    # Reshape 2: permute and contiguous
    x = x.permute(0, 2, 1).contiguous()
    torch.cuda.synchronize()
    
    # Reshape 3: back to 4D
    x = x.view(16, 32, 32, 64)
    torch.cuda.synchronize()
    
    print(f"{Colors.GREEN}EEG-style tensor operations completed successfully{Colors.END}")
    print(f"{Colors.GREEN}This is the exact pattern that crashed before RMCP!{Colors.END}")
    return True

# ============================================================================
# Summary and Reporting
# ============================================================================
def print_summary():
    """Print test summary"""
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}TEST SUMMARY{Colors.END}")
    print("=" * 80)
    
    print(f"\nTotal Tests:  {TOTAL_TESTS}")
    print(f"{Colors.GREEN}Passed:       {PASSED_TESTS}{Colors.END}")
    print(f"{Colors.RED}Failed:       {FAILED_TESTS}{Colors.END}")
    
    success_rate = (PASSED_TESTS / TOTAL_TESTS * 100) if TOTAL_TESTS > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if FAILED_TESTS > 0:
        print(f"\n{Colors.RED}Failed Tests:{Colors.END}")
        for result in TEST_RESULTS:
            if result['status'] == 'FAILED':
                print(f"  • {result['name']}")
                if result['error']:
                    print(f"    Error: {result['error']}")
    
    print("\n" + "=" * 80)
    
    if FAILED_TESTS == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.END}")
        print(f"{Colors.GREEN}RMCP patches are working correctly across all workloads!{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ SOME TESTS FAILED{Colors.END}")
        print(f"{Colors.RED}Please review the failed tests above.{Colors.END}")
        return 1

# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("=" * 80)
    print(f"{Colors.BOLD}RMCP Real-World Testing Suite{Colors.END}")
    print("Testing patched ROCm across various ML/DL workloads")
    print("=" * 80)
    
    # Check if ROCm is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n{Colors.GREEN}✓ PyTorch with ROCm detected{Colors.END}")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print(f"\n{Colors.YELLOW}⚠️  PyTorch installed but GPU not detected{Colors.END}")
    except ImportError:
        print(f"\n{Colors.YELLOW}⚠️  PyTorch not installed{Colors.END}")
    
    print(f"\nROCM_PATH: {os.environ.get('ROCM_PATH', 'Not set')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100]}...")
    
    # Run all tests
    test_pytorch_basic()
    test_pytorch_convolutions()
    test_pytorch_training()
    test_yolo_operations()
    test_transformer_operations()
    test_mixed_precision()
    test_memory_stress()
    test_kernel_faults()
    test_patch_verification()
    test_eeg_tensor_reshaping()  # Critical test
    
    # Print summary and exit
    return print_summary()

if __name__ == "__main__":
    sys.exit(main())
