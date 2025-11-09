#!/usr/bin/env python3
"""
Comprehensive test suite for IMPLICIT_GEMM on gfx1010
Tests all aspects to ensure the solution is robust and reliable
"""

import os
import sys
import time
import torch
import subprocess
from typing import List, Tuple, Dict

# Force IMPLICIT_GEMM configuration
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['MIOPEN_DEBUG_CONV_WINOGRAD'] = '0'
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '0'
os.environ['MIOPEN_DEBUG_CONV_FFT'] = '0'

print("=" * 80)
print("🧪 COMPREHENSIVE IMPLICIT_GEMM TEST SUITE")
print("=" * 80)
print(f"\n📋 Environment Configuration:")
print(f"   MIOPEN_DEBUG_CONV_GEMM: {os.getenv('MIOPEN_DEBUG_CONV_GEMM')}")
print(f"   MIOPEN_DEBUG_CONV_IMPLICIT_GEMM: {os.getenv('MIOPEN_DEBUG_CONV_IMPLICIT_GEMM')}")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print()

# Track all results
test_results = {
    'passed': [],
    'failed': [],
    'total': 0
}

def record_result(test_name: str, success: bool, error: str = None):
    """Record test result"""
    test_results['total'] += 1
    if success:
        test_results['passed'].append(test_name)
        print(f"   ✅ {test_name}")
    else:
        test_results['failed'].append((test_name, error))
        print(f"   ❌ {test_name}: {error}")

def test_basic_sizes():
    """Test 1: Basic Size Range (Previously Failing Sizes)"""
    print("\n" + "=" * 80)
    print("Test 1: Basic Size Range (Previously Failing Sizes)")
    print("=" * 80)
    
    # These sizes are critical - 44x44+ were failing before
    sizes = [32, 40, 42, 43, 44, 45, 48, 50, 56, 64]
    
    for size in sizes:
        try:
            start = time.time()
            x = torch.randn(1, 3, size, size, device='cuda')
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Size {size}x{size}", True)
            print(f"      Time: {elapsed:.3f}s, Output: {y.shape}")
        except Exception as e:
            record_result(f"Size {size}x{size}", False, str(e))

def test_large_sizes():
    """Test 2: Large Sizes (Standard ImageNet and Beyond)"""
    print("\n" + "=" * 80)
    print("Test 2: Large Sizes (Standard ImageNet and Beyond)")
    print("=" * 80)
    
    sizes = [96, 128, 224, 256, 384, 512]
    
    for size in sizes:
        try:
            start = time.time()
            x = torch.randn(1, 3, size, size, device='cuda')
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Size {size}x{size}", True)
            print(f"      Time: {elapsed:.3f}s")
        except Exception as e:
            record_result(f"Size {size}x{size}", False, str(e))

def test_channel_variations():
    """Test 3: Various Channel Configurations"""
    print("\n" + "=" * 80)
    print("Test 3: Various Channel Configurations (64x64)")
    print("=" * 80)
    
    configs = [
        (1, 16), (3, 64), (16, 32), (32, 64), (64, 128),
        (128, 256), (256, 512), (512, 1024)
    ]
    
    for in_ch, out_ch in configs:
        try:
            start = time.time()
            x = torch.randn(1, in_ch, 64, 64, device='cuda')
            conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Channels {in_ch}→{out_ch}", True)
            print(f"      Time: {elapsed:.3f}s")
        except Exception as e:
            record_result(f"Channels {in_ch}→{out_ch}", False, str(e))

def test_kernel_sizes():
    """Test 4: Different Kernel Sizes"""
    print("\n" + "=" * 80)
    print("Test 4: Different Kernel Sizes (64 channels, 64x64)")
    print("=" * 80)
    
    kernel_sizes = [1, 3, 5, 7, 9, 11]
    
    for k in kernel_sizes:
        try:
            start = time.time()
            x = torch.randn(1, 64, 64, 64, device='cuda')
            pad = k // 2
            conv = torch.nn.Conv2d(64, 64, k, padding=pad).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Kernel {k}x{k}", True)
            print(f"      Time: {elapsed:.3f}s")
        except Exception as e:
            record_result(f"Kernel {k}x{k}", False, str(e))

def test_batch_sizes():
    """Test 5: Different Batch Sizes"""
    print("\n" + "=" * 80)
    print("Test 5: Different Batch Sizes (3→64, 64x64)")
    print("=" * 80)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for bs in batch_sizes:
        try:
            start = time.time()
            x = torch.randn(bs, 3, 64, 64, device='cuda')
            conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Batch {bs}", True)
            print(f"      Time: {elapsed:.3f}s")
        except Exception as e:
            record_result(f"Batch {bs}", False, str(e))

def test_stride_and_dilation():
    """Test 6: Stride and Dilation Variations"""
    print("\n" + "=" * 80)
    print("Test 6: Stride and Dilation Variations (64x64)")
    print("=" * 80)
    
    configs = [
        (1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)
    ]
    
    for stride, dilation in configs:
        try:
            start = time.time()
            x = torch.randn(1, 64, 64, 64, device='cuda')
            conv = torch.nn.Conv2d(64, 64, 3, stride=stride, dilation=dilation, padding=dilation).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Stride={stride}, Dilation={dilation}", True)
            print(f"      Time: {elapsed:.3f}s, Output: {y.shape}")
        except Exception as e:
            record_result(f"Stride={stride}, Dilation={dilation}", False, str(e))

def test_groups():
    """Test 7: Grouped Convolutions"""
    print("\n" + "=" * 80)
    print("Test 7: Grouped Convolutions (64x64)")
    print("=" * 80)
    
    group_configs = [
        (64, 64, 1), (64, 64, 2), (64, 64, 4), (64, 64, 8),
        (128, 128, 16), (64, 64, 64)  # Depthwise
    ]
    
    for in_ch, out_ch, groups in group_configs:
        try:
            start = time.time()
            x = torch.randn(1, in_ch, 64, 64, device='cuda')
            conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"Groups={groups} ({in_ch}→{out_ch})", True)
            print(f"      Time: {elapsed:.3f}s")
        except Exception as e:
            record_result(f"Groups={groups} ({in_ch}→{out_ch})", False, str(e))

def test_multiple_layers():
    """Test 8: Multiple Sequential Layers"""
    print("\n" + "=" * 80)
    print("Test 8: Multiple Sequential Convolution Layers")
    print("=" * 80)
    
    layer_counts = [2, 4, 8, 16]
    
    for num_layers in layer_counts:
        try:
            start = time.time()
            x = torch.randn(1, 64, 64, 64, device='cuda')
            
            # Create sequential layers
            layers = []
            for i in range(num_layers):
                layers.append(torch.nn.Conv2d(64, 64, 3, padding=1))
                layers.append(torch.nn.ReLU())
            
            model = torch.nn.Sequential(*layers).cuda()
            y = model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            record_result(f"{num_layers} layers", True)
            print(f"      Time: {elapsed:.3f}s, Output: {y.shape}")
        except Exception as e:
            record_result(f"{num_layers} layers", False, str(e))

def test_memory_intensive():
    """Test 9: Memory Intensive Operations"""
    print("\n" + "=" * 80)
    print("Test 9: Memory Intensive Operations")
    print("=" * 80)
    
    configs = [
        (512, 512, 16, "512x512 with 16 channels"),
        (256, 256, 64, "256x256 with 64 channels"),
        (128, 128, 256, "128x128 with 256 channels"),
    ]
    
    for size, _, channels, desc in configs:
        try:
            start = time.time()
            x = torch.randn(1, channels, size, size, device='cuda')
            conv = torch.nn.Conv2d(channels, channels, 3, padding=1).cuda()
            y = conv(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Check memory
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            
            record_result(desc, True)
            print(f"      Time: {elapsed:.3f}s, Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        except Exception as e:
            record_result(desc, False, str(e))

def test_real_world_models():
    """Test 10: Real-World Model Components"""
    print("\n" + "=" * 80)
    print("Test 10: Real-World Model Components")
    print("=" * 80)
    
    # ResNet-like block
    print("\n   Testing ResNet-like block...")
    try:
        start = time.time()
        x = torch.randn(1, 64, 56, 56, device='cuda')
        
        # ResNet basic block
        conv1 = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        bn1 = torch.nn.BatchNorm2d(64).cuda()
        relu = torch.nn.ReLU().cuda()
        conv2 = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
        bn2 = torch.nn.BatchNorm2d(64).cuda()
        
        identity = x
        out = conv1(x)
        out = bn1(out)
        out = relu(out)
        out = conv2(out)
        out = bn2(out)
        out += identity
        out = relu(out)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        record_result("ResNet Basic Block", True)
        print(f"      Time: {elapsed:.3f}s, Output: {out.shape}")
    except Exception as e:
        record_result("ResNet Basic Block", False, str(e))
    
    # MobileNet-like depthwise separable
    print("\n   Testing MobileNet-like depthwise separable...")
    try:
        start = time.time()
        x = torch.randn(1, 32, 112, 112, device='cuda')
        
        # Depthwise
        dw_conv = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32).cuda()
        bn1 = torch.nn.BatchNorm2d(32).cuda()
        relu1 = torch.nn.ReLU().cuda()
        
        # Pointwise
        pw_conv = torch.nn.Conv2d(32, 64, 1).cuda()
        bn2 = torch.nn.BatchNorm2d(64).cuda()
        relu2 = torch.nn.ReLU().cuda()
        
        out = dw_conv(x)
        out = bn1(out)
        out = relu1(out)
        out = pw_conv(out)
        out = bn2(out)
        out = relu2(out)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        record_result("MobileNet Depthwise Separable", True)
        print(f"      Time: {elapsed:.3f}s, Output: {out.shape}")
    except Exception as e:
        record_result("MobileNet Depthwise Separable", False, str(e))

def test_stress():
    """Test 11: Stress Test - Repeated Operations"""
    print("\n" + "=" * 80)
    print("Test 11: Stress Test - 100 Repeated Operations")
    print("=" * 80)
    
    iterations = 100
    size = 64
    
    try:
        start = time.time()
        x = torch.randn(1, 3, size, size, device='cuda')
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
        
        for i in range(iterations):
            y = conv(x)
            torch.cuda.synchronize()
            if i % 20 == 0:
                print(f"      Iteration {i}/{iterations}...")
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations
        record_result(f"Stress test ({iterations} iterations)", True)
        print(f"      Total: {elapsed:.3f}s, Average: {avg_time*1000:.2f}ms per operation")
    except Exception as e:
        record_result(f"Stress test ({iterations} iterations)", False, str(e))

def test_mixed_operations():
    """Test 12: Mixed Operations in Single Forward Pass"""
    print("\n" + "=" * 80)
    print("Test 12: Mixed Operations in Single Forward Pass")
    print("=" * 80)
    
    try:
        start = time.time()
        x = torch.randn(2, 3, 224, 224, device='cuda')
        
        # Simulate a small CNN
        conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda()
        bn1 = torch.nn.BatchNorm2d(64).cuda()
        relu = torch.nn.ReLU().cuda()
        maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1).cuda()
        
        conv2 = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
        bn2 = torch.nn.BatchNorm2d(128).cuda()
        
        conv3 = torch.nn.Conv2d(128, 256, 3, padding=1).cuda()
        bn3 = torch.nn.BatchNorm2d(256).cuda()
        
        # Forward pass
        out = conv1(x)
        out = bn1(out)
        out = relu(out)
        out = maxpool(out)
        out = conv2(out)
        out = bn2(out)
        out = relu(out)
        out = conv3(out)
        out = bn3(out)
        out = relu(out)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        record_result("Small CNN (7 layers)", True)
        print(f"      Time: {elapsed:.3f}s, Output: {out.shape}")
    except Exception as e:
        record_result("Small CNN (7 layers)", False, str(e))

def print_summary():
    """Print test summary"""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    total = test_results['total']
    passed = len(test_results['passed'])
    failed = len(test_results['failed'])
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    if failed > 0:
        print("\n❌ Failed Tests:")
        for test_name, error in test_results['failed']:
            print(f"   - {test_name}")
            print(f"     Error: {error[:100]}")
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ IMPLICIT_GEMM solution is ROBUST and RELIABLE!")
        print("   - All sizes work (32x32 to 512x512)")
        print("   - All channel configurations work")
        print("   - All kernel sizes work")
        print("   - All batch sizes work")
        print("   - Grouped convolutions work")
        print("   - Real-world model components work")
        print("   - Stress test passed")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    # Run all tests
    test_basic_sizes()
    test_large_sizes()
    test_channel_variations()
    test_kernel_sizes()
    test_batch_sizes()
    test_stride_and_dilation()
    test_groups()
    test_multiple_layers()
    test_memory_intensive()
    test_real_world_models()
    test_stress()
    test_mixed_operations()
    
    # Print summary
    print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if len(test_results['failed']) == 0 else 1)
