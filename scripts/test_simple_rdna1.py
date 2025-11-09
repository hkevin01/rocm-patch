#!/usr/bin/env python3
"""
Simple RDNA1 GPU Test - Bypasses MIOpen kernel compilation
"""
import os
import sys

# Disable MIOpen kernel compilation and use immediate mode
os.environ['MIOPEN_FIND_MODE'] = '1'  # Use immediate mode (no compilation)
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '1'
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '3'
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm-miopen-rdna1/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn

print("=" * 70)
print("RDNA1 GPU Test - Simple Forward Pass (Immediate Mode)")
print("=" * 70)
print()

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA/ROCm not available")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"✓ GPU Device: {device_name}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Using patched MIOpen from: /opt/rocm-miopen-rdna1/lib")
print()

try:
    print("Test 1: Tiny Conv2d (should work immediately)")
    print("-" * 70)
    
    # Very small convolution
    model = nn.Conv2d(1, 2, kernel_size=1).cuda()  # 1x1 conv (simplest)
    x = torch.randn(1, 1, 4, 4).cuda()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Running forward pass...")
    
    y = model(x)
    
    print(f"  Output shape: {y.shape}")
    print("  ✅ TINY Conv2d SUCCESSFUL!")
    print()
    
    print("Test 2: Small Conv2d")
    print("-" * 70)
    
    model2 = nn.Conv2d(1, 8, kernel_size=3, padding=1).cuda()
    x2 = torch.randn(1, 1, 8, 8).cuda()
    
    print(f"  Input shape: {x2.shape}")
    print(f"  Running forward pass...")
    
    y2 = model2(x2)
    
    print(f"  Output shape: {y2.shape}")
    print("  ✅ SMALL Conv2d SUCCESSFUL!")
    print()
    
    print("Test 3: Backward Pass")
    print("-" * 70)
    
    loss = y2.sum()
    print(f"  Computing gradients...")
    loss.backward()
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient exists: {model2.weight.grad is not None}")
    print("  ✅ Backward Pass SUCCESSFUL!")
    print()
    
    print("=" * 70)
    print("🎉 BASIC TESTS PASSED!")
    print("=" * 70)
    print()
    print("The patched MIOpen is working for basic operations.")
    print("Your RDNA1 GPU can run Conv2d operations!")
    print()
    
except RuntimeError as e:
    print()
    print("=" * 70)
    print("❌ TEST FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    if "HSA_STATUS_ERROR" in str(e):
        print("⚠️  HSA error detected - this suggests:")
        print("   - The patch might not be working correctly")
        print("   - RDNA1 detection may not be triggering")
        print("   - Memory allocation issue persists")
    sys.exit(1)
except Exception as e:
    print()
    print("=" * 70)
    print("❌ UNEXPECTED ERROR")
    print("=" * 70)
    print(f"Error: {e}")
    sys.exit(1)
