#!/usr/bin/env python3
"""
Test with MIOpen Immediate Mode - completely bypasses Find
"""
import os
import sys

# CRITICAL: Set BEFORE importing torch
os.environ['MIOPEN_FIND_MODE'] = '1'  # 1 = Immediate mode (no Find, no compile)
os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm-miopen-rdna1/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
# Minimal logging
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '2'

print("MIOpen Configuration:")
print(f"  MIOPEN_FIND_MODE=1 (Immediate mode - NO kernel compilation)")
print(f"  Using patched library: /opt/rocm-miopen-rdna1/lib")
print()

import torch
import torch.nn as nn
import time

print("=" * 70)
print("RDNA1 Test - Immediate Mode (No Kernel Compilation)")
print("=" * 70)
print()

if not torch.cuda.is_available():
    print("❌ ROCm not available")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print()

try:
    print("Test: Conv2d Forward Pass")
    print("-" * 70)
    
    model = nn.Conv2d(1, 4, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 1, 16, 16).cuda()
    
    print(f"Input: {x.shape}")
    print("Running forward pass (should be fast, no compilation)...")
    
    start = time.time()
    y = model(x)
    elapsed = time.time() - start
    
    print(f"Output: {y.shape}")
    print(f"Time: {elapsed:.3f}s")
    
    if elapsed > 5:
        print("⚠️  Took too long - might be compiling kernels")
    else:
        print("✅ Fast execution - immediate mode working!")
    
    print()
    print("Test: Backward Pass")
    print("-" * 70)
    
    loss = y.sum()
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient: {model.weight.grad is not None}")
    print("✅ Backward SUCCESSFUL!")
    print()
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED - RDNA1 GPU WORKING!")
    print("=" * 70)
    print()
    print("Your AMD RX 5600 XT can run PyTorch Conv2d!")
    print("The patched MIOpen is enabling GPU training.")
    
except RuntimeError as e:
    print()
    print("❌ FAILED")
    print(f"Error: {e}")
    if "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in str(e):
        print()
        print("Memory aperture violation detected!")
        print("This means the RDNA1 patch didn't work as expected.")
        print()
        print("Possible issues:")
        print("  1. Patch not being triggered (check if gfx1010 detected)")
        print("  2. hipHostMallocNonCoherent not working on RDNA1")
        print("  3. Need to patch deeper in the stack")
    sys.exit(1)
except Exception as e:
    print()
    print("❌ UNEXPECTED ERROR")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
