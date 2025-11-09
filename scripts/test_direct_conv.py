#!/usr/bin/env python3
"""
Direct Conv2d test - Forces direct algorithm, no kernel search
"""
import os
import sys

# Force MIOpen to use Direct algorithm (no compilation needed)
os.environ['MIOPEN_FIND_ENFORCE'] = 'NONE'  # Disable all find operations
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '1'  # Force direct convolution
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '0'  # Disable iGEMM
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'  # Disable GEMM
os.environ['MIOPEN_DEBUG_CONV_WINOGRAD'] = '0'  # Disable Winograd
os.environ['MIOPEN_DEBUG_CONV_FFT'] = '0'  # Disable FFT
os.environ['MIOPEN_FIND_MODE'] = '3'  # Normal find mode but...
os.environ['MIOPEN_DEBUG_FIND_ONLY_SOLVER'] = 'ConvDirectNaiveConvFwd'  # Only use naive solver
os.environ['MIOPEN_ENABLE_LOGGING'] = '1'
os.environ['MIOPEN_LOG_LEVEL'] = '4'
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm-miopen-rdna1/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

print("Environment configured:")
print(f"  LD_LIBRARY_PATH includes: /opt/rocm-miopen-rdna1/lib")
print(f"  MIOPEN_DEBUG_FIND_ONLY_SOLVER: ConvDirectNaiveConvFwd")
print()

import torch
import torch.nn as nn

print("=" * 70)
print("RDNA1 Direct Convolution Test")
print("=" * 70)
print()

if not torch.cuda.is_available():
    print("❌ ERROR: ROCm not available")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"GPU Device: {device_name}")
print(f"PyTorch: {torch.__version__}")
print()

try:
    print("Running Conv2d with Direct algorithm (bypasses kernel compilation)...")
    print("-" * 70)
    
    # Small test
    model = nn.Conv2d(1, 2, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 1, 8, 8).cuda()
    
    print(f"Input: {x.shape}")
    print("Executing forward pass...")
    
    # This should use the direct naive implementation
    y = model(x)
    
    print(f"Output: {y.shape}")
    print()
    print("✅ Forward pass SUCCESSFUL!")
    print()
    
    # Test backward
    print("Testing backward pass...")
    loss = y.sum()
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print("✅ Backward pass SUCCESSFUL!")
    print()
    
    print("=" * 70)
    print("🎉 SUCCESS! RDNA1 GPU is working!")
    print("=" * 70)
    
except RuntimeError as e:
    print()
    print("❌ FAILED")
    print(f"Error: {e}")
    if "HSA" in str(e):
        print("\n⚠️  Memory aperture violation - patch may need adjustment")
    sys.exit(1)
