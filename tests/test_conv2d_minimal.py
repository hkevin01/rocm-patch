#!/usr/bin/env python3
"""Minimal Conv2d test to reproduce the crash"""
import torch
import torch.nn as nn

print("=" * 70)
print("MINIMAL CONV2D TEST")
print("=" * 70)

# Check GPU
if not torch.cuda.is_available():
    print("❌ No GPU available")
    exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Try the exact operation that crashes
try:
    print("\n→ Creating Conv2d layer...")
    conv = nn.Conv2d(1, 32, (64, 1)).cuda()
    print("✅ Conv2d layer created")
    
    print("\n→ Creating input tensor...")
    x = torch.randn(16, 1, 64, 256).cuda()
    print("✅ Input tensor created")
    
    print("\n→ Running forward pass...")
    y = conv(x)
    print("✅ Forward pass completed")
    
    print("\n→ Squeezing tensor...")
    y = y.squeeze(2)
    print("✅ Squeeze completed")
    
    print("\n" + "=" * 70)
    print("✅ ALL OPERATIONS PASSED - GPU IS WORKING!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ CRASH: {e}")
    exit(1)
