#!/usr/bin/env python3
"""
Complete ROCm Patch Setup Example
==================================

This demonstrates the CORRECT order of operations for ROCm patches.
"""

import sys
import os

# Add patches to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 70)
print("Complete ROCm Setup - Correct Order")
print("=" * 70)
print()

# STEP 1: Setup multiprocessing BEFORE importing anything
print("Step 1: Setup multiprocessing (BEFORE torch import)...")
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
print("✓ Multiprocessing set to 'spawn'")
print()

# STEP 2: Setup environment variables BEFORE importing torch
print("Step 2: Setup environment variables...")
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_USE_SVM'] = '0'
os.environ['HSA_XNACK'] = '0'
print("✓ Environment configured")
print()

# STEP 3: Now import torch
print("Step 3: Import PyTorch...")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
print()

# STEP 4: Patch DataLoader
print("Step 4: Patch DataLoader...")
from patches import patch_dataloader
patch_dataloader()
print()

# STEP 5: Enable MIOpen bypass
print("Step 5: Enable MIOpen bypass...")
from patches.miopen_bypass.conv2d_fallback import (
    enable_miopen_bypass,
    FallbackStrategy,
    SafeConv2d  # Use SafeConv2d instead of nn.Conv2d
)
enable_miopen_bypass(strategy=FallbackStrategy.AUTO)
print()

# Now everything is ready!
print("=" * 70)
print("✅ All patches active - Ready to train!")
print("=" * 70)
print()

# Test 1: Conv2d works (using SafeConv2d)
print("Test 1: Conv2d operation (with GPU bypass)...")
conv = SafeConv2d(3, 64, kernel_size=3, padding=1).cuda()
x = torch.randn(2, 3, 64, 64).cuda()
y = conv(x)
print(f"✓ Conv2d: {x.shape} → {y.shape}")
print()

# Test 2: DataLoader with workers
print("Test 2: DataLoader with 4 workers...")
dataset = TensorDataset(
    torch.randn(50, 3, 32, 32),
    torch.randint(0, 10, (50,))
)
loader = DataLoader(dataset, batch_size=8, num_workers=4)
batch_x, batch_y = next(iter(loader))
print(f"✓ DataLoader: batch shape {batch_x.shape}, {len(loader)} batches")
print()

# Test 3: Training loop (using SafeConv2d)
print("Test 3: Training loop simulation...")
model = nn.Sequential(
    SafeConv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    SafeConv2d(16, 10, 3, padding=1),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i, (batch_x, batch_y) in enumerate(loader):
    if i >= 2:
        break

    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()

    output = model(batch_x)
    loss = torch.nn.functional.cross_entropy(output, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Batch {i+1}: loss={loss.item():.4f}")

print()
print("=" * 70)
print("✅ Everything working correctly!")
print("=" * 70)
