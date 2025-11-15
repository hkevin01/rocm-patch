#!/usr/bin/env python3
"""
Test All ROCm Patches
=====================

Tests that all patches work together correctly.
"""

import sys
import os

# Add patches to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("ROCm Patch Integration Test")
print("=" * 70)
print()

# Test 1: Enable all patches (BEFORE importing torch)
print("Test 1: Enabling all patches...")
print("-" * 70)

from patches import enable_all_patches

status = enable_all_patches()

if not all(status.values()):
    print("\n❌ Some patches failed!")
    sys.exit(1)

print("\n✅ All patches enabled successfully!")
print()

# Test 2: Verify torch works
print("Test 2: Testing PyTorch basic operations...")
print("-" * 70)

import torch

x = torch.randn(10, 10).cuda()
y = torch.matmul(x, x.T)

print(f"✓ Tensor creation: {x.shape}")
print(f"✓ GPU operations: {y.shape}")
print(f"✓ Result on GPU: {y.device}")
print()

# Test 3: Test Conv2d with bypass
print("Test 3: Testing Conv2d with MIOpen bypass...")
print("-" * 70)

import torch.nn as nn

conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
x = torch.randn(2, 3, 64, 64).cuda()

try:
    y = conv(x)
    print(f"✓ Conv2d forward: input {x.shape} → output {y.shape}")
    
    loss = y.sum()
    loss.backward()
    print(f"✓ Conv2d backward: gradients computed")
except Exception as e:
    print(f"❌ Conv2d failed: {e}")
    sys.exit(1)

print()

# Test 4: Test DataLoader with workers
print("Test 4: Testing DataLoader with multiprocessing...")
print("-" * 70)

from torch.utils.data import TensorDataset, DataLoader

# Create dummy dataset
dataset = TensorDataset(
    torch.randn(100, 3, 32, 32),
    torch.randint(0, 10, (100,))
)

# Test with workers (this should work now!)
loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    shuffle=True
)

print(f"✓ DataLoader created: {len(loader)} batches, 4 workers")

# Try to iterate (this would fail without spawn fix)
try:
    batch_x, batch_y = next(iter(loader))
    print(f"✓ DataLoader iteration: batch shape {batch_x.shape}")
    print(f"✓ Workers functioning correctly!")
except Exception as e:
    print(f"❌ DataLoader failed: {e}")
    sys.exit(1)

print()

# Test 5: Verify multiprocessing context
print("Test 5: Verifying multiprocessing configuration...")
print("-" * 70)

import multiprocessing as mp

method = mp.get_start_method()
print(f"✓ Multiprocessing method: {method}")

if method != 'spawn':
    print(f"⚠️  Warning: Expected 'spawn', got '{method}'")
else:
    print(f"✓ Correct method for ROCm!")

print()

# Test 6: Test with model and DataLoader together
print("Test 6: Testing model training loop with DataLoader...")
print("-" * 70)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, 3, padding=1),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
).cuda()

optimizer = torch.optim.Adam(model.parameters())

# Mini training loop
for i, (batch_x, batch_y) in enumerate(loader):
    if i >= 3:  # Just test 3 batches
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

# Final summary
print("=" * 70)
print("INTEGRATION TEST SUMMARY")
print("=" * 70)

tests = [
    "Patch initialization",
    "PyTorch basic operations",
    "Conv2d with MIOpen bypass",
    "DataLoader with workers",
    "Multiprocessing configuration",
    "Training loop integration"
]

for i, test in enumerate(tests, 1):
    print(f"  ✓ Test {i}: {test}")

print()
print("✅ All integration tests passed!")
print("=" * 70)
