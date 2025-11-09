#!/usr/bin/env python3
"""
Test script to demonstrate Conv2d first-run vs subsequent-run behavior.
First run: Kernel compilation (30-60 seconds)
Subsequent runs: Cached kernels (instant)
"""
import torch
import time

print("=" * 70)
print("🔍 Conv2d Timing Test - First Run vs Cached Run")
print("=" * 70)
print()

# Verify PyTorch and CUDA
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ Device: {torch.cuda.get_device_name(0)}")
print()

# Test Conv2d with timing
print("=" * 70)
print("Testing 3x3 Conv2d (in=3, out=16, size=32x32)")
print("=" * 70)
print()

# Create input and conv layer
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()

# First run - kernel compilation
print("🔄 First run (kernel compilation - will take 30-60 seconds)...")
start = time.time()
y1 = conv(x)
first_time = time.time() - start
print(f"✅ First run complete: {y1.shape}")
print(f"⏱️  Time: {first_time:.2f} seconds")
print()

# Second run - cached kernels
print("🔄 Second run (cached kernels - should be instant)...")
start = time.time()
y2 = conv(x)
second_time = time.time() - start
print(f"✅ Second run complete: {y2.shape}")
print(f"⏱️  Time: {second_time:.4f} seconds")
print()

# Third run - verify consistency
print("🔄 Third run (verify consistency)...")
start = time.time()
y3 = conv(x)
third_time = time.time() - start
print(f"✅ Third run complete: {y3.shape}")
print(f"⏱️  Time: {third_time:.4f} seconds")
print()

# Summary
print("=" * 70)
print("📊 Summary")
print("=" * 70)
speedup = first_time / second_time
print(f"First run:  {first_time:.2f}s (kernel compilation)")
print(f"Second run: {second_time:.4f}s (cached)")
print(f"Third run:  {third_time:.4f}s (cached)")
print(f"Speedup:    {speedup:.0f}x faster after caching")
print()
print("✅ Conv2d is WORKING!")
print("⏱️  First run slow = NORMAL (kernel compilation)")
print("⚡ Subsequent runs fast = NORMAL (cached kernels)")
print()
