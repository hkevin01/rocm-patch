#!/usr/bin/env python3
"""Find exact size boundary where convolutions start hanging"""
import subprocess

def test_size(size, timeout=15):
    """Test 16→32 conv at given size"""
    code = f'''
import torch
import torch.nn as nn
conv = nn.Conv2d(16, 32, 3, padding=1).cuda()
x = torch.randn(1, 16, {size}, {size}).cuda()
with torch.no_grad():
    output = conv(x)
print("OK")
'''
    try:
        result = subprocess.run(['python3', '-c', code], 
                              capture_output=True, text=True, timeout=timeout)
        return 'OK' in result.stdout
    except subprocess.TimeoutExpired:
        return False

print("="*70)
print("Finding Size Boundary for Conv2d Hangs (16→32 channels)")
print("="*70)

# Test sizes from 32 to 64
sizes_to_test = [32, 36, 40, 42, 44, 46, 48, 50, 52, 56, 60, 64]

results = {}
for size in sizes_to_test:
    print(f"Testing {size}x{size}... ", end='', flush=True)
    success = test_size(size, timeout=15)
    results[size] = success
    print("✅ OK" if success else "⏱️  HANG")

print("\n" + "="*70)
print("RESULTS")
print("="*70)

working_sizes = [s for s, ok in results.items() if ok]
hanging_sizes = [s for s, ok in results.items() if not ok]

if working_sizes:
    print(f"\n✅ Working sizes: {working_sizes}")
    print(f"   Max working: {max(working_sizes)}x{max(working_sizes)}")

if hanging_sizes:
    print(f"\n⏱️  Hanging sizes: {hanging_sizes}")
    print(f"   Min hanging: {min(hanging_sizes)}x{min(hanging_sizes)}")

if working_sizes and hanging_sizes:
    boundary = max(working_sizes)
    print(f"\n🎯 BOUNDARY FOUND: {boundary}x{boundary} works, {min(hanging_sizes)}x{min(hanging_sizes)} hangs")
    print(f"   Safe range: ≤{boundary}x{boundary}")

print("\n✅ Boundary test complete!")
