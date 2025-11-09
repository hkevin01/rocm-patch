#!/usr/bin/env python3
"""
Test Conv2d WITHOUT restrictive MIOpen environment variables
Let MIOpen choose the best algorithm automatically
"""

import os
import sys
import subprocess
import time

def run_single_test(size, timeout=30):
    """Run a single conv test in subprocess with timeout"""
    test_code = f"""
import torch
import torch.nn as nn

# Clear MIOpen restrictive flags - let it choose best algorithm
if 'MIOPEN_DEBUG_CONV_GEMM' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_GEMM']
if 'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM']
if 'MIOPEN_DEBUG_CONV_WINOGRAD' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_WINOGRAD']
if 'MIOPEN_DEBUG_CONV_DIRECT' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_DIRECT']
if 'MIOPEN_DEBUG_CONV_FFT' in os.environ:
    del os.environ['MIOPEN_DEBUG_CONV_FFT']

print(f"Testing {size}x{size} without MIOpen restrictions...")

conv = nn.Conv2d(16, 32, kernel_size=3, padding=1).cuda()
x = torch.randn(1, 16, {size}, {size}).cuda()
y = conv(x)
print(f"✅ SUCCESS - Output shape: {{y.shape}}")
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            elapsed = result.stdout.split('\n')
            return ('SUCCESS', elapsed)
        else:
            return ('ERROR', result.stderr)
    except subprocess.TimeoutExpired:
        return ('TIMEOUT', f'Exceeded {timeout}s')
    except Exception as e:
        return ('ERROR', str(e))

def main():
    print("=" * 70)
    print("Testing Conv2d WITHOUT MIOpen Restrictions")
    print("Let MIOpen choose the best algorithm automatically")
    print("=" * 70)
    
    # Test sizes that were hanging before
    test_sizes = [32, 40, 42, 44, 48, 56, 64]
    
    results = {}
    
    for size in test_sizes:
        print(f"\nTesting {size}x{size}...")
        start = time.time()
        status, info = run_single_test(size, timeout=30)
        elapsed = time.time() - start
        
        results[size] = (status, elapsed)
        
        if status == 'SUCCESS':
            print(f"   ✅ SUCCESS - {elapsed:.2f}s")
        elif status == 'TIMEOUT':
            print(f"   ⏱️  TIMEOUT - {elapsed:.2f}s")
        else:
            print(f"   ❌ ERROR - {info}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    working = [sz for sz, (st, _) in results.items() if st == 'SUCCESS']
    hanging = [sz for sz, (st, _) in results.items() if st == 'TIMEOUT']
    
    if working:
        print(f"✅ Working: {working}")
        print(f"   Max: {max(working)}x{max(working)}")
    
    if hanging:
        print(f"⏱️  Hanging: {hanging}")
        print(f"   Min hanging: {min(hanging)}x{min(hanging)}")
    
    print("\nConclusion:")
    if len(hanging) < len(test_sizes) // 2:
        print("🎉 Removing MIOpen restrictions HELPED!")
    else:
        print("⚠️  No improvement - issue persists")

if __name__ == "__main__":
    main()
