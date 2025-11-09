#!/usr/bin/env python3
"""
Quick Conv2d test with aggressive timeouts - test each config in isolated runs
"""
import torch
import torch.nn as nn
import time
import signal
import sys

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout!")

def quick_test(name, in_ch, out_ch, size, timeout=20):
    """Ultra-fast test with aggressive timeout"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"Config: {in_ch}→{out_ch}, {size}x{size}, timeout={timeout}s")
    print(f"{'='*60}")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        print("Creating layers...")
        conv = nn.Conv2d(in_ch, out_ch, 3, padding=1).cuda()
        x = torch.randn(1, in_ch, size, size).cuda()
        
        print("Running convolution (single pass)...")
        start = time.time()
        with torch.no_grad():
            output = conv(x)
        elapsed = time.time() - start
        
        signal.alarm(0)
        print(f"✅ SUCCESS - {elapsed:.4f}s")
        return True
        
    except TimeoutException:
        signal.alarm(0)
        print(f"⏱️  TIMEOUT after {timeout}s - HANGING")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"❌ ERROR: {e}")
        return False

print("="*70)
print("QUICK Conv2d Test Suite - RDNA1 (gfx1010)")
print("="*70)
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

results = {}

# Test 1: Baseline small (should work)
print("\n" + "="*70)
print("BASELINE: Small conv (should work)")
print("="*70)
results['16→32, 32x32'] = quick_test("Baseline", 16, 32, 32, timeout=20)

# Test 2: Known problematic size
print("\n" + "="*70)
print("PROBLEMATIC: 16→32, 48x48 (known to hang)")
print("="*70)
results['16→32, 48x48'] = quick_test("Power-of-2, large", 16, 32, 48, timeout=20)

# Test 3: Non-power-of-2 channels
print("\n" + "="*70)
print("WORKAROUND 1: Non-power-of-2 channels")
print("="*70)
results['15→31, 48x48'] = quick_test("Odd channels", 15, 31, 48, timeout=20)

# Test 4: Non-power-of-2 size
print("\n" + "="*70)
print("WORKAROUND 2: Non-power-of-2 size")
print("="*70)
results['16→32, 40x40'] = quick_test("Odd size", 16, 32, 40, timeout=20)

# Test 5: Another odd channel combo
print("\n" + "="*70)
print("WORKAROUND 3: Different odd channels")
print("="*70)
results['17→33, 64x64'] = quick_test("17→33", 17, 33, 64, timeout=20)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

working = [k for k, v in results.items() if v]
hanging = [k for k, v in results.items() if not v]

if working:
    print(f"\n✅ WORKING ({len(working)}):")
    for config in working:
        print(f"   • {config}")

if hanging:
    print(f"\n⏱️  HANGING ({len(hanging)}):")
    for config in hanging:
        print(f"   • {config}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if '15→31, 48x48' in working or '17→33, 64x64' in working:
    print("✅ Non-power-of-2 CHANNELS work as a workaround!")
elif '16→32, 40x40' in working:
    print("✅ Non-power-of-2 SIZES work as a workaround!")
elif len(hanging) > 3:
    print("⚠️  SEVERE: Most configurations hang on this hardware")
    print("   ROCm 5.7 has fundamental issues with RDNA1 for larger convs")
    print("\n   Options:")
    print("   1. Use ONLY small feature maps (≤32x32)")
    print("   2. Move problematic layers to CPU")
    print("   3. Downgrade to Ubuntu 22.04 + ROCm 5.2")
    print("   4. Upgrade to RDNA2/RDNA3 GPU")
else:
    print("✅ Some workarounds available - see working configs above")

print("\n✅ Quick test complete!")
