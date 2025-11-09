#!/usr/bin/env python3
"""
Conv2d test using subprocess to enforce hard timeouts
Each test runs in isolated subprocess that can be killed
"""
import subprocess
import sys
import time

def run_single_test(in_ch, out_ch, size, timeout=15):
    """Run a single conv test in subprocess with hard timeout"""
    test_code = f'''
import torch
import torch.nn as nn
import time

try:
    conv = nn.Conv2d({in_ch}, {out_ch}, 3, padding=1).cuda()
    x = torch.randn(1, {in_ch}, {size}, {size}).cuda()
    
    start = time.time()
    with torch.no_grad():
        output = conv(x)
    elapsed = time.time() - start
    
    print(f"SUCCESS,{{elapsed:.4f}},{{output.shape}}")
except Exception as e:
    print(f"ERROR,{{str(e)}}")
'''
    
    try:
        result = subprocess.run(
            ['python3', '-c', test_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        output = result.stdout.strip()
        if output.startswith('SUCCESS'):
            parts = output.split(',')
            return 'SUCCESS', float(parts[1])
        elif output.startswith('ERROR'):
            return 'ERROR', output.split(',', 1)[1]
        else:
            return 'UNKNOWN', output
            
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', timeout
    except Exception as e:
        return 'EXCEPTION', str(e)

print("="*70)
print("Conv2d Test Suite with Subprocess Isolation")
print("="*70)

# Get device info
device_code = '''
import torch
print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")
'''
result = subprocess.run(['python3', '-c', device_code], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')
print(f"PyTorch: {lines[0]}")
print(f"Device: {lines[1]}")

tests = [
    ("Baseline 16→32, 32x32", 16, 32, 32, 20),
    ("Problematic 16→32, 48x48", 16, 32, 48, 20),
    ("Workaround: 15→31, 48x48", 15, 31, 48, 20),
    ("Workaround: 17→33, 48x48", 17, 33, 48, 20),
    ("Workaround: 16→32, 40x40", 16, 32, 40, 20),
    ("Large: 17→33, 64x64", 17, 33, 64, 20),
    ("Large: 31→63, 64x64", 31, 63, 64, 20),
]

results = []

for name, in_ch, out_ch, size, timeout in tests:
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"Config: {in_ch}→{out_ch}, {size}x{size}, timeout={timeout}s")
    print(f"{'='*70}")
    print("Running... ", end='', flush=True)
    
    status, info = run_single_test(in_ch, out_ch, size, timeout)
    
    if status == 'SUCCESS':
        print(f"✅ SUCCESS - {info:.4f}s")
        results.append((name, True, info))
    elif status == 'TIMEOUT':
        print(f"⏱️  TIMEOUT after {info}s")
        results.append((name, False, None))
    elif status == 'ERROR':
        print(f"❌ ERROR: {info}")
        results.append((name, False, None))
    else:
        print(f"❓ UNKNOWN: {info}")
        results.append((name, False, None))
    
    time.sleep(0.5)  # Brief pause between tests

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

working = [(name, time) for name, success, time in results if success]
failing = [name for name, success, _ in results if not success]

if working:
    print(f"\n✅ WORKING CONFIGURATIONS ({len(working)}):")
    for name, elapsed in working:
        print(f"   • {name:40} - {elapsed:.4f}s")

if failing:
    print(f"\n⏱️  FAILED/TIMEOUT CONFIGURATIONS ({len(failing)}):")
    for name in failing:
        print(f"   • {name}")

# Analysis
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

working_names = [name for name, _ in working]

if any('15→31' in name or '17→33' in name or '31→63' in name for name in working_names):
    print("\n✅ NON-POWER-OF-2 CHANNELS WORK!")
    print("   Solution: Modify model architectures to use odd channel counts")
    print("   Examples: 15, 17, 31, 33, 63, 127, 255")
elif any('40x40' in name for name in working_names):
    print("\n✅ NON-POWER-OF-2 SIZES WORK!")
    print("   Solution: Resize images to non-power-of-2 dimensions")
else:
    print("\n⚠️  SEVERE LIMITATIONS DETECTED")
    print("   ROCm 5.7 + RDNA1 has fundamental issues with larger convolutions")

if len(failing) > 4:
    print("\n⚠️  RECOMMENDATION:")
    print("   Multiple configurations fail - this is a deep hardware/driver issue")
    print("   Options:")
    print("   1. Restrict models to ≤32x32 feature maps")
    print("   2. Use CPU for problematic layers")
    print("   3. Downgrade OS to Ubuntu 22.04 + ROCm 5.2")
    print("   4. Upgrade GPU to RDNA2 or newer")

print("\n✅ Test complete!")
