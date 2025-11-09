#!/usr/bin/env python3
"""
Safe IMPLICIT_GEMM testing with subprocess isolation and timeouts
"""
import os
import sys
import subprocess
import time

def test_size_with_timeout(size, timeout=15):
    """Test a specific size in isolated subprocess with timeout"""
    test_code = f"""
import os
import torch
import time

# Configure IMPLICIT_GEMM
os.environ['MIOPEN_DEBUG_CONV_GEMM'] = '0'
os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
os.environ['MIOPEN_DEBUG_CONV_WINOGRAD'] = '0'
os.environ['MIOPEN_DEBUG_CONV_DIRECT'] = '0'
os.environ['MIOPEN_DEBUG_CONV_FFT'] = '0'

try:
    start = time.time()
    x = torch.randn(1, 3, {size}, {size}, device='cuda')
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"SUCCESS:{size}:{elapsed:.3f}:{y.shape}")
except Exception as e:
    print(f"ERROR:{size}:{{str(e)}}")
    import sys
    sys.exit(1)
"""
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0 and 'SUCCESS' in result.stdout:
            parts = result.stdout.strip().split(':')
            return ('SUCCESS', float(parts[2]), parts[3])
        else:
            return ('ERROR', 0, result.stderr.strip()[:100])
    except subprocess.TimeoutExpired:
        return ('TIMEOUT', timeout, f'Exceeded {timeout}s')
    except Exception as e:
        return ('ERROR', 0, str(e))

print("=" * 80)
print("🔒 SAFE IMPLICIT_GEMM TEST WITH SUBPROCESS ISOLATION")
print("=" * 80)

# Test critical sizes that were previously failing
test_sizes = [32, 40, 42, 43, 44, 45, 48, 56, 64, 96, 128, 224]

results = {'passed': [], 'failed': [], 'timeout': []}

for size in test_sizes:
    print(f"\nTesting {size}x{size}...", end=' ', flush=True)
    status, time_val, info = test_size_with_timeout(size, timeout=15)
    
    if status == 'SUCCESS':
        print(f"✅ ({time_val:.3f}s)")
        results['passed'].append(size)
    elif status == 'TIMEOUT':
        print(f"⏱️  TIMEOUT")
        results['timeout'].append(size)
    else:
        print(f"❌ ERROR")
        print(f"   {info}")
        results['failed'].append(size)
    
    # Pause between tests to prevent issues
    time.sleep(0.5)

print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)
print(f"✅ Passed: {len(results['passed'])} sizes - {results['passed']}")
print(f"❌ Failed: {len(results['failed'])} sizes - {results['failed']}")
print(f"⏱️  Timeout: {len(results['timeout'])} sizes - {results['timeout']}")

if len(results['failed']) > 0 or len(results['timeout']) > 0:
    print("\n⚠️  IMPLICIT_GEMM solution needs investigation!")
    sys.exit(1)
else:
    print("\n🎉 ALL TESTS PASSED! IMPLICIT_GEMM is working correctly!")
    sys.exit(0)
