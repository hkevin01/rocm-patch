#!/usr/bin/env python3
"""
Test Conv2d with MINIMAL ROCm environment
Remove all restrictive MIOpen settings and let it choose best algorithm
"""

import os
import sys
import subprocess
import time

def test_with_minimal_env(size, timeout=30):
    """Test a specific size with minimal ROCm environment (no MIOpen restrictions)"""

    # Create test script to run in subprocess
    test_script = f"""
import sys
import os

# Remove ALL MIOPEN_DEBUG restrictions
for key in list(os.environ.keys()):
    if key.startswith('MIOPEN_DEBUG'):
        del os.environ[key]

# Now import torch (after environment setup)
import torch

size = {size}

try:
    x = torch.randn(1, 3, size, size, device='cuda')
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    torch.cuda.synchronize()
    print(f"SUCCESS: {{y.shape}}")
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy()
        )

        if result.returncode == 0 and 'SUCCESS' in result.stdout:
            return ('SUCCESS', result.stdout.strip())
        else:
            return ('ERROR', result.stderr.strip())

    except subprocess.TimeoutExpired:
        return ('TIMEOUT', f'Exceeded {timeout}s')
    except Exception as e:
        return ('ERROR', str(e))

def main():
    print("=" * 70)
    print("🧪 Testing Conv2d with MINIMAL ROCm Environment")
    print("=" * 70)
    print("\nRemoving ALL MIOpen restrictions")
    print("Letting MIOpen choose best algorithm automatically\n")

    # Clear cache first
    print("Clearing MIOpen cache...")
    cache_dir = os.path.expanduser('~/.cache/miopen')
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared\n")

    # Test critical sizes
    test_sizes = [32, 40, 42, 44, 48, 56, 64, 128]

    results = {}

    for size in test_sizes:
        print(f"Testing {size}x{size}...", end=' ', flush=True)
        start = time.time()
        status, info = test_with_minimal_env(size, timeout=30)
        elapsed = time.time() - start

        results[size] = (status, elapsed)

        if status == 'SUCCESS':
            print(f"✅ {elapsed:.2f}s")
        elif status == 'TIMEOUT':
            print(f"⏱️  TIMEOUT ({elapsed:.2f}s)")
        else:
            print(f"❌ ERROR")
            if info:
                print(f"   {info[:200]}")

    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print()

    success_sizes = [s for s, (st, _) in results.items() if st == 'SUCCESS']
    timeout_sizes = [s for s, (st, _) in results.items() if st == 'TIMEOUT']
    error_sizes = [s for s, (st, _) in results.items() if st == 'ERROR']

    if success_sizes:
        print(f"✅ SUCCESS sizes: {success_sizes}")
    if timeout_sizes:
        print(f"⏱️  TIMEOUT sizes: {timeout_sizes}")
    if error_sizes:
        print(f"❌ ERROR sizes: {error_sizes}")

    print("\n" + "=" * 70)
    print("🔍 CONCLUSION")
    print("=" * 70)
    print()

    if success_sizes:
        max_success = max(success_sizes)
        if max_success > 42:
            print(f"🎉 BREAKTHROUGH! Minimal environment extends boundary to {max_success}x{max_success}!")
            print(f"   Previous restrictive settings were CAUSING the 42x42 limitation!")
        elif max_success == 42:
            print("⚠️  Boundary still at 42x42 - minimal environment didn't help")
        else:
            print(f"⚠️  Boundary decreased to {max_success}x{max_success} - restrictive settings were helping")
    else:
        print("⚠️  Inconclusive results - check errors above")

    return results

if __name__ == '__main__':
    results = main()
