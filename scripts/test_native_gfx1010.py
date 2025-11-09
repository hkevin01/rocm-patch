#!/usr/bin/env python3
"""
Test Conv2d on native gfx1010 (RDNA1) without architecture spoofing
"""
import os
import torch
import time

print("="*60)
print("RDNA1 (gfx1010) Native Conv2d Test")
print("="*60)
print()

# System info
print("System Information:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Device props: {torch.cuda.get_device_properties(0)}")
print()

# Environment check
print("Environment Variables:")
for var in ['HSA_OVERRIDE_GFX_VERSION', 'MIOPEN_FIND_MODE', 'MIOPEN_WORKSPACE_LIMIT_MB',
            'MIOPEN_ENABLE_LOGGING', 'MIOPEN_LOG_LEVEL', 'MIOPEN_USER_DB_PATH',
            'MIOPEN_DISABLE_SOLVERS']:
    val = os.environ.get(var, '(not set)')
    print(f"  {var}: {val}")
print()

if not torch.cuda.is_available():
    print("❌ CUDA not available - cannot run test")
    exit(1)

# Test configurations
test_cases = [
    # (name, batch, in_c, out_c, h, w, kernel, stride, padding)
    ("Small 3x3", 1, 3, 16, 32, 32, 3, 1, 1),
    ("Medium 3x3", 4, 64, 128, 56, 56, 3, 1, 1),
    ("Large 3x3", 8, 128, 256, 28, 28, 3, 1, 1),
    ("1x1 conv", 4, 256, 512, 14, 14, 1, 1, 0),
    ("Strided 3x3", 4, 64, 128, 56, 56, 3, 2, 1),
]

print("Running Conv2d Tests:")
print("-" * 60)

results = []
for name, batch, in_c, out_c, h, w, kernel, stride, padding in test_cases:
    print(f"\nTest: {name}")
    print(f"  Shape: [{batch}, {in_c}, {h}, {w}] -> [{batch}, {out_c}, ?, ?]")
    print(f"  Kernel: {kernel}x{kernel}, stride={stride}, padding={padding}")
    
    try:
        # Create tensors on device
        x = torch.randn(batch, in_c, h, w, device="cuda", dtype=torch.float32)
        conv = torch.nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding).cuda()
        
        # Warmup
        torch.cuda.synchronize()
        _ = conv(x)
        torch.cuda.synchronize()
        
        # Timed run
        start = time.time()
        y = conv(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  ✅ SUCCESS! Output shape: {list(y.shape)}")
        print(f"  Time: {elapsed*1000:.2f} ms")
        results.append((name, "PASS", elapsed))
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results.append((name, "FAIL", str(e)))

print()
print("="*60)
print("Summary:")
print("="*60)
passed = sum(1 for _, status, _ in results if status == "PASS")
failed = sum(1 for _, status, _ in results if status == "FAIL")
print(f"Passed: {passed}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")
print()

if failed > 0:
    print("Failed tests:")
    for name, status, info in results:
        if status == "FAIL":
            print(f"  - {name}: {info}")
