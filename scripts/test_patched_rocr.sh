#!/bin/bash
set -e

echo "========================================="
echo "Testing Patched ROCr Runtime"
echo "========================================="
echo ""

# Test with LD_LIBRARY_PATH to avoid system-wide installation
export LD_LIBRARY_PATH=/opt/rocm-rdna1-rocr/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "🔍 Library path: $LD_LIBRARY_PATH"
echo "🔍 HSA override: $HSA_OVERRIDE_GFX_VERSION"
echo ""

# Simple test
python3 << 'PYTEST'
import torch
import sys

print("🧪 Testing basic CUDA availability...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("")

print("🧪 Testing simple tensor operations...")
try:
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = x + y
    print(f"✅ Tensor operations work: {z.shape}")
except Exception as e:
    print(f"❌ Tensor operations failed: {e}")
    sys.exit(1)

print("")
print("🧪 Testing Conv2d (the critical test)...")
print("⚠️  This may hang if the patch doesn't work...")

try:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Conv2d timed out!")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    x = torch.randn(1, 3, 32, 32).cuda()
    conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    y = conv(x)
    
    signal.alarm(0)  # Cancel alarm
    
    print(f"✅ SUCCESS! Conv2d works: {y.shape}")
    print("")
    print("🎉 All tests passed!")
    
except TimeoutError as e:
    print(f"❌ Conv2d timed out - patch may not be working")
    sys.exit(1)
except Exception as e:
    print(f"❌ Conv2d failed: {e}")
    sys.exit(1)
PYTEST
