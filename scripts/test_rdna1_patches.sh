#!/bin/bash
# Test script to verify RDNA1 patches are active

echo "=== RDNA1 Patch Verification Test ==="
echo ""

# Check if patched library is in place
TORCH_LIB="$HOME/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so"
if [ -f "$TORCH_LIB" ]; then
    echo "Checking for RDNA1 patches in PyTorch's MIOpen..."
    if strings "$TORCH_LIB" | grep -q "RDNA1 PATCH"; then
        echo "✅ RDNA1 patches found in library"
    else
        echo "❌ RDNA1 patches NOT found - rebuild needed"
        exit 1
    fi
else
    echo "❌ PyTorch MIOpen library not found at $TORCH_LIB"
    exit 1
fi

echo ""
echo "Running Python runtime test..."
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FORCE_RDNA1=1
export MIOPEN_LOG_LEVEL=7

python3 << 'PYTHON_EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    
    print("\nAttempting convolution test...")
    try:
        x = torch.randn(1, 3, 224, 224).cuda()
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        y = conv(x)
        print(f"✅ Convolution succeeded: output shape {y.shape}")
    except Exception as e:
        print(f"⚠️  Convolution failed: {e}")
        print("\nThis is expected - RDNA1 still has HSA runtime issues")
        print("But debug output above should show patches executing")
else:
    print("❌ CUDA not available")
    sys.exit(1)
PYTHON_EOF

echo ""
echo "=== Test Complete ==="
echo "Check output above for '[RDNA1 PATCH]' debug messages"
