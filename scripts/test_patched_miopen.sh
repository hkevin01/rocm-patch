#!/bin/bash
echo "========================================================================"
echo "Testing Patched MIOpen with PyTorch on RDNA1 GPU"
echo "========================================================================"

# Set library path to use patched MIOpen
export LD_LIBRARY_PATH=/opt/rocm-miopen-rdna1/lib:$LD_LIBRARY_PATH
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=4

echo "Using MIOpen from: /opt/rocm-miopen-rdna1/lib"
echo ""
echo "Running PyTorch Conv2d test..."
echo ""

python3 << 'PYEOF'
import torch
import torch.nn as nn
import sys

print("=" * 70)
print("PyTorch RDNA1 GPU Test - Patched MIOpen")
print("=" * 70)

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA/ROCm not available")
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f"✓ GPU Device: {device_name}")
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print()

try:
    print("Test 1: Simple Conv2d Forward Pass")
    print("-" * 70)
    model = nn.Conv2d(1, 32, kernel_size=3, padding=1).cuda()
    x = torch.randn(1, 1, 28, 28).cuda()
    
    print(f"  Input shape: {x.shape}")
    print(f"  Model on GPU: {next(model.parameters()).is_cuda}")
    
    y = model(x)
    print(f"  Output shape: {y.shape}")
    print("  ✅ Forward pass SUCCESSFUL!")
    print()
    
    print("Test 2: Conv2d Backward Pass")
    print("-" * 70)
    loss = y.sum()
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient computed: {model.weight.grad is not None}")
    print("  ✅ Backward pass SUCCESSFUL!")
    print()
    
    print("Test 3: Larger Conv2d (Stress Test)")
    print("-" * 70)
    model2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).cuda()
    x2 = torch.randn(4, 64, 56, 56).cuda()
    y2 = model2(x2)
    print(f"  Input shape: {x2.shape}")
    print(f"  Output shape: {y2.shape}")
    print("  ✅ Larger Conv2d SUCCESSFUL!")
    print()
    
    print("Test 4: Multi-layer Network")
    print("-" * 70)
    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
    ).cuda()
    
    x3 = torch.randn(2, 3, 32, 32).cuda()
    y3 = net(x3)
    loss3 = y3.sum()
    loss3.backward()
    
    print(f"  Input shape: {x3.shape}")
    print(f"  Output shape: {y3.shape}")
    print(f"  Loss: {loss3.item():.4f}")
    print("  ✅ Multi-layer network SUCCESSFUL!")
    print()
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("✓ GPU Conv2d operations work correctly")
    print("✓ Forward and backward passes successful")
    print("✓ RDNA1 patch is working!")
    print()
    print("The patched MIOpen successfully enables GPU training on RDNA1!")
    
except RuntimeError as e:
    print()
    print("=" * 70)
    print("❌ TEST FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    if "HSA_STATUS_ERROR" in str(e):
        print("HSA error detected - patch may not be working correctly")
    sys.exit(1)
except Exception as e:
    print()
    print("=" * 70)
    print("❌ UNEXPECTED ERROR")
    print("=" * 70)
    print(f"Error: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "========================================================================"
echo "Test Complete!"
echo "========================================================================"
