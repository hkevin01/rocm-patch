"""
Simple Functional Test for MIOpen Bypass
========================================

Tests that the bypass works for real-world scenarios.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv2d_fallback import (
    SafeConv2d,
    Conv2dBypassConfig,
    FallbackStrategy,
    enable_miopen_bypass,
    patch_model
)

def test_cpu_fallback_basic():
    """Test basic CPU fallback works"""
    print("Test 1: Basic CPU fallback...")
    try:
        config = Conv2dBypassConfig(
            strategy=FallbackStrategy.CPU_FALLBACK,
            verbose=True
        )
        
        conv = SafeConv2d(3, 64, kernel_size=3, padding=1, config=config).cuda()
        x = torch.randn(1, 3, 44, 44).cuda()
        
        # This should work via CPU fallback
        y = conv(x)
        
        # Test gradient flow
        loss = y.sum()
        loss.backward()
        
        stats = conv.get_bypass_stats()
        print(f"  ✅ Forward pass works: output shape {y.shape}")
        print(f"  ✅ Gradients computed: weight.grad shape {conv.weight.grad.shape}")
        print(f"  ✅ Bypass stats: {stats['bypass_count']}/{stats['total_forwards']} bypassed")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def test_auto_strategy():
    """Test AUTO strategy (IMPLICIT_GEMM + CPU fallback)"""
    print("\nTest 2: AUTO strategy...")
    try:
        # Enable AUTO strategy
        enable_miopen_bypass(strategy=FallbackStrategy.AUTO)
        
        # Test multiple sizes
        sizes = [32, 64, 128, 224]
        for size in sizes:
            x = torch.randn(1, 3, size, size).cuda()
            conv = SafeConv2d(3, 64, kernel_size=3, padding=1).cuda()
            y = conv(x)
            print(f"  ✅ Size {size}×{size}: output shape {y.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def test_model_patching():
    """Test patching a real model"""
    print("\nTest 3: Model patching...")
    try:
        # Create a simple ResNet-like block
        class ResNetBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=1)
            
            def forward(self, x):
                identity = self.conv3(x)
                out = self.conv1(x)
                out = self.conv2(out)
                # Pad to match dimensions
                if out.size(1) != identity.size(1):
                    return identity
                return out + identity
        
        model = ResNetBlock()
        
        # Patch with CPU_FALLBACK
        config = Conv2dBypassConfig(strategy=FallbackStrategy.CPU_FALLBACK)
        patch_model(model, config)
        
        model = model.cuda()
        
        # Test forward pass
        x = torch.randn(1, 64, 44, 44).cuda()
        y = model(x)
        
        # Check that layers were patched
        patched_count = sum(1 for m in model.modules() if isinstance(m, SafeConv2d))
        print(f"  ✅ Patched {patched_count} Conv2d layers")
        print(f"  ✅ Forward pass works: output shape {y.shape}")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_selective_bypass():
    """Test SELECTIVE strategy (bypass only large sizes)"""
    print("\nTest 4: SELECTIVE strategy...")
    try:
        config = Conv2dBypassConfig(
            strategy=FallbackStrategy.SELECTIVE,
            size_threshold=42,
            verbose=False
        )
        
        conv = SafeConv2d(3, 64, kernel_size=3, padding=1, config=config).cuda()
        
        # Small size (should not bypass)
        x_small = torch.randn(1, 3, 32, 32).cuda()
        y_small = conv(x_small)
        
        # Large size (should bypass)
        x_large = torch.randn(1, 3, 64, 64).cuda()
        y_large = conv(x_large)
        
        stats = conv.get_bypass_stats()
        print(f"  ✅ Small size (32×32): works")
        print(f"  ✅ Large size (64×64): works")
        print(f"  ✅ Bypass stats: {stats['bypass_count']}/{stats['total_forwards']} bypassed")
        print(f"  ✅ Bypass rate: {stats['bypass_rate']:.1f}%")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def test_statistics_tracking():
    """Test that statistics are tracked correctly"""
    print("\nTest 5: Statistics tracking...")
    try:
        config = Conv2dBypassConfig(
            strategy=FallbackStrategy.CPU_FALLBACK,
            verbose=False
        )
        
        conv = SafeConv2d(3, 64, kernel_size=3, padding=1, config=config).cuda()
        
        # Run multiple forward passes
        for i in range(5):
            x = torch.randn(1, 3, 64, 64).cuda()
            y = conv(x)
        
        stats = conv.get_bypass_stats()
        
        assert stats['total_forwards'] == 5, f"Expected 5 forwards, got {stats['total_forwards']}"
        assert stats['bypass_count'] == 5, f"Expected 5 bypasses, got {stats['bypass_count']}"
        assert stats['bypass_rate'] == 100.0, f"Expected 100% bypass rate, got {stats['bypass_rate']}"
        
        print(f"  ✅ Total forwards: {stats['total_forwards']}")
        print(f"  ✅ Bypass count: {stats['bypass_count']}")
        print(f"  ✅ Bypass rate: {stats['bypass_rate']:.1f}%")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("MIOpen Bypass - Simple Functional Tests")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run tests
    results = []
    results.append(("CPU Fallback Basic", test_cpu_fallback_basic()))
    results.append(("AUTO Strategy", test_auto_strategy()))
    results.append(("Model Patching", test_model_patching()))
    results.append(("SELECTIVE Strategy", test_selective_bypass()))
    results.append(("Statistics Tracking", test_statistics_tracking()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print("=" * 70)
    
    # Exit code
    sys.exit(0 if passed == total else 1)
