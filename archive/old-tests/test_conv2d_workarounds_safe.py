#!/usr/bin/env python3
"""
Test Conv2d workarounds for RDNA1 (gfx1010) with timeout protection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import signal
import sys

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Test timed out!")

def test_conv_with_timeout(name, in_ch, out_ch, size, batch=1, timeout_sec=30):
    """Test a single convolution with timeout using signal.alarm"""
    print(f"\n--- Test: {name} ---")
    print(f"    Config: {in_ch}→{out_ch} channels, {size}x{size} input, batch={batch}")
    print(f"    Timeout: {timeout_sec}s")
    
    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        # Create conv layer and input
        conv = nn.Conv2d(in_ch, out_ch, 3, padding=1).cuda()
        x = torch.randn(batch, in_ch, size, size).cuda()
        
        # Warmup
        print(f"    Running warmup...")
        with torch.no_grad():
            _ = conv(x)
        
        # Timed run
        print(f"    Running timed test...")
        start = time.time()
        with torch.no_grad():
            output = conv(x)
        elapsed = time.time() - start
        
        # Cancel the alarm
        signal.alarm(0)
        
        print(f"    ✅ PASS - Time: {elapsed:.4f}s")
        print(f"    Output shape: {output.shape}")
        return True, elapsed
        
    except TimeoutException:
        signal.alarm(0)
        print(f"    ⏱️  TIMEOUT - Test exceeded {timeout_sec}s (likely hanging)")
        return False, None
    except Exception as e:
        signal.alarm(0)
        print(f"    ❌ FAIL - Error: {e}")
        return False, None

print("=" * 70)
print("Conv2d Workaround Tests for RDNA1 (gfx1010) - WITH TIMEOUTS")
print("=" * 70)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 70)
print("PART 1: Small Conv (Known to work - baseline)")
print("=" * 70)

test_conv_with_timeout(
    "Small conv 16→32, 32x32 (should work)",
    in_ch=16, out_ch=32, size=32, timeout_sec=30
)

print("\n" + "=" * 70)
print("PART 2: Non-Power-of-2 Channel Tests")
print("=" * 70)
print("Strategy: Use odd channel counts to avoid problematic GEMM paths")

results = []

# Test various non-power-of-2 configurations
configs = [
    ("15→31 channels, 48x48", 15, 31, 48),
    ("17→31 channels, 48x48", 17, 31, 48),
    ("15→31 channels, 64x64", 15, 31, 64),
    ("17→33 channels, 64x64", 17, 33, 64),
    ("31→63 channels, 48x48", 31, 63, 48),
]

for name, in_ch, out_ch, size in configs:
    success, elapsed = test_conv_with_timeout(name, in_ch, out_ch, size, timeout_sec=45)
    results.append((name, success, elapsed))
    time.sleep(1)  # Brief pause between tests

print("\n" + "=" * 70)
print("PART 3: Smaller Size Tests (Even with power-of-2 channels)")
print("=" * 70)
print("Strategy: Test if 40x40 or 44x44 works (not power-of-2 sizes)")

size_configs = [
    ("16→32 channels, 40x40", 16, 32, 40),
    ("16→32 channels, 44x44", 16, 32, 44),
    ("16→32 channels, 36x36", 16, 32, 36),
]

for name, in_ch, out_ch, size in size_configs:
    success, elapsed = test_conv_with_timeout(name, in_ch, out_ch, size, timeout_sec=45)
    results.append((name, success, elapsed))
    time.sleep(1)

print("\n" + "=" * 70)
print("PART 4: Tiling Approach (for known-hanging configs)")
print("=" * 70)

def conv2d_tiled(x, conv, tile_size=32):
    """Apply convolution using tiling - simplified version"""
    B, C, H, W = x.shape
    out_channels = conv.out_channels
    
    # Simple tiling without overlap (will have boundary artifacts)
    output_patches = []
    
    for i in range(0, H, tile_size):
        row_patches = []
        for j in range(0, W, tile_size):
            i_end = min(i + tile_size, H)
            j_end = min(j + tile_size, W)
            
            tile = x[:, :, i:i_end, j:j_end]
            tile_out = conv(tile)
            row_patches.append(tile_out)
        
        # Concatenate row
        row = torch.cat(row_patches, dim=3)
        output_patches.append(row)
    
    # Concatenate all rows
    output = torch.cat(output_patches, dim=2)
    return output

print(f"\n--- Test: 16→32, 48x48 WITH TILING (32x32 tiles) ---")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)
try:
    conv = nn.Conv2d(16, 32, 3, padding=1).cuda()
    x = torch.randn(1, 16, 48, 48).cuda()
    
    print(f"    Running tiled convolution...")
    start = time.time()
    with torch.no_grad():
        output = conv2d_tiled(x, conv, tile_size=32)
    elapsed = time.time() - start
    
    signal.alarm(0)
    print(f"    ✅ PASS - Time: {elapsed:.4f}s")
    print(f"    Output shape: {output.shape}")
    results.append(("16→32 48x48 TILED", True, elapsed))
except TimeoutException:
    signal.alarm(0)
    print(f"    ⏱️  TIMEOUT - Tiling approach also hanging")
    results.append(("16→32 48x48 TILED", False, None))
except Exception as e:
    signal.alarm(0)
    print(f"    ❌ FAIL - Error: {e}")
    results.append(("16→32 48x48 TILED", False, None))

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

working = []
hanging = []

for name, success, elapsed in results:
    if success:
        working.append((name, elapsed))
        print(f"✅ {name:40} - {elapsed:.4f}s")
    else:
        hanging.append(name)
        print(f"⏱️  {name:40} - TIMEOUT/HANG")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

if working:
    print(f"\n✅ WORKING CONFIGURATIONS ({len(working)}):")
    for name, elapsed in working:
        print(f"   - {name}")
    
    print(f"\nFastest working config: {min(working, key=lambda x: x[1])[0]}")

if hanging:
    print(f"\n⏱️  HANGING CONFIGURATIONS ({len(hanging)}):")
    for name in hanging:
        print(f"   - {name}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if any("31" in name or "33" in name or "63" in name for name, _, _ in results if _):
    print("✅ Non-power-of-2 channels appear to work!")
    print("   Recommendation: Modify models to use odd channel counts")
    print("   Examples: 15, 17, 31, 33, 63, 127, 255")
elif any("40x40" in name or "44x44" in name or "36x36" in name for name, _, _ in results if _):
    print("✅ Non-power-of-2 sizes appear to work!")
    print("   Recommendation: Resize inputs to non-power-of-2 dimensions")
else:
    print("⚠️  Multiple configurations still hanging")
    print("   This may be a deeper MIOpen/rocBLAS issue with ROCm 5.7")
    print("   Consider:")
    print("   1. Staying with small feature maps (≤32x32)")
    print("   2. Using CPU for problematic layers")
    print("   3. Upgrading hardware to RDNA2/RDNA3")

print("\n✅ Test suite complete!")
