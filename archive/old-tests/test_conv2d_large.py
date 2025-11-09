#!/usr/bin/env python3
"""
Run a set of Conv2d tests that previously hung on ROCm 5.7/GEMM.
Tests:
 - Medium: 16->32 channels, 48x48
 - Large: 16->32 channels, 64x64
 - Very large: 64->64 channels, 224x224 (ResNet-like)
Each test has a timeout wrapper; if it hangs we'll detect and abort.
"""
import torch
import time
import signal
import sys

# Timeout helper
class TimeoutException(Exception):
    pass

def alarm_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, alarm_handler)

def run_conv(in_ch, out_ch, H, W, timeout=60, batch=1):
    print(f"\n--- Test: {in_ch}→{out_ch}, {H}x{W}, batch={batch}, timeout={timeout}s ---")
    try:
        signal.alarm(timeout)
        x = torch.randn(batch, in_ch, H, W).cuda()
        conv = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1).cuda()
        start = time.time()
        y = conv(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        signal.alarm(0)
        print(f"✅ Completed in {elapsed:.2f}s, output: {y.shape}")
        return True
    except TimeoutException:
        print(f"❌ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    print("Conv2d large-tests -- environment:\n")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Couldn't import torch or query device: {e}")
        sys.exit(1)

    tests = [
        (16, 32, 48, 48, 180),   # medium - previously hung
        (16, 32, 64, 64, 300),   # larger - previously hung
        (64, 64, 224, 224, 600), # very large - ResNet test
    ]

    summary = {}
    for in_ch, out_ch, H, W, tout in tests:
        ok = run_conv(in_ch, out_ch, H, W, timeout=tout)
        summary[f"{in_ch}->{out_ch}_{H}x{W}"] = ok

    print("\n=== SUMMARY ===")
    for k,v in summary.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
