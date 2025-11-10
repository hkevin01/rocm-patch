"""
Performance Test: GPU Unfold vs CPU Fallback
=============================================

Compare the new GPU unfold+matmul approach vs old CPU fallback.
"""

import torch
import torch.nn as nn
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conv2d_fallback import SafeConv2d, Conv2dBypassConfig, FallbackStrategy

def benchmark_gpu_unfold():
    """Test GPU unfold+matmul performance"""
    print("=" * 70)
    print("GPU Unfold+Matmul Performance Test")
    print("=" * 70)
    
    config = Conv2dBypassConfig(
        strategy=FallbackStrategy.GPU_UNFOLD,
        verbose=False
    )
    
    conv = SafeConv2d(3, 64, kernel_size=3, padding=1, config=config).cuda()
    
    # Test different sizes
    sizes = [32, 64, 128, 224]
    
    for size in sizes:
        x = torch.randn(4, 3, size, size).cuda()  # Batch of 4
        
        # Warmup
        for _ in range(5):
            y = conv(x)
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            y = conv(x)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = 4 / avg_time  # Images per second (batch size 4)
        
        print(f"Size {size:3d}×{size:<3d}: {avg_time*1000:6.2f} ms/batch  ({throughput:6.1f} img/s)")

def benchmark_cpu_fallback():
    """Test CPU fallback performance (for comparison)"""
    print("\n" + "=" * 70)
    print("CPU Fallback Performance Test (for comparison)")
    print("=" * 70)
    
    # Manually implement CPU fallback
    conv_gpu = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
    
    sizes = [32, 64, 128, 224]
    
    for size in sizes:
        x = torch.randn(4, 3, size, size).cuda()
        
        # Warmup
        for _ in range(5):
            x_cpu = x.cpu()
            conv_cpu = conv_gpu.cpu()
            y_cpu = conv_cpu(x_cpu)
            y = y_cpu.cuda()
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            start = time.time()
            x_cpu = x.cpu()
            conv_cpu = conv_gpu.cpu()
            y_cpu = conv_cpu(x_cpu)
            y = y_cpu.cuda()
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        throughput = 4 / avg_time
        
        print(f"Size {size:3d}×{size:<3d}: {avg_time*1000:6.2f} ms/batch  ({throughput:6.1f} img/s)")

def compare_speedup():
    """Direct comparison"""
    print("\n" + "=" * 70)
    print("Speedup Comparison: GPU Unfold vs CPU Fallback")
    print("=" * 70)
    
    config_gpu = Conv2dBypassConfig(strategy=FallbackStrategy.GPU_UNFOLD, verbose=False)
    conv_safe = SafeConv2d(3, 64, kernel_size=3, padding=1, config=config_gpu).cuda()
    conv_cpu_ref = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    
    size = 64
    x = torch.randn(4, 3, size, size).cuda()
    
    # GPU unfold timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        y = conv_safe(x)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 50
    
    # CPU fallback timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        x_cpu = x.cpu()
        conv_cpu = conv_cpu_ref.cpu()
        y_cpu = conv_cpu(x_cpu)
        y = y_cpu.cuda()
    torch.cuda.synchronize()
    cpu_time = (time.time() - start) / 50
    
    speedup = cpu_time / gpu_time
    
    print(f"\nSize: {size}×{size}, Batch: 4, Iterations: 50")
    print(f"GPU Unfold+Matmul: {gpu_time*1000:.2f} ms/batch")
    print(f"CPU Fallback:      {cpu_time*1000:.2f} ms/batch")
    print(f"Speedup:           {speedup:.2f}x faster 🚀")
    print()
    print("✅ GPU unfold+matmul keeps everything on GPU!")
    print("✅ No PCIe transfer overhead!")
    print("✅ Uses optimized rocBLAS matmul!")

if __name__ == "__main__":
    print("🔬 MIOpen Bypass Performance Analysis")
    print("Testing GPU-only unfold+matmul vs CPU fallback\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")
    
    benchmark_gpu_unfold()
    benchmark_cpu_fallback()
    compare_speedup()
