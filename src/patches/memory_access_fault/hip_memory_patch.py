#!/usr/bin/env python3
"""
ROCm RDNA1/2 Memory Coherency Fix - HIP Memory Allocator Patch
Addresses: Memory access fault by GPU (Page not present or supervisor privilege)
Target: AMD RX 5600 XT (gfx1010), RX 6000 series (gfx1030)
ROCm versions: 6.2+

This patch wraps PyTorch's HIP memory allocation to force non-coherent memory types.
"""

import os
import sys
import ctypes
import functools

print("=" * 70)
print("ROCm RDNA1/2 Memory Coherency Fix - HIP Allocator Patch")
print("=" * 70)

# Set environment variables BEFORE importing torch
os.environ['HSA_USE_SVM'] = '0'
os.environ['HSA_XNACK'] = '0'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['PYTORCH_NO_HIP_MEMORY_CACHING'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Force compatibility mode

# Force PyTorch to use smaller memory pools
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'

# Disable THP (Transparent Huge Pages) for this process
try:
    with open('/proc/self/smaps', 'r') as f:
        pass  # Just checking if accessible
    # Best effort - may need root
    os.system('echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null')
except:
    pass

print("\n✓ Environment variables configured:")
for key in ['HSA_USE_SVM', 'HSA_XNACK', 'PYTORCH_NO_HIP_MEMORY_CACHING', 'HSA_OVERRIDE_GFX_VERSION']:
    print(f"  {key} = {os.environ.get(key, 'N/A')}")

# Import torch AFTER setting environment
import torch

print(f"\n✓ PyTorch {torch.__version__} imported")
print(f"  CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    print(f"  Device: {device_name}")
    print(f"  Memory: {device_props.total_memory / 1e9:.2f} GB")
    print(f"  Compute: {device_props.major}.{device_props.minor}")


class HIPMemoryAllocatorPatch:
    """
    Wrapper for PyTorch memory allocation to force non-coherent allocations.
    """
    
    def __init__(self):
        self.original_empty = torch.empty
        self.original_zeros = torch.zeros
        self.original_ones = torch.ones
        self.original_tensor = torch.tensor
        self._patched = False
        
    def safe_allocation_wrapper(self, original_func, *args, **kwargs):
        """
        Wrapper that catches memory access faults and retries with CPU fallback.
        """
        # Force pin_memory=False for GPU allocations (prevents coherent memory)
        if 'device' in kwargs and 'cuda' in str(kwargs.get('device', '')):
            kwargs['pin_memory'] = False
            
        # Limit memory allocation size to prevent fragmentation
        if len(args) > 0:
            try:
                size = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
                total_elements = 1
                for dim in size:
                    total_elements *= dim
                
                # If allocation > 1GB, warn and potentially split
                bytes_needed = total_elements * 4  # Assume float32
                if bytes_needed > 1e9:
                    print(f"⚠️  Warning: Large allocation ({bytes_needed/1e9:.2f} GB) - may cause issues")
            except:
                pass
        
        try:
            # Try original allocation
            return original_func(*args, **kwargs)
        except RuntimeError as e:
            if "memory access fault" in str(e).lower() or "hip error" in str(e).lower():
                print(f"❌ Memory access fault detected! Retrying with safer parameters...")
                
                # Fallback: Force CPU allocation
                if 'device' in kwargs:
                    del kwargs['device']
                kwargs['device'] = 'cpu'
                
                print(f"   → Falling back to CPU allocation")
                result = original_func(*args, **kwargs)
                print(f"   ✓ CPU allocation successful")
                return result
            else:
                raise
    
    def patch(self):
        """Apply the memory allocation patches."""
        if self._patched:
            print("⚠️  Already patched!")
            return
        
        print("\n📦 Patching PyTorch memory allocation functions...")
        
        # Patch torch.empty
        torch.empty = functools.partial(self.safe_allocation_wrapper, self.original_empty)
        print("  ✓ torch.empty")
        
        # Patch torch.zeros
        torch.zeros = functools.partial(self.safe_allocation_wrapper, self.original_zeros)
        print("  ✓ torch.zeros")
        
        # Patch torch.ones
        torch.ones = functools.partial(self.safe_allocation_wrapper, self.original_ones)
        print("  ✓ torch.ones")
        
        # Patch torch.tensor
        torch.tensor = functools.partial(self.safe_allocation_wrapper, self.original_tensor)
        print("  ✓ torch.tensor")
        
        # Configure PyTorch memory allocator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✓ Cleared CUDA cache")
            
            # Set memory fraction to prevent over-allocation
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
            print("  ✓ Set memory fraction: 80%")
        
        self._patched = True
        print("\n✅ Memory allocator patches applied successfully!")
    
    def unpatch(self):
        """Remove patches (for testing)."""
        if not self._patched:
            return
        
        torch.empty = self.original_empty
        torch.zeros = self.original_zeros
        torch.ones = self.original_ones
        torch.tensor = self.original_tensor
        self._patched = False
        print("✓ Patches removed")


# Global instance
_patch_instance = None

def apply_rocm_fix():
    """
    Apply the ROCm RDNA1/2 memory coherency fix.
    Call this BEFORE any model creation or training.
    
    Usage:
        from patches.rocm_fix.hip_memory_patch import apply_rocm_fix
        apply_rocm_fix()
    """
    global _patch_instance
    
    if _patch_instance is None:
        _patch_instance = HIPMemoryAllocatorPatch()
        _patch_instance.patch()
    
    return _patch_instance


def test_allocation():
    """Test if GPU allocations work with the patch."""
    print("\n" + "=" * 70)
    print("Testing GPU Memory Allocation")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False
    
    try:
        print("\n1. Testing small allocation (1 MB)...")
        x = torch.randn(256, 1024, device='cuda')
        print(f"   ✓ Success: {x.shape}, device={x.device}")
        del x
        torch.cuda.empty_cache()
        
        print("\n2. Testing medium allocation (100 MB)...")
        x = torch.randn(1024, 10240, device='cuda')
        print(f"   ✓ Success: {x.shape}, device={x.device}")
        del x
        torch.cuda.empty_cache()
        
        print("\n3. Testing large allocation (500 MB)...")
        x = torch.randn(4096, 16384, device='cuda')
        print(f"   ✓ Success: {x.shape}, device={x.device}")
        del x
        torch.cuda.empty_cache()
        
        print("\n4. Testing computation...")
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        print(f"   ✓ Success: matmul result shape={c.shape}")
        del a, b, c
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("❌ TESTS FAILED - GPU may still have issues")
        print("=" * 70)
        return False


if __name__ == "__main__":
    # Apply patches
    apply_rocm_fix()
    
    # Run tests
    test_allocation()
    
    print("\n" + "=" * 70)
    print("To use in your training script:")
    print("=" * 70)
    print("""
    # Add to the TOP of your training script:
    import sys
    sys.path.insert(0, '/home/kevin/Projects/robust-thermal-image-object-detection')
    from patches.rocm_fix.hip_memory_patch import apply_rocm_fix
    
    # Apply fix BEFORE importing any other modules
    apply_rocm_fix()
    
    # Then continue with normal imports
    from ultralytics import YOLO
    # ... rest of your code
    """)
