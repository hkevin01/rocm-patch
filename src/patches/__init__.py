"""
ROCm Patch Collection
=====================

Collection of patches for AMD ROCm platform issues affecting RDNA1/2/3 and CDNA architectures.

Available Patches:
    - memory_access_fault: Fix for "Memory access fault - Page not present" on RDNA1/2

Quick Start:
    # Apply memory access fault patch
    from rocm_patch.patches.memory_access_fault import apply_patch
    apply_patch()
    
    # Apply all available patches
    from rocm_patch.patches import apply_all_patches
    apply_all_patches()

For detailed documentation, see docs/issues/
"""

from .memory_access_fault import apply_patch as apply_memory_access_fault_patch

__all__ = ['apply_memory_access_fault_patch', 'apply_all_patches']

def apply_all_patches():
    """
    Apply all available ROCm patches.
    
    This is a convenience function that applies all patches in the collection.
    Individual patches can be applied separately if needed.
    
    Returns:
        dict: Dictionary of applied patches and their results
    """
    results = {}
    
    print("=== Applying ROCm Patches ===\n")
    
    # Apply memory access fault patch
    try:
        print("1. Memory Access Fault Patch (RDNA1/2)...")
        results['memory_access_fault'] = apply_memory_access_fault_patch()
        print("   ✅ Success\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        results['memory_access_fault'] = None
    
    print("=== Patch Application Complete ===\n")
    
    return results
