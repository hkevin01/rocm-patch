"""
ROCm RDNA1/2 Memory Access Fault Patch
======================================

Comprehensive fix for "Memory access fault by GPU - Page not present or supervisor privilege"
affecting AMD RDNA1/2 consumer GPUs (RX 5000/6000 series) on ROCm 6.2+.

Usage:
    from rocm_patch.patches.memory_access_fault import apply_patch
    
    # Apply all patches (recommended)
    apply_patch()
    
    # Or import individual components
    from rocm_patch.patches.memory_access_fault import hip_memory_patch
    hip_memory_patch.apply_rocm_fix()

Components:
    - hip_memory_patch: Python wrapper for PyTorch memory allocation
    - kernel_params.sh: System-level kernel module configuration (requires root)

For more information, see docs/issues/thermal-object-detection-memory-faults.md
"""

from .hip_memory_patch import apply_rocm_fix

__all__ = ['apply_rocm_fix', 'apply_patch']

def apply_patch():
    """
    Apply the ROCm RDNA1/2 memory access fault patch.
    
    This is the main entry point that applies the HIP memory allocator fix.
    For kernel-level fixes, run kernel_params.sh separately with sudo.
    
    Returns:
        HIPMemoryAllocatorPatch instance
    """
    return apply_rocm_fix()
