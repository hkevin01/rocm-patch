"""
ROCm Compatibility Utilities
=============================

Configures PyTorch for optimal ROCm performance and compatibility.

Key Features:
    - Sets multiprocessing to 'spawn' (required for ROCm)
    - Patches DataLoader with optimal settings
    - Provides diagnostic utilities

Usage:
    # Import FIRST in your training script (before torch)
    from utils.rocm_compat import setup_rocm_multiprocessing, patch_dataloader
    
    setup_rocm_multiprocessing()
    
    import torch
    # ... rest of imports ...
    
    patch_dataloader()  # After torch import
    
    # Now use DataLoaders normally with num_workers > 0

Author: ROCm Patch Project
Date: November 2025
"""

import multiprocessing as mp
import sys
import os
from typing import Optional


def setup_rocm_multiprocessing(force: bool = True, verbose: bool = True) -> None:
    """
    Configure multiprocessing for ROCm compatibility.
    
    CRITICAL: Must be called BEFORE importing torch!
    
    ROCm/HIP does not support Python's default 'fork' method.
    This function sets the start method to 'spawn' which creates
    fresh Python processes with independent ROCm contexts.
    
    Args:
        force: If True, override existing start method (default: True)
        verbose: Print confirmation message (default: True)
    
    Example:
        >>> from utils.rocm_compat import setup_rocm_multiprocessing
        >>> setup_rocm_multiprocessing()
        ✅ Multiprocessing set to 'spawn' for ROCm compatibility
        >>> import torch  # Import after setup
    """
    try:
        current_method = mp.get_start_method(allow_none=True)
        
        if current_method == 'spawn':
            if verbose:
                print("✅ Multiprocessing already set to 'spawn'")
            return
        
        mp.set_start_method('spawn', force=force)
        
        if verbose:
            print("✅ Multiprocessing set to 'spawn' for ROCm compatibility")
            if current_method:
                print(f"   (changed from '{current_method}')")
    
    except RuntimeError as e:
        if verbose:
            print(f"⚠️  Could not set multiprocessing method: {e}")
            print(f"   Current method: {mp.get_start_method()}")


def patch_dataloader(verbose: bool = True) -> None:
    """
    Monkey-patch PyTorch DataLoader for ROCm-optimal settings.
    
    Automatically adds to DataLoaders with num_workers > 0:
        - multiprocessing_context='spawn' (required for ROCm)
        - persistent_workers=True (faster, workers stay alive)
    
    Must be called AFTER importing torch.
    
    Args:
        verbose: Print confirmation message (default: True)
    
    Example:
        >>> import torch
        >>> from utils.rocm_compat import patch_dataloader
        >>> patch_dataloader()
        ✅ DataLoader patched for ROCm (spawn + persistent_workers)
        >>> 
        >>> # Now all DataLoaders automatically use correct settings
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        if verbose:
            print("⚠️  torch not imported yet, skipping DataLoader patch")
        return
    
    # Check if already patched
    if hasattr(DataLoader.__init__, '_rocm_patched'):
        if verbose:
            print("✅ DataLoader already patched for ROCm")
        return
    
    # Store original init
    _original_init = DataLoader.__init__
    
    def _rocm_dataloader_init(self, *args, **kwargs):
        """Patched DataLoader.__init__ with ROCm-optimal settings"""
        num_workers = kwargs.get('num_workers', 0)
        
        if num_workers > 0:
            # Force spawn context for ROCm compatibility
            if 'multiprocessing_context' not in kwargs:
                kwargs['multiprocessing_context'] = 'spawn'
            
            # Enable persistent workers for better performance
            if 'persistent_workers' not in kwargs:
                kwargs['persistent_workers'] = True
        
        # Call original init
        _original_init(self, *args, **kwargs)
    
    # Mark as patched
    _rocm_dataloader_init._rocm_patched = True
    
    # Apply patch
    DataLoader.__init__ = _rocm_dataloader_init
    
    if verbose:
        print("✅ DataLoader patched for ROCm (spawn + persistent_workers)")


def verify_rocm_setup(raise_on_error: bool = False) -> bool:
    """
    Verify ROCm multiprocessing setup is correct.
    
    Checks:
        - Multiprocessing start method is 'spawn'
        - torch is imported (if checking DataLoader)
        - ROCm/CUDA is available
    
    Args:
        raise_on_error: If True, raise exception on verification failure
    
    Returns:
        True if setup is correct, False otherwise
    
    Example:
        >>> from utils.rocm_compat import verify_rocm_setup
        >>> if not verify_rocm_setup():
        ...     print("Setup not correct!")
    """
    issues = []
    
    # Check multiprocessing method
    method = mp.get_start_method()
    if method != 'spawn':
        issues.append(f"Multiprocessing method is '{method}', should be 'spawn'")
    
    # Check torch availability
    if 'torch' in sys.modules:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA/ROCm not available")
    
    # Print results
    if issues:
        print("❌ ROCm setup verification FAILED:")
        for issue in issues:
            print(f"   - {issue}")
        
        if raise_on_error:
            raise RuntimeError(f"ROCm setup verification failed: {', '.join(issues)}")
        return False
    else:
        print("✅ ROCm setup verification PASSED")
        return True


def print_rocm_info() -> None:
    """
    Print comprehensive ROCm configuration information.
    
    Displays:
        - PyTorch version
        - GPU information
        - Multiprocessing settings
        - Environment variables
        - System resources
    
    Example:
        >>> from utils.rocm_compat import print_rocm_info
        >>> print_rocm_info()
        ======================================================================
        ROCm Configuration
        ======================================================================
        PyTorch version: 1.13.1+rocm5.2
        CUDA available: True
        GPU: AMD Radeon RX 5600 XT
        ...
    """
    print("=" * 70)
    print("ROCm Configuration")
    print("=" * 70)
    
    # PyTorch info
    if 'torch' in sys.modules:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"GPU memory: {props.total_memory / 1e9:.1f} GB")
            print(f"GPU capability: {props.major}.{props.minor}")
    else:
        print("PyTorch: not imported yet")
    
    # Multiprocessing info
    print(f"\nMultiprocessing start method: {mp.get_start_method()}")
    print(f"Number of CPU cores: {mp.cpu_count()}")
    
    # Environment variables
    print("\nROCm Environment Variables:")
    rocm_vars = {
        'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM': 'MIOpen algorithm selection',
        'HSA_OVERRIDE_GFX_VERSION': 'GPU architecture override',
        'ROCM_PATH': 'ROCm installation path',
        'HIP_VISIBLE_DEVICES': 'Visible GPU devices',
        'MIOPEN_ENABLE_LOGGING': 'MIOpen debug logging',
    }
    
    for var, description in rocm_vars.items():
        value = os.environ.get(var, 'not set')
        print(f"  {var:30s} = {value:20s} # {description}")
    
    print("=" * 70)


def get_optimal_num_workers(
    gpu_memory_gb: Optional[float] = None,
    cpu_cores: Optional[int] = None
) -> int:
    """
    Recommend optimal number of DataLoader workers for system.
    
    Args:
        gpu_memory_gb: GPU memory in GB (auto-detected if None)
        cpu_cores: Number of CPU cores (auto-detected if None)
    
    Returns:
        Recommended number of workers (typically 2-8)
    
    Example:
        >>> from utils.rocm_compat import get_optimal_num_workers
        >>> num_workers = get_optimal_num_workers()
        >>> loader = DataLoader(dataset, num_workers=num_workers)
    """
    if cpu_cores is None:
        cpu_cores = mp.cpu_count()
    
    if gpu_memory_gb is None and 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # General recommendations:
    # - 4 workers is tested and works well
    # - More workers = more memory usage
    # - Diminishing returns after 4-8 workers
    
    if gpu_memory_gb and gpu_memory_gb < 4:
        # Low memory GPU: 2 workers
        return 2
    elif cpu_cores <= 4:
        # Limited CPU: 2 workers
        return 2
    else:
        # Standard: 4 workers (tested optimal)
        return 4


# Auto-setup on import (if called before torch)
def _auto_setup():
    """Automatically setup spawn method if torch not yet imported"""
    if 'torch' not in sys.modules:
        setup_rocm_multiprocessing(verbose=False)


# Run auto-setup
_auto_setup()


# Convenience: all-in-one setup
def setup_rocm(verbose: bool = True) -> None:
    """
    One-function setup for ROCm compatibility.
    
    Performs:
        1. Set multiprocessing to spawn (if torch not imported)
        2. Patch DataLoader (if torch is imported)
        3. Print configuration info (if verbose)
    
    Args:
        verbose: Print information and confirmation (default: True)
    
    Example:
        >>> from utils.rocm_compat import setup_rocm
        >>> setup_rocm()
        >>> import torch  # If not already imported
    """
    # Setup multiprocessing
    if 'torch' not in sys.modules:
        setup_rocm_multiprocessing(verbose=verbose)
    else:
        if verbose:
            print("⚠️  torch already imported, can't change multiprocessing method")
    
    # Patch DataLoader if torch available
    if 'torch' in sys.modules:
        patch_dataloader(verbose=verbose)
    
    # Print info
    if verbose:
        print()
        print_rocm_info()


if __name__ == "__main__":
    # Test/demo the module
    print("ROCm Compatibility Utilities - Test Mode")
    print()
    
    # Show info
    print_rocm_info()
    print()
    
    # Verify setup
    verify_rocm_setup()
    print()
    
    # Recommend workers
    num_workers = get_optimal_num_workers()
    print(f"Recommended num_workers: {num_workers}")
