"""
ROCm Patch for RDNA1 GPUs - Complete Initialization
====================================================

This module provides a single entry point for all ROCm patches and workarounds.

Key Features:
    1. MIOpen Conv2d bypass (GPU-only unfold+matmul)
    2. Memory access fault prevention
    3. Multiprocessing setup for DataLoader compatibility
    4. Environment configuration

Usage:
    from rocm_patch import enable_all_patches
    enable_all_patches()
    
    # Now use PyTorch normally
    import torch
    model = YourModel().cuda()

Author: ROCm Patch Project
Date: November 2025
License: MIT
"""

import os
import sys
import warnings

__version__ = "1.1.0"
__author__ = "ROCm Patch Project"

# Version that includes multiprocessing fixes
ROCM_PATCH_VERSION = "1.1.0"


def setup_multiprocessing():
    """
    Configure multiprocessing for ROCm compatibility.
    
    Critical for DataLoader with num_workers > 0.
    Must be called BEFORE importing torch!
    
    Changes:
        - Sets multiprocessing start method to 'spawn' (not 'fork')
        - 'fork' causes CUDA initialization errors with ROCm
        - 'spawn' creates fresh Python process (clean CUDA context)
    
    Returns:
        bool: True if setup successful, False if torch already imported
    """
    if 'torch' in sys.modules:
        warnings.warn(
            "⚠️  torch already imported! Multiprocessing setup may not work correctly.\n"
            "   Call enable_all_patches() or setup_multiprocessing() BEFORE importing torch!",
            RuntimeWarning
        )
        return False
    
    import multiprocessing as mp
    
    # Force spawn method (required for ROCm)
    try:
        mp.set_start_method('spawn', force=True)
        print("✓ Multiprocessing start method set to 'spawn' (ROCm compatible)")
        return True
    except RuntimeError as e:
        # Already set - verify it's spawn
        if mp.get_start_method() == 'spawn':
            print("✓ Multiprocessing already configured with 'spawn'")
            return True
        else:
            warnings.warn(
                f"⚠️  Multiprocessing set to '{mp.get_start_method()}', expected 'spawn'!\n"
                f"   DataLoader with workers may fail. Error: {e}",
                RuntimeWarning
            )
            return False


def setup_environment():
    """
    Configure ROCm environment variables.
    
    Must be called BEFORE importing torch!
    
    Sets:
        - MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1 (stable convolution algorithm)
        - HSA_OVERRIDE_GFX_VERSION=10.3.0 (RDNA1 compatibility)
        - HSA_USE_SVM=0 (disable shared virtual memory)
        - HSA_XNACK=0 (disable page migration)
    """
    env_vars = {
        'MIOPEN_DEBUG_CONV_IMPLICIT_GEMM': '1',
        'HSA_OVERRIDE_GFX_VERSION': '10.3.0',
        'HSA_USE_SVM': '0',
        'HSA_XNACK': '0',
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"✓ Set {key}={value}")
        else:
            print(f"  {key} already set to '{os.environ[key]}'")


def patch_dataloader():
    """
    Monkey-patch torch.utils.data.DataLoader for ROCm compatibility.
    
    Must be called AFTER importing torch!
    
    Changes:
        - Adds multiprocessing_context='spawn' by default
        - Enables persistent_workers=True by default (if num_workers > 0)
        - Prevents 'fork' method which breaks ROCm CUDA contexts
    
    Returns:
        bool: True if patched successfully
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        
        # Save original __init__
        _original_init = DataLoader.__init__
        
        def _patched_init(self, *args, **kwargs):
            """Patched DataLoader.__init__ with ROCm-safe defaults"""
            
            # Get num_workers (default is 0)
            num_workers = kwargs.get('num_workers', 0)
            
            # Force spawn context if using workers and not explicitly set
            if num_workers > 0:
                if 'multiprocessing_context' not in kwargs:
                    import multiprocessing as mp
                    kwargs['multiprocessing_context'] = mp.get_context('spawn')
                    
                # Enable persistent workers for better performance
                if 'persistent_workers' not in kwargs:
                    kwargs['persistent_workers'] = True
            
            # Call original init with modified kwargs
            _original_init(self, *args, **kwargs)
        
        # Apply patch
        DataLoader.__init__ = _patched_init
        
        print("✓ DataLoader patched: multiprocessing_context='spawn', persistent_workers=True")
        return True
        
    except ImportError:
        warnings.warn(
            "⚠️  Could not patch DataLoader - torch not imported yet.\n"
            "   Call this after importing torch, or use enable_all_patches().",
            RuntimeWarning
        )
        return False
    except Exception as e:
        warnings.warn(f"⚠️  Failed to patch DataLoader: {e}", RuntimeWarning)
        return False


def enable_miopen_bypass(strategy=None):
    """
    Enable MIOpen Conv2d bypass with GPU unfold+matmul.
    
    Args:
        strategy: FallbackStrategy enum or None for AUTO
    
    Returns:
        bool: True if enabled successfully
    """
    try:
        from .miopen_bypass.conv2d_fallback import enable_miopen_bypass as _enable
        from .miopen_bypass.conv2d_fallback import FallbackStrategy
        
        if strategy is None:
            strategy = FallbackStrategy.AUTO
        
        _enable(strategy=strategy)
        print(f"✓ MIOpen bypass enabled: strategy={strategy.value}")
        return True
        
    except ImportError as e:
        warnings.warn(f"⚠️  Could not enable MIOpen bypass: {e}", RuntimeWarning)
        return False


def enable_all_patches(miopen_strategy=None):
    """
    Enable all ROCm patches in correct order.
    
    This is the recommended entry point for most users.
    
    Order of operations:
        1. Setup multiprocessing (BEFORE torch import)
        2. Setup environment variables (BEFORE torch import)
        3. Import torch
        4. Patch DataLoader
        5. Enable MIOpen bypass
    
    Args:
        miopen_strategy: FallbackStrategy for MIOpen bypass (default: AUTO)
    
    Returns:
        dict: Status of each patch
    
    Example:
        >>> from rocm_patch import enable_all_patches
        >>> enable_all_patches()
        >>> 
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> 
        >>> # Now DataLoader with workers=4 works!
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """
    print("=" * 70)
    print("ROCm Patch v{} - Enabling All Patches".format(ROCM_PATCH_VERSION))
    print("=" * 70)
    print()
    
    status = {}
    
    # Step 1: Multiprocessing (before torch import)
    print("Step 1: Configuring multiprocessing...")
    status['multiprocessing'] = setup_multiprocessing()
    print()
    
    # Step 2: Environment (before torch import)
    print("Step 2: Configuring environment variables...")
    setup_environment()
    status['environment'] = True
    print()
    
    # Step 3: Import torch
    print("Step 3: Importing PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        status['torch_import'] = True
    except Exception as e:
        print(f"❌ Failed to import torch: {e}")
        status['torch_import'] = False
        return status
    print()
    
    # Step 4: Patch DataLoader
    print("Step 4: Patching DataLoader...")
    status['dataloader'] = patch_dataloader()
    print()
    
    # Step 5: Enable MIOpen bypass
    print("Step 5: Enabling MIOpen bypass...")
    status['miopen_bypass'] = enable_miopen_bypass(strategy=miopen_strategy)
    print()
    
    # Summary
    print("=" * 70)
    print("Patch Summary:")
    print("=" * 70)
    for name, enabled in status.items():
        symbol = "✓" if enabled else "✗"
        print(f"  {symbol} {name}")
    
    all_ok = all(status.values())
    if all_ok:
        print("\n✅ All patches enabled successfully!")
    else:
        print("\n⚠️  Some patches failed - check warnings above")
    
    print("=" * 70)
    print()
    
    return status


# Convenience imports
try:
    from .miopen_bypass.conv2d_fallback import (
        SafeConv2d,
        Conv2dBypassConfig,
        FallbackStrategy,
        patch_model,
        patch_torch_nn,
        enable_miopen_bypass,
        print_bypass_report
    )
    __all__ = [
        'enable_all_patches',
        'setup_multiprocessing',
        'setup_environment',
        'patch_dataloader',
        'enable_miopen_bypass',
        'SafeConv2d',
        'Conv2dBypassConfig',
        'FallbackStrategy',
        'patch_model',
        'patch_torch_nn',
        'print_bypass_report',
    ]
except ImportError:
    __all__ = [
        'enable_all_patches',
        'setup_multiprocessing',
        'setup_environment',
        'patch_dataloader',
    ]
