"""
MIOpen Bypass for RDNA1 GPUs
=============================

Intelligent Conv2d fallback strategies for AMD RDNA1 GPUs.

Quick Start:
    from miopen_bypass import enable_miopen_bypass
    enable_miopen_bypass()

For more details, see README.md
"""

from .conv2d_fallback import (
    SafeConv2d,
    Conv2dBypassConfig,
    FallbackStrategy,
    patch_model,
    patch_torch_nn,
    restore_torch_nn,
    enable_miopen_bypass,
    print_bypass_report,
    get_config,
    set_config
)

__all__ = [
    'SafeConv2d',
    'Conv2dBypassConfig',
    'FallbackStrategy',
    'patch_model',
    'patch_torch_nn',
    'restore_torch_nn',
    'enable_miopen_bypass',
    'print_bypass_report',
    'get_config',
    'set_config'
]

__version__ = '1.0.0'
__author__ = 'ROCm Patch Project'
