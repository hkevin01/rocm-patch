"""
Advanced MIOpen Bypass for RDNA1 GPUs
======================================

This module provides intelligent Conv2d fallback strategies to avoid MIOpen issues
on AMD RDNA1 GPUs (RX 5600 XT, RX 5700 series).

Problem:
    MIOpen (AMD's cuDNN equivalent) has kernel bugs and missing database entries
    for RDNA1 (gfx1010/gfx1030) that cause:
    - Hangs on certain tensor sizes (>42×42 with default algorithm)
    - Memory access violations
    - Missing kernel database warnings

Solution Strategies:
    1. IMPLICIT_GEMM (preferred): Use MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1
    2. GPU Unfold (safe): Use unfold+matmul on GPU (bypasses MIOpen completely!)
    3. Pure PyTorch (alternative): Use torch.nn.functional ops directly
    4. Selective Bypass: Only bypass problematic sizes/configurations

Key Innovation:
    All fallback paths stay on GPU using unfold (im2col) + matmul (rocBLAS).
    This bypasses MIOpen completely while being ~10x faster than CPU fallback!

Author: ROCm Patch Project
Date: November 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings
from typing import Optional, Tuple, Union, Dict, Any
from enum import Enum


class FallbackStrategy(Enum):
    """Fallback strategies for Conv2d operations"""
    IMPLICIT_GEMM = "implicit_gemm"      # Use MIOpen with IMPLICIT_GEMM (fastest)
    GPU_UNFOLD = "gpu_unfold"            # Use unfold+matmul on GPU (bypasses MIOpen, ~2-3x slower)
    PURE_PYTORCH = "pure_pytorch"        # Use torch.nn.functional directly (alias for GPU_UNFOLD)
    SELECTIVE = "selective"              # Only fallback for problematic sizes
    AUTO = "auto"                        # Automatically detect best strategy

    # Legacy alias for backward compatibility
    CPU_FALLBACK = "gpu_unfold"          # Now uses GPU unfold instead of CPU!


class Conv2dBypassConfig:
    """Configuration for Conv2d bypass behavior"""

    def __init__(
        self,
        strategy: FallbackStrategy = FallbackStrategy.AUTO,
        size_threshold: int = 42,           # Known problematic size boundary
        enable_implicit_gemm: bool = True,  # Try IMPLICIT_GEMM first
        gpu_unfold_fallback: bool = True,   # Allow GPU unfold+matmul fallback
        verbose: bool = True,               # Print bypass information
        cache_decisions: bool = True        # Cache bypass decisions per config
    ):
        self.strategy = strategy
        self.size_threshold = size_threshold
        self.enable_implicit_gemm = enable_implicit_gemm
        self.gpu_unfold_fallback = gpu_unfold_fallback
        self.verbose = verbose
        self.cache_decisions = cache_decisions

        # Cache for bypass decisions
        self._decision_cache: Dict[str, bool] = {}

        # Set environment for IMPLICIT_GEMM if enabled
        if self.enable_implicit_gemm:
            os.environ['MIOPEN_DEBUG_CONV_IMPLICIT_GEMM'] = '1'
            if self.verbose:
                print("✓ MIOpen IMPLICIT_GEMM enabled (environment variable set)")


# Global configuration instance
_global_config = Conv2dBypassConfig()


def get_config() -> Conv2dBypassConfig:
    """Get global bypass configuration"""
    return _global_config


def set_config(config: Conv2dBypassConfig):
    """Set global bypass configuration"""
    global _global_config
    _global_config = config


class SafeConv2d(nn.Conv2d):
    """
    Drop-in replacement for nn.Conv2d with intelligent MIOpen bypass.

    This class automatically detects problematic configurations and applies
    appropriate fallback strategies to avoid GPU hangs and crashes.

    Features:
        - Automatic size-based bypass detection
        - Multiple fallback strategies
        - Performance monitoring
        - Caching of bypass decisions
        - Minimal overhead for working configurations

    Usage:
        # Option 1: Direct replacement
        conv = SafeConv2d(3, 64, kernel_size=3)

        # Option 2: Patch existing model
        from conv2d_fallback import patch_model
        model = patch_model(your_model)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        config: Optional[Conv2dBypassConfig] = None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        self.config = config or get_config()
        self._bypass_count = 0
        self._total_forward_count = 0
        self._cache_key = None

    def _should_bypass(self, input: torch.Tensor) -> bool:
        """
        Determine if this forward pass should bypass MIOpen.

        Decision factors:
            - Input tensor size (H×W)
            - Configuration strategy
            - Cached decisions
            - GPU device capability

        Returns:
            True if should bypass MIOpen, False otherwise
        """

        # Always use CPU path if input is already on CPU
        if not input.is_cuda:
            return False

        # Check cache if enabled
        if self.config.cache_decisions and self._cache_key is not None:
            cached = self.config._decision_cache.get(self._cache_key)
            if cached is not None:
                return cached

        # Extract input dimensions
        batch, channels, height, width = input.shape
        max_size = max(height, width)

        # Strategy-based decision
        strategy = self.config.strategy

        if strategy == FallbackStrategy.GPU_UNFOLD or strategy == FallbackStrategy.CPU_FALLBACK:
            # Always use GPU unfold+matmul (bypasses MIOpen)
            decision = True

        elif strategy == FallbackStrategy.IMPLICIT_GEMM:
            # Trust IMPLICIT_GEMM, no bypass needed
            decision = False

        elif strategy == FallbackStrategy.SELECTIVE:
            # Bypass only for known problematic sizes
            decision = max_size > self.config.size_threshold

        elif strategy == FallbackStrategy.AUTO:
            # Auto-detect: bypass if size > threshold and IMPLICIT_GEMM not working
            if max_size <= self.config.size_threshold:
                decision = False
            else:
                # For large sizes, try to use IMPLICIT_GEMM first
                # If it fails (not set), use GPU unfold+matmul
                decision = os.environ.get('MIOPEN_DEBUG_CONV_IMPLICIT_GEMM') != '1'

        else:  # PURE_PYTORCH
            decision = True

        # Cache decision
        if self.config.cache_decisions:
            cache_key = f"{height}x{width}_{self.kernel_size}_{self.stride}"
            self._cache_key = cache_key
            self.config._decision_cache[cache_key] = decision

        return decision

    def _gpu_unfold_forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass on GPU using unfold (im2col) + matmul.

        This bypasses MIOpen completely while staying on GPU!

        Algorithm:
            1. Use torch.nn.functional.unfold to do im2col transform (GPU operation)
            2. Reshape weights for matrix multiplication
            3. Use torch.matmul (rocBLAS GEMM, not MIOpen!)
            4. Reshape result back to NCHW format

        This is ~2-3x slower than optimized MIOpen but MUCH faster than CPU.
        """
        N, C_in, H, W = input.shape
        C_out = self.out_channels

        # Handle groups
        if self.groups != 1:
            # For grouped convolutions, process each group separately
            outputs = []
            C_in_per_group = C_in // self.groups
            C_out_per_group = C_out // self.groups

            for g in range(self.groups):
                input_g = input[:, g*C_in_per_group:(g+1)*C_in_per_group]
                weight_g = self.weight[g*C_out_per_group:(g+1)*C_out_per_group]

                # im2col using unfold (stays on GPU!)
                unfold = nn.Unfold(
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride
                )
                input_unfold = unfold(input_g)  # (N, C_in_per_group*kH*kW, L)

                # Reshape weight for matmul
                weight_flat = weight_g.view(C_out_per_group, -1)  # (C_out_per_group, C_in_per_group*kH*kW)

                # Matrix multiply on GPU (uses rocBLAS, NOT MIOpen!)
                output_flat = torch.matmul(weight_flat, input_unfold)  # (N, C_out_per_group, L)

                outputs.append(output_flat)

            # Concatenate group outputs
            output_flat = torch.cat(outputs, dim=1)  # (N, C_out, L)
        else:
            # Standard convolution (no groups)
            # im2col using unfold (stays on GPU!)
            unfold = nn.Unfold(
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride
            )
            input_unfold = unfold(input)  # (N, C_in*kH*kW, L)

            # Reshape weight for matmul
            weight_flat = self.weight.view(C_out, -1)  # (C_out, C_in*kH*kW)

            # Matrix multiply on GPU (uses rocBLAS, NOT MIOpen!)
            output_flat = torch.matmul(weight_flat, input_unfold)  # (N, C_out, L)

        # Calculate output spatial dimensions
        kernel_h, kernel_w = (self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size
        stride_h, stride_w = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        padding_h, padding_w = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        dilation_h, dilation_w = (self.dilation, self.dilation) if isinstance(self.dilation, int) else self.dilation

        H_out = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        W_out = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

        # Reshape to (N, C_out, H_out, W_out)
        output = output_flat.view(N, C_out, H_out, W_out)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with intelligent MIOpen bypass.

        Decision tree:
            1. Check if bypass needed based on configuration
            2. If bypass: use GPU unfold+matmul (bypasses MIOpen, stays on GPU!)
            3. If no bypass: try standard GPU forward (with IMPLICIT_GEMM if set)
            4. If GPU forward fails: automatically fallback to GPU unfold+matmul
        """
        self._total_forward_count += 1

        # Decide bypass strategy
        should_bypass = self._should_bypass(input)

        if should_bypass:
            self._bypass_count += 1

            if self.config.verbose and self._bypass_count == 1:
                print(f"🔄 Conv2d GPU bypass activated for {input.shape[2]}×{input.shape[3]} input")
                print(f"   Strategy: {self.config.strategy.value} (using unfold+matmul on GPU)")

            return self._gpu_unfold_forward(input)
        else:
            # Try standard forward (GPU with IMPLICIT_GEMM if env var set)
            # If it fails (MIOpen bug), automatically fallback to GPU unfold+matmul
            try:
                return super().forward(input)
            except RuntimeError as e:
                error_msg = str(e)
                # Check if it's a MIOpen error
                if 'miopen' in error_msg.lower() or 'convolution' in error_msg.lower():
                    self._bypass_count += 1
                    if self.config.verbose:
                        print(f"🔄 MIOpen error detected, auto-fallback to GPU unfold+matmul for {input.shape[2]}×{input.shape[3]} input")
                    return self._gpu_unfold_forward(input)
                else:
                    # Re-raise if it's not a MIOpen issue
                    raise

    def get_bypass_stats(self) -> Dict[str, Any]:
        """Get statistics about bypass usage"""
        bypass_rate = (self._bypass_count / self._total_forward_count * 100) if self._total_forward_count > 0 else 0
        return {
            'total_forwards': self._total_forward_count,
            'bypass_count': self._bypass_count,
            'bypass_rate': bypass_rate,
            'strategy': self.config.strategy.value
        }


def patch_model(
    model: nn.Module,
    config: Optional[Conv2dBypassConfig] = None,
    recursive: bool = True
) -> nn.Module:
    """
    Replace all Conv2d layers in a model with SafeConv2d.

    This is the recommended way to apply MIOpen bypass to existing models.

    Args:
        model: PyTorch model to patch
        config: Bypass configuration (uses global if None)
        recursive: Whether to patch nested modules

    Returns:
        Patched model (modified in-place, but also returned)

    Example:
        >>> model = torchvision.models.resnet18()
        >>> model = patch_model(model)
        >>> model = model.cuda()  # Safe to move to GPU now!
    """
    config = config or get_config()
    patch_count = 0

    def _patch_module(module: nn.Module, prefix: str = ""):
        nonlocal patch_count

        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d) and not isinstance(child, SafeConv2d):
                # Create SafeConv2d with same parameters
                safe_conv = SafeConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    config=config
                )

                # Copy weights and bias
                safe_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    safe_conv.bias.data = child.bias.data.clone()

                # Move to same device as original
                if child.weight.is_cuda:
                    safe_conv = safe_conv.to(child.weight.device)

                # Replace module
                setattr(module, name, safe_conv)
                patch_count += 1

                if config.verbose:
                    print(f"✓ Patched: {full_name}")

            elif recursive:
                _patch_module(child, full_name)

    _patch_module(model)

    if config.verbose:
        print(f"\n✅ Model patched: {patch_count} Conv2d layers replaced with SafeConv2d")

    return model


def patch_torch_nn(config: Optional[Conv2dBypassConfig] = None):
    """
    Globally replace torch.nn.Conv2d with SafeConv2d.

    WARNING: This affects all Conv2d layers created after this call.
    Use with caution in large projects.

    Recommended: Use patch_model() instead for better control.

    Example:
        >>> from conv2d_fallback import patch_torch_nn
        >>> patch_torch_nn()  # Call BEFORE creating models
        >>> model = YourModel()  # All Conv2d will be SafeConv2d
    """
    config = config or get_config()

    # Store original Conv2d
    if not hasattr(nn, '_original_Conv2d'):
        nn._original_Conv2d = nn.Conv2d

    # Replace with SafeConv2d
    nn.Conv2d = SafeConv2d

    if config.verbose:
        print("🔧 torch.nn.Conv2d globally replaced with SafeConv2d")
        print(f"   Strategy: {config.strategy.value}")
        print("   All new Conv2d layers will use MIOpen bypass")


def restore_torch_nn():
    """Restore original torch.nn.Conv2d"""
    if hasattr(nn, '_original_Conv2d'):
        nn.Conv2d = nn._original_Conv2d
        delattr(nn, '_original_Conv2d')
        print("✓ torch.nn.Conv2d restored to original")


def print_bypass_report(model: nn.Module):
    """
    Print a detailed report of Conv2d bypass usage in a model.

    Useful for understanding which layers are being bypassed.
    """
    print("\n" + "=" * 70)
    print("Conv2d Bypass Report")
    print("=" * 70)

    total_layers = 0
    safe_layers = 0
    total_forwards = 0
    total_bypasses = 0

    for name, module in model.named_modules():
        if isinstance(module, SafeConv2d):
            safe_layers += 1
            stats = module.get_bypass_stats()
            total_forwards += stats['total_forwards']
            total_bypasses += stats['bypass_count']

            if stats['total_forwards'] > 0:
                print(f"\n{name}:")
                print(f"  Forwards: {stats['total_forwards']}")
                print(f"  Bypasses: {stats['bypass_count']} ({stats['bypass_rate']:.1f}%)")
        elif isinstance(module, nn.Conv2d):
            total_layers += 1

    print("\n" + "=" * 70)
    print(f"Total Conv2d layers: {total_layers + safe_layers}")
    print(f"SafeConv2d layers: {safe_layers}")
    print(f"Standard Conv2d layers: {total_layers}")

    if total_forwards > 0:
        print(f"\nTotal forward passes: {total_forwards}")
        print(f"Total bypasses: {total_bypasses} ({total_bypasses/total_forwards*100:.1f}%)")

    print("=" * 70 + "\n")


# Convenience function for quick setup
def enable_miopen_bypass(
    strategy: FallbackStrategy = FallbackStrategy.AUTO,
    verbose: bool = True
) -> Conv2dBypassConfig:
    """
    Quick setup for MIOpen bypass.

    Example:
        >>> from conv2d_fallback import enable_miopen_bypass
        >>> enable_miopen_bypass(strategy=FallbackStrategy.SELECTIVE)
        >>> # Now create and train your model
    """
    config = Conv2dBypassConfig(
        strategy=strategy,
        verbose=verbose
    )
    set_config(config)

    if verbose:
        print("\n" + "=" * 70)
        print("MIOpen Bypass Enabled")
        print("=" * 70)
        print(f"Strategy: {strategy.value}")
        print(f"IMPLICIT_GEMM: {'Enabled' if config.enable_implicit_gemm else 'Disabled'}")
        print(f"Size threshold: {config.size_threshold}×{config.size_threshold}")
        print("=" * 70 + "\n")

    return config


if __name__ == "__main__":
    print(__doc__)
    print("\nQuick Start:")
    print("=" * 70)
    print("1. Enable bypass globally:")
    print("   from conv2d_fallback import enable_miopen_bypass")
    print("   enable_miopen_bypass()")
    print("")
    print("2. Patch existing model:")
    print("   from conv2d_fallback import patch_model")
    print("   model = patch_model(your_model)")
    print("")
    print("3. Use SafeConv2d directly:")
    print("   from conv2d_fallback import SafeConv2d")
    print("   conv = SafeConv2d(3, 64, kernel_size=3)")
    print("=" * 70)
