"""
RDNA1-Compatible PyTorch Layers
Overrides Conv2d to avoid MIOpen's cache-coherent memory allocations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

try:
    import rdna1_conv2d
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False
    print("WARNING: rdna1_conv2d extension not compiled. Using CPU fallback.")


class RDNA1Conv2d(nn.Conv2d):
    """
    Drop-in replacement for nn.Conv2d that works on RDNA1 GPUs.
    
    This avoids MIOpen's cache-coherent memory allocations by:
    1. Moving tensors to CPU
    2. Running convolution on CPU (no MIOpen)
    3. Moving result back to GPU
    
    Usage:
        Replace: conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        With:    conv = RDNA1Conv2d(in_channels, out_channels, kernel_size)
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
        dtype=None
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
        
        self.use_extension = EXTENSION_AVAILABLE
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that avoids MIOpen.
        
        Strategy:
        1. Keep weights on GPU (no MIOpen calls)
        2. Move input to CPU
        3. Move weights to CPU  
        4. Compute on CPU (uses optimized CPU kernels)
        5. Move result back to GPU
        
        This is slower than GPU convolution but faster than crashing!
        """
        
        if not input.is_cuda:
            # Already on CPU, use parent implementation
            return super().forward(input)
        
        # Save original device
        original_device = input.device
        
        # Move to CPU for computation
        input_cpu = input.cpu()
        weight_cpu = self.weight.cpu()
        bias_cpu = self.bias.cpu() if self.bias is not None else None
        
        # Use standard CPU convolution (no MIOpen)
        if self.padding_mode != 'zeros':
            # Handle special padding modes
            output_cpu = self._conv_forward(
                nn.functional.pad(input_cpu, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight_cpu,
                bias_cpu
            )
        else:
            output_cpu = self._conv_forward(input_cpu, weight_cpu, bias_cpu)
        
        # Move result back to GPU
        output = output_cpu.to(original_device)
        
        return output


def patch_model_for_rdna1(model: nn.Module) -> nn.Module:
    """
    Recursively replace all Conv2d layers with RDNA1Conv2d.
    
    Usage:
        model = YourModel()
        model = patch_model_for_rdna1(model)
        model = model.cuda()  # Now safe to move to GPU!
    
    Args:
        model: PyTorch model to patch
        
    Returns:
        Patched model with RDNA1-compatible Conv2d layers
    """
    
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and not isinstance(module, RDNA1Conv2d):
            # Get Conv2d parameters
            new_conv = RDNA1Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode
            )
            
            # Copy weights and bias
            new_conv.weight = module.weight
            if module.bias is not None:
                new_conv.bias = module.bias
            
            # Replace module
            setattr(model, name, new_conv)
            print(f"✓ Patched {name}: Conv2d -> RDNA1Conv2d")
        else:
            # Recursively patch child modules
            patch_model_for_rdna1(module)
    
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("RDNA1 Conv2d Layer Test")
    print("=" * 70)
    
    # Test if running on RDNA1
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU: {device_name}")
        
        if "5600" in device_name or "5700" in device_name:
            print("✓ RDNA1 GPU detected - this layer is for you!")
        else:
            print("⚠ Not RDNA1 - this layer works but isn't necessary")
    else:
        print("No GPU available")
    
    print("\n" + "=" * 70)
    print("Testing RDNA1Conv2d...")
    print("=" * 70)
    
    # Create test layer
    conv = RDNA1Conv2d(1, 32, (64, 1))
    
    if torch.cuda.is_available():
        print("\n→ Moving layer to GPU...")
        conv = conv.cuda()
        print("✓ Layer on GPU")
        
        print("\n→ Creating input tensor...")
        x = torch.randn(1, 1, 128, 64).cuda()
        print(f"✓ Input shape: {x.shape}")
        
        print("\n→ Running forward pass...")
        try:
            y = conv(x)
            print(f"✓ SUCCESS! Output shape: {y.shape}")
            print("\n🎉 RDNA1Conv2d works on your GPU!")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Skipping GPU test (no GPU available)")

