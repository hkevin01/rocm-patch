"""
RMCP Immediate Workaround - CPU Fallback for Problematic Operations
Import this at the top of your training script to avoid Conv2d crashes
"""
import torch
import torch.nn as nn
import warnings

class SafeConv2d(nn.Conv2d):
    """Conv2d wrapper that uses CPU for forward pass to avoid GPU crashes"""

    def forward(self, x):
        # Move to CPU, run conv, move back to GPU
        device = x.device
        if device.type == 'cuda':
            # Move both input and layer to CPU
            self_cpu = self.cpu()
            x_cpu = x.cpu()
            result_cpu = nn.Conv2d.forward(self_cpu, x_cpu)
            # Move layer back to GPU
            self.to(device)
            return result_cpu.to(device)
        else:
            return super().forward(x)

def patch_conv2d():
    """Replace torch.nn.Conv2d with safe version"""
    print("🔧 RMCP: Patching Conv2d to use CPU fallback")
    torch.nn.Conv2d = SafeConv2d
    print("✅ RMCP: Conv2d patched - training will use CPU for convolutions")
    print("⚠️  Performance: ~10x slower but stable")

if __name__ == "__main__":
    print("RMCP Workaround Module")
    print("=" * 70)
    print("Usage:")
    print("  import sys")
    print("  sys.path.insert(0, '/home/kevin/Projects/rocm-patch/src')")
    print("  from rmcp_workaround import patch_conv2d")
    print("  patch_conv2d()  # Call BEFORE creating any models")
