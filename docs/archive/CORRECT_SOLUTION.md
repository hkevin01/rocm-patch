# Correct RDNA1 (gfx1010) Solution for Conv2d

## The Real Problem (Corrected Understanding)

### What We Have
- **Hardware**: RX 5600 XT (Navi 10, gfx1010, RDNA1)
- **Memory Model**: Coarse-grained SVM (unified addressing, but no fine-grained coherence)
- **Capability**: Full device memory + explicit synchronization (perfect for Conv2d)

### What We're Doing Wrong
- **HSA_OVERRIDE_GFX_VERSION=10.3.0** makes ROCm think we have gfx1030 (RDNA2)
- **gfx1030 expects fine-grained SVM** which gfx1010 doesn't have
- **ROCm tries to use fine-grained memory operations** → crashes or hangs

### What We Should Do
**Use device memory with proper MIOpen configuration** - NO OVERRIDE NEEDED!

## Why Our Patches Failed

### ❌ All HSA Runtime Patches
**Wrong Approach**: Trying to "fix" memory model at runtime
**Reality**: Memory model is set by hardware, can't be changed
**Result**: System crashes because we're lying to the memory subsystem

### ✅ Correct Approach
**Don't spoof architecture** - configure MIOpen to work with native gfx1010

## The Actual Solution

### Option 1: MIOpen Configuration (RECOMMENDED)

Conv2d doesn't need fine-grained SVM! It just needs:
1. Device memory allocation (already works)
2. Proper kernel selection
3. Adequate workspace

**Implementation**:
```bash
# ~/.bashrc or launch script
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=SEARCH
export MIOPEN_WORKSPACE_LIMIT_MB=4096
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=6
export MIOPEN_USER_DB_PATH=~/.config/miopen

# If specific solvers fail, blacklist them:
# export MIOPEN_DISABLE_SOLVERS=ConvAsmImplicitGemm

# Use native architecture (no override!)
unset HSA_OVERRIDE_GFX_VERSION
```

**Test**:
```python
import torch
x = torch.randn(32, 64, 56, 56, device="cuda", dtype=torch.float32)
conv = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
y = conv(x)  # Should work!
```

### Option 2: Build PyTorch with Native gfx1010 Support

If MIOpen doesn't have gfx1010 kernels:

```bash
# Build PyTorch from source
export PYTORCH_ROCM_ARCH="gfx1010"
export USE_ROCM=1
export USE_MIOPEN=1
python setup.py install
```

This generates kernels for gfx1010 instead of relying on gfx1030 spoofing.

### Option 3: Fallback to im2col+GEMM

If MIOpen conv kernels are problematic:

```python
import torch
import torch.nn.functional as F

def conv2d_fallback(input, weight, bias=None, stride=1, padding=0):
    """Fallback conv using im2col + matrix multiply"""
    # Use unfold to do im2col
    unfold = torch.nn.Unfold(
        kernel_size=weight.shape[2:],
        padding=padding,
        stride=stride
    )
    input_unfold = unfold(input)  # (N, C*kH*kW, L)
    
    # Reshape weight for matmul
    weight_flat = weight.view(weight.size(0), -1)  # (out_c, in_c*kH*kW)
    
    # Matrix multiply
    output = weight_flat @ input_unfold  # (N, out_c, L)
    
    # Reshape back
    out_h = (input.size(2) + 2*padding - weight.size(2)) // stride + 1
    out_w = (input.size(3) + 2*padding - weight.size(3)) // stride + 1
    output = output.view(input.size(0), weight.size(0), out_h, out_w)
    
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    
    return output
```

## What Each Solution Provides

### Option 1: MIOpen Config ✅ FASTEST
**Pros**:
- Uses optimized assembly kernels
- No recompilation needed
- Respects hardware capabilities

**Cons**:
- Need to tune for specific shapes
- May require solver blacklisting

### Option 2: Native Build ✅ MOST ROBUST
**Pros**:
- Native gfx1010 support
- No architecture spoofing
- Generates optimal kernels

**Cons**:
- Long compile time
- Need to rebuild for updates

### Option 3: Fallback ✅ SAFEST
**Pros**:
- Always works
- No special config needed
- Easy to debug

**Cons**:
- Slower than optimized kernels
- More memory usage

## Testing Plan

### Step 1: Test without Override
```bash
unset HSA_OVERRIDE_GFX_VERSION
export MIOPEN_FIND_MODE=1
python test_conv2d.py
```

**Expected**: May fail with "No suitable kernel" or be slow

### Step 2: Configure MIOpen
```bash
export MIOPEN_WORKSPACE_LIMIT_MB=4096
export MIOPEN_USER_DB_PATH=~/.config/miopen
python test_conv2d.py
```

**Expected**: Slower first run (tuning), then fast

### Step 3: Blacklist Problem Solvers
```bash
export MIOPEN_DISABLE_SOLVERS=ConvAsmImplicitGemm
python test_conv2d.py
```

**Expected**: May find alternative working solver

### Step 4: Use Fallback if Needed
```python
# Replace torch.nn.Conv2d with custom fallback
# Test performance vs correctness
```

## Why This is Better Than Kernel Patching

### ❌ Kernel/HSA Patches
- Lie to hardware about capabilities
- Cause system instability
- Can't emulate missing hardware features

### ✅ Proper Configuration
- Respects hardware limits
- Uses available features correctly
- Stable and maintainable

## Implementation Priority

1. **[HIGH]** Test with MIOPEN_FIND_MODE=1 (no override)
2. **[HIGH]** Configure workspace limits
3. **[MEDIUM]** Blacklist problematic solvers
4. **[MEDIUM]** Implement fallback for edge cases
5. **[LOW]** Build PyTorch from source if needed

## Next Steps

- [ ] Remove all HSA runtime patches
- [ ] Test native gfx1010 with MIOpen config
- [ ] Document working solver combinations
- [ ] Create shape-specific fallback routing
- [ ] (Optional) Build PyTorch with gfx1010 support

