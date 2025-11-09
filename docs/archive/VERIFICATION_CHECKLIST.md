# RDNA1 ROCm Patch - Verification Checklist

Use this checklist to verify the project status and ensure everything is properly configured.

---

## ✅ Installation Verification

### ROCm Installation
```bash
# Check ROCm version (should be 6.2.4)
- [ ] /opt/rocm/bin/hipcc --version
      Expected: HIP version: 6.2.41133

# Check ROCm info
- [ ] /opt/rocm/bin/rocminfo | grep "Name:" | head -1
      Expected: Should list your AMD GPU

# Verify user groups
- [ ] groups | grep -E "render|video"
      Expected: Both render and video groups present
```

### PyTorch Version Check
```bash
# Check PyTorch and ROCm version
- [ ] python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'ROCm: {torch.version.hip}')"
      Expected: PyTorch: 2.5.1+rocm6.2, ROCm: 6.2.41133
```

---

## ✅ Build Verification

### MIOpen Source
```bash
# Check if source exists
- [ ] ls -la /tmp/MIOpen/src/hip/handlehip.cpp
- [ ] ls -la /tmp/MIOpen/src/convolution_api.cpp

# Verify patches are in source
- [ ] grep -n "RDNA1" /tmp/MIOpen/src/hip/handlehip.cpp
      Expected: Should find RDNA1 detection code
      
- [ ] grep -n "RDNA1 PATCH" /tmp/MIOpen/src/convolution_api.cpp
      Expected: Should find "[RDNA1 PATCH]" messages
```

### Build Directory
```bash
# Check build directory exists
- [ ] ls -la /tmp/MIOpen/build_rdna1/

# Check if build completed
- [ ] ls -la /tmp/MIOpen/build_rdna1/Makefile
      Expected: Makefile should exist (indicates CMake succeeded)
```

### Built Library
```bash
# Check custom installation
- [ ] ls -lh /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0
      Expected: ~447MB file

# Check patches in library
- [ ] strings /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0 | grep "RDNA1 PATCH"
      Expected: Should find "[RDNA1 PATCH]" strings
```

---

## ✅ Deployment Verification

### PyTorch Integration
```bash
# Check PyTorch's library
- [ ] ls -lh ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so
      Expected: ~447MB file (same size as custom build)

# Check backup exists
- [ ] ls -la ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so.original
      Expected: ~1.4GB file (original library)

# Verify MD5 matches
- [ ] md5sum ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so \
            /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0
      Expected: Both should have same MD5 hash

# Check patches in PyTorch's library
- [ ] strings ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so | grep "RDNA1 PATCH"
      Expected: Should find "[RDNA1 PATCH]" strings
```

### Library Dependencies
```bash
# Check library links to correct ROCm
- [ ] ldd ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so | grep rocm
      Expected: Should show /opt/rocm libraries
```

---

## ✅ Runtime Verification

### Environment Variables
```bash
# Check environment setup
- [ ] echo $HSA_OVERRIDE_GFX_VERSION
      Expected: 10.3.0

- [ ] echo $MIOPEN_FORCE_RDNA1
      Expected: 1
```

### Python Runtime Test
```bash
# Run automated test
- [ ] cd ~/Projects/rocm-patch/scripts
- [ ] ./test_rdna1_patches.sh
      Expected: Should show "✅ RDNA1 patches found in library"
```

### Manual Python Test
```python
# Run this Python code:
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Arch: {torch.cuda.get_device_properties(0).gcnArchName}")
```

**Expected Output:**
```
- [ ] PyTorch version shows 2.5.1+rocm6.2
- [ ] CUDA available is True
- [ ] GPU name is "AMD Radeon RX 5600 XT" or similar
- [ ] Arch shows gfx1010 (or gfx1030 if spoofed)
```

### Patch Execution Test
```bash
# Set environment and run convolution test
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FORCE_RDNA1=1
export MIOPEN_LOG_LEVEL=7

python3 << 'EOF'
import torch
x = torch.randn(1, 3, 224, 224).cuda()
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
try:
    y = conv(x)
    print("Convolution succeeded (unexpected!)")
except Exception as e:
    print(f"Convolution failed (expected): {type(e).__name__}")
