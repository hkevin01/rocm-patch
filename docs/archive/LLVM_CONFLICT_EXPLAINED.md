# LLVM Conflict Explanation & Real Solutions

## 🔬 What is an LLVM Conflict?

### The Problem

**LLVM (Low Level Virtual Machine)** is the compiler infrastructure that:
1. Compiles GPU kernels (OpenCL, HIP code) to GPU bytecode
2. Generates bitcode files (`.bc`) for GPU libraries
3. Links different GPU code modules together

**The Conflict**:
```
ROCm 7.0.2 (installed) → Uses LLVM 20 → Creates LLVM 20 bitcode
ROCm 6.2.x (we want)   → Uses LLVM 16 → Expects LLVM 16 bitcode
System LLVM            → LLVM 16, 17, 18 available
```

**What happens when we try to build**:
```
1. We use system LLVM 16 to compile ROCm 6.2.x sources ✅
2. Build tries to link against `/opt/rocm/amdgcn/bitcode/opencl.bc` ❌
3. That file was compiled with LLVM 20 (from ROCm 7.0.2)
4. LLVM 16 can't read LLVM 20 bitcode → BUILD FAILS
```

**Error message breakdown**:
```
fatal error: cannot open file '/opt/rocm/amdgcn/bitcode/opencl.bc': 
Invalid attribute group entry (Producer: 'LLVM20.0.0git' Reader: 'LLVM 16.0.6')
                                         ↑ Made with LLVM 20  ↑ Reading with LLVM 16
```

## 🎯 Why Can't We Just Override/Workaround?

### Attempt 1: Use Different LLVM Version
**Problem**: Bitcode format is incompatible across major versions
```bash
LLVM 16 cannot read LLVM 20 bitcode (forward incompatibility)
LLVM 20 can sometimes read LLVM 16 bitcode (backward compatible)
```
**Workaround**: Use LLVM 20 from ROCm 7.0.2 to build ROCm 6.2.x
**Issue**: ROCm 6.2.x source code doesn't compile with LLVM 20 (API changes)

### Attempt 2: Method Overriding/Monkey Patching
**Problem**: The issue is at compile-time, not runtime
- Method overriding works for Python/runtime languages
- GPU kernels are compiled to machine code at build time
- Once compiled, you can't "override" GPU instructions

**Analogy**:
```
Runtime override: Like changing a recipe while cooking ✅
Compile-time issue: Like trying to use ingredients that don't exist yet ❌
```

### Attempt 3: LD_PRELOAD Library Intercept
**What we tried**: Create `libhip_rdna_fix.so` to intercept memory calls
**Problem**: 
1. Convolution kernels are pre-compiled in MIOpen library
2. LD_PRELOAD can't modify already-compiled GPU code
3. The crash happens inside MIOpen's kernel execution

**Analogy**: Trying to change a movie after it's been filmed

### Attempt 4: Environment Variables
**What we tried**: HSA_*, MIOPEN_*, etc.
**Problem**: These control runtime behavior, not memory allocation strategies
**Result**: Crash still happens in the same place

## 🔍 The Root Cause

The crash happens in **MIOpen** (ROCm's convolution library):

```
1. PyTorch calls: conv2d(input)
2. PyTorch/HIP calls: MIOpen convolution kernel
3. MIOpen kernel allocates GPU memory
4. MIOpen uses MTYPE_CC (cache-coherent) by default in ROCm 6.2+
5. RDNA1/2 hardware can't handle MTYPE_CC
6. → HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
7. → CRASH
```

**Where the fix needs to be**:
```
❌ Environment variables    (too late - after MIOpen loads)
❌ LD_PRELOAD intercept     (can't modify compiled kernels)
❌ Python monkey patching   (GPU kernels already compiled)
✅ MIOpen source code       (change MTYPE_CC → MTYPE_NC before compile)
✅ HIP runtime source       (force non-coherent allocations)
✅ Kernel driver            (reject coherent memory requests)
```

## 💡 Real Solutions

### Solution 1: Build MIOpen from Source (PROPER FIX)

**Time**: 3-4 hours  
**Complexity**: High  
**Result**: Full GPU acceleration

**Steps**:
```bash
# 1. Remove ROCm 7.0.2 temporarily
sudo apt remove rocm-dkms rocm-dev rocm-libs
# or rename /opt/rocm-7.0.2 to avoid conflicts

# 2. Install ROCm 6.2.x base (without MIOpen)
# Download from AMD archives

# 3. Build MIOpen with MTYPE_NC patch
git clone -b rocm-6.2.x https://github.com/ROCmSoftwarePlatform/MIOpen.git
cd MIOpen
# Apply patch to force MTYPE_NC
make -j$(nproc)
sudo make install

# 4. Test
python3 -c "import torch; conv = torch.nn.Conv2d(1,32,(64,1)).cuda(); ..."
```

**Patch to apply** (in MIOpen source):
```cpp
// File: src/ocl/convolutionocl.cpp
// Change memory flags from CL_MEM_READ_WRITE to CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
// This forces non-coherent allocations
```

### Solution 2: Use ROCm 5.7 (EASIEST)

**Time**: 30 minutes  
**Complexity**: Low  
**Result**: Full GPU acceleration

**Problem**: ROCm 5.7 not in Ubuntu 24.04 repos

**Workaround**:
```bash
# Download ROCm 5.7 debs from AMD archive
wget https://repo.radeon.com/rocm/apt/5.7/pool/main/r/rocm-dkms/...
sudo dpkg -i rocm-*.deb
sudo reboot
```

ROCm 5.7 uses MTYPE_NC by default → No RDNA1/2 crashes

### Solution 3: Docker with ROCm 5.7

**Time**: 1 hour  
**Complexity**: Medium  
**Result**: Full GPU acceleration in container

```dockerfile
FROM rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# Your training code here
# GPU passthrough: docker run --device=/dev/kfd --device=/dev/dri ...
```

**Pros**:
- Isolated from system ROCm 7.0.2
- Works with RDNA1/2
- Easy to switch versions

**Cons**:
- Docker overhead (~5-10%)
- Need GPU passthrough setup

### Solution 4: Patch MIOpen Binary (HACK)

**Time**: 2 hours  
**Complexity**: Very High  
**Result**: May work, may break

**Approach**:
```bash
# 1. Disassemble MIOpen library
objdump -d /opt/rocm/lib/libMIOpen.so > miopen.asm

# 2. Find memory allocation calls
grep -A10 "hip.*Malloc" miopen.asm

# 3. Use hex editor to change memory type flags
# 4. Hope it doesn't crash
```

**Not recommended** - fragile, may break on updates

### Solution 5: Use PyTorch CPU Backend (TEMPORARY)

**Time**: Immediate  
**Complexity**: None  
**Result**: Stable but 10x slower

```python
# Already created: src/rmcp_workaround.py
from rmcp_workaround import patch_conv2d
patch_conv2d()
```

Works now, but too slow for real training.

## 📊 Comparison

| Solution | Time | Complexity | Speed | Stability | Recommendation |
|----------|------|------------|-------|-----------|----------------|
| Build MIOpen | 3-4h | High | 100% | High | ⭐⭐⭐ Best |
| ROCm 5.7 | 30m | Low | 100% | High | ⭐⭐⭐⭐⭐ Easiest |
| Docker | 1h | Medium | 95% | High | ⭐⭐⭐⭐ Practical |
| Binary Patch | 2h | Very High | 100% | Low | ⭐ Risky |
| CPU Fallback | 0m | None | 10% | High | ⭐⭐ Temporary |

## 🎯 Recommended Path Forward

### Option A: Docker with ROCm 5.7 (FASTEST TO WORKING GPU)

```bash
# 1. Install Docker with GPU support
sudo apt install docker.io
sudo usermod -aG docker $USER
sudo reboot

# 2. Pull ROCm 5.7 PyTorch image
docker pull rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1

# 3. Test
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1 \
    python3 -c "import torch; print(torch.cuda.is_available())"

# 4. Run training in container
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    -v /home/kevin/Projects:/workspace \
    rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1 \
    bash
```

**Time to working GPU**: 30 minutes

### Option B: Download ROCm 5.7 Packages

```bash
# Create download script
cat > download_rocm57.sh << 'SCRIPT'
#!/bin/bash
ROCM_VERSION="5.7"
UBUNTU_VERSION="ubuntu20.04"
BASE_URL="https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/pool/main"

mkdir -p ~/rocm57_debs
cd ~/rocm57_debs

# Download essential packages
wget ${BASE_URL}/r/rocm-dkms/rocm-dkms_${ROCM_VERSION}*.deb
wget ${BASE_URL}/r/rocm-dev/rocm-dev_${ROCM_VERSION}*.deb
wget ${BASE_URL}/h/hip-runtime-amd/hip-runtime-amd_${ROCM_VERSION}*.deb
wget ${BASE_URL}/m/miopen-hip/miopen-hip_${ROCM_VERSION}*.deb

echo "Downloaded ROCm 5.7 packages to ~/rocm57_debs"
echo "To install:"
echo "  sudo apt remove rocm-dkms  # Remove ROCm 7.0"
echo "  sudo dpkg -i *.deb"
echo "  sudo apt-get install -f  # Fix dependencies"
echo "  sudo reboot"
SCRIPT

chmod +x download_rocm57.sh
./download_rocm57.sh
```

**Time to working GPU**: 1 hour

## 🔑 Key Takeaways

1. **LLVM conflicts are compile-time issues** - can't be fixed with runtime overrides
2. **The crash happens in pre-compiled MIOpen kernels** - need to recompile with different flags
3. **ROCm 6.2+ breaks RDNA1/2** - architectural change, not a bug
4. **ROCm 5.7 works perfectly** - uses MTYPE_NC by default
5. **Docker is the fastest path** - isolated environment, no system changes

## 📋 Next Steps

**I recommend trying Docker first** (30 minutes to working GPU):
```bash
cd /home/kevin/Projects/rocm-patch/scripts
cat > test_docker_rocm.sh << 'SCRIPT'
#!/bin/bash
docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v /home/kevin/Projects:/workspace \
    rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1 \
    python3 -c "
import torch
import torch.nn as nn
print('GPU Available:', torch.cuda.is_available())
print('GPU Name:', torch.cuda.get_device_name(0))

# The critical test
conv = nn.Conv2d(1, 32, (64, 1)).cuda()
x = torch.randn(16, 1, 64, 256).cuda()
y = conv(x)
print('✅ Conv2d works!')
"
SCRIPT
chmod +x test_docker_rocm.sh
```

Would you like me to help set up Docker with ROCm 5.7?

---

**Status**: LLVM conflict understood, Docker solution ready to implement
**Date**: November 6, 2025
