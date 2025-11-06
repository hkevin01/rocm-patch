# ROCm Source Patching Scripts

Automated scripts for patching ROCm at the source level to fix RDNA1/2 memory coherency issues.

## Overview

These scripts implement a complete source-level patching solution for ROCm that fixes memory access faults on AMD RDNA1 (RX 5000 series) and RDNA2 (RX 6000 series) GPUs. The patches modify ROCm runtime components and the amdgpu kernel module to use safe memory configurations.

## Scripts

### 1. `patch_rocm_source.sh` - ROCm Runtime Patcher

**Purpose**: Clones, patches, and builds ROCm runtime components (HIP, ROCR, ROCT)

**What it does**:
- Sets up build environment with all dependencies
- Clones ROCm source repositories (HIP, ROCR, ROCT, CLR)
- Creates and applies patches for memory coherency fixes
- Builds patched ROCm from source
- Installs to `/opt/rocm-patched`
- Configures system environment

**Time**: 2-3 hours (mostly compilation)

**Disk Space**: ~10GB

**Usage**:
```bash
cd ~/Projects/rocm-patch/scripts
./patch_rocm_source.sh
```

**What gets patched**:
- **HIP Runtime** (`hip_memory.cpp`): Forces non-coherent allocations for RDNA1/2
- **ROCR Runtime** (`amd_gpu_agent.cpp`): Detects RDNA GPUs and applies safe defaults
- **Memory Allocator**: Conservative fragment sizes, disabled aggressive caching

### 2. `patch_kernel_module.sh` - Kernel Module Patcher

**Purpose**: Patches the amdgpu kernel module for system-level memory fixes

**What it does**:
- Checks prerequisites (kernel headers, build tools)
- Creates kernel patch for `gmc_v10_0.c`
- Builds patched amdgpu.ko module
- Backs up original module
- Installs patched module
- Requires reboot to take effect

**Time**: 15-30 minutes

**Requires**: Reboot after installation

**Usage**:
```bash
cd ~/Projects/rocm-patch/scripts
./patch_kernel_module.sh
# Reboot system after completion
```

**What gets patched**:
- **gmc_v10_0.c**: RDNA detection and safe memory defaults at driver level
- Forces non-coherent aperture base
- Sets conservative VM fragment size (512KB)
- Disables aggressive retry behavior

### 3. `test_patched_rocm.sh` - Test Suite

**Purpose**: Comprehensive testing of patched ROCm installation

**What it tests**:
1. ROCm environment and paths
2. `rocminfo` functionality
3. HIP compilation
4. HIP memory operations (critical for RDNA issues)
5. PyTorch integration (if available)
6. Kernel memory fault detection
7. Patch verification

**Time**: 5-10 minutes

**Usage**:
```bash
cd ~/Projects/rocm-patch/scripts

# Test default installation
./test_patched_rocm.sh

# Test specific ROCm path
ROCM_PATH=/opt/rocm-patched ./test_patched_rocm.sh
```

**Output**: Pass/fail for each test with detailed logs in `/tmp/rocm-patch-tests/`

## Execution Workflow

### Complete Installation

```bash
# 1. Clone this repository
cd ~/Projects
git clone <repo-url> rocm-patch
cd rocm-patch/scripts

# 2. Patch and build ROCm runtime (2-3 hours)
./patch_rocm_source.sh

# 3. (Optional) Patch kernel module
./patch_kernel_module.sh
sudo reboot  # Required for kernel module

# 4. Test installation
./test_patched_rocm.sh

# 5. Configure your environment
source /etc/profile.d/rocm-patched.sh

# 6. Test with your workloads
```

### Quick Test (No Build)

If you just want to test patches without building:

```bash
# Test existing ROCm installation
ROCM_PATH=/opt/rocm ./test_patched_rocm.sh
```

## Installation Locations

- **Patched ROCm**: `/opt/rocm-patched`
- **Source code**: `~/rocm-source-patches/`
- **Patch files**: `~/Projects/rocm-patch/patches/`
- **Environment**: `/etc/profile.d/rocm-patched.sh`

## Environment Variables

After installation, the following environment variables are configured:

```bash
export ROCM_PATH=/opt/rocm-patched
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# ROCm optimizations for RDNA1/2
export HSA_USE_SVM=0
export HSA_XNACK=0
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
```

## Verification

After installation, verify patches are active:

```bash
# Check ROCm version
/opt/rocm-patched/bin/rocminfo

# Check for patch messages
sudo dmesg | grep -i "patch\|rdna"

# Run test suite
./test_patched_rocm.sh

# Test with PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Build Failures

**Problem**: ROCm source build fails

**Solutions**:
1. Ensure all dependencies installed: `sudo apt-get install build-essential cmake git python3-dev`
2. Check disk space: `df -h` (need ~10GB free)
3. Check ROCm version compatibility: Edit `ROCM_VERSION` in script
4. Review build logs in `~/rocm-source-patches/*/build/`

### Module Build Failures

**Problem**: Kernel module build fails

**Solutions**:
1. Ensure kernel headers match running kernel: `uname -r` vs `/usr/src/linux-headers-*`
2. Install amdgpu-dkms: `sudo apt-get install amdgpu-dkms`
3. Check patch compatibility with your kernel version

### Test Failures

**Problem**: Memory tests fail

**Solutions**:
1. Check kernel logs: `sudo dmesg | tail -50`
2. Verify GPU detection: `/opt/rocm-patched/bin/rocminfo | grep -i gfx`
3. Ensure environment variables set: `echo $ROCM_PATH`
4. Try rebooting (especially after kernel module patch)

### PyTorch Issues

**Problem**: PyTorch doesn't detect GPU

**Solutions**:
1. Reinstall PyTorch with ROCm support: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0`
2. Check ROCM_HOME: `export ROCM_HOME=/opt/rocm-patched`
3. Rebuild PyTorch from source with patched ROCm

## Rollback

### Revert to Original ROCm

```bash
# Switch alternatives
sudo update-alternatives --config rocm
# Select original ROCm installation

# Or set directly
export ROCM_PATH=/opt/rocm-6.2.2
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### Revert Kernel Module

```bash
# Find backup
BACKUP=$(find /lib/modules -name 'amdgpu.ko*.backup')
echo $BACKUP

# Restore (adjust path for your system)
sudo cp $BACKUP /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
sudo depmod -a
sudo reboot
```

## Technical Details

### Patches Applied

**HIP Runtime Patch** (`001-hip-rdna-memory-coherency.patch`):
- Adds `isRDNA1or2()` detection function
- Checks GPU architecture name for gfx101x-103x
- Forces non-coherent memory allocations
- Logs patch application to stderr

**ROCR Runtime Patch** (`002-rocr-rdna-memory-type.patch`):
- Detects RDNA GPUs by GFX version in GPU agent initialization
- Sets `rdna_workaround_active_` flag
- Applies conservative memory settings
- Logs workaround activation

**Kernel Module Patch** (`003-amdgpu-rdna-memory-defaults.patch`):
- Adds `gmc_v10_0_apply_rdna_workarounds()` function
- Detects RDNA by IP version check
- Forces non-coherent aperture base
- Sets VM fragment size to 512KB
- Disables aggressive retry (noretry=0)

### Why Source-Level Patching?

**Advantages over application-level workarounds**:
1. **System-wide**: Fixes all applications automatically
2. **Performance**: No runtime overhead from wrapper libraries
3. **Transparency**: Applications don't need modification
4. **Completeness**: Fixes issues at multiple levels (kernel, runtime, HIP)

**Compared to workarounds**:
- Application patches: Must patch each app individually
- LD_PRELOAD wrappers: Runtime overhead, complex management
- Kernel parameters: Limited effectiveness, boot-only changes

## Support

- **Issues**: See `docs/issues/` for documented ROCm issues
- **Strategy**: See `docs/ROCM_SOURCE_PATCHING_STRATEGY.md`
- **Community**: ROCm GitHub issue #5051 (401+ similar reports)

## Contributing

If you improve these patches or find issues:

1. Test thoroughly with the test suite
2. Document changes clearly
3. Consider submitting to upstream ROCm
4. Share results with the community

## License

These patches are provided as-is for fixing known ROCm issues. Distribute freely, improve openly.
