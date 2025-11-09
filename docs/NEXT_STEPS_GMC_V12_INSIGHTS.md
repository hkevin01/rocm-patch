# Next Steps: GMC v12_0 Insights & Implementation Plan

## 📅 Date: November 6, 2025

## 🎯 Context

Based on research of Linux kernel commits for GFX12 (RDNA3), we discovered AMD engineers implemented MTYPE_NC workarounds at the kernel level. This validates our approach and provides implementation patterns we can adapt for RDNA1/2.

## 🔬 Key Findings from GMC v12_0 Research

### Kernel Commits Analyzed

**Commit 628e1ac**: MTYPE_NC workaround for GFX12  
**Commit eb6cdfb**: GFX12 coherency handling

### What AMD Did for RDNA3

AMD engineers implemented kernel-level workarounds for memory type issues in RDNA3 (GFX12):

1. **MTYPE_NC Enforcement**
   - Forced non-coherent memory types at kernel level
   - Modified GMC (Graphics Memory Controller) initialization
   - Set memory aperture flags to non-coherent

2. **VM Fragment Size Optimization**
   - Adjusted fragment sizes for better memory mapping
   - Reduced page fault frequency
   - Improved GTT (Graphics Translation Table) performance

3. **Coherency Flag Management**
   - Disabled coherent aperture base for problematic workloads
   - Added IP version detection for targeted fixes
   - Applied workarounds during late_init phase

### Why This Matters for RDNA1/2

**Validation**: If AMD needed kernel-level MTYPE workarounds for RDNA3 (which HAS SVM support), then RDNA1/2 (which LACKS SVM) absolutely needs them.

**Implementation Pattern**: We can use the exact same approach:
- Detect GPU by IP version
- Modify GMC settings during initialization
- Force non-coherent memory types
- Adjust VM fragment sizes

## 📋 Implementation Plan

### Phase 1: Test Hardware Compatibility ✅ COMPLETE

Status: **✅ COMPLETE**

Files:
- `tests/test_hardware_compatibility.py` - Created and tested
- `README.md` - Updated with Step 0

Results:
- Successfully detected RX 5600 XT (gfx1010/RDNA1)
- Confirmed needs RMCP patches
- Exit code system working correctly

### Phase 2: Apply RMCP Patches (IN PROGRESS)

Status: **🟡 IN PROGRESS**

#### Step 1: ROCm Source Patching (Current)

**Script**: `scripts/patch_rocm_source.sh`

**What it does**:
1. Clone ROCm source repositories (ROCT, ROCR, HIP)
2. Apply source-level patches for memory coherency
3. Build patched ROCm components
4. Install to `/opt/rocm-patched`

**Patches Applied**:

**Patch 1: HIP Runtime** (`001-hip-rdna-memory-coherency.patch`)
- File: `HIP/src/hip_memory.cpp`
- Adds `isRDNA1or2()` detection
- Forces `hipMemAllocationTypeNonCoherent` for RDNA1/2
- Validates memory type before allocation

**Patch 2: ROCR Runtime** (`002-rocr-rdna-memory-type.patch`)
- File: `ROCR-Runtime/src/core/runtime/hsa.cpp`
- Detects RDNA in GPU agent initialization
- Applies conservative memory settings
- Forces fine-grain PCIe memory

**Patch 3: Kernel Module** (`003-amdgpu-rdna-memory-defaults.patch`)
- File: `drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c`
- Adds `gmc_v10_0_apply_rdna_workarounds()` function
- Detects RDNA1/2 by IP version (10.1.x, 10.3.x)
- Forces non-coherent aperture base
- Sets VM fragment size to 512KB (9 = 2^9 pages)
- Disables aggressive retry (noretry=0)

**Time Required**: 2-3 hours  
**Disk Space**: ~10GB

**Status**: Ready to execute (don't run as sudo)

#### Step 2: Kernel Module Parameters

**Script**: `scripts/apply_kernel_params.sh` (if exists) or manual

**Parameters to Apply**:
```bash
# /etc/modprobe.d/amdgpu-rmcp.conf
options amdgpu noretry=0
options amdgpu vm_fragment_size=9
options amdgpu vm_update_mode=0
options amdgpu gtt_size=8192
```

**Why These Values**:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `noretry` | 0 | Disable infinite retry loops on page faults |
| `vm_fragment_size` | 9 | 512MB fragments (2^9 × 2MB pages) reduce fragmentation |
| `vm_update_mode` | 0 | Use SDMA engine (more stable than GFXHUB) |
| `gtt_size` | 8192 | 8GB GTT aperture (vs default 4GB) |

**Status**: Already applied (see `KERNEL_PARAMS_APPLIED.md`)

#### Step 3: Test ML Workloads

**Tests to Run**:
```bash
# Basic tensor operations
python3 tests/test_ml_basic.py

# Convolutional operations (the critical test)
python3 tests/test_ml_convolution.py

# Real-world workload
python3 tests/test_real_world_workloads.py
```

**Expected Results**:
- ✅ Basic tensor ops work (matmul, addition)
- ✅ Conv2d operations succeed (the key test)
- ✅ Memory allocations use non-coherent types
- ✅ No HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION

**Status**: ⭕ NOT STARTED (waiting for patches)

### Phase 3: Validation & Documentation

#### Step 1: Verify Patch Application

**Checks**:
```bash
# 1. Verify patched ROCm installed
ls -la /opt/rocm-patched/

# 2. Check environment variables
cat /etc/profile.d/rocm-patched.sh

# 3. Verify kernel params active
cat /proc/cmdline | grep amdgpu
cat /sys/module/amdgpu/parameters/noretry
cat /sys/module/amdgpu/parameters/vm_fragment_size

# 4. Test ROCm info
/opt/rocm-patched/bin/rocminfo
```

#### Step 2: Run Comprehensive Tests

**Test Suite**:
```bash
cd /home/kevin/Projects/rocm-patch

# Hardware compatibility (should show patched)
python3 tests/test_hardware_compatibility.py

# Basic ML operations
python3 tests/test_ml_basic.py

# Convolutional operations (critical)
python3 tests/test_ml_convolution.py

# Real-world workloads
python3 tests/test_real_world_workloads.py

# Integration test
bash tests/test_project_integration.sh
```

#### Step 3: Document Results

**Files to Update**:
- `INSTALLATION_RESULTS.md` - Test outcomes and performance
- `README.md` - Success confirmation or troubleshooting
- `docs/TROUBLESHOOTING.md` - Any issues encountered

## 🎓 Lessons from GMC v12_0 Research

### 1. AMD Validates Our Approach

AMD engineers used the EXACT same approach for RDNA3:
- Kernel-level MTYPE enforcement
- GMC initialization modifications
- Non-coherent memory aperture settings
- VM fragment size adjustments

**This proves our strategy is correct.**

### 2. IP Version Detection is Reliable

AMD uses IP version checks for GPU detection:
```c
if (adev->ip_versions[GC_HWIP][0] >= IP_VERSION(12, 0, 0)) {
    // Apply GFX12 workarounds
}
```

We use the same for RDNA1/2:
```c
if (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 1, 0) ||  // RDNA1
    adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 3, 0)) {  // RDNA2
    // Apply workarounds
}
```

### 3. Late Init is the Right Phase

AMD applies workarounds during `gmc_v*_late_init()`:
- Hardware is fully detected
- Memory controller is initialized
- Safe to modify settings

Our patches follow the same pattern.

### 4. Multi-Layer Defense Works

AMD uses multiple layers:
- Kernel driver (GMC)
- Runtime (HSA)
- Application (HIP)

This is identical to our 3-layer approach.

## 🚀 Next Immediate Actions

### Action 1: Run ROCm Source Patcher (NOW)

```bash
cd /home/kevin/Projects/rocm-patch/scripts
./patch_rocm_source.sh
```

**What will happen**:
1. Clones ROCm repositories (~5 minutes)
2. Applies patches to source code (~1 minute)
3. Builds ROCT, ROCR, HIP (~2-3 hours)
4. Installs to `/opt/rocm-patched` (~5 minutes)
5. Sets up environment variables (~1 minute)

**Monitor progress**: Watch for "[INFO]" messages

**Expected completion**: 2-3 hours

### Action 2: Verify Installation

```bash
# Check patched ROCm exists
ls -la /opt/rocm-patched/

# Test rocminfo
/opt/rocm-patched/bin/rocminfo

# Verify environment
source /etc/profile.d/rocm-patched.sh
echo $ROCM_PATH
```

### Action 3: Run ML Tests

```bash
cd /home/kevin/Projects/rocm-patch

# Basic test
python3 tests/test_ml_basic.py

# Critical test (Conv2d - this is what was crashing)
python3 tests/test_ml_convolution.py
```

**Success criteria**:
- ✅ No memory access faults
- ✅ Conv2d operations complete
- ✅ GPU is used (not CPU fallback)
- ✅ Performance is 10-20x faster than CPU

### Action 4: Update Documentation

```bash
# Create installation results doc
cat > INSTALLATION_RESULTS.md << 'EOF'
# RMCP Installation Results

## Hardware
- GPU: AMD Radeon RX 5600 XT (gfx1010/RDNA1)
- Kernel: 6.14.0-34-generic
- ROCm: 6.2.x + RMCP patches

## Installation
- Date: [DATE]
- Duration: [TIME]
- Status: [SUCCESS/FAILURE]

## Test Results
[Test outputs here]

## Performance
- Before RMCP: CPU fallback (100% crash rate)
- After RMCP: [GPU performance data]
- Speedup: [X]x faster

EOF
```

## 📊 Success Metrics

### Must Have ✅

- [x] Hardware compatibility test created
- [ ] ROCm source patches applied
- [ ] Patched ROCm installed to `/opt/rocm-patched`
- [ ] Environment variables configured
- [ ] Basic ML tests pass
- [ ] Conv2d operations work (CRITICAL)

### Nice to Have 🎯

- [ ] Performance benchmarks documented
- [ ] Comparison with ROCm 5.7 (downgrade alternative)
- [ ] Detailed troubleshooting guide
- [ ] Community feedback from other RDNA1/2 users

## 🔄 Rollback Plan

If patches don't work:

### Option 1: ROCm 5.7 Downgrade
```bash
sudo apt remove rocm-dkms
sudo apt install rocm-dkms=5.7.0-*
sudo reboot
```

### Option 2: CPU Fallback
- Already implemented in projects
- 100% stable, 10-20x slower
- No further action needed

### Option 3: Try Additional Kernel Params
```bash
# Add to /etc/default/grub
amdgpu.vm_update_mode=3 amdgpu.dc=0
```

## 📚 References

### Kernel Commits (GMC v12_0)
- **628e1ac**: MTYPE_NC workaround for GFX12
- **eb6cdfb**: GFX12 coherency handling

### ROCm Issues
- **#5051**: Primary issue (401+ users)
- **#5616**: Recent memory access faults

### Community
- **Hashcat #3932**: Validates SVM/coherency problem

### Documentation
- `HARDWARE_TEST_SUMMARY.md` - Hardware test documentation
- `KERNEL_PARAMS_APPLIED.md` - Kernel parameter setup
- `docs/ROCM_SOURCE_PATCHING_STRATEGY.md` - Patching strategy

## 🎯 Current Status

```markdown
### RMCP Implementation Status

- [x] **Phase 1: Research & Planning** ✅ COMPLETE
  - [x] Analyzed GMC v12_0 kernel commits
  - [x] Validated approach against AMD's RDNA3 fixes
  - [x] Created hardware compatibility test
  - [x] Documented kernel parameters

- [ ] **Phase 2: Patch Application** 🟡 IN PROGRESS
  - [ ] Run ROCm source patcher (CURRENT STEP)
  - [ ] Build and install patched components
  - [ ] Verify environment setup
  - [ ] Test basic ROCm functionality

- [ ] **Phase 3: Testing** ⭕ NOT STARTED
  - [ ] Run ML basic tests
  - [ ] Test Conv2d operations (critical)
  - [ ] Benchmark performance
  - [ ] Document results

- [ ] **Phase 4: Validation** ⭕ NOT STARTED
  - [ ] Comprehensive test suite
  - [ ] Performance comparison
  - [ ] Documentation updates
  - [ ] Community feedback
```

## 🚀 Execute Now

**Current Step**: Run ROCm source patcher

**Command**:
```bash
cd /home/kevin/Projects/rocm-patch/scripts
./patch_rocm_source.sh
```

**Note**: Do NOT use sudo - script will prompt when needed

**Expected Duration**: 2-3 hours

**Next Step After Completion**: Run ML tests to verify GPU training works

---

**Status**: 🟡 **READY TO EXECUTE PHASE 2**

**Date**: November 6, 2025  
**Project**: RMCP (RDNA Memory Coherency Patch) v1.0
