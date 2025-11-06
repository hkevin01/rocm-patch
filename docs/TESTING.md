# RMCP Testing Documentation

## Current Status

**Date**: November 6, 2024  
**Status**: Pre-Patch Testing Completed - Crash Confirmed  
**GPU**: AMD Radeon RX 5600 XT (RDNA1 gfx1010)  
**ROCm Version**: 6.2 (unpatched)

---

## Test Results Summary

### Phase 1: Pre-Patch Testing ✅ COMPLETE

#### Test 1: Real-World Workload Test
**File**: `tests/test_real_world_workloads.py`  
**Status**: ❌ **CRASHED** (Expected - validates problem exists)

```
[TEST 1] PyTorch Basic Tensor Operations
✅ PASSED (0.70s) - Basic operations work

[TEST 2] PyTorch Convolutional Operations
❌ CRASHED - Memory aperture violation
Error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
Code: 0x29
Message: "The agent attempted to access memory beyond the largest legal address"
Exit Code: 134 (SIGABRT - Core Dumped)
```

**Analysis**:
- Basic tensor operations (matmul, chain ops) work fine
- **Convolutional operations crash immediately** (as documented)
- This confirms the exact issue we're patching: RDNA1/2 GPUs cannot handle cache-coherent memory with convolutions
- The crash happens at the ROCm runtime level before kernel execution

---

## Testing Plan

### Phase 1: Pre-Patch Baseline ✅
- [x] Verify crash occurs on unpatched ROCm
- [x] Document exact error message
- [x] Confirm GPU is RDNA1/2 (gfx1010)
- [x] Verify PyTorch with ROCm is installed
- [x] Baseline test shows 100% crash on convolutions

### Phase 2: Patch Installation ⭕
- [ ] Run `scripts/patch_rocm_source.sh`
  - Clone ROCm sources (HIP, ROCR, ROCT, CLR)
  - Apply memory coherency patches
  - Build patched ROCm (~2-3 hours)
  - Install to `/opt/rocm-patched`
- [ ] Run `scripts/patch_kernel_module.sh`
  - Patch amdgpu kernel module
  - Backup original module
  - Install patched module
  - **Reboot required**
- [ ] Configure environment
  - Set `ROCM_PATH=/opt/rocm-patched`
  - Set `HSA_USE_SVM=0`
  - Set `HSA_XNACK=0`
  - Update `LD_LIBRARY_PATH`

### Phase 3: Post-Patch Validation ⭕
- [ ] **Test 1**: Run `test_real_world_workloads.py` again
  - Expected: All 10 tests pass
  - Critical: Test #2 (convolutions) must pass
  - Critical: Test #10 (EEG tensor reshaping) must pass
- [ ] **Test 2**: Run `test_project_integration.sh`
  - System-wide RMCP verification
  - PyTorch GPU test
  - GPU memory allocation test
  - Kernel log analysis
- [ ] **Test 3**: Check for no regressions
  - Basic operations still work
  - No new crashes introduced
  - Performance maintained

### Phase 4: Real-World Project Testing ⭕
- [ ] **EEG2025 Project** (`/home/kevin/Projects/eeg2025`)
  - Challenge 1: EEGNeX spatial convolution training
  - Challenge 2: SAM+TCN training
  - Expected: No crashes, GPU utilization >90%
  - Before RMCP: 100% crash on spatial conv
  - After RMCP: Should train successfully
- [ ] **Thermal Project** (`/home/kevin/Projects/robust-thermal-image-object-detection`)
  - YOLO training on thermal images
  - Expected: 99% stability, 8-10x speedup vs CPU
  - Before RMCP: "Page not present" on every batch
  - After RMCP: Full training without crashes

### Phase 5: Long-Term Stability ⭕
- [ ] 24+ hour training session
- [ ] Monitor for memory leaks
- [ ] Check kernel logs periodically
- [ ] Verify no degradation over time

---

## Test Files

### 1. `tests/test_real_world_workloads.py`
**Purpose**: Comprehensive ML/DL workload testing  
**Tests**: 10 categories covering PyTorch, YOLO, transformers, mixed precision  
**Critical Tests**:
- Test #2: Convolutions (crashes 100% before RMCP)
- Test #10: EEG tensor reshaping (exact pattern from eeg2025)

**Usage**:
```bash
cd /home/kevin/Projects/rocm-patch
python3 tests/test_real_world_workloads.py
```

### 2. `tests/test_project_integration.sh`
**Purpose**: System-wide RMCP validation and project integration  
**Tests**: 8 categories including environment, GPU, kernel logs, projects  

**Usage**:
```bash
cd /home/kevin/Projects/rocm-patch
./tests/test_project_integration.sh
```

### 3. `scripts/test_patched_rocm.sh`
**Purpose**: Basic ROCm installation validation  
**Tests**: 7 categories (environment, rocminfo, HIP, memory, PyTorch)  

**Usage**:
```bash
cd /home/kevin/Projects/rocm-patch
./scripts/test_patched_rocm.sh
```

---

## Expected Results

### Before RMCP (Current State)
```
PyTorch Basic Operations:        ✅ PASS
PyTorch Convolutions:             ❌ CRASH (aperture violation)
PyTorch Training:                 ❌ CRASH (memory fault)
YOLO Operations:                  ❌ CRASH (page not present)
Transformer Operations:           ❌ CRASH (memory fault)
Mixed Precision:                  ❌ CRASH (memory fault)
Memory Stress:                    ❌ CRASH (memory fault)
Kernel Faults:                    ❌ DETECTED
Patch Verification:               ❌ NOT INSTALLED
EEG Tensor Reshaping:             ❌ CRASH (spatial conv)

Success Rate: 10% (1/10 tests)
GPU Training: IMPOSSIBLE - 100% crash rate
```

### After RMCP (Expected)
```
PyTorch Basic Operations:        ✅ PASS
PyTorch Convolutions:             ✅ PASS (non-coherent memory)
PyTorch Training:                 ✅ PASS (full backprop)
YOLO Operations:                  ✅ PASS (feature extraction)
Transformer Operations:           ✅ PASS (attention)
Mixed Precision:                  ✅ PASS (AMP)
Memory Stress:                    ✅ PASS (50% allocation)
Kernel Faults:                    ✅ NONE DETECTED
Patch Verification:               ✅ RMCP ACTIVE
EEG Tensor Reshaping:             ✅ PASS (spatial conv)

Success Rate: 100% (10/10 tests)
GPU Training: ENABLED - 0% crash rate
Performance: 10-20x speedup vs CPU fallback
```

---

## Performance Metrics

### Expected Improvements

| Metric | Before RMCP | After RMCP | Improvement |
|--------|------------|------------|-------------|
| Conv2D Operations | 0% success | 100% success | ∞ (was impossible) |
| Training Stability | 0% | 99% | +99pp |
| GPU Utilization | 0% (forced CPU) | 90%+ | Enabled |
| Training Speed | 1x (CPU only) | 10-20x (GPU) | 10-20x faster |
| Crash Rate | 100% | 0% | -100pp |
| Affected Users | 401+ | 0 | Issue resolved |

---

## Troubleshooting

### If Tests Still Crash After Patching

1. **Verify RMCP is active**:
   ```bash
   echo $ROCM_PATH  # Should be /opt/rocm-patched
   echo $HSA_USE_SVM  # Should be 0
   rocminfo | grep "Name:"  # Should show GPU
   ```

2. **Check patch application**:
   ```bash
   cd /opt/rocm-patched
   strings lib/libamdhip64.so | grep "RDNA"  # Should show detection code
   ```

3. **Check kernel module**:
   ```bash
   lsmod | grep amdgpu  # Should be loaded
   modinfo amdgpu | grep "version"  # Check version
   ```

4. **Check kernel logs**:
   ```bash
   sudo dmesg | tail -50 | grep -i "amdgpu\|memory"
   ```

### If Basic Ops Work But Conv Crashes

This indicates patches were applied but may not be fully active:
- Verify environment variables are set in current shell
- Try rebooting to ensure kernel module loads
- Check `/etc/profile.d/rocm-patched.sh` exists and is sourced

### If Everything Crashes

This indicates PyTorch may be using wrong ROCm:
- Reinstall PyTorch to link against patched ROCm
- Or set `LD_PRELOAD` to force patched libraries

---

## Next Steps

1. **Install RMCP patches** (2-4 hours total):
   ```bash
   cd /home/kevin/Projects/rocm-patch
   sudo ./scripts/patch_rocm_source.sh      # 2-3 hours
   sudo ./scripts/patch_kernel_module.sh    # 5-10 minutes
   sudo reboot                               # Required for kernel
   ```

2. **Verify installation**:
   ```bash
   source /etc/profile.d/rocm-patched.sh
   ./tests/test_project_integration.sh
   ```

3. **Run full test suite**:
   ```bash
   python3 tests/test_real_world_workloads.py
   ```

4. **Test real projects**:
   ```bash
   cd /home/kevin/Projects/eeg2025
   # Run training - should work now
   
   cd /home/kevin/Projects/robust-thermal-image-object-detection
   # Run YOLO training - should work now
   ```

5. **Document results** in `TESTING.md`

---

## Contributing Test Results

If you test RMCP, please share your results:
1. GPU model and architecture (RDNA1/2)
2. ROCm version
3. Test results (pass/fail/crash)
4. Performance improvements observed
5. Any issues encountered

Create an issue or PR on the RMCP GitHub repository with your findings.

---

## References

- [Issue #1: EEG2025 Tensor Operations](../docs/issues/eeg2025-tensor-operations.md)
- [Issue #2: Thermal Memory Faults](../docs/issues/thermal-object-detection-memory-faults.md)
- [ROCm GitHub Issue #5051](https://github.com/ROCm/ROCm/issues/5051) - 401+ affected users
- [Source Patching Strategy](../docs/ROCM_SOURCE_PATCHING_STRATEGY.md)
