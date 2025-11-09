# RMCP Testing & Validation Checklist

**Project**: RMCP v1.0  
**Purpose**: Track testing progress across 5 phases  
**Status**: Phase 1 Complete ✅

---

## Phase 1: Pre-Patch Baseline Testing ✅

**Status**: COMPLETE (100%)  
**Date Completed**: November 6, 2024

- [x] Verify crash occurs on unpatched ROCm
- [x] Document exact error messages
- [x] Confirm GPU is RDNA1/2 (gfx1010)
- [x] Verify PyTorch with ROCm installed
- [x] Run test_real_world_workloads.py
- [x] Document baseline: 100% crash on convolutions
- [x] Create TESTING.md documentation

**Results**:
```
✅ Basic operations: PASS
❌ Convolutions: CRASH (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)
Crash Rate: 100% on Conv2d operations
Exit Code: 134 (SIGABRT)
```

---

## Phase 2: Patch Installation ⭕

**Status**: READY TO START  
**Estimated Time**: 2-4 hours total

### Part A: ROCm Source Patches (2-3 hours)
- [ ] Install build dependencies
- [ ] Run `sudo ./scripts/patch_rocm_source.sh`
- [ ] Monitor clone progress (HIP, ROCR, ROCT, CLR)
- [ ] Monitor patch application
- [ ] Monitor CMake build (may take 2+ hours)
- [ ] Verify installation to `/opt/rocm-patched`
- [ ] Check for build errors

### Part B: Kernel Module Patch (10 minutes)
- [ ] Run `sudo ./scripts/patch_kernel_module.sh`
- [ ] Verify kernel headers available
- [ ] Monitor module compilation
- [ ] Backup original module
- [ ] Install patched module
- [ ] Verify module installed

### Part C: System Configuration
- [ ] Reboot system (required for kernel module)
- [ ] Source environment: `source /etc/profile.d/rocm-patched.sh`
- [ ] Verify `ROCM_PATH=/opt/rocm-patched`
- [ ] Verify `HSA_USE_SVM=0`
- [ ] Verify `HSA_XNACK=0`
- [ ] Run `rocminfo` to check GPU detection

---

## Phase 3: Post-Patch Validation ⭕

**Status**: PENDING  
**Estimated Time**: 30-60 minutes

### Test Suite Execution
- [ ] Run `test_real_world_workloads.py`
  - [ ] Test 1: PyTorch basic ops (should still pass)
  - [ ] Test 2: Convolutions (CRITICAL - must pass now)
  - [ ] Test 3: Training loop (should pass)
  - [ ] Test 4: YOLO operations (should pass)
  - [ ] Test 5: Transformers (should pass)
  - [ ] Test 6: Mixed precision (should pass)
  - [ ] Test 7: Memory stress (should pass)
  - [ ] Test 8: Kernel faults (should be none)
  - [ ] Test 9: Patch verification (RMCP active)
  - [ ] Test 10: EEG reshaping (CRITICAL - must pass)

**Expected**: 10/10 tests pass (vs 1/10 before)

### Integration Testing
- [ ] Run `test_project_integration.sh`
- [ ] Check RMCP environment variables
- [ ] Test PyTorch GPU availability
- [ ] Test GPU memory allocation (30% of VRAM)
- [ ] Check kernel logs (no new errors)
- [ ] Verify no regressions

---

## Phase 4: Real-World Project Testing ⭕

**Status**: PENDING  
**Estimated Time**: 2-4 hours

### EEG2025 Project Testing
- [ ] Navigate to `/home/kevin/Projects/eeg2025`
- [ ] Verify project structure
- [ ] Test Challenge 1: EEGNeX spatial conv
  - [ ] Load model
  - [ ] Run training epoch
  - [ ] Verify no crashes on spatial conv
  - [ ] Check GPU utilization >90%
  - [ ] Monitor for memory faults
- [ ] Test Challenge 2: SAM+TCN
  - [ ] Load model
  - [ ] Run training epoch
  - [ ] Verify stability
- [ ] Document training speed improvement
- [ ] Document crash rate (should be 0%)

### Thermal Project Testing
- [ ] Navigate to `/home/kevin/Projects/robust-thermal-image-object-detection`
- [ ] Verify project structure
- [ ] Test YOLO training
  - [ ] Load YOLO model
  - [ ] Run training batch
  - [ ] Verify no "page not present" errors
  - [ ] Check GPU utilization
  - [ ] Monitor kernel logs
- [ ] Run multiple epochs (stability test)
- [ ] Document speedup vs CPU (expect 8-10x)
- [ ] Document crash rate (should be 0%)

---

## Phase 5: Long-Term Stability ⭕

**Status**: PENDING  
**Estimated Time**: 24+ hours

### Extended Testing
- [ ] Start 24-hour training session
- [ ] Monitor GPU memory usage hourly
- [ ] Check for memory leaks
- [ ] Check kernel logs every 4 hours
- [ ] Verify no performance degradation
- [ ] Document any issues

### Performance Benchmarking
- [ ] Baseline CPU training speed
- [ ] Measure GPU training speed
- [ ] Calculate speedup ratio
- [ ] Measure GPU utilization
- [ ] Measure memory efficiency
- [ ] Document results

---

## Success Criteria

### Must Pass
- ✅ Phase 1: Baseline crash confirmed
- ⭕ Phase 2: Patches installed successfully
- ⭕ Phase 3: All 10 tests pass (especially Test 2 & 10)
- ⭕ Phase 4: Both projects train without crashes
- ⭕ Phase 5: 24+ hours stable operation

### Performance Targets
- ⭕ Conv2D success: 0% → 100%
- ⭕ Training stability: 0% → 99%+
- ⭕ GPU acceleration: 10-20x speedup
- ⭕ Crash rate: 100% → 0%

---

## Notes

**Current Status**: Phase 1 complete, ready for Phase 2  
**GPU**: AMD Radeon RX 5600 XT (RDNA1 gfx1010)  
**ROCm**: 6.2 (unpatched)  
**PyTorch**: 2.5.1+rocm6.2

**Next Action**: Run `sudo ./scripts/patch_rocm_source.sh`

---

## Progress Tracking

```
Phase 1: ████████████████████ 100% COMPLETE
Phase 2: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0%   READY
Phase 3: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0%   PENDING
Phase 4: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0%   PENDING
Phase 5: ⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0%   PENDING

Overall: ████⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 20% (1/5 phases)
```
