# RMCP Testing & Validation TODO

**Project**: RMCP (RDNA Memory Coherency Patch) v1.0  
**Date**: November 6, 2024  
**Current Phase**: Testing Phase 1 Complete - Ready for Phase 2

---

## 📊 Progress Overview

```
Phase 1: Testing Framework    ✅ COMPLETE (100%)
Phase 2: Patch Installation    ⏳ READY    (0%)
Phase 3: Validation           ⭕ PENDING  (0%)
Phase 4: Real Projects        ⭕ PENDING  (0%)
Phase 5: Stability            ⭕ PENDING  (0%)

Overall: 20% (1/5 phases)
```

---

## Phase 1: Testing Framework ✅ COMPLETE

### 1.1 Create Test Suites ✅
- [x] Create `test_real_world_workloads.py` (700 lines)
- [x] Create `test_project_integration.sh` (400 lines)
- [x] Create `docs/TESTING.md` documentation
- [x] Create `TESTING_PHASE_COMPLETE.md` summary

### 1.2 Baseline Testing ✅
- [x] Run tests on unpatched ROCm
- [x] Confirm crash on Conv2d operations
- [x] Document error messages
- [x] Validate 100% crash rate

### 1.3 Documentation ✅
- [x] Document expected results
- [x] Create performance metrics tables
- [x] Write troubleshooting guide
- [x] Define 5-phase test plan

---

## Phase 2: Patch Installation ⏳ READY

### 2.1 Prepare Environment
- [ ] **Task**: Check available disk space
  - **Command**: `df -h /opt`
  - **Required**: At least 15GB free
  - **Time**: 1 minute
  
- [ ] **Task**: Install build dependencies
  - **Command**: Check if cmake, gcc, g++ installed
  - **Action**: `sudo apt install cmake build-essential`
  - **Time**: 5 minutes

- [ ] **Task**: Backup current ROCm
  - **Command**: `sudo cp -r /opt/rocm /opt/rocm.backup`
  - **Time**: 5 minutes

### 2.2 ROCm Source Patching (2-3 hours)
- [ ] **Task**: Run patch script
  - **Command**: `sudo ./scripts/patch_rocm_source.sh`
  - **Expected**: Clone repos, apply patches, build ROCm
  - **Time**: 2-3 hours
  - **Monitor**: Check for build errors

- [ ] **Task**: Verify patched installation
  - **Command**: `ls -la /opt/rocm-patched`
  - **Expected**: Directory exists with lib/, include/, bin/
  - **Time**: 1 minute

### 2.3 Kernel Module Patching (10 minutes)
- [ ] **Task**: Run kernel patch script
  - **Command**: `sudo ./scripts/patch_kernel_module.sh`
  - **Expected**: Patch amdgpu module, install
  - **Time**: 10 minutes

- [ ] **Task**: Verify module backup
  - **Command**: `ls -la /lib/modules/$(uname -r)/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.orig`
  - **Expected**: Backup file exists
  - **Time**: 1 minute

### 2.4 System Configuration
- [ ] **Task**: Reboot system
  - **Command**: `sudo reboot`
  - **Required**: For kernel module to load
  - **Time**: 2 minutes

- [ ] **Task**: After reboot, load environment
  - **Command**: `source /etc/profile.d/rocm-patched.sh`
  - **Expected**: ROCM_PATH set to /opt/rocm-patched
  - **Time**: 1 second

- [ ] **Task**: Verify environment variables
  - **Commands**:
    ```bash
    echo $ROCM_PATH        # Should be /opt/rocm-patched
    echo $HSA_USE_SVM      # Should be 0
    echo $HSA_XNACK        # Should be 0
    ```
  - **Time**: 1 minute

---

## Phase 3: Post-Patch Validation ⭕ PENDING

### 3.1 Basic Validation
- [ ] **Task**: Check ROCm info
  - **Command**: `rocminfo | grep "Name:"`
  - **Expected**: Shows AMD Radeon RX 5600 XT
  - **Time**: 1 minute

- [ ] **Task**: Check HIP version
  - **Command**: `hipconfig --version`
  - **Expected**: Shows patched version
  - **Time**: 1 minute

- [ ] **Task**: Verify kernel module loaded
  - **Command**: `lsmod | grep amdgpu`
  - **Expected**: amdgpu module loaded
  - **Time**: 1 minute

### 3.2 Integration Tests
- [ ] **Task**: Run integration test suite
  - **Command**: `./tests/test_project_integration.sh`
  - **Expected**: All 8 tests pass
  - **Time**: 5 minutes

- [ ] **Task**: Check for kernel errors
  - **Command**: `sudo dmesg | tail -50 | grep -i "amdgpu\|memory"`
  - **Expected**: No memory fault errors
  - **Time**: 1 minute

### 3.3 Comprehensive Workload Tests
- [ ] **Task**: Run real-world workload tests
  - **Command**: `python3 tests/test_real_world_workloads.py`
  - **Expected**: 10/10 tests pass (vs 1/10 before)
  - **Critical**: Test #2 (convolutions) must pass
  - **Critical**: Test #10 (EEG reshaping) must pass
  - **Time**: 10 minutes

- [ ] **Task**: Verify no crashes
  - **Expected**: All tests complete without core dumps
  - **Expected**: No HSA_STATUS_ERROR messages
  - **Time**: Included in test run

---

## Phase 4: Real-World Projects ⭕ PENDING

### 4.1 EEG2025 Project Testing
- [ ] **Task**: Navigate to project
  - **Command**: `cd /home/kevin/Projects/eeg2025`
  - **Time**: 1 second

- [ ] **Task**: Check GPU detection
  - **Command**: `python3 -c "import torch; print(torch.cuda.is_available())"`
  - **Expected**: True
  - **Time**: 5 seconds

- [ ] **Task**: Test spatial convolution (critical test)
  - **Create test file**: `test_spatial_conv.py`
  - **Code**:
    ```python
    import torch
    import torch.nn as nn
    
    eeg_input = torch.randn(16, 1, 64, 256).cuda()
    spatial_conv = nn.Conv2d(1, 32, (64, 1)).cuda()
    output = spatial_conv(eeg_input)
    output = output.squeeze(2)  # This crashed before RMCP
    print("✅ Spatial convolution SUCCESS!")
    ```
  - **Expected**: No crash, prints success message
  - **Before RMCP**: 100% crash
  - **Time**: 2 minutes

- [ ] **Task**: Run short training test (5 epochs)
  - **Command**: Run EEGNeX training with small batch
  - **Expected**: Training completes without crash
  - **Expected**: GPU utilization >90%
  - **Time**: 10-15 minutes

- [ ] **Task**: Monitor GPU during training
  - **Command**: `watch -n 1 rocm-smi`
  - **Expected**: GPU memory allocated, compute active
  - **Time**: During training

### 4.2 Thermal Object Detection Project Testing
- [ ] **Task**: Navigate to project
  - **Command**: `cd /home/kevin/Projects/robust-thermal-image-object-detection`
  - **Time**: 1 second

- [ ] **Task**: Test YOLO backbone (critical test)
  - **Create test file**: `test_yolo_backbone.py`
  - **Code**:
    ```python
    import torch
    import torch.nn as nn
    
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
    ).cuda()
    
    x = torch.randn(2, 3, 416, 416).cuda()
    output = backbone(x)  # This crashed before RMCP
    torch.cuda.synchronize()
    print("✅ YOLO backbone SUCCESS!")
    ```
  - **Expected**: No crash, prints success message
  - **Before RMCP**: 100% crash on first forward pass
  - **Time**: 2 minutes

- [ ] **Task**: Run short training test (10 batches)
  - **Command**: Run YOLO training with small dataset
  - **Expected**: Training completes without crash
  - **Expected**: No "Page not present" errors
  - **Time**: 10-15 minutes

- [ ] **Task**: Check kernel logs after training
  - **Command**: `sudo dmesg | tail -100 | grep -i "memory fault"`
  - **Expected**: No memory fault messages
  - **Time**: 1 minute

### 4.3 Performance Validation
- [ ] **Task**: Compare training speeds
  - **Measure**: EEG2025 epoch time (GPU vs historical CPU)
  - **Expected**: 10-20x faster with GPU
  - **Time**: During training

- [ ] **Task**: Monitor memory usage
  - **Command**: `rocm-smi --showmeminfo`
  - **Expected**: Memory allocated and released properly
  - **Time**: 2 minutes

---

## Phase 5: Long-Term Stability ⭕ PENDING

### 5.1 Extended Training Test
- [ ] **Task**: Run 24-hour training session
  - **Project**: Choose EEG2025 or thermal
  - **Expected**: No crashes, no degradation
  - **Time**: 24+ hours

- [ ] **Task**: Monitor system during long run
  - **Check every 4 hours**:
    - GPU temperature
    - Memory usage
    - Kernel logs for errors
  - **Time**: Periodic checks

### 5.2 Stress Testing
- [ ] **Task**: Run memory stress test
  - **Command**: From `test_real_world_workloads.py`
  - **Test**: Allocate 50% GPU memory repeatedly
  - **Expected**: No leaks, no crashes
  - **Time**: 30 minutes

- [ ] **Task**: Run mixed workload test
  - **Action**: Train multiple models simultaneously
  - **Expected**: All complete successfully
  - **Time**: 1 hour

### 5.3 Final Validation
- [ ] **Task**: Compare results before/after RMCP
  - **Document**:
    - Crash rate: 100% → 0%
    - Training speed: 1x (CPU) → 10-20x (GPU)
    - Success rate: 0% → 99%
  - **Time**: 30 minutes to compile results

- [ ] **Task**: Update TESTING.md with results
  - **Add**: Actual test results
  - **Add**: Performance measurements
  - **Add**: Any issues encountered
  - **Time**: 30 minutes

---

## 🚀 Quick Commands Reference

### Start Phase 2 (Patching)
```bash
cd /home/kevin/Projects/rocm-patch
sudo ./scripts/patch_rocm_source.sh      # 2-3 hours
sudo ./scripts/patch_kernel_module.sh    # 10 minutes
sudo reboot                               # Required
```

### After Reboot (Phase 3)
```bash
source /etc/profile.d/rocm-patched.sh
cd /home/kevin/Projects/rocm-patch
./tests/test_project_integration.sh
python3 tests/test_real_world_workloads.py
```

### Test Real Projects (Phase 4)
```bash
# EEG2025
cd /home/kevin/Projects/eeg2025
# Run training

# Thermal
cd /home/kevin/Projects/robust-thermal-image-object-detection
# Run YOLO training
```

---

## 📈 Success Metrics

### Before RMCP (Confirmed)
- ✅ Basic tensor ops: PASS
- ❌ Convolutions: CRASH (aperture violation)
- ❌ Training: IMPOSSIBLE
- ❌ GPU utilization: 0%

### After RMCP (Expected)
- ✅ Basic tensor ops: PASS
- ✅ Convolutions: PASS
- ✅ Training: ENABLED
- ✅ GPU utilization: 90%+
- ✅ Speed: 10-20x faster
- ✅ Crash rate: 0%

---

**Next Action**: Start Phase 2 by running `sudo ./scripts/patch_rocm_source.sh`

**Estimated Total Time**: 4-6 hours (including 24-hour long test)
