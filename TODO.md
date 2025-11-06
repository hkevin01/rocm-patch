# RMCP Testing & Validation TODO

**Project**: RMCP (RDNA Memory Coherency Patch) v1.0  
**Last Updated**: November 6, 2024  
**Current Phase**: Phase 2 - Patch Installation

---

## 📊 Overall Progress

```
Phase 1: Pre-Patch Baseline        ✅ COMPLETE (100%)
Phase 2: Patch Installation         ⏳ IN PROGRESS (0%)
Phase 3: Post-Patch Validation      ⭕ PENDING (0%)
Phase 4: Real-World Projects        ⭕ PENDING (0%)
Phase 5: Long-Term Stability        ⭕ PENDING (0%)

Overall: 20% Complete (1/5 phases)
```

---

## Phase 1: Pre-Patch Baseline ✅ COMPLETE

**Goal**: Validate that the crash exists on unpatched ROCm  
**Status**: ✅ Complete  
**Date**: November 6, 2024

### Completed Tasks

- [x] Verify GPU is RDNA1/2 (confirmed: RX 5600 XT gfx1010)
- [x] Confirm PyTorch with ROCm installed (2.5.1+rocm6.2)
- [x] Run baseline test suite
- [x] Document crash on convolutions
- [x] Capture error message (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)
- [x] Confirm 100% crash rate on Conv2d operations
- [x] Create testing framework (3 test suites, 1,500+ lines)
- [x] Document expected results in TESTING.md

### Test Results Summary

```
Test 1: Basic tensor operations       ✅ PASS (0.70s)
Test 2: Convolutional operations       ❌ CRASH (exit 134)
Error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (0x29)

Baseline validates: RMCP is needed to fix this crash
```

---

## Phase 2: Patch Installation ⏳ IN PROGRESS

**Goal**: Install RMCP patches to fix ROCm  
**Status**: ⏳ Ready to Start  
**Estimated Time**: 2-4 hours

### Step 1: Prepare Environment

- [ ] Ensure 10GB+ free disk space
  ```bash
  df -h /opt
  ```
- [ ] Verify ROCm 6.2 or 7.0 installed
  ```bash
  ls /opt/rocm*
  ```
- [ ] Check kernel headers installed
  ```bash
  ls /usr/src/linux-headers-$(uname -r)
  ```
- [ ] Backup current ROCm (optional but recommended)
  ```bash
  sudo cp -r /opt/rocm /opt/rocm-backup-$(date +%Y%m%d)
  ```

### Step 2: Run ROCm Source Patcher

**Script**: `scripts/patch_rocm_source.sh`  
**Time**: 2-3 hours  
**Disk**: ~10GB

- [ ] Navigate to project directory
  ```bash
  cd /home/kevin/Projects/rocm-patch
  ```
- [ ] Review patcher script
  ```bash
  cat scripts/patch_rocm_source.sh | head -50
  ```
- [ ] Run patcher with sudo
  ```bash
  sudo ./scripts/patch_rocm_source.sh
  ```
- [ ] Monitor progress (will show):
  - [ ] Installing dependencies (cmake, git, build-essential)
  - [ ] Creating workspace at /tmp/rocm-build
  - [ ] Cloning ROCm sources (HIP, ROCR, ROCT, CLR)
  - [ ] Creating patch files inline
  - [ ] Applying patches to source code
  - [ ] Building ROCT-Thunk-Interface (~10 min)
  - [ ] Building ROCR-Runtime (~15 min)
  - [ ] Building HIP/CLR (~90 min)
  - [ ] Installing to /opt/rocm-patched
  - [ ] Creating environment script
  - [ ] Running basic tests

- [ ] Verify installation completed successfully
  ```bash
  ls -la /opt/rocm-patched/lib/libamdhip64.so
  ```

### Step 3: Run Kernel Module Patcher

**Script**: `scripts/patch_kernel_module.sh`  
**Time**: 5-10 minutes  
**Reboot**: Required after this step

- [ ] Navigate to project directory
  ```bash
  cd /home/kevin/Projects/rocm-patch
  ```
- [ ] Run kernel patcher with sudo
  ```bash
  sudo ./scripts/patch_kernel_module.sh
  ```
- [ ] Monitor progress (will show):
  - [ ] Checking kernel headers
  - [ ] Creating kernel patch file
  - [ ] Downloading amdgpu driver source
  - [ ] Applying memory coherency patch
  - [ ] Building amdgpu.ko module
  - [ ] Backing up original module
  - [ ] Installing patched module
  - [ ] Creating modprobe configuration

- [ ] Verify module installation
  ```bash
  modinfo amdgpu | grep -i filename
  ```

### Step 4: Reboot System

**Required**: Kernel module changes require reboot

- [ ] Save all work in other projects
- [ ] Commit any uncommitted changes
- [ ] Reboot system
  ```bash
  sudo reboot
  ```
- [ ] After reboot, verify login works
- [ ] Open terminal

### Step 5: Configure Environment

**Goal**: Set up environment variables for patched ROCm

- [ ] Source RMCP environment script
  ```bash
  source /etc/profile.d/rocm-patched.sh
  ```
- [ ] Verify ROCM_PATH set correctly
  ```bash
  echo $ROCM_PATH
  # Should show: /opt/rocm-patched
  ```
- [ ] Verify HSA_USE_SVM disabled
  ```bash
  echo $HSA_USE_SVM
  # Should show: 0
  ```
- [ ] Verify HSA_XNACK disabled
  ```bash
  echo $HSA_XNACK
  # Should show: 0
  ```
- [ ] Check LD_LIBRARY_PATH includes patched libs
  ```bash
  echo $LD_LIBRARY_PATH | grep rocm-patched
  ```
- [ ] Add environment script to your shell profile
  ```bash
  echo 'source /etc/profile.d/rocm-patched.sh' >> ~/.bashrc
  ```

### Step 6: Verify Patched ROCm

- [ ] Check rocminfo works
  ```bash
  rocminfo | grep "Name:"
  # Should show your GPU
  ```
- [ ] Check HIP compiler
  ```bash
  hipcc --version
  ```
- [ ] Verify GPU is detected
  ```bash
  rocm-smi
  ```
- [ ] Check for RDNA patches in library
  ```bash
  strings /opt/rocm-patched/lib/libamdhip64.so | grep -i rdna
  ```

---

## Phase 3: Post-Patch Validation ⭕ PENDING

**Goal**: Verify RMCP patches fix the crashes  
**Status**: ⭕ Pending Phase 2 completion  
**Estimated Time**: 30 minutes

### Step 1: Run Integration Tests

- [ ] Navigate to project
  ```bash
  cd /home/kevin/Projects/rocm-patch
  ```
- [ ] Run project integration test
  ```bash
  ./tests/test_project_integration.sh
  ```
- [ ] Verify all tests pass:
  - [ ] System-wide RMCP verification
  - [ ] PyTorch GPU test
  - [ ] GPU memory allocation test
  - [ ] Kernel log analysis (no errors)
  - [ ] EEG2025 project detected
  - [ ] Thermal project detected

### Step 2: Run Comprehensive Workload Tests

**Critical**: This test crashed at Test #2 before RMCP

- [ ] Run real-world workload tests
  ```bash
  cd /home/kevin/Projects/rocm-patch
  python3 tests/test_real_world_workloads.py
  ```
- [ ] Verify all 10 tests pass:
  - [ ] Test 1: PyTorch Basic Operations (was ✅ before)
  - [ ] Test 2: PyTorch Convolutions (was ❌ CRASH before) **CRITICAL**
  - [ ] Test 3: PyTorch Training Loop
  - [ ] Test 4: YOLO Operations
  - [ ] Test 5: Transformer Operations
  - [ ] Test 6: Mixed Precision
  - [ ] Test 7: Memory Stress Test
  - [ ] Test 8: Kernel Fault Detection
  - [ ] Test 9: Patch Verification
  - [ ] Test 10: EEG Tensor Reshaping **CRITICAL**

- [ ] Verify success rate is 100% (was 10% before)
- [ ] Confirm no core dumps
- [ ] Check no memory errors in dmesg
  ```bash
  sudo dmesg | tail -50 | grep -i "memory\|fault"
  ```

### Step 3: Run Basic ROCm Tests

- [ ] Run basic patched ROCm test suite
  ```bash
  cd /home/kevin/Projects/rocm-patch
  ./scripts/test_patched_rocm.sh
  ```
- [ ] Verify all categories pass:
  - [ ] Environment variables
  - [ ] rocminfo output
  - [ ] HIP compilation
  - [ ] Memory allocation
  - [ ] PyTorch GPU detection
  - [ ] Kernel fault check
  - [ ] Patch verification

### Step 4: Regression Testing

- [ ] Verify basic operations still work
  ```bash
  python3 -c "import torch; x=torch.randn(100,100).cuda(); print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
  ```
- [ ] Check no new errors introduced
- [ ] Verify performance is maintained

---

## Phase 4: Real-World Projects ⭕ PENDING

**Goal**: Test RMCP in actual production projects  
**Status**: ⭕ Pending Phase 3 completion  
**Estimated Time**: 1-2 hours

### Project 1: EEG2025 - Brain Signal Classification

**Path**: `/home/kevin/Projects/eeg2025`  
**Problem**: Spatial convolution crash on `Conv2d(1, 32, (64, 1))`  
**Expected**: GPU training works, 10-20x faster than CPU

#### Setup

- [ ] Navigate to eeg2025 project
  ```bash
  cd /home/kevin/Projects/eeg2025
  ```
- [ ] Verify environment
  ```bash
  source /etc/profile.d/rocm-patched.sh
  echo $ROCM_PATH
  ```
- [ ] Check PyTorch detects GPU
  ```bash
  python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
  ```

#### Test Challenge 1: EEGNeX Model

- [ ] Locate training script
  ```bash
  ls *train*.py *main*.py *run*.py
  ```
- [ ] Review spatial convolution code
  ```bash
  grep -r "Conv2d.*64.*1" .
  ```
- [ ] Run short training test (1-2 epochs)
  ```bash
  # Adjust command based on project structure
  python3 train.py --epochs 2 --device cuda
  ```
- [ ] Verify training completes without crash
- [ ] Check GPU utilization
  ```bash
  rocm-smi --showuse
  ```
- [ ] Monitor for crashes
  ```bash
  # In another terminal
  watch -n 1 'sudo dmesg | tail -10'
  ```

#### Test Challenge 2: SAM+TCN Model

- [ ] Run SAM+TCN training test
- [ ] Verify temporal convolutions work
- [ ] Check no memory faults

#### Validation

- [ ] Training completes successfully
- [ ] GPU utilization >90%
- [ ] No crashes or core dumps
- [ ] No memory errors in dmesg
- [ ] Performance improvement documented

### Project 2: Thermal Object Detection - YOLO Training

**Path**: `/home/kevin/Projects/robust-thermal-image-object-detection`  
**Problem**: "Page not present" on every YOLO training batch  
**Expected**: 99% stability, 8-10x speedup

#### Setup

- [ ] Navigate to thermal project
  ```bash
  cd /home/kevin/Projects/robust-thermal-image-object-detection
  ```
- [ ] Verify environment
  ```bash
  source /etc/profile.d/rocm-patched.sh
  ```
- [ ] Check YOLO model files exist
  ```bash
  ls *yolo* *train* *detect*
  ```

#### Test YOLO Training

- [ ] Locate training script
- [ ] Run short training test (10-20 batches)
  ```bash
  # Adjust command based on project structure
  python3 train.py --epochs 1 --batch 4 --device cuda
  ```
- [ ] Verify no "page not present" errors
- [ ] Check no memory access faults
- [ ] Monitor kernel log
  ```bash
  sudo dmesg -w
  ```

#### Test YOLO Inference

- [ ] Run detection on sample images
- [ ] Verify GPU acceleration works
- [ ] Check output quality

#### Validation

- [ ] Training completes without crashes
- [ ] No page fault errors
- [ ] GPU utilization high (>80%)
- [ ] 8-10x faster than CPU fallback
- [ ] Detection accuracy maintained

---

## Phase 5: Long-Term Stability ⭕ PENDING

**Goal**: Ensure RMCP is production-ready  
**Status**: ⭕ Pending Phase 4 completion  
**Estimated Time**: 24+ hours

### Extended Training Test

- [ ] Set up 24-hour training session
  ```bash
  cd /home/kevin/Projects/eeg2025
  nohup python3 train.py --epochs 100 > training.log 2>&1 &
  ```
- [ ] Monitor GPU temperature
  ```bash
  watch -n 5 'rocm-smi --showtemp'
  ```
- [ ] Check memory usage periodically
  ```bash
  watch -n 60 'rocm-smi --showmeminfo'
  ```
- [ ] Monitor for crashes
  ```bash
  tail -f training.log
  ```

### Memory Leak Testing

- [ ] Run repeated allocation/deallocation
- [ ] Monitor GPU memory over time
- [ ] Check for gradual memory increase
- [ ] Verify GPU memory releases correctly

### Kernel Log Monitoring

- [ ] Set up continuous monitoring
  ```bash
  sudo dmesg -w | tee kernel-monitor.log
  ```
- [ ] Check for memory fault messages
- [ ] Look for GPU reset events
- [ ] Verify no warnings accumulate

### Performance Benchmarking

- [ ] Measure training time (EEG2025)
  - [ ] Time per epoch
  - [ ] Samples per second
  - [ ] GPU utilization average
- [ ] Measure training time (Thermal YOLO)
  - [ ] Time per epoch
  - [ ] Batches per second
  - [ ] GPU memory usage
- [ ] Compare to CPU baseline
- [ ] Document speedup achieved

### Validation Metrics

- [ ] Zero crashes over 24+ hours
- [ ] No memory leaks detected
- [ ] Stable GPU temperature
- [ ] Consistent performance
- [ ] No kernel warnings/errors

---

## 📈 Success Criteria

### Phase 1 Success ✅
- [x] Crash confirmed on unpatched ROCm
- [x] Error documented
- [x] Testing framework created

### Phase 2 Success (To Be Determined)
- [ ] Patches installed successfully
- [ ] Environment configured correctly
- [ ] No installation errors
- [ ] Patched ROCm detected

### Phase 3 Success (To Be Determined)
- [ ] All integration tests pass
- [ ] Test #2 (convolutions) passes (was crashing)
- [ ] Test #10 (EEG reshaping) passes (was crashing)
- [ ] Success rate: 100% (was 10%)
- [ ] No regressions

### Phase 4 Success (To Be Determined)
- [ ] EEG2025 training completes without crashes
- [ ] Thermal YOLO training completes without crashes
- [ ] GPU utilization >90%
- [ ] 10-20x speedup vs CPU achieved
- [ ] No memory faults

### Phase 5 Success (To Be Determined)
- [ ] 24+ hour stability achieved
- [ ] No memory leaks
- [ ] Consistent performance
- [ ] Production-ready

---

## 🚨 Rollback Plan

If patches cause issues:

### Quick Rollback

```bash
# 1. Switch back to original ROCm
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

# 2. Remove environment script
sudo rm /etc/profile.d/rocm-patched.sh

# 3. Restore original kernel module (if backed up)
sudo cp /lib/modules/$(uname -r)/updates/dkms/amdgpu.ko.backup \
        /lib/modules/$(uname -r)/updates/dkms/amdgpu.ko
sudo depmod -a
sudo reboot
```

### Full Removal

```bash
# Remove patched ROCm
sudo rm -rf /opt/rocm-patched

# Restore original ROCm paths
unset ROCM_PATH HSA_USE_SVM HSA_XNACK

# Reboot
sudo reboot
```

---

## 📝 Notes

### Important Reminders

- **Always backup** before patching
- **Reboot required** after kernel module patch
- **Environment variables** must be set in every new shell
- **Test incrementally** - don't skip phases
- **Document results** - update this TODO as you progress

### Troubleshooting

If tests still crash after patching:
1. Verify environment variables are set
2. Check patched libraries are being used (`ldd`)
3. Review kernel logs for errors
4. Try rebooting to ensure kernel module loads
5. See `docs/TESTING.md` for detailed troubleshooting

### Time Estimates

- Phase 1: ✅ Complete (2 hours)
- Phase 2: 2-4 hours (mostly compilation)
- Phase 3: 30 minutes (testing)
- Phase 4: 1-2 hours (project testing)
- Phase 5: 24+ hours (stability)

**Total**: ~28-32 hours (mostly unattended)

---

## ✅ Completion Checklist

Final validation before marking RMCP as complete:

- [ ] All 5 phases complete
- [ ] All tests passing
- [ ] EEG2025 project works
- [ ] Thermal project works
- [ ] 24+ hour stability achieved
- [ ] Documentation updated
- [ ] Results documented in TESTING.md
- [ ] GitHub repository updated

---

**Last Updated**: November 6, 2024  
**Next Action**: Begin Phase 2 - Run `sudo ./scripts/patch_rocm_source.sh`
