# ROCm 5.2 Installation TODO List

**Status**: Ready for execution  
**Estimated Time**: 30 minutes  
**Risk Level**: Low (backups included, can revert)

---

## Pre-Installation (5 min)

- [ ] **Read documentation** (optional but recommended)
  - [ ] `QUICK_START_ROCM52.md` - Quick overview
  - [ ] `DECISION_ROCM_52_VS_57.md` - Full analysis
  - [ ] `ROCM_VERSION_ANALYSIS.md` - Research details

- [ ] **Verify prerequisites**
  - [ ] Currently on ROCm 5.7.1 ✅ (confirmed)
  - [ ] Have sudo access
  - [ ] Internet connection stable
  - [ ] ~2GB free disk space

---

## Installation (25 min)

- [ ] **Run installation script**
  ```bash
  cd ~/Projects/rocm-patch
  ./install_rocm52.sh
  ```

- [ ] **Script will automatically**:
  - [ ] Backup ROCm 5.7 configuration
  - [ ] Backup PyTorch versions
  - [ ] Backup MIOpen cache
  - [ ] Remove ROCm 5.7 completely
  - [ ] Install ROCm 5.2 from repository
  - [ ] Install PyTorch 2.0+rocm5.2
  - [ ] Configure environment for RDNA1
  - [ ] Run test suite

- [ ] **Review test results**
  - [ ] Test 1: ROCm info (should show gfx1010)
  - [ ] Test 2: PyTorch CUDA (should be available)
  - [ ] Test 3: Small Conv2d (should work - baseline)
  - [ ] Test 4: Medium Conv2d (currently hangs in 5.7, should work in 5.2!)

---

## Post-Installation (Critical!)

- [ ] **Log out and log back in** (required for group changes)

- [ ] **Verify environment**
  ```bash
  echo $HSA_OVERRIDE_GFX_VERSION  # Should be 10.3.0
  echo $ROCM_PATH                 # Should be /opt/rocm
  python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
  ```

- [ ] **Test the hang bug is fixed**
  ```bash
  # This HANGS in ROCm 5.7, should work in 5.2
  python3 -c "
  import torch
  print('Testing 16→32 channels, 64x64 (hangs in 5.7)...')
  x = torch.randn(1, 16, 64, 64).cuda()
  conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
  y = conv(x)
  print(f'✅ SUCCESS! Output: {y.shape}')
  "
  ```

- [ ] **Test larger sizes**
  ```bash
  python3 test_conv2d_timing.py
  ```

---

## Verification Checklist

### ✅ Success Indicators
- [ ] `rocminfo` shows gfx1010
- [ ] PyTorch CUDA is available
- [ ] Small Conv2d works (baseline)
- [ ] Medium Conv2d (64x64) works **without hanging**
- [ ] Large Conv2d (128x128) works
- [ ] No 30-minute hangs
- [ ] First-run timing is reasonable (<60 seconds)

### ❌ Failure Indicators
If you see these, something went wrong:
- [ ] ROCm not found
- [ ] PyTorch CUDA not available
- [ ] Medium Conv2d still hangs (>30 seconds)
- [ ] Segmentation faults
- [ ] "Out of Memory" errors (different issue)

---

## If Successful

- [ ] **Update documentation**
  - [ ] Update `README.md` to reflect ROCm 5.2
  - [ ] Mark `INVESTIGATION_FINAL_SUMMARY.md` as superseded
  - [ ] Create success report

- [ ] **Test real workloads**
  - [ ] Run your actual models
  - [ ] Test training/inference
  - [ ] Verify performance

- [ ] **Celebrate!** 🎉
  - [ ] Large feature maps work
  - [ ] Hardware is properly utilized
  - [ ] Problem solved

---

## If Unsuccessful (10% chance)

- [ ] **Gather diagnostic information**
  ```bash
  # Check ROCm version
  /opt/rocm/bin/rocminfo | head -20
  
  # Check PyTorch version
  python3 -c "import torch; print(torch.__version__)"
  
  # Check environment
  env | grep -E '(ROCM|HIP|HSA|MIOPEN)'
  
  # Check for errors
  dmesg | tail -50
  ```

- [ ] **Try troubleshooting**
  - [ ] Verify group membership: `groups` (should include video, render)
  - [ ] Clear MIOpen cache: `rm -rf /tmp/miopen-cache-$USER`
  - [ ] Reboot system
  - [ ] Re-run tests

- [ ] **Revert to ROCm 5.7** (if needed)
  ```bash
  # Restore backup
  sudo cp ~/rocm-rdna1-57.sh.backup /etc/profile.d/rocm-rdna1-57.sh
  
  # Reinstall ROCm 5.7
  ./install_rocm57.sh
  ```

---

## Next Steps After Success

- [ ] **Test with real workloads**
  - [ ] Stable Diffusion
  - [ ] Training models
  - [ ] Inference tasks

- [ ] **Consider PyTorch 2.2.2** (if needed)
  - [ ] Build from source with ROCm 5.2
  - [ ] Or use PyTorch 2.0 (works fine for most tasks)

- [ ] **Document your experience**
  - [ ] Report back if it fixed the issue
  - [ ] Note any unexpected behavior
  - [ ] Help other RDNA1 users

---

## Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| Read docs | 5-10 min | Understand the change |
| Run script | 20-25 min | Installation complete |
| Log out/in | 1 min | Environment active |
| Test | 5 min | Verify it works |
| **Total** | **~30 min** | **Problem solved!** |

---

## Decision Point

**Ready to proceed?**

- ✅ **YES** → Run `./install_rocm52.sh`
- ❌ **NO** → Read documentation first, then decide
- ❓ **UNSURE** → Ask questions, review evidence

**Confidence**: 90% success rate based on community evidence

**Risk**: Low (backups included, can revert)

**Reward**: Large feature maps work, hardware fully utilized

---

## After Installation Report

Once complete, update this section:

**Installation Date**: ___________  
**Time Taken**: ___________  
**Success**: ☐ YES ☐ NO  

**Test Results**:
- Small Conv2d (32x32): ☐ Pass ☐ Fail  
- Medium Conv2d (64x64): ☐ Pass ☐ Fail ☐ Hang  
- Large Conv2d (128x128): ☐ Pass ☐ Fail ☐ Hang  

**Notes**: ___________

---

**Last Updated**: 2025-01-XX  
**Status**: Ready for execution
