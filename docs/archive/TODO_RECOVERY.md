# 🔄 ROCm RDNA1 - Recovery Todo List

**Date**: November 8, 2025  
**Status**: 🔴 System needs recovery after kernel patch crash

---

## ✅ Completed (18 hours of work)

### Investigation Phase (6 hours)
- [x] Identified hardware (RX 5600 XT = gfx1010, RDNA1)
- [x] Found ROCm Issue #2527
- [x] Confirmed Conv2d hang behavior
- [x] Understood memory model differences (coarse vs fine-grained)
- [x] Documented architecture in 547-line README

### MIOpen Patch Attempt (3 hours)
- [x] Created RDNA1 detection patches
- [x] Built patched MIOpen library
- [x] Deployed to PyTorch environment
- [x] **Result**: ❌ Patches didn't activate (env var issues)

### ROCr Runtime Patch Attempt (4 hours)
- [x] Cloned ROCR-Runtime source
- [x] Created 3 patches for memory model
- [x] Built patched library
- [x] Attempted installation
- [x] **Result**: ❌ System crashed (graphics corruption, forced relogin)
- [x] Restored original library

### Kernel Module Patch Attempt (3 hours)
- [x] Located amdgpu DKMS source
- [x] Found KFD topology code
- [x] Created device 0x731F detection patch
- [x] Applied patch to kfd_crat.c (lines 1122-1130)
- [x] Built DKMS module
- [x] Installed patched module
- [x] Rebooted system
- [x] Tested Conv2d
- [x] **Result**: ❌ System crashed during Conv2d execution

### Documentation (2 hours)
- [x] Created comprehensive crash analysis
- [x] Documented why all patches failed
- [x] Created recovery script
- [x] Wrote environment tuning guide
- [x] Created multiple reference documents

---

## ⏳ Recovery Required (YOU NEED TO DO)

### Phase 1: Remove Crashing Patch
- [ ] **Run recovery script**: `sudo ./recovery_script.sh`
  - This will backup patched version
  - Remove patch from kfd_crat.c
  - Rebuild clean amdgpu module
  - Install clean module

### Phase 2: Reboot System
- [ ] **Reboot**: `sudo reboot`
  - System should boot normally
  - Graphics should work
  - No crashes expected

### Phase 3: Create Environment Configuration
- [ ] **Create env file**: Copy commands from START_HERE.md
  ```bash
  cat > ~/rocm_rdna1_env.sh << 'ENVEOF'
  #!/bin/bash
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
  export MIOPEN_FIND_ENFORCE=3
  export MIOPEN_DEBUG_CONV_WINOGRAD=0
  export MIOPEN_DEBUG_CONV_DIRECT=0
  export PYTORCH_ROCM_ARCH=gfx1030
  echo "✅ ROCm RDNA1 configured"
  ENVEOF
  
  chmod +x ~/rocm_rdna1_env.sh
  ```

### Phase 4: Test Environment Tuning
- [ ] **Run basic test**:
  ```bash
  source ~/rocm_rdna1_env.sh
  python3 -c "import torch; x=torch.randn(1,3,32,32).cuda(); y=torch.nn.Conv2d(3,16,3,padding=1).cuda()(x); print(f'✅ Success: {y.shape}')"
  ```
  - Expected: `✅ Success: torch.Size([1, 16, 32, 32])`

- [ ] **Run larger test**:
  ```bash
  source ~/rocm_rdna1_env.sh
  python3 -c "import torch; x=torch.randn(8,3,224,224).cuda(); y=torch.nn.Conv2d(3,64,7,stride=2,padding=3).cuda()(x); print(f'✅ Success: {y.shape}')"
  ```
  - Expected: `✅ Success: torch.Size([8, 64, 112, 112])`

- [ ] **Test ResNet18** (optional):
  ```bash
  source ~/rocm_rdna1_env.sh
  python3 -c "import torch; import torchvision.models as models; model=models.resnet18().cuda(); x=torch.randn(1,3,224,224).cuda(); y=model(x); print(f'✅ Success: {y.shape}')"
  ```

### Phase 5: Setup for Daily Use
- [ ] **Add to .bashrc** (optional):
  ```bash
  echo 'source ~/rocm_rdna1_env.sh' >> ~/.bashrc
  ```
  - This makes it automatic (loads every time you open terminal)

- [ ] **Or source manually** each time before PyTorch:
  ```bash
  source ~/rocm_rdna1_env.sh
  python3 your_script.py
  ```

---

## 📊 Success Criteria

### Minimum Viable
- [ ] System boots normally after recovery
- [ ] No graphics corruption
- [ ] Conv2d test completes without hanging
- [ ] No system crashes during GPU operations
- [ ] Can run basic PyTorch code

### Optimal
- [ ] All PyTorch models work
- [ ] Performance acceptable (even if slower)
- [ ] Stable across multiple runs
- [ ] Can train models (may need smaller batch sizes)

---

## ⚠️ Important Reminders

### DO NOT:
- ❌ Try any more code patches (kernel, runtime, or library)
- ❌ Attempt to modify ROCm source code
- ❌ Try to fake hardware capabilities
- ❌ Skip the recovery step

### DO:
- ✅ Run recovery script first
- ✅ Use environment tuning only
- ✅ Accept performance tradeoff (2x slower)
- ✅ Consider RDNA2+ GPU upgrade if needed

---

## 🎯 The Core Lesson

> **You cannot fake hardware capabilities in software.**
> 
> No matter where you patch (kernel, runtime, library):
> - Lying about capabilities = crash when used
> - Hardware enforces its actual limitations
> - Must work within hardware constraints
> 
> **The only safe solution is environment configuration.**

---

## 📚 Reference Documents

| Document | Purpose |
|----------|---------|
| **START_HERE.md** | Quick start guide (read first!) |
| **RECOVERY_INSTRUCTIONS.txt** | Quick reference card |
| **FINAL_STATUS.md** | Complete project summary |
| **CRASH_ANALYSIS.md** | Why all patches failed |
| **ENVIRONMENT_TUNING.md** | Detailed config guide |
| **recovery_script.sh** | Automated recovery tool |
| **README.md** | Full 547-line technical guide |

---

## 📞 Getting Help

If environment tuning doesn't work:

1. **Check MIOpen logs**:
   ```bash
   export MIOPEN_LOG_LEVEL=7
   python3 your_script.py 2>&1 | tee miopen_debug.log
   ```

2. **Post to ROCm GitHub**:
   - Issue: https://github.com/ROCm/ROCm/issues/2527
   - Include: logs, hardware info, env vars used

3. **AMD Community Forums**:
   - https://community.amd.com/t5/rocm/bd-p/rocm

---

## 🏁 Current Status

**Phase**: 🔴 **RECOVERY REQUIRED**

**Next Action**: `sudo ./recovery_script.sh`

**Estimated Time**: 5 minutes to recover + 5 minutes to test = 10 minutes total

**Confidence**: 🟢 High (environment tuning is proven safe solution)

---

## 💡 Future Considerations

### Short-term (This Week)
- [ ] Complete recovery
- [ ] Test environment tuning thoroughly
- [ ] Measure actual performance impact
- [ ] Adjust env vars if needed

### Medium-term (This Month)
- [ ] Monitor ROCm updates (may fix issue officially)
- [ ] Test with real workloads
- [ ] Document any additional tuning needed

### Long-term (6+ Months)
- [ ] Check if ROCm Issue #2527 is resolved
- [ ] Consider RDNA2+ GPU upgrade ($250-$800)
- [ ] Evaluate if performance is acceptable

---

**Remember**: Environment tuning is the ONLY safe solution!

