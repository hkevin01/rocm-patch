# TODO: ROCm 5.2 Migration for RDNA1 (RX 5600 XT)

**Date**: November 8, 2025  
**Goal**: Migrate from ROCm 5.7 to ROCm 5.2 to fix Conv2d hang bug  
**Reason**: ROCm 5.3+ introduced regression for gfx1010; 5.2 is last working version

---

## Current Status (ROCm 5.7)

- [x] ROCm 5.7.1 installed
- [x] PyTorch 2.2.2+rocm5.7 installed
- [x] Basic Conv2d works (≤32x32 inputs)
- [x] Environment variables configured
- [ ] **BROKEN**: Conv2d hangs on 16→32 channels with >32x32 inputs
- [ ] **BROKEN**: MIOpen GEMM kernel compilation freezes

## Migration Plan

### Phase 1: Backup Current Setup ⚠️ CRITICAL

- [ ] 1.1 Document current environment variables
  ```bash
  env | grep -E "ROCM|MIOPEN|HSA|HIP" > current_env_backup.txt
  ```

- [ ] 1.2 Backup current MIOpen cache
  ```bash
  cp -r ~/.config/miopen ~/.config/miopen_backup_57
  ```

- [ ] 1.3 Test and record current working configs
  ```bash
  python3 test_conv2d_timing.py > test_results_rocm57.txt
  ```

- [ ] 1.4 Save list of installed ROCm packages
  ```bash
  dpkg -l | grep rocm > installed_rocm57_packages.txt
  ```

### Phase 2: Uninstall ROCm 5.7

- [ ] 2.1 Remove ROCm 5.7 packages
  ```bash
  sudo apt remove --purge -y 'rocm*' 'hip*' 'miopen*' 'rocblas*'
  ```

- [ ] 2.2 Clean up configuration files
  ```bash
  sudo rm -rf /opt/rocm*
  sudo rm -rf /etc/apt/sources.list.d/rocm.list*
  ```

- [ ] 2.3 Remove environment configuration
  ```bash
  sudo rm /etc/profile.d/rocm-rdna1-57.sh
  ```

- [ ] 2.4 Clean apt cache
  ```bash
  sudo apt autoremove -y
  sudo apt autoclean
  ```

### Phase 3: Install ROCm 5.2

- [ ] 3.1 Add ROCm 5.2 repository
  ```bash
  wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
  echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list
  sudo apt update
  ```

- [ ] 3.2 Install ROCm 5.2 core packages
  ```bash
  sudo apt install -y rocm-dev rocm-libs rocminfo rocm-smi hip-base hip-runtime-amd
  ```

- [ ] 3.3 Install MIOpen for ROCm 5.2
  ```bash
  sudo apt install -y miopen-hip rocblas
  ```

- [ ] 3.4 Verify installation
  ```bash
  rocminfo | grep gfx1010
  /opt/rocm/bin/rocm-smi
  ```

### Phase 4: Install PyTorch for ROCm 5.2

- [ ] 4.1 Uninstall current PyTorch
  ```bash
  pip uninstall -y torch torchvision torchaudio
  ```

- [ ] 4.2 Install PyTorch 2.2.2+rocm5.2
  ```bash
  pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2
  ```

- [ ] 4.3 Verify PyTorch installation
  ```bash
  python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
  ```

### Phase 5: Configure Environment for ROCm 5.2

- [ ] 5.1 Create new environment script
  ```bash
  sudo ./install_rocm52.sh
  ```

- [ ] 5.2 Verify environment variables
  ```bash
  source /etc/profile.d/rocm-rdna1-52.sh
  env | grep -E "ROCM|MIOPEN|HSA|HIP"
  ```

- [ ] 5.3 Check for conflicts
  ```bash
  ./verify_setup.sh
  ```

### Phase 6: Testing

- [ ] 6.1 Test basic GPU availability
  ```bash
  python3 -c "import torch; print(torch.cuda.get_device_name(0))"
  ```

- [ ] 6.2 Test small Conv2d (known to work)
  ```bash
  python3 -c "
  import torch
  x = torch.randn(1, 3, 32, 32).cuda()
  conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
  print(conv(x).shape)
  "
  ```

- [ ] 6.3 **CRITICAL TEST**: Test 16→32, 48x48 (previously hung)
  ```bash
  timeout 30 python3 -c "
  import torch
  x = torch.randn(1, 16, 48, 48).cuda()
  conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
  print('SUCCESS:', conv(x).shape)
  "
  ```

- [ ] 6.4 Test 16→32, 64x64 (previously hung)
  ```bash
  timeout 30 python3 -c "
  import torch
  x = torch.randn(1, 16, 64, 64).cuda()
  conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
  print('SUCCESS:', conv(x).shape)
  "
  ```

- [ ] 6.5 Test ResNet-like configurations
  ```bash
  python3 -c "
  import torch
  x = torch.randn(1, 64, 56, 56).cuda()
  conv = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
  print('SUCCESS:', conv(x).shape)
  "
  ```

- [ ] 6.6 Run full timing test suite
  ```bash
  python3 test_conv2d_timing.py
  ```

- [ ] 6.7 Test ResNet50 if available
  ```bash
  python3 -c "
  import torch, torchvision
  model = torchvision.models.resnet50().cuda().eval()
  x = torch.randn(1, 3, 224, 224).cuda()
  with torch.no_grad():
      print('ResNet50 SUCCESS:', model(x).shape)
  "
  ```

### Phase 7: Verification & Documentation

- [ ] 7.1 Compare results with ROCm 5.7
  ```bash
  diff test_results_rocm57.txt test_results_rocm52.txt
  ```

- [ ] 7.2 Document any differences

- [ ] 7.3 Update README.md with ROCm 5.2 instructions

- [ ] 7.4 Create rollback instructions

- [ ] 7.5 Archive ROCm 5.7 documentation

### Phase 8: Rollback Plan (If 5.2 Fails)

- [ ] 8.1 Uninstall ROCm 5.2
- [ ] 8.2 Reinstall ROCm 5.7
- [ ] 8.3 Restore PyTorch 2.2.2+rocm5.7
- [ ] 8.4 Restore environment configuration
- [ ] 8.5 Document why 5.2 didn't work

---

## Expected Outcomes

### If ROCm 5.2 Works ✅

- ✅ Conv2d works on ALL tensor sizes
- ✅ No hangs on 16→32, 48x48+
- ✅ ResNet50 runs successfully
- ✅ ImageNet models work
- ✅ YOLO models work
- ✅ Full GPU utilization

### If ROCm 5.2 Doesn't Work ❌

- Document specific failures
- Compare with 5.7 behavior
- Research other ROCm versions (5.1, 5.0, 4.5)
- Consider hardware upgrade to RDNA2

---

## Success Criteria

**ROCm 5.2 is considered successful if**:

1. ✅ All Phase 6 tests pass without timeout
2. ✅ Conv2d 16→32, 64x64 completes in <5 seconds
3. ✅ No MIOpen kernel hangs
4. ✅ ResNet50 inference works
5. ✅ Stable across multiple runs

---

## Timeline

- **Backup**: 15 minutes
- **Uninstall**: 10 minutes
- **Install**: 30 minutes
- **Configure**: 10 minutes
- **Testing**: 30 minutes
- **Documentation**: 30 minutes

**Total**: ~2 hours

---

## Notes

- **Community reports**: Multiple users confirmed ROCm 5.2 works for gfx1010
- **Regression point**: Between 5.2 and 5.3
- **PyTorch compatibility**: 2.2.2 works with both 5.2 and 5.7
- **Risk**: Low (can rollback to 5.7)

---

**Ready to proceed?** Start with Phase 1 (Backup)!
