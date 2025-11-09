# Action Plan: Enable Large Feature Maps on RDNA1

**GPU**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)  
**Goal**: Run Conv2d operations with medium/large tensor sizes (48x48+, 16→32+ channels)  
**Current Limitation**: ROCm 5.7 hangs on medium/large tensors  
**Root Cause**: ROCm 5.3+ regression broke gfx1010 support

---

## 📊 Executive Decision Required

Based on extensive GitHub and Reddit research, we have **3 viable options**:

| Option | ROCm Version | PyTorch Version | Effort | Stability | Features | Large Feature Maps |
|--------|--------------|-----------------|--------|-----------|----------|-------------------|
| **Option 1** | 5.2 | 1.13.1 | ⭐ Low | ✅ Excellent | ❌ Old | ✅ Works |
| **Option 2** | 6.2+ | 2.4+ | ⭐⭐⭐⭐ High | ⚠️ Good | ✅ Latest | ✅ Works |
| **Option 3** | 5.7 | 2.2.2 | - | ⚠️ Partial | ⚠️ Some | ❌ Hangs |

**Recommendation**: Start with **Option 1** (ROCm 5.2) for immediate working solution, then optionally pursue **Option 2** if latest features are needed.

---

## Option 1: ROCm 5.2 + PyTorch 1.13.1 (QUICK WIN)

### ✅ Pros
- **Proven working** by Stable Diffusion community
- **No hangs** on large feature maps
- **Fast installation** (1-2 hours max)
- **Stable** and predictable
- Used successfully by thousands of users

### ❌ Cons
- Old PyTorch version (1.13.1 vs 2.2.2)
- Missing newer features (flash attention, memory efficient attention)
- Python 3.10 required (wheels only available for 3.10)

### 📋 Implementation Steps

```bash
# Step 1: Backup current configuration
sudo cp /etc/profile.d/rocm-rdna1-57.sh /etc/profile.d/rocm-rdna1-57.sh.backup

# Step 2: Uninstall ROCm 5.7
sudo apt-get purge -y rocm-core rocm-dev rocm-libs
sudo apt-get autoremove -y

# Step 3: Install ROCm 5.2
# Download from: https://repo.radeon.com/rocm/apt/5.2/
wget https://repo.radeon.com/rocm/apt/5.2/rocm-install.sh
sudo bash rocm-install.sh --rocm-version=5.2

# Step 4: Create new environment variables script
sudo tee /etc/profile.d/rocm-rdna1-52.sh << 'SCRIPT'
#!/bin/bash
# ROCm 5.2 Configuration for RDNA1 (RX 5600 XT / RX 5700 XT)
# gfx1010 (Navi 10) - Last fully working ROCm version

# Detect RDNA1 GPU
if lspci | grep -iE 'VGA.*Radeon.*(731F|731E|7310|7312)' > /dev/null 2>&1; then
    # Core ROCm paths
    export ROCM_PATH=/opt/rocm-5.2.0
    export PATH=$ROCM_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
    
    # GPU architecture spoofing (gfx1010 → gfx1030)
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_ROCM_ARCH=gfx1010
    
    # Force GEMM-only convolutions
    export MIOPEN_DEBUG_CONV_GEMM=1
    export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
    export MIOPEN_DEBUG_CONV_WINOGRAD=0
    export MIOPEN_DEBUG_CONV_DIRECT=0
    export MIOPEN_DEBUG_CONV_FFT=0
    
    # MIOpen database settings
    export MIOPEN_FIND_MODE=NORMAL
    export MIOPEN_FIND_ENFORCE=NONE
    export MIOPEN_USER_DB_PATH=/tmp/miopen-cache-$USER
    
    # Memory management (critical for RDNA1)
    export HIP_FORCE_COARSE_GRAIN=1
    export HSA_ENABLE_SDMA=0
    export HSA_USE_SVM=0
    export HSA_XNACK=0
fi
SCRIPT

# Step 5: Make executable and source
sudo chmod +x /etc/profile.d/rocm-rdna1-52.sh
source /etc/profile.d/rocm-rdna1-52.sh

# Step 6: Install PyTorch 1.13.1 + ROCm 5.2
# Create Python 3.10 virtual environment
python3.10 -m venv ~/pytorch-rocm52-venv
source ~/pytorch-rocm52-venv/bin/activate

# Install PyTorch wheels
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2

# Step 7: Test large feature maps
python3 << 'PYTHON'
import torch
import time

print("Testing ROCm 5.2 + PyTorch 1.13.1 with large feature maps...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test medium size (previously hung on ROCm 5.7)
print("\n[TEST 1] Medium: 16→32 channels, 48x48 input")
x = torch.randn(1, 16, 48, 48).cuda()
conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
start = time.time()
y = conv(x)
elapsed = time.time() - start
print(f"✅ SUCCESS! Time: {elapsed:.4f}s, Output: {y.shape}")

# Test large size (definitely hung on ROCm 5.7)
print("\n[TEST 2] Large: 32→64 channels, 64x64 input")
x = torch.randn(1, 32, 64, 64).cuda()
conv = torch.nn.Conv2d(32, 64, 3, padding=1).cuda()
start = time.time()
y = conv(x)
elapsed = time.time() - start
print(f"✅ SUCCESS! Time: {elapsed:.4f}s, Output: {y.shape}")

# Test extra large
print("\n[TEST 3] Extra Large: 64→128 channels, 128x128 input")
x = torch.randn(1, 64, 128, 128).cuda()
conv = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
start = time.time()
y = conv(x)
elapsed = time.time() - start
print(f"✅ SUCCESS! Time: {elapsed:.4f}s, Output: {y.shape}")

print("\n✅ All tests passed! Large feature maps work on ROCm 5.2")
PYTHON
```

### ⏱️ Estimated Time
- **Installation**: 1-2 hours
- **Testing**: 10 minutes
- **Total**: ~2 hours

### 🎯 Success Criteria
- ✅ Medium Conv2d (16→32, 48x48) completes in <5 seconds
- ✅ Large Conv2d (32→64, 64x64) completes in <10 seconds
- ✅ No hangs or timeouts
- ✅ Stable across multiple runs

---

## Option 2: ROCm 6.2 + PyTorch from Source (ADVANCED)

### ✅ Pros
- **Latest PyTorch** (2.4+)
- **Latest features** (flash attention, etc.)
- **Better performance** (reported by community)
- **Future-proof**

### ❌ Cons
- **Long build time** (4-8 hours)
- **Complex process** (multiple dependencies)
- **Must rebuild** for each PyTorch update
- **Experimental** (less tested than ROCm 5.2)

### 📋 Implementation Steps

```bash
# TODO: Create automated build script (install_rocm62_pytorch.sh)
# Based on @Zakhrov's instructions from GitHub discussion #4030

# High-level overview:
# 1. Install ROCm 6.2.4 from AMD repos
# 2. Install build dependencies (GCC 10+, cmake, ninja, etc.)
# 3. Clone PyTorch repository
# 4. Build AMD-specific code
# 5. Compile PyTorch with PYTORCH_ROCM_ARCH=gfx1010
# 6. Test large feature maps

# Full script will be created if user selects this option
```

### ⏱️ Estimated Time
- **Installation**: 6-10 hours (mostly automated)
- **Testing**: 10 minutes
- **Total**: ~8 hours

### 🎯 Success Criteria
- ✅ PyTorch 2.4+ installed and working
- ✅ Large Conv2d operations work
- ✅ Flash attention available
- ✅ No hangs or crashes

---

## Option 3: Stay on ROCm 5.7 (NOT RECOMMENDED)

### Status
- ✅ Works for small tensors (≤32x32)
- ❌ Hangs on medium/large tensors
- ⚠️ Limited usefulness for real applications

**Verdict**: Only suitable if you're working exclusively with small models (unlikely).

---

## 🚀 Recommended Workflow

### Phase 1: Quick Win (ROCm 5.2)
1. **Today**: Implement Option 1 (ROCm 5.2 + PyTorch 1.13.1)
2. **Verify**: Test large feature maps work
3. **Document**: Update README with new configuration
4. **Use**: Run real applications (Stable Diffusion, etc.)

### Phase 2: Future Upgrade (Optional)
1. **If needed**: Implement Option 2 (ROCm 6.2 + PyTorch 2.4)
2. **Compare**: Benchmark performance vs ROCm 5.2
3. **Decide**: Keep whichever works best for your use case

---

## 📝 Documentation Updates Required

After implementing Option 1 or 2:

1. **README.md**: Update ROCm version recommendation
2. **QUICK_REFERENCE.md**: Update environment variables
3. **verify_setup.sh**: Update version checks
4. **test_conv2d_timing.py**: Add large tensor tests
5. **Create new docs**:
   - `ROCM_52_SOLUTION.md` (for Option 1)
   - `ROCM_62_BUILD_GUIDE.md` (for Option 2)

---

## 🎯 Todo List

```markdown
### Phase 1: Research (COMPLETE ✅)
- [x] Search GitHub for RDNA1 solutions
- [x] Read GitHub issue #2527
- [x] Read GitHub discussion #4030
- [x] Search Reddit for community solutions
- [x] Analyze Stable Diffusion community workarounds
- [x] Document all findings
- [x] Create action plan

### Phase 2: Implementation (PENDING)
- [ ] **DECISION REQUIRED**: Choose Option 1 or Option 2
- [ ] Backup current ROCm 5.7 configuration
- [ ] Implement chosen solution
- [ ] Test small Conv2d operations
- [ ] Test medium Conv2d operations (16→32, 48x48)
- [ ] Test large Conv2d operations (32→64, 64x64)
- [ ] Test extra large Conv2d operations (64→128, 128x128)
- [ ] Run verify_setup.sh
- [ ] Run real-world application tests

### Phase 3: Documentation (PENDING)
- [ ] Update README.md with correct ROCm version
- [ ] Update QUICK_REFERENCE.md
- [ ] Update verify_setup.sh
- [ ] Update test_conv2d_timing.py
- [ ] Create solution-specific documentation
- [ ] Update PROJECT_STRUCTURE.md
- [ ] Mark INVESTIGATION_FINAL_SUMMARY.md as superseded

### Phase 4: Validation (PENDING)
- [ ] Test with Stable Diffusion
- [ ] Test with ComfyUI
- [ ] Test with real deep learning models
- [ ] Verify stability over time
- [ ] Confirm no regressions
```

---

## 💡 Key Insights

1. **User was RIGHT**: RX 5600 XT hardware is fully capable
2. **We chose the WRONG ROCm version**: 5.7 has the regression
3. **Solution EXISTS**: ROCm 5.2 is proven working
4. **Community proof**: Stable Diffusion users run large models successfully
5. **No hardware limitation**: This is purely a software issue

---

## 🔗 References

All research findings documented in: `RESEARCH_FINDINGS_RDNA1_SOLUTIONS.md`

---

## ⚡ Next Steps

**USER DECISION REQUIRED**:

Which option do you prefer?

**Option 1 (RECOMMENDED)**: ROCm 5.2 + PyTorch 1.13.1
- ✅ Fast (2 hours)
- ✅ Proven working
- ❌ Old PyTorch

**Option 2 (ADVANCED)**: ROCm 6.2 + PyTorch 2.4 from source
- ✅ Latest features
- ❌ Long build (8 hours)
- ⚠️ More complex

**Please advise which path you'd like to take, and I'll implement it immediately.**
