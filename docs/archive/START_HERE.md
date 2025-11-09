# 🚨 START HERE - ROCm RDNA1 Recovery Guide

**Your system CRASHED after kernel patch. Follow these steps to recover.**

---

## 🎯 What You Need to Know

After 16 hours of testing, we learned:

> **You CANNOT fake hardware capabilities in software.**
> All patches (kernel, runtime, library) caused crashes.
> The ONLY safe solution is environment configuration.

---

## ⚡ QUICK RECOVERY (5 minutes)

### 1. Remove Crashing Patch
```bash
cd ~/Projects/rocm-patch
sudo ./recovery_script.sh
```

### 2. Reboot
```bash
sudo reboot
```

### 3. Create Environment File
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

### 4. Test It Works
```bash
source ~/rocm_rdna1_env.sh
python3 -c "import torch; x=torch.randn(1,3,32,32).cuda(); y=torch.nn.Conv2d(3,16,3,padding=1).cuda()(x); print(f'✅ Success: {y.shape}')"
```

**Expected output**: `✅ Success: torch.Size([1, 16, 32, 32])`

---

## 📊 What We Tried (All Failed)

| Attempt | Result | Reason |
|---------|--------|--------|
| MIOpen Patch | ❌ Failed | Env vars didn't work |
| ROCr Runtime Patch | ❌ **CRASH** | Graphics corruption |
| Kernel Module Patch | ❌ **CRASH** | System unstable |
| **Environment Tuning** | ✅ **WORKS** | Safe, stable |

---

## 📚 Full Documentation

- **RECOVERY_INSTRUCTIONS.txt** - Quick reference card
- **FINAL_STATUS.md** - Complete summary
- **ENVIRONMENT_TUNING.md** - Detailed config guide  
- **CRASH_ANALYSIS.md** - Why patches failed
- **README.md** - Full 547-line technical guide

---

## ⚠️ Performance Warning

Environment tuning is **50-100% slower** than native RDNA2.

**Why?**
- Uses GEMM (matrix multiply) fallback
- Not optimized direct convolutions
- But it's **STABLE** (no crashes)

**Tradeoff**: Slow but working > Fast but crashing

---

## 💡 Long-term Solution

Consider upgrading to RDNA2+ GPU:

- **RX 6600 XT** ($250) - Budget, native support
- **RX 7600 XT** ($400) - Mid-range, RDNA3
- **RX 7900 XT** ($800) - High-end, best performance

All RDNA2+ GPUs work with ROCm without any workarounds.

---

## ✅ Success Checklist

After recovery:

- [ ] System boots normally
- [ ] Graphics work (no corruption)
- [ ] Conv2d test passes
- [ ] No crashes during use
- [ ] Can run PyTorch models

---

## 🔄 Daily Usage

**Every time before using PyTorch:**

```bash
source ~/rocm_rdna1_env.sh
python3 your_script.py
```

**Or add to .bashrc for automatic loading:**

```bash
echo 'source ~/rocm_rdna1_env.sh' >> ~/.bashrc
```

---

## 🎓 Key Lesson Learned

> Hardware capabilities are enforced by hardware, not software.
> Lying about capabilities = crash when those capabilities are used.
> The only solution is to work within hardware limitations.

---

## 📞 Need Help?

1. **Check logs**: `export MIOPEN_LOG_LEVEL=7`
2. **ROCm GitHub**: https://github.com/ROCm/ROCm/issues/2527
3. **AMD Forums**: https://community.amd.com/t5/rocm/bd-p/rocm

---

**Current Status**: 🔴 Needs recovery  
**Time to Fix**: 5 minutes  
**Next Step**: `sudo ./recovery_script.sh`

**Remember**: Environment tuning is the ONLY safe solution!

