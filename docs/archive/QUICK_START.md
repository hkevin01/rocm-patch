# 🚀 Quick Start - Recovery & Fix

## ⚠️ Current Situation
Your system has a **crashing kernel patch** installed!

---

## 🔧 Recovery (Do This NOW)

### Step 1: Remove Patch (5 min)
```bash
sudo ./recovery_script.sh
```

### Step 2: Reboot (1 min)
```bash
sudo reboot
```

---

## ✅ After Reboot - Setup Solution (5 min)

### Step 3: Create Environment File
```bash
cat > ~/rocm_rdna1_env.sh << 'ENVEOF'
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_LOG_LEVEL=4
echo "✅ ROCm RDNA1 environment configured"
ENVEOF

chmod +x ~/rocm_rdna1_env.sh
```

### Step 4: Test It
```bash
source ~/rocm_rdna1_env.sh

python3 -c "
import torch
print('Testing Conv2d...')
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
print(f'✅ SUCCESS! Shape: {y.shape}')
"
```

---

## 📚 Full Documentation

- **CRASH_ANALYSIS.md** - Why patches failed
- **ENVIRONMENT_TUNING.md** - Complete guide
- **COMPLETE_TODO_LIST.md** - Full project status
- **README.md** - 547-line comprehensive guide

---

## 🎯 Expected Result

✅ Conv2d works (may be 2x slower, but **STABLE**)  
✅ No crashes  
✅ No system instability

---

**Start NOW**: `sudo ./recovery_script.sh`

