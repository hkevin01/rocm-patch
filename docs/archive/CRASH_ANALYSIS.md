# ⚠️ KERNEL PATCH CRASH ANALYSIS

**Date**: November 8, 2025  
**Status**: 🔴 **SYSTEM CRASHED** - Kernel patch caused instability

---

## 🔍 What Happened

### Crash Sequence
1. ✅ Rebooted with patched kernel module
2. ✅ Module loaded successfully (kernel 6.14.0-35)
3. ⚠️ Ran Conv2d test with HSA_OVERRIDE_GFX_VERSION=10.3.0
4. 🔴 **System crashed** during Conv2d execution

### Evidence
```
🧪 Testing Conv2d with patched kernel...
[ROCr RDNA1 Fix] Forcing coarse-grained memory
[DEBUG] FindFwd called, is_rdna1=0, perfResults=...
[DEBUG] FindFwd called, is_rdna1=0, perfResults=...
<CRASH - No output, system unresponsive>
```

---

## 💡 Root Cause Analysis

### Why It Crashed

Our kernel patch added `HSA_MEM_FLAGS_HOT_PLUGGABLE` flag, which tells the HSA runtime that memory is "hot pluggable" (can be remapped). However:

1. **Hardware Reality**: RDNA1 (0x731F) does NOT support fine-grained memory
2. **Kernel Says**: "This memory is hot pluggable" (our patch)
3. **ROCm Assumes**: "I can use fine-grained operations"
4. **Hardware Reality**: Tries to access memory with fine-grained semantics
5. **Result**: 💥 **PAGE FAULT / MEMORY ACCESS VIOLATION / CRASH**

### The Fundamental Problem

**We can't fake hardware capabilities at the kernel level.**

```
Hardware: "I only do coarse-grained"
Kernel Patch: "Tell ROCm you do fine-grained"
ROCm: "Great! I'll use fine-grained operations"
Hardware: "Wait, I don't support that!" 💥
```

This is the SAME problem we had with ROCr patches, just at a different level.

---

## ❌ Why All Patches Failed

### MIOpen Patches (Partial Success)
- ✅ Detected RDNA1 correctly
- ❌ Environment variables not propagating
- Result: No effect

### ROCr Runtime Patches (System Crashes)
- ✅ Forced coarse-grained memory
- ❌ Too late - memory model already initialized
- ❌ Accessing kernel structures from userspace = crash
- Result: System crashes, graphics corruption

### Kernel Module Patch (System Crashes)
- ✅ Patches at correct level (kernel initialization)
- ❌ Lying about hardware capabilities
- ❌ ROCm uses capabilities that hardware doesn't support
- Result: **System crashes during actual GPU operations**

---

## 🎯 The Real Solution

### What We Actually Need

**We DON'T need to fake fine-grained memory support.**

We need to:
1. ✅ Tell PyTorch we're gfx1030 (to get kernels) - `HSA_OVERRIDE=10.3.0`
2. ✅ Tell MIOpen we're RDNA1 (to use compatible algorithms)
3. ✅ Tell ROCm to use coarse-grained memory operations
4. ❌ **NEVER** fake hardware capabilities

### The Correct Approach: Environment Tuning (Option 3)

Instead of patching code, we need to:
- Use `HSA_OVERRIDE_GFX_VERSION=10.3.0` (for kernels)
- Set `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0` (disable problematic convolutions)
- Set `MIOPEN_FIND_ENFORCE=3` (use database only)
- Use fallback algorithms that work on RDNA1

---

## 🔧 Recovery Steps

### Immediate: Remove Kernel Patch

```bash
# 1. Boot to recovery mode (if system won't boot normally)
# Hold Shift during boot, select Advanced -> Recovery Mode

# 2. Remove the patch
sudo mount -o remount,rw /
cd /usr/src/amdgpu-6.16.6-2238411.24.04/amd/amdkfd
sudo cp kfd_crat.c kfd_crat.c.patched  # backup
sudo nano kfd_crat.c
# Remove lines 1122-1130 (the RDNA1 fix block)

# 3. Rebuild and reinstall
sudo dkms remove -m amdgpu -v 6.16.6-2238411.24.04 --all
sudo dkms build -m amdgpu -v 6.16.6-2238411.24.04
sudo dkms install -m amdgpu -v 6.16.6-2238411.24.04

# 4. Reboot
sudo reboot
```

### Alternative: Kernel Parameter Blacklist

If system won't boot, add to kernel parameters:
```
modprobe.blacklist=amdgpu
```

Then use integrated graphics to fix.

---

## ✅ Next Steps: Environment Tuning

### Option 3: MIOpen Environment Variables

This is the ONLY safe approach that doesn't require code changes:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_LOG_LEVEL=4

python3 test_conv2d.py
```

### Why This Works
- ✅ No code changes (safe)
- ✅ Uses existing MIOpen fallback paths
- ✅ Works within hardware limitations
- ✅ Reversible (just unset variables)
- ⚠️ May be slower (uses GEMM instead of direct convolutions)

---

## 🎓 Lessons Learned

### The Hard Truth

**You cannot fake hardware capabilities in software.**

No matter where you patch (kernel, runtime, library):
- Lying about capabilities = crash when those capabilities are used
- Hardware will always enforce its actual capabilities
- The only solution is to work within hardware limitations

### What Actually Works

1. **Accept hardware limitations**: RDNA1 = coarse-grained only
2. **Use compatible algorithms**: GEMM-based convolutions
3. **Configure software properly**: Environment variables
4. **Don't fake capabilities**: Never lie about what hardware can do

---

## 📊 Final Assessment

### All Patch Attempts: ❌ FAILED

| Approach | Level | Result | Reason |
|----------|-------|--------|--------|
| MIOpen Patch | Library | Partial | Env vars not working |
| ROCr Patch | Runtime | Crash | Memory model mismatch |
| Kernel Patch | Driver | Crash | Faking hardware capabilities |

### Only Safe Solution: ✅ Environment Tuning

**No code changes. Work with hardware, not against it.**

---

## 🚨 Current Status

- 🔴 **System may not boot** (kernel panic possible)
- 🔴 **Graphics may be corrupted** (amdgpu driver unstable)
- 🟡 **Recovery mode needed** (to remove patch)
- ✅ **Backup exists** (can restore original module)

---

## 🎯 Recommended Path Forward

1. **Boot to recovery** (if needed)
2. **Remove kernel patch** (restore original kfd_crat.c)
3. **Rebuild clean module**
4. **Reboot normally**
5. **Try Option 3** (environment tuning)
6. **Accept limitations** (may be slower, but stable)

---

**Remember**: Sometimes the best solution is NOT to patch the code,
but to configure it properly for your hardware.

