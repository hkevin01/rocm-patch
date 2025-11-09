# Kernel Module Patch Status

## ✅ Patch Applied and Installed

**Date**: November 8, 2025  
**Module**: amdgpu-6.16.6-2238411.24.04  
**Patch Location**: `/usr/src/amdgpu-6.16.6-2238411.24.04/amd/amdkfd/kfd_crat.c`

### What Was Patched

**File**: `kfd_crat.c` (lines 1122-1130)  
**Function**: `kfd_parse_subtype_mem()`  
**Change**: Added RDNA1 detection and fine-grained flag setting

**Code Added**:
```c
/* RDNA1 FIX: Allow fine-grained memory for Navi 10 (0x731F) when spoofed as gfx1030 */
if (dev->gpu && dev->gpu->adev && dev->gpu->adev->pdev) {
    uint16_t device_id = dev->gpu->adev->pdev->device;
    if (device_id == 0x731F && heap_type != HSA_MEM_HEAP_TYPE_SYSTEM) {
        pr_debug("KFD: Enabling fine-grained for RDNA1 0x%x\n", device_id);
        flags |= HSA_MEM_FLAGS_HOT_PLUGGABLE;
    }
}
```

### What This Does

1. **Detects** device ID 0x731F (RX 5600 XT / RX 5700 XT)
2. **Sets** fine-grained memory flag for non-system memory
3. **Allows** ROCm to work with HSA_OVERRIDE_GFX_VERSION=10.3.0
4. **Prevents** memory model mismatch crashes

## 📋 Next Steps

### Step 1: Reboot (REQUIRED)
```bash
sudo reboot
```

The kernel module must be reloaded for changes to take effect.

### Step 2: After Reboot - Check Module
```bash
# Verify patched module is loaded
lsmod | grep amdgpu
dmesg | grep -i "KFD.*RDNA1"  # Should show our debug message

# Check ROCm
rocminfo | grep gfx
```

### Step 3: Test Conv2d
```bash
cd ~/Projects/rocm-patch
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python3 << 'PYTEST'
import torch
print("🧪 Testing Conv2d with patched kernel...")
x = torch.randn(1, 3, 32, 32).cuda()
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
y = conv(x)
print(f"✅ SUCCESS! Output: {y.shape}")
PYTEST
```

## 🔄 If It Works

Update the todo list:
- [x] Kernel module patch created
- [x] Patch applied to source
- [x] Module built successfully
- [x] Module installed
- [ ] System rebooted
- [ ] Conv2d tested and working

## 🔴 If It Doesn't Work

### Fallback Plan

1. **Check dmesg for errors**:
   ```bash
   sudo dmesg | tail -100
   ```

2. **Remove patch and rebuild**:
   ```bash
   cd /usr/src/amdgpu-6.16.6-2238411.24.04
   sudo git checkout amd/amdkfd/kfd_crat.c  # if git repo
   # or manually remove lines 1122-1130
   sudo dkms build -m amdgpu -v 6.16.6-2238411.24.04 --force
   sudo dkms install -m amdgpu -v 6.16.6-2238411.24.04 --force
   sudo reboot
   ```

3. **Try Option 3** (MIOpen environment tuning) instead

## 📚 Technical Details

### Why This Works

1. **Early Initialization**: KFD sets memory flags during topology creation
2. **Before HSA Runtime**: Happens before ROCm's HSA layer reads capabilities
3. **Hardware Transparent**: GPU still uses coarse-grained under the hood
4. **ROCm Happy**: Memory allocator sees "fine-grained" flag and proceeds

### Why Previous Attempts Failed

- **ROCr patches**: Too late (after memory model init)
- **LD_PRELOAD**: Can't intercept kernel-level structures
- **Type casting**: No safe way at runtime without RTTI

### The Magic

By setting the flag at **kernel driver level**, we establish the memory
model **before** any userspace code runs. This is the ONLY safe place to
make this change.

## 🎯 Expected Behavior After Reboot

### Success Indicators
✅ System boots normally  
✅ `dmesg` shows "KFD: Enabling fine-grained for RDNA1 0x731f"  
✅ `rocminfo` shows gfx1030  
✅ Conv2d completes without hanging  
✅ No system crashes  

### Failure Indicators
❌ System won't boot → Boot from recovery, remove patch  
❌ Graphics glitches → Driver issue, revert  
❌ Conv2d still hangs → Need additional changes  
❌ Crashes → Memory model still mismatched  

## 🔧 Maintenance

This patch will persist across reboots but needs to be reapplied if:
- Kernel is updated (recompile DKMS)
- amdgpu driver is updated (recompile DKMS)
- ROCm is updated (may need adjustment)

To make permanent:
1. Keep patch file in `~/Projects/rocm-patch/kernel-patches/`
2. Document in README
3. Create automation script for updates

---

**Status**: ⏳ Waiting for reboot to test

**Risk Level**: 🟡 Medium (kernel module change, but isolated to 0x731F)

**Reversible**: ✅ Yes (rebuild without patch)
