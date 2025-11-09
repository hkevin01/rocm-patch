# Installing ROCm 5.2 on Ubuntu 24.04 with Compatibility Layer

**For**: AMD RDNA1 GPUs (RX 5600 XT, RX 5700 XT, RX 5700)  
**Date**: November 9, 2025

---

## Overview

ROCm 5.2 was built for Ubuntu 22.04 and requires libraries not present in Ubuntu 24.04. This guide installs a compatibility layer to bridge the gap WITHOUT downgrading your OS.

---

## What This Does

### Compatibility Layer (`install_rocm52_compat_libs.sh`)
- Adds Ubuntu 22.04 (Jammy) package sources **with pinning**
- Installs only: `libtinfo5`, `libncurses5` from Ubuntu 22.04
- Creates `python` → `python3` symlink
- **Does NOT** downgrade any Ubuntu 24.04 packages
- Safe: apt pinning prevents system package conflicts

### ROCm Installation (`install_rocm52_with_compat.sh`)
- Removes existing ROCm (if any)
- Installs ROCm 5.2 from AMD repository
- Configures environment for RDNA1 (gfx1010)
- Installs PyTorch 2.2.2+rocm5.2
- Creates `/etc/profile.d/rocm-rdna1-52.sh`

---

## Prerequisites

- Ubuntu 24.04.x LTS
- AMD RDNA1 GPU (gfx1010):
  - RX 5600 XT (Device ID: 1002:731f) ✅
  - RX 5700 XT (Device ID: 1002:731e)
  - RX 5700 (Device ID: 1002:7310)
- Sudo access
- Internet connection

---

## Installation Steps

### Step 1: Install Compatibility Libraries

```bash
cd /home/kevin/Projects/rocm-patch
sudo ./install_rocm52_compat_libs.sh
```

**This will:**
1. Add Ubuntu 22.04 package sources (pinned for safety)
2. Install `libtinfo5` and `libncurses5`
3. Create `python` symlink
4. Verify installation

**Time**: 2-3 minutes

### Step 2: Install ROCm 5.2

```bash
sudo ./install_rocm52_with_compat.sh
```

**This will:**
1. Check for compatibility libraries
2. Remove existing ROCm
3. Install ROCm 5.2 packages
4. Configure environment for RDNA1
5. Install PyTorch 2.2.2+rocm5.2
6. Prompt for reboot

**Time**: 10-15 minutes

### Step 3: Reboot

```bash
sudo reboot
```

**Required**: ROCm kernel modules need to load.

### Step 4: Verify Installation

After reboot:

```bash
# Load environment
source /etc/profile.d/rocm-rdna1-52.sh

# Check ROCm
rocm-smi
rocminfo | grep "Name:" | head -5

# Check PyTorch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Run tests
python3 test_conv2d_subprocess.py
```

**Expected output:**
```
PyTorch: 2.2.2+rocm5.2
CUDA available: True
Device: AMD Radeon RX 5600 XT
```

---

## What Gets Changed

### System Changes

| Item | Change | Reversible? |
|------|--------|-------------|
| `/etc/apt/sources.list.d/jammy-compat.list` | Added (Ubuntu 22.04 sources) | Yes (remove file) |
| `/etc/apt/preferences.d/jammy-compat` | Added (apt pinning) | Yes (remove file) |
| `libtinfo5`, `libncurses5` | Installed from Ubuntu 22.04 | Yes (`apt remove`) |
| `/usr/bin/python` | Symlink to python3 | Yes (`rm /usr/bin/python`) |
| `/etc/apt/sources.list.d/rocm.list` | Added (ROCm 5.2 repo) | Yes (remove file) |
| ROCm 5.2 packages | Installed to `/opt/rocm-5.2.0` | Yes (`apt remove rocm*`) |
| `/etc/profile.d/rocm-rdna1-52.sh` | Created | Yes (remove file) |
| PyTorch | Updated to 2.2.2+rocm5.2 | Yes (reinstall other version) |

**Backups created:**
- `/etc/apt/sources.list.backup`
- `~/rocm-backups/`

---

## Safety Features

### Apt Pinning
The compatibility layer uses apt pinning to ensure:
- Ubuntu 24.04 packages have priority 990 (highest)
- Only specific compat libs from Ubuntu 22.04 (priority 500)
- All other Ubuntu 22.04 packages blocked (priority 100)

**Result**: Your Ubuntu 24.04 system remains intact!

### Verification Checks
- GPU detection (fails if no RDNA1)
- Compatibility lib check (fails if not installed)
- User confirmation at each step
- Installation logs saved

---

## Rollback / Uninstall

### Remove ROCm 5.2 Only
```bash
sudo apt remove --purge rocm* hip* miopen* rocblas* -y
sudo rm /etc/profile.d/rocm-rdna1-52.sh
sudo rm /etc/apt/sources.list.d/rocm.list
```

### Remove Compatibility Layer
```bash
sudo apt remove libtinfo5 libncurses5 -y
sudo rm /etc/apt/sources.list.d/jammy-compat.list
sudo rm /etc/apt/preferences.d/jammy-compat
sudo rm /usr/bin/python  # if you want
```

### Restore from Backup
```bash
sudo cp /etc/apt/sources.list.backup /etc/apt/sources.list
cp -r ~/rocm-backups/* /etc/profile.d/  # if needed
```

---

## Troubleshooting

### Issue: "libtinfo5 not found" after compatibility script
**Solution**: The apt update failed. Check internet connection and retry:
```bash
sudo apt update
sudo apt install libtinfo5 libncurses5
```

### Issue: ROCm install fails with dependency errors
**Solution**: Run compatibility script first:
```bash
sudo ./install_rocm52_compat_libs.sh
```

### Issue: PyTorch can't find GPU after reboot
**Solution**: Load environment:
```bash
source /etc/profile.d/rocm-rdna1-52.sh
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Conv2d still hangs on large feature maps
**Expected**: ROCm 5.2 may improve the boundary but likely still has limits.
- Test with: `python3 test_size_boundary.py`
- Expected: May work up to 48x48 or higher (needs testing)
- If still hangs at 44x44: This is a known RDNA1 limitation

---

## Known Limitations

Even with ROCm 5.2, RDNA1 (gfx1010) has known issues:
1. **Conv2d size limits**: May still have boundary (needs testing after install)
2. **Not officially supported**: AMD doesn't officially support gfx1010 in newer ROCm
3. **Community solution**: Based on community reports, not AMD official

**After installation, test the boundary:**
```bash
python3 test_size_boundary.py
```

---

## Expected Improvements

Based on community reports, ROCm 5.2 may:
- ✅ Extend working size beyond 42x42 (possibly to 64x64)
- ✅ Better Tensile/rocBLAS compatibility for gfx1010
- ✅ More stable for non-power-of-2 configurations
- ⚠️  **Not guaranteed** - needs testing on your hardware

---

## Testing After Installation

Run comprehensive tests:
```bash
# Quick test (20s per config)
python3 test_conv2d_subprocess.py

# Boundary test (find exact limits)
python3 test_size_boundary.py

# Timing test (performance)
python3 test_conv2d_timing.py
```

Compare results with `FINAL_FINDINGS.md` to see improvements.

---

## Next Steps

1. **Install compatibility layer** → `sudo ./install_rocm52_compat_libs.sh`
2. **Install ROCm 5.2** → `sudo ./install_rocm52_with_compat.sh`
3. **Reboot** → `sudo reboot`
4. **Test** → `python3 test_conv2d_subprocess.py`
5. **Document results** → Update findings if boundaries improved

---

## Support

If ROCm 5.2 still doesn't resolve the issue:
- See `FINAL_FINDINGS.md` for alternative solutions
- Consider hardware upgrade to RDNA2/RDNA3
- Use workarounds (≤42x42 restriction, CPU fallback)

Good luck! 🚀
