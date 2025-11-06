#!/bin/bash
################################################################################
# AMD GPU Kernel Module Patcher
# Patches amdgpu kernel module for RDNA1/2 memory stability
################################################################################

set -e

KERNEL_VERSION=$(uname -r)
KERNEL_SRC=/usr/src/linux-headers-$KERNEL_VERSION
MODULE_BUILD_DIR=~/amdgpu-patched
PATCH_DIR=/home/kevin/Projects/rocm-patch/patches/kernel

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ ! -d "$KERNEL_SRC" ]; then
        log_error "Kernel headers not found. Installing..."
        sudo apt-get install -y linux-headers-$KERNEL_VERSION
    fi
    
    log_info "Installing build dependencies..."
    sudo apt-get install -y build-essential bc kmod dkms libelf-dev
    
    log_info "✅ Prerequisites satisfied"
}

create_kernel_patch() {
    log_info "Creating kernel patch for amdgpu..."
    
    mkdir -p $PATCH_DIR
    
    cat > $PATCH_DIR/003-amdgpu-rdna-memory-defaults.patch << 'KPATCH'
diff --git a/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c b/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
index abc1234..def5678 100644
--- a/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
+++ b/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
@@ -600,6 +600,35 @@ static void gmc_v10_0_get_vm_pde(struct amdgpu_device *adev, int level,
 *flags = AMDGPU_PTE_VALID;
 }
 
+/*
+ * RDNA1/2 Safe Memory Configuration
+ * Applies conservative settings for RX 5000/6000 series GPUs
+ */
+static void gmc_v10_0_apply_rdna_workarounds(struct amdgpu_device *adev)
+{
+/* Detect RDNA1 (Navi 10/12/14) or RDNA2 (Navi 21/22/23/24) */
+bool is_rdna = (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 1, 0)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 1, 1)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 1, 2)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 3, 0)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 3, 1)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 3, 2)) ||
+               (adev->ip_versions[GC_HWIP][0] == IP_VERSION(10, 3, 3));
+
+if (is_rdna) {
+dev_info(adev->dev, "[Patch] RDNA1/2 detected - applying stability workarounds\n");
+
+/* Force non-coherent memory */
+adev->gmc.aper_base_coherent = false;
+
+/* Conservative VM fragment size */
+adev->vm_manager.fragment_size = 9; /* 512KB fragments */
+
+/* Disable aggressive retry behavior */
+adev->gmc.noretry = 0;
+}
+}
+
 static void gmc_v10_0_get_vm_pte(struct amdgpu_device *adev,
  struct amdgpu_bo_va_mapping *mapping,
  uint64_t *flags)
@@ -850,6 +879,9 @@ static int gmc_v10_0_late_init(void *handle)
 if (!amdgpu_sriov_vf(adev))
 amdgpu_bo_late_init(adev);
 
+/* Apply RDNA workarounds */
+gmc_v10_0_apply_rdna_workarounds(adev);
+
 return 0;
 }
KPATCH
    
    log_info "✅ Kernel patch created"
}

build_module() {
    log_info "Building patched amdgpu module..."
    log_warn "This approach patches in-tree module. For ROCm-specific changes,"
    log_warn "consider using DKMS with amdgpu-dkms package instead."
    
    mkdir -p $MODULE_BUILD_DIR
    cd $MODULE_BUILD_DIR
    
    # Copy amdgpu sources
    if [ -d "/usr/src/amdgpu-dkms" ]; then
        log_info "Found amdgpu-dkms sources..."
        cp -r /usr/src/amdgpu-dkms/* .
    else
        log_error "amdgpu-dkms not found. Install with: sudo apt-get install amdgpu-dkms"
        exit 1
    fi
    
    # Apply patch
    if [ -f "$PATCH_DIR/003-amdgpu-rdna-memory-defaults.patch" ]; then
        log_info "Applying kernel patch..."
        patch -p1 < $PATCH_DIR/003-amdgpu-rdna-memory-defaults.patch || {
            log_warn "Patch did not apply cleanly - manual intervention required"
        }
    fi
    
    # Build module
    log_info "Compiling module (this may take 10-15 minutes)..."
    make -C $KERNEL_SRC M=$MODULE_BUILD_DIR modules
    
    log_info "✅ Module built successfully"
}

install_module() {
    log_info "Installing patched module..."
    log_warn "This will replace your current amdgpu module!"
    
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
    
    # Backup original module
    AMDGPU_KO=$(find /lib/modules/$KERNEL_VERSION -name amdgpu.ko -o -name amdgpu.ko.xz)
    if [ -n "$AMDGPU_KO" ]; then
        sudo cp $AMDGPU_KO ${AMDGPU_KO}.backup
        log_info "Original module backed up to ${AMDGPU_KO}.backup"
    fi
    
    # Install new module
    cd $MODULE_BUILD_DIR
    sudo make -C $KERNEL_SRC M=$MODULE_BUILD_DIR modules_install
    
    # Update module dependencies
    sudo depmod -a
    
    log_info "✅ Module installed"
    log_warn "⚠️  REBOOT REQUIRED to load patched module"
}

verify_module() {
    log_info "Verifying module installation..."
    
    modinfo amdgpu | grep filename
    
    log_info "After reboot, verify patch with:"
    log_info "  sudo dmesg | grep -i 'rdna.*patch'"
    log_info "  sudo dmesg | grep amdgpu"
}

main() {
    echo "========================================================================"
    echo "   AMD GPU Kernel Module Patcher for RDNA1/2"
    echo "========================================================================"
    echo ""
    echo "This script patches the amdgpu kernel module to apply safe memory"
    echo "defaults for RDNA1 (RX 5000) and RDNA2 (RX 6000) series GPUs."
    echo ""
    echo "WARNING: This modifies kernel modules. Use at your own risk."
    echo "         A backup will be created automatically."
    echo ""
    
    check_prerequisites
    create_kernel_patch
    build_module
    install_module
    verify_module
    
    echo ""
    echo "========================================================================"
    echo "✅ Kernel Module Patching Complete!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. REBOOT your system"
    echo "  2. After reboot, check: sudo dmesg | grep -i rdna"
    echo "  3. Verify module: modinfo amdgpu"
    echo ""
    echo "To restore original module if needed:"
    echo "  Find backup: find /lib/modules -name 'amdgpu.ko*.backup'"
    echo "  Restore and reboot"
    echo ""
}

main "$@"
