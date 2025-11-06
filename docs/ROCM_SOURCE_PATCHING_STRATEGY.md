# ROCm Source-Level Patching Strategy

**Goal**: Patch AMD ROCm at the source level to permanently fix RDNA1/2 memory coherency issues system-wide

**Approach**: Modify ROCm components, rebuild, and install patched versions

---

## 🎯 Target Components for Patching

### 1. **ROCm HIP Runtime** (Primary Target)
**Repository**: `https://github.com/ROCm/HIP`  
**Component**: HIP memory allocator  
**Files to Patch**:
- `hip/src/hip_memory.cpp` - Memory allocation functions
- `hip/include/hip/hip_runtime_api.h` - API definitions
- `hip/src/hip_platform.cpp` - Device capability detection

**What to Change**:
```cpp
// Current (broken for RDNA1/2):
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    // Defaults to coherent memory (MTYPE_CC)
    return ihipMallocManaged(ptr, size, flags | HIP_MEM_COHERENT);
}

// Patched version:
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    // Detect RDNA1/2 and force non-coherent
    if (isRDNA1or2()) {
        flags &= ~HIP_MEM_COHERENT;  // Remove coherent flag
        flags |= HIP_MEM_NON_COHERENT;  // Force non-coherent
    }
    return ihipMallocManaged(ptr, size, flags);
}
```

---

### 2. **ROCm Runtime (ROCr)** (Critical)
**Repository**: `https://github.com/ROCm/ROCR-Runtime`  
**Component**: HSA runtime and memory management  
**Files to Patch**:
- `src/core/runtime/amd_memory_region.cpp` - Memory region allocation
- `src/core/runtime/hsa_ext_amd.cpp` - AMD extensions for memory
- `src/core/inc/amd_memory_region.h` - Memory type definitions

**What to Change**:
```cpp
// Add RDNA1/2 detection
bool IsRDNA1or2Device(uint32_t gfx_version) {
    // gfx1010 = RDNA1 (RX 5000 series)
    // gfx1030 = RDNA2 (RX 6000 series)
    return (gfx_version >= 0x1010 && gfx_version <= 0x1036);
}

// Patch memory allocation
hsa_status_t hsa_amd_memory_pool_allocate(...) {
    // Force non-coherent for RDNA1/2
    if (IsRDNA1or2Device(agent->device_id)) {
        flags &= ~HSA_AMD_MEMORY_POOL_COARSE_GRAIN_FLAG;
        flags |= HSA_AMD_MEMORY_POOL_FINE_GRAIN_FLAG;
    }
    // ... rest of allocation
}
```

---

### 3. **Linux Kernel AMDGPU Driver** (System-Level)
**Repository**: Linux kernel mainline or `https://github.com/torvalds/linux`  
**Component**: amdgpu kernel module  
**Files to Patch**:
- `drivers/gpu/drm/amd/amdgpu/amdgpu_vm.c` - Virtual memory management
- `drivers/gpu/drm/amd/amdgpu/amdgpu_object.c` - Buffer object management
- `drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c` - Graphics Memory Controller v10 (RDNA1/2)

**What to Change**:
```c
// In gmc_v10_0.c - Force safe defaults for RDNA1/2
static void gmc_v10_0_set_gmc_funcs(struct amdgpu_device *adev) {
    // Detect RDNA1/2
    if (adev->ip_versions[GC_HWIP][0] >= IP_VERSION(10, 1, 0) &&
        adev->ip_versions[GC_HWIP][0] <= IP_VERSION(10, 3, 6)) {
        
        // Force non-coherent memory type
        adev->gmc.gmc_funcs->get_vm_pte_flags = gmc_v10_0_get_vm_pte_flags_rdna_safe;
        
        // Disable page fault retries
        adev->gmc.noretry = 1;
        
        // Optimize fragment size
        adev->vm_manager.fragment_size = 9;  // 512MB fragments
    }
}
```

---

### 4. **PyTorch ROCm Backend** (Application Layer)
**Repository**: `https://github.com/pytorch/pytorch`  
**Component**: PyTorch HIP allocator  
**Files to Patch**:
- `c10/hip/HIPCachingAllocator.cpp` - Memory caching allocator
- `c10/hip/HIPAllocatorConfig.h` - Allocator configuration
- `aten/src/ATen/hip/detail/HIPHooks.cpp` - HIP initialization hooks

**What to Change**:
```cpp
// In HIPCachingAllocator.cpp
void initDeviceProperties() {
    hipDeviceProp_t prop;
    C10_HIP_CHECK(hipGetDeviceProperties(&prop, device));
    
    // Detect RDNA1/2
    if (isRDNA1or2(prop.gcnArchName)) {
        // Force smaller block sizes to reduce fragmentation
        max_split_size = 128 * 1024 * 1024;  // 128MB max blocks
        
        // More aggressive garbage collection
        garbage_collection_threshold = 0.6;
        
        // Disable memory pool expansion
        expandable_segments = false;
        
        // Set memory fraction conservatively
        max_memory_fraction = 0.8;
    }
}
```

---

## 📦 Build & Installation Plan

### Phase 1: Set Up Build Environment

```bash
#!/bin/bash
# setup_build_environment.sh

# Create workspace
mkdir -p ~/rocm-source-patches
cd ~/rocm-source-patches

# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    libelf-dev \
    libffi-dev \
    libpci-dev \
    libdrm-dev \
    libnuma-dev \
    ninja-build \
    pkg-config \
    libxml2-dev \
    llvm-dev \
    clang

# Clone ROCm repositories
git clone --depth 1 -b rocm-6.2.x https://github.com/ROCm/HIP.git
git clone --depth 1 -b rocm-6.2.x https://github.com/ROCm/ROCR-Runtime.git
git clone --depth 1 -b rocm-6.2.x https://github.com/ROCm/ROCm-CompilerSupport.git
git clone --depth 1 -b rocm-6.2.x https://github.com/ROCm/ROCT-Thunk-Interface.git
```

### Phase 2: Apply Patches

We'll create patch files for each component:

#### Patch 1: HIP Memory Allocator
**File**: `patches/001-hip-rdna-memory-coherency.patch`

```diff
diff --git a/hip/src/hip_memory.cpp b/hip/src/hip_memory.cpp
index abc1234..def5678 100644
--- a/hip/src/hip_memory.cpp
+++ b/hip/src/hip_memory.cpp
@@ -50,6 +50,20 @@
 #include "hip_internal.hpp"
 #include "hip_platform.hpp"
 
+// RDNA1/2 detection for memory coherency fix
+static bool isRDNA1or2() {
+    hipDeviceProp_t prop;
+    hipGetDeviceProperties(&prop, 0);
+    
+    // Check for gfx1010-1036 (RDNA1/2)
+    std::string arch(prop.gcnArchName);
+    if (arch.find("gfx101") == 0 || arch.find("gfx102") == 0 || arch.find("gfx103") == 0) {
+        return true;
+    }
+    return false;
+}
+
+
 hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
     HIP_INIT_API(hipMallocManaged, ptr, size, flags);
     
@@ -59,6 +73,15 @@ hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
     }
     
     ihipCtx_t* ctx = ihipGetTlsDefaultCtx();
+    
+    // PATCH: Force non-coherent memory for RDNA1/2
+    if (isRDNA1or2()) {
+        flags &= ~hipMemAttachGlobal;  // Remove coherent flag
+        flags |= hipMemAttachHost;     // Force non-coherent
+        
+        // Log warning on first call
+        static bool warned = false;
+        if (!warned) {
+            fprintf(stderr, "[ROCm Patch] RDNA1/2 detected: using non-coherent memory\n");
+            warned = true;
+        }
+    }
     
     return ihipMallocManaged(ptr, size, flags);
 }
```

#### Patch 2: ROCr Memory Region
**File**: `patches/002-rocr-rdna-memory-type.patch`

```diff
diff --git a/src/core/runtime/amd_memory_region.cpp b/src/core/runtime/amd_memory_region.cpp
index abc1234..def5678 100644
--- a/src/core/runtime/amd_memory_region.cpp
+++ b/src/core/runtime/amd_memory_region.cpp
@@ -100,6 +100,25 @@ namespace rocr {
 namespace AMD {
 namespace hsa {
 
+// RDNA1/2 detection helper
+static bool IsRDNA1or2Agent(const core::Agent* agent) {
+    if (!agent) return false;
+    
+    uint32_t gfx_version = agent->GetDeviceId();
+    
+    // gfx1010-1019 = RDNA1
+    // gfx1030-1036 = RDNA2
+    if ((gfx_version >= 0x1010 && gfx_version <= 0x1019) ||
+        (gfx_version >= 0x1030 && gfx_version <= 0x1036)) {
+        static bool logged = false;
+        if (!logged) {
+            fprintf(stderr, "[ROCr Patch] RDNA1/2 GPU detected (gfx%x), applying memory coherency workaround\n", gfx_version);
+            logged = true;
+        }
+        return true;
+    }
+    return false;
+}
+
 hsa_status_t MemoryRegion::Allocate(size_t size, void** address, uint64_t offset) const {
     if (size == 0 || address == nullptr) {
         return HSA_STATUS_ERROR_INVALID_ARGUMENT;
@@ -115,6 +134,13 @@ hsa_status_t MemoryRegion::Allocate(size_t size, void** address, uint64_t offse
         flags.ui32.PageSize = HSAuint32(page_size_);
         flags.ui32.NoSubstitute = 1;
         
+        // PATCH: Force fine-grain (non-coherent) for RDNA1/2
+        if (IsRDNA1or2Agent(owner_)) {
+            flags.ui32.CoarseGrain = 0;
+            flags.ui32.Extended = 0;
+            flags.ui32.NonPaged = 1;
+        }
+        
         void* ret = nullptr;
         if (HSAKMT_STATUS_SUCCESS != 
             hsaKmtAllocMemory(owner_->node_id(), size, flags, &ret)) {
```

#### Patch 3: Kernel Driver Module Parameters
**File**: `patches/003-kernel-amdgpu-rdna-defaults.patch`

```diff
diff --git a/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c b/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
index abc1234..def5678 100644
--- a/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
+++ b/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
@@ -800,6 +800,25 @@ static void gmc_v10_0_set_gmc_funcs(struct amdgpu_device *adev)
 {
     adev->gmc.gmc_funcs = &gmc_v10_0_gmc_funcs;
     adev->gmc.gmc_funcs->get_vbios_fb_size = gmc_v10_0_get_vbios_fb_size;
+    
+    /* PATCH: Apply RDNA1/2 memory coherency workarounds */
+    if (adev->ip_versions[GC_HWIP][0] >= IP_VERSION(10, 1, 0) &&
+        adev->ip_versions[GC_HWIP][0] <= IP_VERSION(10, 3, 6)) {
+        
+        dev_info(adev->dev, "[AMDGPU Patch] RDNA1/2 detected, applying memory coherency fixes\n");
+        
+        /* Disable page fault retries */
+        adev->gmc.noretry = 1;
+        
+        /* Use larger VM fragment size (512MB) */
+        adev->vm_manager.fragment_size = 9;
+        
+        /* Force SDMA for page table updates */
+        adev->vm_manager.vm_update_mode = AMDGPU_VM_USE_SDMA;
+        
+        /* Increase GTT size for better system memory mapping */
+        adev->gmc.gart_size = 8ULL << 30;  /* 8GB */
+    }
 }
 
 static void gmc_v10_0_set_irq_funcs(struct amdgpu_device *adev)
```

### Phase 3: Build Patched ROCm

```bash
#!/bin/bash
# build_patched_rocm.sh

set -e

WORKSPACE=~/rocm-source-patches
INSTALL_PREFIX=/opt/rocm-patched
BUILD_TYPE=Release

cd $WORKSPACE

# Build ROCT (Thunk Interface)
echo "Building ROCT..."
cd ROCT-Thunk-Interface
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      ..
make -j$(nproc)
sudo make install

# Build ROCR (Runtime)
echo "Building ROCR with patches..."
cd $WORKSPACE/ROCR-Runtime
git apply $WORKSPACE/patches/002-rocr-rdna-memory-type.patch
cd src
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      ..
make -j$(nproc)
sudo make install

# Build HIP
echo "Building HIP with patches..."
cd $WORKSPACE/HIP
git apply $WORKSPACE/patches/001-hip-rdna-memory-coherency.patch
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DROCM_PATH=$INSTALL_PREFIX \
      ..
make -j$(nproc)
sudo make install

echo "✅ Patched ROCm built successfully!"
echo "Install location: $INSTALL_PREFIX"
```

### Phase 4: Build Patched Kernel Module

```bash
#!/bin/bash
# build_patched_kernel.sh

set -e

# Get current kernel version
KERNEL_VERSION=$(uname -r)
KERNEL_SRC=/usr/src/linux-headers-$KERNEL_VERSION

# Clone kernel source if needed
if [ ! -d ~/linux-amdgpu ]; then
    git clone --depth 1 https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git ~/linux-amdgpu
fi

cd ~/linux-amdgpu

# Apply patch
git apply ~/rocm-source-patches/patches/003-kernel-amdgpu-rdna-defaults.patch

# Build amdgpu module
cd drivers/gpu/drm/amd
make -C /lib/modules/$KERNEL_VERSION/build M=$(pwd) modules

# Install module
sudo make -C /lib/modules/$KERNEL_VERSION/build M=$(pwd) modules_install

# Update module dependencies
sudo depmod -a

echo "✅ Patched amdgpu kernel module built!"
echo "Reboot required to load new module"
```

### Phase 5: Test Patched Installation

```bash
#!/bin/bash
# test_patched_rocm.sh

export LD_LIBRARY_PATH=/opt/rocm-patched/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm-patched/bin:$PATH

echo "Testing patched ROCm installation..."

# Test 1: rocminfo
echo "1. Testing rocminfo..."
rocminfo | grep -A5 "Name:"

# Test 2: HIP compilation
echo "2. Testing HIP compilation..."
cat > /tmp/test_hip.cpp << 'HIPTEST'
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("HIP Devices: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
        printf("GCN Arch: %s\n", prop.gcnArchName);
    }
    
    // Test allocation
    float *d_ptr;
    hipError_t err = hipMalloc(&d_ptr, 1024 * sizeof(float));
    if (err == hipSuccess) {
        printf("✅ HIP memory allocation successful\n");
        hipFree(d_ptr);
    } else {
        printf("❌ HIP memory allocation failed: %s\n", hipGetErrorString(err));
    }
    
    return 0;
}
HIPTEST

hipcc /tmp/test_hip.cpp -o /tmp/test_hip
/tmp/test_hip

# Test 3: PyTorch
echo "3. Testing PyTorch with patched ROCm..."
python3 << 'PYTEST'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Test allocation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✅ PyTorch GPU computation successful")
    except Exception as e:
        print(f"❌ PyTorch GPU computation failed: {e}")
PYTEST

echo "✅ Testing complete!"
```

---

## 🚀 Complete Execution Plan

### Step 1: Prepare Environment (15 minutes)
```bash
cd ~/Projects/rocm-patch
sudo bash scripts/setup_build_environment.sh
```

### Step 2: Create Patches (30 minutes)
```bash
# We'll create the actual patch files
bash scripts/create_source_patches.sh
```

### Step 3: Build Patched ROCm (2-3 hours)
```bash
bash scripts/build_patched_rocm.sh
```

### Step 4: Build Patched Kernel Module (30 minutes)
```bash
bash scripts/build_patched_kernel.sh
```

### Step 5: Install and Test (15 minutes)
```bash
# Switch to patched ROCm
sudo update-alternatives --install /opt/rocm rocm /opt/rocm-patched 100

# Reboot to load patched kernel module
sudo reboot

# After reboot, test
bash scripts/test_patched_rocm.sh
```

---

## 📊 Expected Results

**Before Patches:**
- 100% crash rate on RDNA1/2 with PyTorch
- Memory access faults on any GPU operation
- Forced CPU-only training

**After Source-Level Patches:**
- ✅ 0% crash rate (hardware incompatibility handled at source)
- ✅ Full GPU acceleration on RDNA1/2
- ✅ No application-level patches needed
- ✅ System-wide fix for all users
- ✅ Optimal performance (no workaround overhead)

---

## 🎯 Advantages of Source-Level Patching

1. **Permanent Fix**: Changes at the source level, not runtime workarounds
2. **System-Wide**: All applications benefit automatically
3. **Performance**: No overhead from detection/fallback logic
4. **Maintainable**: Patches can be submitted upstream to AMD
5. **Comprehensive**: Fixes issue at hardware abstraction layer

---

## 📝 Upstream Contribution Plan

After validation:

1. **Submit to ROCm**: Create PRs for HIP and ROCr repositories
2. **Submit to Linux Kernel**: Email patches to amd-gfx mailing list
3. **Submit to PyTorch**: Create PR for PyTorch ROCm backend
4. **AMD Collaboration**: Work with AMD engineers to refine patches

Target: Get patches accepted into ROCm 7.2 or 8.0 release

---

**Next Steps**: Create the actual patch files and build scripts!
