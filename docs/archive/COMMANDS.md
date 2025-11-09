# Command Reference for RDNA1 MIOpen Patching

This document lists all commands used in the project for easy reference and reproduction.

## Table of Contents
1. [ROCm Installation](#rocm-installation)
2. [Source Preparation](#source-preparation)
3. [Applying Patches](#applying-patches)
4. [Building MIOpen](#building-miopen)
5. [Deployment](#deployment)
6. [Testing & Verification](#testing--verification)
7. [Environment Variables](#environment-variables)

---

## ROCm Installation

### Automated Script
```bash
cd ~/Projects/rocm-patch/scripts
./install_rocm_6.2.4.sh
```

### Manual Installation
```bash
# Remove existing ROCm
sudo apt-get remove --purge -y 'rocm-*' 'hip-*' 'miopen-*'
sudo apt-get autoremove -y

# Add repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 noble main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm 6.2.4
sudo apt-get update
sudo apt-get install -y rocm-hip-sdk rocm-dev miopen-hip miopen-hip-dev

# Add user to groups
sudo usermod -a -G render,video $USER

# Verify installation
/opt/rocm/bin/hipcc --version
/opt/rocm/bin/rocminfo | grep "Name:" | head -1
```

---

## Source Preparation

### Clone MIOpen
```bash
cd /tmp
git clone https://github.com/ROCm/MIOpen.git
cd MIOpen
git checkout rocm-6.2.4  # Or appropriate version tag

# Create build directory
mkdir -p build_rdna1
```

---

## Applying Patches

### Patch 1: handlehip.cpp (Memory Allocation)

```bash
cat > /tmp/miopen_handlehip.patch << 'PATCH_EOF'
--- a/src/hip/handlehip.cpp
+++ b/src/hip/handlehip.cpp
@@ -103,6 +103,39 @@ void HandleImpl::Init()
         }
     }
 
+    // Detect RDNA1 architecture
+    bool is_rdna1 = false;
+    hipDeviceProp_t props;
+    if(hipGetDeviceProperties(&props, device_id) == hipSuccess)
+    {
+        std::string gcn_arch = props.gcnArchName;
+        if(gcn_arch.find("gfx1010") != std::string::npos ||
+           gcn_arch.find("gfx1011") != std::string::npos ||
+           gcn_arch.find("gfx1012") != std::string::npos)
+        {
+            is_rdna1 = true;
+            fprintf(stderr, "[RDNA1] Detected architecture: %s\n", gcn_arch.c_str());
+        }
+    }
+
+    // Use non-coherent memory for RDNA1
+    unsigned int flags = hipHostMallocDefault;
+    if(is_rdna1)
+    {
+        flags = hipHostMallocNonCoherent | hipHostMallocMapped;
+        fprintf(stderr, "[RDNA1] Using non-coherent memory flags\n");
+    }
+
+    // Try standard allocation first
+    void* test_ptr = nullptr;
+    hipError_t err = hipHostMalloc(&test_ptr, size, flags);
+    if(err == hipSuccess && test_ptr != nullptr)
+    {
+        hipHostFree(test_ptr);
+    }
+    else if(is_rdna1)
+    {
+        // Fallback for RDNA1
+        fprintf(stderr, "[RDNA1] Standard allocation failed, trying hipExtMallocWithFlags\n");
+        hipExtMallocWithFlags(&test_ptr, size, hipDeviceMallocUncached);
+        if(test_ptr != nullptr) hipFree(test_ptr);
+    }
 }
PATCH_EOF

cd /tmp/MIOpen
patch -p1 < /tmp/miopen_handlehip.patch
```

### Patch 2: convolution_api.cpp (Skip Find Mode)

```bash
cat > /tmp/miopen_convolution_api.patch << 'PATCH_EOF'
--- a/src/convolution_api.cpp
+++ b/src/convolution_api.cpp
@@ -47,6 +47,24 @@
 
 extern "C" miopenStatus_t miopenFindConvolutionForwardAlgorithm(...)
 {
+    // RDNA1 detection helper
+    static bool is_rdna1_cached = false;
+    static bool is_rdna1_value = false;
+    auto is_gpu_rdna1 = []() {
+        if(is_rdna1_cached) return is_rdna1_value;
+        
+        const char* force_rdna1 = getenv("MIOPEN_FORCE_RDNA1");
+        if(force_rdna1 && atoi(force_rdna1) == 1) {
+            is_rdna1_value = true;
+            is_rdna1_cached = true;
+            return true;
+        }
+        
+        // Additional detection logic here
+        is_rdna1_cached = true;
+        return is_rdna1_value;
+    };
+
     fprintf(stderr, "[DEBUG] FindFwd called, is_rdna1=%d\n", is_gpu_rdna1() ? 1 : 0);
     
     if(is_gpu_rdna1())
@@ -580,6 +598,17 @@ extern "C" miopenStatus_t miopenFindConvolutionBackwardDataAlgorithm(...)
 {
+    if(is_gpu_rdna1())
+    {
+        fprintf(stderr, "[RDNA1 PATCH] Skipping backward data Find\n");
+        *returnedAlgoCount = 1;
+        perfResults[0].bwd_data_algo = miopenConvolutionBwdDataAlgoDirect;
+        perfResults[0].time = 0.0f;
+        perfResults[0].memory = 0;
+        return miopenStatusSuccess;
+    }
+    
     // Original implementation...
 }
PATCH_EOF

cd /tmp/MIOpen
patch -p1 < /tmp/miopen_convolution_api.patch
```

### Quick Patch Application (Edit Files Directly)

See `README.md` sections 2.2 and 2.3 for exact line numbers and code to add.

---

## Building MIOpen

### Automated Build Script
```bash
cd ~/Projects/rocm-patch/scripts
./rebuild_miopen.sh
```

### Manual Build
```bash
cd /tmp/MIOpen/build_rdna1

# Configure
cmake \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm-miopen-rdna1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DMIOPEN_BACKEND=HIP \
  -DMIOPEN_USE_MLIR=OFF \
  -DMIOPEN_USE_HIPBLASLT=OFF \
  -DMIOPEN_ENABLE_AI_KERNEL_TUNING=OFF \
  -DMIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK=OFF \
  -DMIOPEN_USE_COMPOSABLEKERNEL=OFF \
  -DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__" \
  ..

# Build (30-45 minutes)
make -j$(nproc)

# Install
sudo make install
```

---

## Deployment

### Replace PyTorch's MIOpen Library
```bash
# Backup original
TORCH_LIB="$HOME/.local/lib/python3.12/site-packages/torch/lib"
cp "$TORCH_LIB/libMIOpen.so" "$TORCH_LIB/libMIOpen.so.original"

# Deploy patched version
cp /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0 "$TORCH_LIB/libMIOpen.so"

# Verify
ls -lh "$TORCH_LIB/libMIOpen.so"
md5sum "$TORCH_LIB/libMIOpen.so" /opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0
```

### Restore Original Library
```bash
TORCH_LIB="$HOME/.local/lib/python3.12/site-packages/torch/lib"
cp "$TORCH_LIB/libMIOpen.so.original" "$TORCH_LIB/libMIOpen.so"
```

---

## Testing & Verification

### Automated Test Script
```bash
cd ~/Projects/rocm-patch/scripts
./test_rdna1_patches.sh
```

### Manual Verification

#### Check for Patches in Library
```bash
strings ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so | grep "RDNA1 PATCH"
```

#### Library Dependencies
```bash
ldd ~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so | grep -E "rocm|hip"
```

#### Python Runtime Test
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FORCE_RDNA1=1
export MIOPEN_LOG_LEVEL=7

python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1, 3, 224, 224).cuda()
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    print(f"Output: {y.shape}")
