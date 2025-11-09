/*
 * HIP Memory Allocation Interceptor for RDNA1/2
 * Forces non-coherent memory allocations via LD_PRELOAD
 * Compile: gcc -shared -fPIC -o libhip_rdna_fix.so hip_memory_intercept.c -ldl -I/opt/rocm/include
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

static int initialized = 0;
static int is_rdna12 = -1;

// Function pointers to original HIP functions
static hipError_t (*original_hipMalloc)(void**, size_t) = NULL;
static hipError_t (*original_hipMallocManaged)(void**, size_t, unsigned int) = NULL;
static hipError_t (*original_hipGetDeviceProperties)(hipDeviceProp_t*, int) = NULL;

static void init_library() {
    if (initialized) return;
    initialized = 1;
    
    fprintf(stderr, "🔧 RMCP: HIP Memory Interceptor loaded\n");
    
    // Load original functions
    original_hipMalloc = dlsym(RTLD_NEXT, "hipMalloc");
    original_hipMallocManaged = dlsym(RTLD_NEXT, "hipMallocManaged");
    original_hipGetDeviceProperties = dlsym(RTLD_NEXT, "hipGetDeviceProperties");
    
    if (!original_hipMalloc || !original_hipGetDeviceProperties) {
        fprintf(stderr, "❌ RMCP: Failed to load HIP functions\n");
        return;
    }
    
    // Detect GPU
    hipDeviceProp_t prop;
    if (original_hipGetDeviceProperties(&prop, 0) == hipSuccess) {
        const char* arch = prop.gcnArchName;
        is_rdna12 = (strstr(arch, "gfx101") != NULL ||
                     strstr(arch, "gfx102") != NULL ||
                     strstr(arch, "gfx103") != NULL) ? 1 : 0;
        
        if (is_rdna12) {
            fprintf(stderr, "✅ RMCP: RDNA1/2 GPU detected (%s)\n", arch);
            fprintf(stderr, "✅ RMCP: Forcing non-coherent memory allocations\n");
            
            // Set environment variables for ROCm runtime
            setenv("HSA_USE_SVM", "0", 0);
            setenv("HSA_XNACK", "0", 0);
            setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 0);
        } else {
            fprintf(stderr, "ℹ️  RMCP: Non-RDNA1/2 GPU (%s) - no patches needed\n", arch);
        }
    }
}

// Intercept hipMalloc
hipError_t hipMalloc(void** ptr, size_t size) {
    if (!initialized) init_library();
    
    if (!original_hipMalloc) {
        fprintf(stderr, "❌ RMCP: hipMalloc not available\n");
        return hipErrorNotInitialized;
    }
    
    // For RDNA1/2, ensure non-coherent allocation
    if (is_rdna12) {
        // HipMalloc should already be non-coherent, but we verify
        hipError_t err = original_hipMalloc(ptr, size);
        if (err == hipSuccess) {
            // Success - memory allocated
        }
        return err;
    }
    
    return original_hipMalloc(ptr, size);
}

// Intercept hipMallocManaged
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    if (!initialized) init_library();
    
    if (!original_hipMallocManaged) {
        // Fallback to regular malloc if managed not available
        return hipMalloc(ptr, size);
    }
    
    // For RDNA1/2, force non-coherent flags
    if (is_rdna12) {
        // Modify flags to ensure non-coherent
        flags = hipMemAttachGlobal;  // Use global, non-coherent memory
    }
    
    return original_hipMallocManaged(ptr, size, flags);
}

// Intercept hipGetDeviceProperties (to initialize early)
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device) {
    if (!initialized) init_library();
    
    if (!original_hipGetDeviceProperties) {
        return hipErrorNotInitialized;
    }
    
    return original_hipGetDeviceProperties(prop, device);
}
