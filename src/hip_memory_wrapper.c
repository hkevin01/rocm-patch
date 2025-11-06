/*
 * RMCP HIP Memory Wrapper
 * Intercepts hipMalloc to force non-coherent memory on RDNA1/2
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime.h>

// Original hipMalloc function pointer
static hipError_t (*original_hipMalloc)(void**, size_t) = NULL;

// Check if GPU is RDNA1/2
static int is_rdna12_cached = -1;

static int is_rdna12() {
    if (is_rdna12_cached != -1) {
        return is_rdna12_cached;
    }
    
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        is_rdna12_cached = 0;
        return 0;
    }
    
    // Check for gfx101x, gfx102x, gfx103x (RDNA1/2)
    const char* arch = prop.gcnArchName;
    is_rdna12_cached = (strstr(arch, "gfx101") != NULL ||
                         strstr(arch, "gfx102") != NULL ||
                         strstr(arch, "gfx103") != NULL) ? 1 : 0;
    
    if (is_rdna12_cached) {
        fprintf(stderr, "🔧 RMCP: RDNA1/2 GPU detected (%s) - applying memory fix\n", arch);
    }
    
    return is_rdna12_cached;
}

// Intercepted hipMalloc
hipError_t hipMalloc(void** ptr, size_t size) {
    // Load original function if not loaded
    if (original_hipMalloc == NULL) {
        original_hipMalloc = (hipError_t (*)(void**, size_t))dlsym(RTLD_NEXT, "hipMalloc");
        if (original_hipMalloc == NULL) {
            fprintf(stderr, "RMCP ERROR: Could not load original hipMalloc\n");
            return hipErrorUnknown;
        }
    }
    
    // For RDNA1/2, use hipMallocManaged with non-coherent flag
    if (is_rdna12()) {
        // Try to use fine-grain memory
        hipError_t err = hipMallocManaged(ptr, size, hipMemAttachGlobal);
        if (err == hipSuccess) {
            // Advise to use non-coherent memory
            hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, 0);
        }
        return err;
    }
    
    // For other GPUs, use original hipMalloc
    return original_hipMalloc(ptr, size);
}

// Constructor - runs when library is loaded
__attribute__((constructor))
static void rmcp_init() {
    fprintf(stderr, "🚀 RMCP HIP Memory Wrapper loaded\n");
}
