#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

static int rdna1_override_enabled = -1;

// Check if we should enable override
static int should_override() {
    if (rdna1_override_enabled != -1) {
        return rdna1_override_enabled;
    }

    // Check environment variable
    char* env = getenv("RDNA1_MEMORY_OVERRIDE");
    if (env && strcmp(env, "1") == 0) {
        rdna1_override_enabled = 1;
        fprintf(stderr, "[RDNA1_OVERRIDE] Enabled via environment variable\n");
    } else {
        rdna1_override_enabled = 0;
    }

    return rdna1_override_enabled;
}

// Override hipMalloc to use coarse-grained memory
hipError_t hipMalloc(void** ptr, size_t size) {
    static hipError_t (*real_hipMalloc)(void**, size_t) = NULL;
    static int checked_rdna1 = 0;
    static int is_rdna1_device = 0;

    if (!real_hipMalloc) {
        real_hipMalloc = dlsym(RTLD_NEXT, "hipMalloc");
        if (!real_hipMalloc) {
            fprintf(stderr, "[RDNA1_OVERRIDE] ERROR: Could not find real hipMalloc\n");
            return hipErrorNotFound;
        }
    }

    if (!should_override()) {
        return real_hipMalloc(ptr, size);
    }

    // Use real hipMalloc first - this is safer and lets HIP initialize properly
    // We'll focus on MIOpen-level memory, not all allocations
    return real_hipMalloc(ptr, size);
}

// Override hipMallocManaged
hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    static hipError_t (*real_hipMallocManaged)(void**, size_t, unsigned int) = NULL;

    if (!real_hipMallocManaged) {
        real_hipMallocManaged = dlsym(RTLD_NEXT, "hipMallocManaged");
    }

    if (!should_override()) {
        return real_hipMallocManaged(ptr, size, flags);
    }

    fprintf(stderr, "[RDNA1_OVERRIDE] hipMallocManaged called, redirecting to hipMalloc\n");

    // Redirect to our overridden hipMalloc
    return hipMalloc(ptr, size);
}

// Constructor
__attribute__((constructor))
static void rdna1_override_init() {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║         RDNA1 Memory Override Library Loaded                  ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "This library intercepts HIP memory allocations and attempts\n");
    fprintf(stderr, "to use coarse-grained memory for RDNA1 GPUs.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Set RDNA1_MEMORY_OVERRIDE=1 to enable (currently: %s)\n",
            getenv("RDNA1_MEMORY_OVERRIDE") ? getenv("RDNA1_MEMORY_OVERRIDE") : "not set");
    fprintf(stderr, "\n");
}
