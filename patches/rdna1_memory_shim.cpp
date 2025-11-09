/*
 * RDNA1 Memory Fix Shim - LD_PRELOAD Library
 * 
 * This library intercepts HSA memory allocation functions and forces
 * coarse-grained memory for RDNA1 GPUs (gfx1010) when spoofed as gfx1030.
 * 
 * Safer than patching ROCr-Runtime directly - won't crash the system!
 * 
 * Build:
 *   g++ -shared -fPIC -O2 -o librdna1_fix.so rdna1_memory_shim.cpp -ldl
 * 
 * Use:
 *   export LD_PRELOAD=/path/to/librdna1_fix.so
 *   export HSA_OVERRIDE_GFX_VERSION=10.3.0
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Device ID for RX 5600 XT (RDNA1)
#define RX_5600_XT_DEVICE_ID 0x731F
#define RX_5500_XT_DEVICE_ID 0x7340

static bool is_rdna1_gpu = false;
static bool checked_gpu = false;
static bool verbose = false;

// Check if we're running on RDNA1 hardware
static void check_rdna1_device() {
    if (checked_gpu) return;
    checked_gpu = true;
    
    verbose = getenv("RDNA1_FIX_VERBOSE") != NULL;
    
    // Check via sysfs
    FILE* fp = fopen("/sys/class/kfd/kfd/topology/nodes/1/properties", "r");
    if (!fp) {
        if (verbose) fprintf(stderr, "[RDNA1 Fix] Could not open KFD properties\n");
        return;
    }
    
    char line[256];
    uint32_t device_id = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "device_id %u", &device_id) == 1 || 
            sscanf(line, "device_id 0x%x", &device_id) == 1) {
            break;
        }
    }
    fclose(fp);
    
    // Check if it's RDNA1
    is_rdna1_gpu = (device_id == RX_5600_XT_DEVICE_ID || 
                    device_id == RX_5500_XT_DEVICE_ID ||
                    (device_id >= 0x7310 && device_id <= 0x736F));
    
    if (is_rdna1_gpu) {
        fprintf(stderr, "[RDNA1 Fix] ✅ Detected RDNA1 GPU (device_id=0x%x)\n", device_id);
        fprintf(stderr, "[RDNA1 Fix] Will force coarse-grained memory allocations\n");
    } else if (verbose) {
        fprintf(stderr, "[RDNA1 Fix] Not RDNA1 (device_id=0x%x), shim inactive\n", device_id);
    }
}

// Intercept hsaKmtAllocMemory
typedef int (*hsaKmtAllocMemory_t)(uint32_t NodeId, uint64_t Size, void* Flags, void** MemoryAddress);
static hsaKmtAllocMemory_t real_hsaKmtAllocMemory = NULL;

extern "C" int hsaKmtAllocMemory(uint32_t NodeId, uint64_t Size, void* Flags, void** MemoryAddress) {
    check_rdna1_device();
    
    if (!real_hsaKmtAllocMemory) {
        real_hsaKmtAllocMemory = (hsaKmtAllocMemory_t)dlsym(RTLD_NEXT, "hsaKmtAllocMemory");
        if (!real_hsaKmtAllocMemory) {
            fprintf(stderr, "[RDNA1 Fix] ERROR: Could not find hsaKmtAllocMemory\n");
            return -1;
        }
    }
    
    // If RDNA1, force coarse-grained memory
    if (is_rdna1_gpu && Flags) {
        // HsaMemFlags structure (from KFD)
        struct {
            union {
                struct {
                    uint32_t NonPaged : 1;
                    uint32_t CachePolicy : 2;
                    uint32_t ReadOnly : 1;
                    uint32_t PageSize : 2;
                    uint32_t HostAccess : 1;
                    uint32_t NoSubstitute : 1;
                    uint32_t GDSMemory : 1;
                    uint32_t Scratch : 1;
                    uint32_t MMIO : 1;
                    uint32_t Reserved : 4;
                    uint32_t CoarseGrain : 1;
                    uint32_t Uncached : 1;
                    uint32_t Reserved2 : 14;
                    uint32_t ExecuteAccess : 1;
                } ui32;
                uint32_t Value;
            };
        }* flags = (decltype(flags))Flags;
        
        // Force coarse-grained memory
        if (flags->ui32.CoarseGrain == 0) {
            if (verbose) {
                fprintf(stderr, "[RDNA1 Fix] Forcing CoarseGrain=1 for NodeId=%u, Size=%lu\n", 
                        NodeId, (unsigned long)Size);
            }
            flags->ui32.CoarseGrain = 1;
        }
    }
    
    return real_hsaKmtAllocMemory(NodeId, Size, Flags, MemoryAddress);
}

// Constructor - runs when library is loaded
__attribute__((constructor))
static void rdna1_fix_init() {
    fprintf(stderr, "[RDNA1 Fix] Shim library loaded\n");
    check_rdna1_device();
}

