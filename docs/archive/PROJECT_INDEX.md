# RDNA1 ROCm Patch Project - Complete File Index

**Last Updated**: November 7, 2025  
**Project Status**: 90% Complete - MIOpen patches functional, blocked by HSA runtime limitations

---

## Quick Navigation

- **Getting Started**: [README.md](#documentation)
- **Build Instructions**: [scripts/rebuild_miopen.sh](#automation-scripts)
- **Command Reference**: [COMMANDS.md](#documentation)
- **Technical Summary**: [SUMMARY.md](#documentation)

---

## Core Documentation

### Main Documentation Files
- **README.md** - Main project documentation with step-by-step instructions
- **SUMMARY.md** - Comprehensive technical summary of the project
- **COMMANDS.md** - Complete command reference for reproduction
- **CONTRIBUTING.md** - Contribution guidelines

### Status & Summary Documents
- **CURRENT_STATUS.md** - Current project status
- **FINAL_PROJECT_STATUS.md** - Final status report
- **COMPLETION_SUMMARY.md** - Project completion summary
- **INVESTIGATION_COMPLETE.md** - Investigation findings
- **PROJECT_COMPLETE.md** - Project completion documentation

### Technical Documentation
- **MIOPEN_RDNA1_PATCH.md** - MIOpen patching details
- **FINAL_SOLUTION.md** - Final solution documentation
- **GPU_SOLUTION_STATUS.md** - GPU solution status
- **PHASE_2_SUMMARY.md** - Phase 2 summary
- **POST_BUILD_STEPS.md** - Post-build steps

### Testing & Results
- **MTYPE_TEST_RESULTS.md** - Memory type testing results
- **HARDWARE_TEST_SUMMARY.md** - Hardware testing summary
- **TESTING_PHASE_COMPLETE.md** - Testing phase documentation

---

## Automation Scripts

Location: `scripts/`

### ROCm Installation
- **install_rocm_6.2.4.sh** - Install ROCm 6.2.4 to match PyTorch
  - Removes existing ROCm installations
  - Adds ROCm 6.2.4 repository
  - Installs rocm-hip-sdk and miopen-hip

### MIOpen Building
- **rebuild_miopen.sh** - Complete MIOpen rebuild script
  - Configures CMake with correct flags
  - Builds MIOpen with RDNA1 patches
  - Installs to /opt/rocm-miopen-rdna1
  - Replaces PyTorch's library automatically

- **build_miopen_rdna1.sh** - Alternative build script
- **resume_build.sh** - Resume interrupted builds
- **check_build_status.sh** - Check build progress

### Testing Scripts
- **test_rdna1_patches.sh** - Verify patches are active
  - Checks for patch strings in library
  - Runs Python runtime test
  - Displays debug output

- **test_patched_miopen.sh** - Test patched MIOpen
- **test_simple_rdna1.py** - Simple RDNA1 test
- **test_direct_conv.py** - Direct convolution test
- **test_immediate_mode.py** - Immediate mode test
- **test_no_find.py** - Skip-Find mode test

### Environment Setup
- **setup_gpu_env.sh** - Set up GPU environment variables
- **patch_rocm_environment.sh** - Patch ROCm environment

### Legacy/Alternative Scripts
- **patch_rocm_source.sh** - Patch ROCm source code
- **patch_kernel_module.sh** - Kernel module patching
- **apply_mtype_fix.sh** - Memory type fix application
- **patch_rocm_isolated.sh** - Isolated ROCm patching
- **test_docker_rocm57.sh** - Docker ROCm 5.7 testing

---

## Source Code

Location: `src/`

### Main Source Files
- **hip_memory_intercept.c** - HIP memory interception library
- **hip_memory_wrapper.c** - HIP memory wrapper functions
- **libhip_rdna_fix.so** - Compiled HIP RDNA1 fix library
- **rmcp_workaround.py** - RMCP workaround module
- **__init__.py** - Package initialization

### Patches Module
Location: `src/patches/`
- **__init__.py** - Patches module initialization
- **memory_access_fault/** - Memory access fault patches
  - **hip_memory_patch.py** - HIP memory patching
  - **kernel_params.sh** - Kernel parameter setup
  - **README.md** - Patch documentation
  - **__init__.py** - Module initialization

### Utilities
Location: `src/utils/`
- **gpu_detection.py** - GPU detection utilities
- **system_info.py** - System information gathering
- **__init__.py** - Utils module initialization

---

## PyTorch Extensions

Location: `pytorch_extensions/`

Custom PyTorch layers for RDNA1 compatibility:
- **rdna1_conv2d.cpp** - C++ RDNA1 Conv2D implementation
- **rdna1_layers.py** - Python RDNA1 layer wrappers (v1)
- **rdna1_layers_v2.py** - Enhanced version (v2)
- **rdna1_layers_v3.py** - Latest version (v3)
- **setup.py** - Extension build configuration

---

## Testing Suite

Location: `tests/`

- **test_conv2d_minimal.py** - Minimal Conv2D test
- **test_hardware_compatibility.py** - Hardware compatibility tests
- **test_project_integration.sh** - Full integration tests
- **test_real_world_workloads.py** - Real-world workload tests

---

## Configuration Files

### Docker Configuration
Location: `configs/docker/`
- **Dockerfile** - Docker image definition
- **docker-compose.yml** - Docker Compose configuration

### Project Configuration
- **setup.py** - Python package setup
- **requirements.txt** - Python dependencies
- **LICENSE** - MIT License
- **.gitignore** - Git ignore patterns

---

## Patches & Libraries

### ROCm Source Patches
Location: `patches/rocm-source/`
- **001-hip-rdna-memory-coherency.patch** - HIP memory coherency patch
- **002-rocr-rdna-memory-type.patch** - ROCR memory type patch

### Compiled Libraries
Location: `lib/`
- **librmcp_hip_wrapper.so** - RMCP HIP wrapper library

---

## Data & Assets

### Training Data
Location: `data/MNIST/raw/`
- MNIST dataset files (train/test images and labels)

### Assets
Location: `assets/`
- **README.md** - Assets documentation

---

## Memory Bank

Location: `memory-bank/`

Project memory and context:
- **app-description.md** - Application description
- **change-log.md** - Change log
- **CRITICAL_REQUIREMENT.md** - Critical requirements
- **architecture-decisions/** - Architecture decision records
- **implementation-plans/** - Implementation plans

---

## Helper Scripts & Utilities

### Installation
- **install.sh** - Main installation script

### Runtime
- **run_pytorch_rdna1.sh** - Run PyTorch with RDNA1 settings
- **train_mnist.py** - MNIST training example

---

## Important Technical Documentation

### Problem Analysis
- **LLVM_CONFLICT_EXPLAINED.md** - LLVM version conflict explanation
- **KERNEL_MTYPE_SOLUTION.md** - Kernel memory type solution
- **KERNEL_PARAMS_APPLIED.md** - Applied kernel parameters
- **GPU_FIX_REQUIRED.md** - GPU fix requirements

### Quickstart Guides
- **QUICKSTART.md** - Quick start guide
- **MTYPE_FIX_QUICKSTART.md** - Memory type fix quickstart
- **RDNA1_CONV2D_SOLUTION.md** - RDNA1 Conv2D solution guide

### Implementation Details
- **IMPLEMENTATION_COMPLETE.md** - Implementation completion status
- **docs/ROCM_SOURCE_PATCHING_STRATEGY.md** - ROCm source patching strategy
- **docs/NEXT_STEPS_GMC_V12_INSIGHTS.md** - GMC V12 insights
- **docs/project-plan.md** - Overall project plan
- **docs/TESTING.md** - Testing documentation

### Issue Documentation
Location: `docs/issues/`
- **eeg2025-tensor-operations.md** - EEG tensor operations issue
- **thermal-object-detection-memory-faults.md** - Thermal detection memory faults
- **README.md** - Issues overview

---

## Log Files

- **patch_installation.log** - Patch installation log
- **test_results_after_kernel_params.log** - Test results after kernel params

---

## Key File Locations for Rebuilding

### Source Code to Patch
Located in: `/tmp/MIOpen/`
- **src/hip/handlehip.cpp** (lines 106-140) - Memory allocation patches
- **src/convolution_api.cpp** (lines 50-65, 585+, 1200+, 1380+) - Skip-Find patches

### Build Directory
- `/tmp/MIOpen/build_rdna1/` - CMake build directory

### Installation Locations
- `/opt/rocm-miopen-rdna1/` - Custom MIOpen installation
- `/opt/rocm-miopen-rdna1/lib/libMIOpen.so.1.0` - Patched library (447MB)

### PyTorch Integration
- `~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so` - Active library
- `~/.local/lib/python3.12/site-packages/torch/lib/libMIOpen.so.original` - Backup

---

## Workflow Reference

### 1. Initial Setup
```bash
cd ~/Projects/rocm-patch/scripts
./install_rocm_6.2.4.sh  # Install ROCm 6.2.4
```

### 2. Apply Patches
Follow instructions in `README.md` sections 2.2 and 2.3 to manually edit:
- `/tmp/MIOpen/src/hip/handlehip.cpp`
- `/tmp/MIOpen/src/convolution_api.cpp`

### 3. Build
```bash
./rebuild_miopen.sh  # Automated build and deployment
```

### 4. Test
```bash
./test_rdna1_patches.sh  # Verify patches are active
```

### 5. Use
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FORCE_RDNA1=1
python3 your_script.py
```

---

## Environment Variables

### Essential Variables
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0  # Spoof as RDNA2
MIOPEN_FORCE_RDNA1=1             # Force RDNA1 detection
MIOPEN_LOG_LEVEL=7               # Maximum debug output
```

### Optional Variables
```bash
MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
MIOPEN_DEBUG_CONV_WINOGRAD=0
MIOPEN_DEBUG_CONV_FFT=0
```

---

## Project Statistics

- **Total Files**: ~150+
- **Documentation Files**: 40+
- **Scripts**: 20+
- **Source Files**: 15+
- **Test Files**: 10+
- **Size**: ~500KB documentation + 64MB MNIST data + compiled libraries

---

## Critical Success Indicators

✅ **Working:**
- ROCm 6.2.4 installation matching PyTorch
- MIOpen patches compile successfully
- Patches detected in runtime library
- RDNA1 detection functions correctly
- Skip-Find logic executes

⚠️ **Partial:**
- Memory allocation patches present but ineffective at HSA level

❌ **Blocked:**
- HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION persists
- Root cause: RDNA1 lacks fine-grained SVM support
- Fix requires: HSA/HIP runtime level patches or kernel module changes

---

## Contact & Support

- **Repository**: https://github.com/hkevin01/rocm-patch
- **Issues**: https://github.com/hkevin01/rocm-patch/issues
- **Documentation**: See README.md and SUMMARY.md

---

## License

MIT License - See LICENSE file

---

**Note**: This project successfully patches MIOpen but is fundamentally limited by RDNA1 hardware architecture. For production GPU acceleration, RDNA2 (RX 6000 series) or newer is recommended.
