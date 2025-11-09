# ROCm Patch Repository - Application Description

## ⚠️ CRITICAL PROJECT REQUIREMENT ⚠️

**PRIMARY GOAL: FIX GPU TO WORK WITH PYTORCH - NOT CPU FALLBACK**

This project's purpose is to make the AMD RX 5600 XT (RDNA1/gfx1010) GPU work with PyTorch for actual GPU training.

**NEVER suggest CPU training as a solution.** The user explicitly rejected CPU fallback multiple times.

The goal is **GPU acceleration working natively**, not workarounds that avoid the GPU.

---

## Overview
A comprehensive patch collection and deployment system for AMD ROCm (Radeon Open Compute) platform issues. This repository addresses common problems, bugs, and compatibility issues found in ROCm installations across various GPU architectures and Linux distributions.

**Current Focus**: Fixing RDNA1 (gfx1010) GPU compatibility with PyTorch/ROCm to enable actual GPU training, not CPU fallback.

## Core Features

### 1. **Automated Patch Detection & Application**
- Automatic detection of ROCm version and GPU architecture
- Smart patch selection based on system configuration
- Safe application with automatic rollback capabilities

### 2. **Comprehensive Issue Coverage**
- Memory access fault patches (GPU node errors)
- VRAM allocation and detection fixes
- HIP kernel compatibility patches
- Dependency resolution for installation issues
- DKMS build failure fixes for newer kernels

### 3. **System-Wide Deployment**
- Safe system-wide patch installation
- Integration with package managers (apt, dnf, zypper)
- Persistent patches across ROCm updates
- Modular patch selection

### 4. **Testing & Validation Framework**
- Comprehensive test suite for each patch
- Boundary condition testing
- Performance benchmarking
- Multi-GPU and multi-node testing support

### 5. **Monitoring & Logging**
- Detailed patch application logging
- Performance metrics collection
- Error tracking and reporting
- Recovery and rollback capabilities

## Target Users

### Primary Users
- **Data Scientists & ML Engineers**: Running PyTorch, TensorFlow on AMD GPUs
- **HPC Developers**: Working with ROCm for high-performance computing
- **Computer Vision Researchers**: Using ROCm for thermal imaging and object detection
- **System Administrators**: Managing ROCm installations across multiple systems

### Secondary Users
- **GPU Application Developers**: Building HIP/OpenCL applications
- **ROCm Contributors**: Testing and developing ROCm improvements

## Technical Stack

### Core Technologies
- **Language**: Python 3.10+, Bash, C++ (for HIP patches)
- **Package Management**: setuptools, pip, apt, dnf
- **Testing**: pytest, unittest, bash test framework
- **Container**: Docker with ROCm base images
- **CI/CD**: GitHub Actions

### ROCm Components Addressed
- ROCm Runtime (rocminfo, rocm-smi)
- HIP (Heterogeneous-Compute Interface for Portability)
- ROCm kernel drivers (amdgpu)
- ROCm libraries (rocBLAS, rocFFT, MIOpen)
- PyTorch/TensorFlow ROCm builds

### Supported ROCm Versions
- ROCm 5.7.x (LTS)
- ROCm 6.0.x - 6.4.x
- ROCm 7.0.x - 7.1.x (Latest)

### Supported GPU Architectures
- **RDNA2**: RX 6000 series (gfx1030, gfx1031)
- **RDNA3**: RX 7000 series (gfx1100, gfx1101, gfx1102, gfx1151)
- **CDNA2**: MI200 series (gfx90a)
- **CDNA3**: MI300 series (gfx940, gfx941, gfx942)

### Supported Operating Systems
- Ubuntu 20.04, 22.04, 24.04, 25.04
- RHEL/Rocky/AlmaLinux 8.x, 9.x
- SLES 15 SP4+
- WSL2 (Windows Subsystem for Linux)

## Project Goals

### Short-term Goals (3-6 months)
1. Document and patch top 20 most common ROCm issues
2. Create automated system-wide installation script
3. Establish comprehensive test coverage (>80%)
4. Support ROCm 7.1.x with all major GPU architectures
5. Create Docker-based testing environment

### Long-term Goals (6-12 months)
1. Integration with official ROCm issue tracker
2. Automated patch generation from bug reports
3. Community contribution framework
4. Performance optimization patches
5. Upstream contribution to ROCm project

## Architecture Principles
- **Modularity**: Each patch is independent and self-contained
- **Safety**: All patches include rollback mechanisms
- **Compatibility**: Backward compatible with older ROCm versions
- **Testing**: Every patch has associated tests
- **Documentation**: Comprehensive docs for each issue and patch

## Integration Points
- ROCm installation directories (/opt/rocm)
- System libraries (/usr/lib, /usr/lib64)
- Kernel modules (amdgpu DKMS)
- Python site-packages for ML frameworks
- Environment variables (PATH, LD_LIBRARY_PATH)
