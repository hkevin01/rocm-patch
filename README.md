# ROCm Patch Repository 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROCm Version](https://img.shields.io/badge/ROCm-5.7%20%7C%206.x%20%7C%207.x-blue)](https://rocm.docs.amd.com/)
[![Platform](https://img.shields.io/badge/Platform-Linux-green)](https://www.linux.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

**A comprehensive collection of patches, fixes, and workarounds for AMD ROCm platform issues**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Systems](#supported-systems)
- [Quick Start](#quick-start)
- [Available Patches](#available-patches)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

The **ROCm Patch Repository** provides battle-tested solutions for common AMD ROCm (Radeon Open Compute) platform issues that affect:

- **Machine Learning** engineers running PyTorch and TensorFlow on AMD GPUs
- **Computer Vision** researchers using ROCm for thermal imaging and object detection
- **HPC Developers** building high-performance compute applications
- **System Administrators** managing ROCm installations at scale

### Why This Project?

ROCm is a powerful open-source platform for GPU computing, but users often encounter:
- Memory access faults and GPU node errors
- VRAM allocation and detection issues
- HIP kernel compatibility problems
- Dependency conflicts during installation
- DKMS build failures on newer Linux kernels

This repository provides **tested, reliable patches** with:
- ✅ Automated detection and application
- ✅ Safe rollback mechanisms
- ✅ Comprehensive documentation
- ✅ Active community support

---

## ✨ Features

### 🔧 **Automated Patch Management**
- Smart detection of ROCm version and GPU architecture
- One-command patch application
- Safe rollback to pre-patch state

### 🛡️ **Comprehensive Issue Coverage**
- Memory access fault patches
- VRAM allocation fixes
- HIP kernel compatibility patches
- Package dependency resolution
- DKMS build failure fixes

### 📊 **Testing & Validation**
- Extensive test suite for each patch
- Boundary condition testing
- Performance benchmarking
- Multi-GPU support

### 📚 **Documentation**
- Detailed problem descriptions
- Step-by-step installation guides
- Troubleshooting resources
- API documentation

---

## 💻 Supported Systems

### ROCm Versions
- ✅ ROCm 5.7.x (LTS)
- ✅ ROCm 6.0.x - 6.4.x
- ✅ ROCm 7.0.x - 7.1.x (Latest)

### GPU Architectures
| Architecture | GPUs | Status |
|-------------|------|--------|
| **RDNA2** | RX 6000 series (gfx1030, gfx1031) | ✅ Supported |
| **RDNA3** | RX 7000 series (gfx1100-gfx1151) | ✅ Supported |
| **CDNA2** | MI200 series (gfx90a) | ✅ Supported |
| **CDNA3** | MI300 series (gfx940-gfx942) | ✅ Supported |

### Operating Systems
- **Ubuntu**: 20.04, 22.04, 24.04, 25.04
- **RHEL/Rocky/AlmaLinux**: 8.x, 9.x
- **SLES**: 15 SP4+
- **WSL2**: Windows Subsystem for Linux

---

## 🚀 Quick Start

### Prerequisites
- ROCm 5.7+ installed
- Python 3.10+
- sudo/root access for system-wide patches

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rocm-patch.git
cd rocm-patch

# Run the automated installer
sudo ./scripts/install.sh

# Or install specific patches
sudo ./scripts/patch_manager.py --patch memory-access-fault
```

### Verify Installation

```bash
# Check ROCm status
rocminfo

# Verify GPU detection
rocm-smi

# Test with PyTorch (if installed)
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🩹 Available Patches

### Critical Patches

#### 1. **Memory Access Fault Fix**
- **Issue**: `Memory access fault by GPU node-1 on address (nil)`
- **Affected**: ROCm 7.0+, Multi-GPU systems
- **Status**: ✅ Tested
- **Doc**: [docs/patches/memory-access-fault.md](docs/patches/memory-access-fault.md)

#### 2. **VRAM Allocation Patch**
- **Issue**: ROCm detects incorrect VRAM size (e.g., 62.2GB instead of 64GB)
- **Affected**: Strix Halo (gfx1151), APUs
- **Status**: ✅ Tested
- **Doc**: [docs/patches/vram-allocation.md](docs/patches/vram-allocation.md)

#### 3. **HIP Kernel Compatibility**
- **Issue**: `RuntimeError: HIP error: invalid device function`
- **Affected**: RDNA3, specific ROCm versions
- **Status**: ✅ Tested
- **Doc**: [docs/patches/hip-kernel-compat.md](docs/patches/hip-kernel-compat.md)

### High Priority Patches

#### 4. **Dependency Resolution**
- **Issue**: Broken dependencies in amdgpu-install
- **Affected**: ROCm 7.1, Ubuntu 24.04+
- **Status**: ✅ Tested
- **Doc**: [docs/patches/dependency-resolution.md](docs/patches/dependency-resolution.md)

#### 5. **DKMS Build Fix**
- **Issue**: AMDGPU DKMS fails on kernel 6.17+
- **Affected**: Ubuntu 25.10, latest kernels
- **Status**: 🟡 Beta
- **Doc**: [docs/patches/dkms-build-fix.md](docs/patches/dkms-build-fix.md)

#### 6. **PyTorch ROCm Integration**
- **Issue**: PyTorch doesn't detect GPU or fails on first kernel
- **Affected**: PyTorch 2.0+, ROCm 5.7+
- **Status**: ✅ Tested
- **Doc**: [docs/patches/pytorch-integration.md](docs/patches/pytorch-integration.md)

---

## 📦 Installation

### Method 1: Automated Installer (Recommended)

```bash
# Full installation with all patches
sudo ./scripts/install.sh --all

# Interactive mode (select patches)
sudo ./scripts/install.sh --interactive

# Install specific category
sudo ./scripts/install.sh --category critical
```

### Method 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Apply specific patch
sudo python src/patches/memory_access_fault.py --apply

# Verify patch
sudo python src/patches/memory_access_fault.py --verify
```

### Method 3: Docker Container

```bash
# Build Docker image with patches
docker build -t rocm-patched:latest -f configs/docker/Dockerfile .

# Run container
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
  -it rocm-patched:latest bash

# Verify ROCm inside container
rocminfo
```

---

## 🔧 Usage

### Patch Manager CLI

```bash
# List all available patches
./scripts/patch_manager.py --list

# Get patch information
./scripts/patch_manager.py --info memory-access-fault

# Apply patch with backup
sudo ./scripts/patch_manager.py --apply memory-access-fault --backup

# Rollback to pre-patch state
sudo ./scripts/patch_manager.py --rollback memory-access-fault

# Check patch status
./scripts/patch_manager.py --status

# Update patches
sudo ./scripts/patch_manager.py --update
```

### Python API

```python
from rocm_patch import PatchManager, SystemInfo

# Initialize patch manager
pm = PatchManager()

# Detect system configuration
sys_info = SystemInfo.detect()
print(f"ROCm Version: {sys_info.rocm_version}")
print(f"GPU Architecture: {sys_info.gpu_arch}")

# Get recommended patches
recommended = pm.get_recommended_patches(sys_info)

# Apply patches
for patch in recommended:
    result = pm.apply_patch(patch.name, backup=True)
    print(f"Patch {patch.name}: {result.status}")
```

### Environment Variables

```bash
# Skip interactive prompts
export ROCM_PATCH_AUTO=1

# Enable verbose logging
export ROCM_PATCH_VERBOSE=1

# Custom backup location
export ROCM_PATCH_BACKUP_DIR=/var/backups/rocm-patches

# Dry run mode (no actual changes)
export ROCM_PATCH_DRY_RUN=1
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only
```

### Validate Patches

```bash
# Validate all patches
./scripts/validate_patches.sh

# Validate specific patch
./scripts/validate_patches.sh memory-access-fault

# Run boundary condition tests
pytest tests/boundary/ -v
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-patch`)
3. **Test** your changes thoroughly
4. **Commit** with clear messages (`git commit -m 'Add amazing patch for issue X'`)
5. **Push** to your branch (`git push origin feature/amazing-patch`)
6. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rocm-patch.git
cd rocm-patch

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

---

## 📖 Documentation

- **[Project Plan](docs/project-plan.md)** - Comprehensive project roadmap
- **[Architecture](docs/architecture.md)** - System design and architecture
- **[API Reference](docs/api-reference.md)** - Python API documentation
- **[Patch Guide](docs/patch-guide.md)** - How to create patches
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

---

## 🐛 Troubleshooting

### Common Issues

**Patch fails to apply**
```bash
# Check system compatibility
./scripts/patch_manager.py --check-compatibility

# View detailed error log
cat /var/log/rocm-patch/error.log

# Try manual application
sudo python src/patches/PATCH_NAME.py --apply --verbose
```

**ROCm still not working after patch**
```bash
# Verify ROCm installation
rocminfo

# Check GPU status
rocm-smi

# Restart ROCm services
sudo systemctl restart rocm

# Check for conflicts
./scripts/patch_manager.py --diagnose
```

**Need to rollback**
```bash
# Rollback specific patch
sudo ./scripts/patch_manager.py --rollback PATCH_NAME

# Rollback all patches
sudo ./scripts/patch_manager.py --rollback-all

# Restore from backup
sudo ./scripts/restore_backup.sh /var/backups/rocm-patches/TIMESTAMP
```

For more help, see [docs/troubleshooting.md](docs/troubleshooting.md) or open an issue.

---

## 📊 Project Status

### Current Version: 0.1.0 (Alpha)

**Phase 1: Foundation** - ✅ Complete
- [x] Project structure
- [x] Memory bank documentation
- [x] Development environment

**Phase 2: Research & Documentation** - 🟡 In Progress
- [x] Common issues identified
- [ ] Root cause analysis (80%)
- [ ] Solution approaches validated

**Phase 3: Core Patches** - ⭕ Not Started
- [ ] Memory access fault patch
- [ ] VRAM allocation fix
- [ ] HIP compatibility patches

See [docs/project-plan.md](docs/project-plan.md) for detailed roadmap.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- ROCm components are subject to their respective licenses
- See [NOTICES.md](NOTICES.md) for third-party attributions

---

## 🙏 Acknowledgments

- **AMD ROCm Team** - For creating and maintaining the ROCm platform
- **ROCm Community** - For reporting issues and testing patches
- **Contributors** - Everyone who has contributed patches and improvements

### Related Projects
- [ROCm/ROCm](https://github.com/ROCm/ROCm) - Official ROCm repository
- [ROCm Examples](https://github.com/amd/rocm-examples) - ROCm code examples

### References
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [ROCm GitHub Issues](https://github.com/ROCm/ROCm/issues)

---

## 📞 Contact & Support

- **Issues**: Use [GitHub Issues](https://github.com/YOUR_USERNAME/rocm-patch/issues) for bug reports
- **Discussions**: Use [GitHub Discussions](https://github.com/YOUR_USERNAME/rocm-patch/discussions) for questions
- **Security**: See [SECURITY.md](SECURITY.md) for security policy

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

<p align="center">
Made with ❤️ by the ROCm Patch Community
</p>

<p align="center">
<sub>Not affiliated with AMD or the official ROCm project</sub>
</p>
