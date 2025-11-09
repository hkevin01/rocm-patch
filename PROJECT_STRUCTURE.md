# Project Structure

This document describes the organization of the ROCm-Patch project.

## Root Directory

```
rocm-patch/
├── README.md                    # Main documentation and solution guide
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # Project license
├── requirements.txt             # Python dependencies
├── setup.py                     # Python package setup
│
├── tests/                       # Active test suite
│   ├── test_implicit_gemm_safe.py
│   └── test_implicit_gemm_comprehensive.py
│
├── docs/                        # Documentation
│   ├── BREAKTHROUGH.md          # Discovery process documentation
│   └── SOLUTION_SUMMARY.md      # Quick reference guide
│
├── scripts/                     # Active utility scripts
│   └── (installation helpers)
│
├── src/                         # Source code
│   └── (PyTorch extensions/patches)
│
├── configs/                     # Configuration files
│   └── (ROCm environment configs)
│
├── venv-py310-rocm52/          # Python 3.10 virtual environment
│   └── (PyTorch 1.13.1+rocm5.2 installation)
│
├── downloads/                   # Downloaded packages
│   └── torch-1.13.1+rocm5.2-cp310-cp310-linux_x86_64.whl
│
├── test-results/               # Test output logs
│   └── test_implicit_gemm_results.log
│
└── archive/                    # Historical files (not part of solution)
    ├── old-docs/               # Previous documentation attempts
    ├── old-tests/              # Earlier test scripts
    ├── old-scripts/            # Obsolete installation scripts
    └── installation-scripts/   # Various ROCm install attempts
```

## Directory Purposes

### Active Directories

- **`/`** (root): Core documentation (README, LICENSE, CONTRIBUTING)
- **`tests/`**: Working test suite for the final solution
- **`docs/`**: Supporting documentation
- **`scripts/`**: Useful utility scripts
- **`src/`**: Source code and PyTorch extensions
- **`configs/`**: Configuration files for ROCm environment
- **`venv-py310-rocm52/`**: Python 3.10 virtual environment with correct PyTorch version
- **`downloads/`**: Downloaded binary packages
- **`test-results/`**: Test execution logs and results

### Archive Directory

The `archive/` directory contains **historical artifacts** from the development process:

- **`old-docs/`**: 29 documentation files from various attempted solutions
  - Migration checklists for ROCm 5.2
  - Investigation summaries
  - Multiple TODO lists and action plans
  - Version analysis documents
  - Research findings
  
- **`old-tests/`**: 9 test scripts from debugging phase
  - Various workaround attempts
  - Size boundary tests
  - Timing and subprocess tests
  
- **`installation-scripts/`**: 6 installation scripts for different ROCm versions
  - ROCm 5.2 installation variations
  - ROCm 5.7 installation
  - Compatibility library installers
  
- **`old-scripts/`**: Miscellaneous helper scripts from development

**Note**: Archive contents are kept for historical reference but are **not required** for the working solution.

## Working Solution Files

The **minimal set of files** needed for the solution:

1. **`README.md`** - Complete installation and usage guide
2. **`venv-py310-rocm52/`** - Virtual environment with PyTorch 1.13.1+rocm5.2
3. **`tests/test_implicit_gemm_safe.py`** - Verification test script
4. **`docs/SOLUTION_SUMMARY.md`** - Quick reference
5. **Environment configuration** - `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1`

Everything else is either:
- Supporting documentation (BREAKTHROUGH.md, CONTRIBUTING.md)
- Historical artifacts (archive/)
- Development infrastructure (.git/, .github/, .vscode/)

## File Count Summary

| Category | Count | Location |
|----------|-------|----------|
| Active Documentation | 4 | Root + docs/ |
| Active Tests | 2 | tests/ |
| Virtual Environment | 1 | venv-py310-rocm52/ |
| Archived Documentation | 29 | archive/old-docs/ |
| Archived Tests | 9 | archive/old-tests/ |
| Archived Scripts | 6 | archive/installation-scripts/ |
| **Total Archived** | **44+** | **archive/** |

## Development History

This project went through **multiple attempted solutions** before arriving at the working configuration:

1. **ROCm 5.7 + PyTorch 2.2.2+rocm5.7** - Initial attempt (failed)
2. **ROCm 6.2.4 + Latest PyTorch** - Upgrade attempt (failed)
3. **ROCm 5.2.0 + PyTorch 2.2.2+rocm5.7** - Version mismatch (failed)
4. **ROCm 5.2.0 + PyTorch 1.13.1+rocm5.2 + Python 3.10 + IMPLICIT_GEMM** - ✅ **Working Solution**

The archive directory preserves the documentation from attempts 1-3 for reference.

## Navigation Guide

- **Getting Started**: Read `README.md`
- **Quick Setup**: See `docs/SOLUTION_SUMMARY.md`
- **Testing**: Run `tests/test_implicit_gemm_safe.py`
- **Discovery Process**: See `docs/BREAKTHROUGH.md`
- **Historical Context**: Browse `archive/old-docs/`
- **Contributing**: Read `CONTRIBUTING.md`

---

**Last Updated**: November 9, 2025
**Status**: Organized and cleaned
