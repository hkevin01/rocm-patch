# Project Structure

## Root Directory

```
rocm-patch/
├── README.md                    # Main documentation (15 KB)
├── README_ROCM57.md            # Detailed ROCm 5.7 guide (4.4 KB)
├── SOLUTION_ROCM57.md          # Solution summary (4.0 KB)
├── PROJECT_STRUCTURE.md        # This file (2.8 KB)
├── INSTALL.md                  # Installation instructions (7.7 KB)
├── CONTRIBUTING.md             # Contribution guidelines (4.4 KB)
├── LICENSE                     # Project license (1.1 KB)
├── requirements.txt            # Python dependencies (504 B)
├── setup.py                    # Python package setup (2.7 KB)
│
├── install_rocm57.sh           # ⭐ Main installer (3.5 KB)
├── verify_setup.sh             # ⭐ Setup verification (3.0 KB)
├── test_conv2d_timing.py       # ⭐ Conv2d timing test (2.1 KB)
│
├── assets/                     # Project assets
├── configs/                    # Configuration files
├── data/                       # Data files
├── docs/                       # Documentation
│   ├── archive/                # ⚠️ Archived investigation files (60+ files)
│   ├── issues/                 # Issue-specific documentation
│   └── project-plan.md         # Project planning
│
├── kernel-patches/             # ⚠️ Failed kernel patches (archived)
├── lib/                        # Libraries
├── memory-bank/                # Memory/context files
├── patches/                    # ⚠️ Failed patches (archived)
├── pytorch_extensions/         # PyTorch extensions
├── scripts/                    # Helper scripts (test/setup scripts moved here)
├── src/                        # Source code
└── tests/                      # Test files
```

**⭐ Key Files** - Main files you need to use
**⚠️ Archived** - Historical investigation files (not needed for working solution)

## Key Files You Need

### 1. Installation & Verification
- `install_rocm57.sh` - **Main installer** - Sets up ROCm 5.7 configuration for RDNA1
- `verify_setup.sh` - **Verification script** - Checks if everything is configured correctly

### 2. Testing
- `test_conv2d_timing.py` - **Timing test** - Shows first-run (30-60s) vs cached (<0.01s) behavior

### 3. Documentation
- `README.md` - **Main documentation** - Complete guide with version info and environment variables
- `README_ROCM57.md` - **Detailed guide** - ROCm 5.7 specific setup and troubleshooting
- `SOLUTION_ROCM57.md` - **Solution summary** - Quick overview of what was done
- `PROJECT_STRUCTURE.md` - **This file** - Project organization

### 4. System Configuration (Auto-Created)
- `/etc/profile.d/rocm-rdna1-57.sh` - **System-wide config** - Auto-loaded environment variables

## What Each File Does

### install_rocm57.sh
- Detects RDNA1 GPU (RX 5600 XT, RX 5700 XT)
- Creates `/etc/profile.d/rocm-rdna1-57.sh`
- Sets up all environment variables
- Verifies PyTorch version

### verify_setup.sh
- Checks if configuration file exists
- Verifies all environment variables
- Detects GPU
- Checks PyTorch version
- Confirms CUDA availability

### test_conv2d_timing.py
- Creates a Conv2d layer
- Measures first run time (kernel compilation)
- Measures subsequent run times (cached kernels)
- Shows speedup after caching

### /etc/profile.d/rocm-rdna1-57.sh
- Auto-detects RDNA1 GPUs
- Sets 15+ environment variables
- Forces GEMM algorithms only
- Forces coarse-grained memory
- Loads automatically on terminal startup

## Archived Files

Historical investigation files have been moved to `docs/archive/`:
- All status files (PROJECT_STATUS.md, FINAL_STATUS.md, etc.)
- Investigation summaries
- Failed approach documentation
- Build logs
- Old test scripts

These are kept for reference but are not part of the working solution.

## Working Solution

The working solution consists of:
1. ROCm 5.7 installation
2. PyTorch 2.2.2+rocm5.7 installation
3. System-wide environment configuration via `/etc/profile.d/rocm-rdna1-57.sh`

No patches or kernel modifications are required.
