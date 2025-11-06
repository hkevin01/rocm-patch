#!/bin/bash
# ROCm Patch Installation Script
# Comprehensive fix for AMD RDNA1/2 GPU issues on ROCm 6.2+

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_LOG="/tmp/rocm-patch-install-$(date +%s).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$INSTALL_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$INSTALL_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$INSTALL_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$INSTALL_LOG"
}

# Print banner
cat << 'BANNER'
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           ROCm Patch Installation Script                    ║
║                                                              ║
║   Fixes for AMD RDNA1/2 GPU Memory Access Faults            ║
║   Compatible with ROCm 6.2, 6.3, 7.0+                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
BANNER

echo ""
log "Installation log: $INSTALL_LOG"
echo ""

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    log_error "Cannot detect OS. /etc/os-release not found."
    exit 1
fi

source /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    log_warning "This script is optimized for Ubuntu. Detected: $ID $VERSION_ID"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log "Detected OS: $PRETTY_NAME"

# Check for AMD GPU
log_info "Checking for AMD GPU..."
if command -v rocminfo &> /dev/null; then
    GPU_INFO=$(rocminfo 2>/dev/null | grep -i "gfx" | head -1)
    if [ -n "$GPU_INFO" ]; then
        log "AMD GPU detected: $GPU_INFO"
    else
        log_warning "ROCm installed but no GPU detected. Continuing anyway..."
    fi
elif command -v lspci &> /dev/null; then
    if lspci | grep -i "VGA.*AMD" > /dev/null; then
        GPU_MODEL=$(lspci | grep -i "VGA.*AMD" | head -1)
        log "AMD GPU detected: $GPU_MODEL"
    else
        log_warning "No AMD GPU detected via lspci"
    fi
else
    log_warning "Cannot detect GPU (no rocminfo or lspci)"
fi

echo ""
log "Select installation type:"
echo "  1) Full installation (kernel + Python patches) [RECOMMENDED]"
echo "  2) Kernel patches only (requires reboot)"
echo "  3) Python patches only (no reboot required)"
echo "  4) Test/verify existing installation"
echo ""
read -p "Enter choice [1-4]: " INSTALL_TYPE

case $INSTALL_TYPE in
    1)
        INSTALL_KERNEL=true
        INSTALL_PYTHON=true
        ;;
    2)
        INSTALL_KERNEL=true
        INSTALL_PYTHON=false
        ;;
    3)
        INSTALL_KERNEL=false
        INSTALL_PYTHON=true
        ;;
    4)
        INSTALL_KERNEL=false
        INSTALL_PYTHON=false
        TEST_ONLY=true
        ;;
    *)
        log_error "Invalid choice"
        exit 1
        ;;
esac

# Install kernel patches
if [ "$INSTALL_KERNEL" = true ]; then
    echo ""
    log "=== Installing Kernel Patches ==="
    
    if [ "$EUID" -ne 0 ]; then
        log_error "Kernel patches require root access. Please run with sudo."
        exit 1
    fi
    
    KERNEL_SCRIPT="$SCRIPT_DIR/src/patches/memory_access_fault/kernel_params.sh"
    
    if [ ! -f "$KERNEL_SCRIPT" ]; then
        log_error "Kernel script not found: $KERNEL_SCRIPT"
        exit 1
    fi
    
    log "Running kernel parameters script..."
    bash "$KERNEL_SCRIPT" | tee -a "$INSTALL_LOG"
    
    log "✅ Kernel patches installed successfully"
    log_warning "REBOOT REQUIRED for kernel patches to take effect"
fi

# Install Python patches
if [ "$INSTALL_PYTHON" = true ]; then
    echo ""
    log "=== Installing Python Package ==="
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.10+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log "Python version: $PYTHON_VERSION"
    
    # Install in development mode
    log "Installing rocm-patch package in development mode..."
    cd "$SCRIPT_DIR"
    
    if python3 -m pip install -e . >> "$INSTALL_LOG" 2>&1; then
        log "✅ Python package installed successfully"
    else
        log_error "Failed to install Python package"
        log_info "See log: $INSTALL_LOG"
        exit 1
    fi
fi

# Test installation
if [ "$TEST_ONLY" = true ] || [ "$INSTALL_PYTHON" = true ]; then
    echo ""
    log "=== Testing Installation ==="
    
    log "Running patch test..."
    
    # Create test script
    TEST_SCRIPT="/tmp/rocm-patch-test-$$.py"
    cat > "$TEST_SCRIPT" << 'PYTEST'
#!/usr/bin/env python3
import sys

try:
    # Test import
    from rocm_patch.patches.memory_access_fault import apply_patch
    print("✓ Import successful")
    
    # Test patch application
    result = apply_patch()
    print("✓ Patch applied successfully")
    
    # Test PyTorch import (if available)
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported")
        
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✓ GPU detected: {device}")
        else:
            print("! No GPU detected (may be normal)")
    except ImportError:
        print("! PyTorch not installed (install for full functionality)")
    
    print("\n✅ All tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTEST
    
    if python3 "$TEST_SCRIPT" 2>&1 | tee -a "$INSTALL_LOG"; then
        log "✅ Tests passed"
    else
        log_error "Tests failed"
        log_info "See log: $INSTALL_LOG"
    fi
    
    rm "$TEST_SCRIPT"
fi

# Print summary
echo ""
log "=== Installation Summary ==="
echo ""

if [ "$INSTALL_KERNEL" = true ]; then
    echo "✅ Kernel patches: INSTALLED"
    echo "   Location: /etc/modprobe.d/amdgpu-fix.conf"
    echo "   Status: REBOOT REQUIRED"
    echo ""
fi

if [ "$INSTALL_PYTHON" = true ]; then
    echo "✅ Python package: INSTALLED"
    echo "   Usage: from rocm_patch.patches.memory_access_fault import apply_patch"
    echo "   Status: READY TO USE"
    echo ""
fi

echo "📚 Documentation: $SCRIPT_DIR/docs/issues/"
echo "📝 Log file: $INSTALL_LOG"
echo ""

# Next steps
if [ "$INSTALL_KERNEL" = true ]; then
    log "=== Next Steps ==="
    echo ""
    echo "1. REBOOT your system:"
    echo "   sudo reboot"
    echo ""
    echo "2. After reboot, verify kernel parameters:"
    echo "   cat /sys/module/amdgpu/parameters/noretry"
    echo "   (should show: 0)"
    echo ""
    echo "3. Test your ML training:"
    echo "   python3 -c 'from rocm_patch.patches.memory_access_fault import apply_patch; apply_patch()'"
    echo ""
else
    log "=== Usage ==="
    echo ""
    echo "Add to the TOP of your training script:"
    echo ""
    echo "  from rocm_patch.patches.memory_access_fault import apply_patch"
    echo "  apply_patch()  # Apply BEFORE importing torch/models"
    echo ""
    echo "Example:"
    echo ""
    cat << 'EXAMPLE'
  #!/usr/bin/env python3
  from rocm_patch.patches.memory_access_fault import apply_patch
  apply_patch()
  
  import torch
  from ultralytics import YOLO
  # ... rest of your code
EXAMPLE
    echo ""
fi

log "Installation complete! 🚀"
