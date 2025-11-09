#!/bin/bash

# AMD RX 5600 XT (RDNA1) PyTorch Compatibility Script
# This script sets up the environment for PyTorch on unsupported RDNA1 GPUs

echo "====================================="
echo "RDNA1 PyTorch Compatibility Wrapper"
echo "====================================="
echo ""
echo "GPU: AMD Radeon RX 5600 XT (gfx1010)"
echo "Applying workarounds..."
echo ""

# Export environment variables
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_LOG_LEVEL=3

# Check if Python script is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script.py> [args...]"
    echo ""
    echo "Example: $0 train_mnist.py"
    exit 1
fi

# Run the Python script
echo "Running: python3 $@"
echo ""
exec python3 "$@"
