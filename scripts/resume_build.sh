#!/bin/bash
echo "========================================================================"
echo "Resuming MIOpen Build"
echo "========================================================================"
cd /tmp/MIOpen/build_rdna1 || exit 1
echo "Build directory: $(pwd)"
echo "Starting build with $(nproc) parallel jobs..."
echo ""
make -j$(nproc) 2>&1 | tee -a /tmp/miopen_build.log
