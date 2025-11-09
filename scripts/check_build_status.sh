#!/bin/bash
echo "========================================================================"
echo "MIOpen Build Status Check"
echo "========================================================================"

if ps aux | grep -E "make.*build_rdna1" | grep -v grep > /dev/null; then
    echo "✓ Build is RUNNING"
    echo ""
    echo "Latest progress:"
    tail -n 20 /tmp/miopen_build_final.log | grep -E "^\[" | tail -5
    echo ""
    echo "To monitor live: tail -f /tmp/miopen_build_final.log"
else
    echo "✗ Build is NOT running"
    echo ""
    if [ -f /tmp/miopen_build_final.log ]; then
        echo "Last log entries:"
        tail -n 10 /tmp/miopen_build_final.log
        echo ""
        # Check if build completed successfully
        if tail -20 /tmp/miopen_build_final.log | grep -q "Built target MIOpen"; then
            echo "🎉 BUILD COMPLETED SUCCESSFULLY!"
        elif tail -20 /tmp/miopen_build_final.log | grep -qi "error"; then
            echo "❌ BUILD FAILED - Check log for errors"
        fi
    fi
fi
