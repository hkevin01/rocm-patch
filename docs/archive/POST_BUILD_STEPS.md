# Post-Build Steps - MIOpen RDNA1 Patch

## Overview
Once the MIOpen build completes, follow these steps to install and test the patched library.

## Step 1: Verify Build Completion

Check if build finished successfully:
```bash
./scripts/check_build_status.sh
```

Or check manually:
```bash
tail -50 /tmp/miopen_build_final.log | grep -E "(Built target MIOpen|Error|error)"
```

## Step 2: Install Patched MIOpen

Install to `/opt/rocm-miopen-rdna1/`:
```bash
cd /tmp/MIOpen/build_rdna1
sudo make install
```

Verify installation:
```bash
ls -lh /opt/rocm-miopen-rdna1/lib/libMIOpen.so*
```

## Step 3: Test with PyTorch

Run the automated test script:
```bash
cd /home/kevin/Projects/rocm-patch
./scripts/test_patched_miopen.sh
```

This will test:
- Simple Conv2d forward pass
- Backward pass with gradients
- Larger Conv2d operations
- Multi-layer networks

## Step 4: Verify GPU Usage

While test is running, check GPU utilization:
```bash
watch -n 1 rocm-smi --showuse
```

You should see GPU activity during Conv2d operations.

## Step 5: Make Permanent (Optional)

To use the patched MIOpen by default, add to your `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=/opt/rocm-miopen-rdna1/lib:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

## Expected Results

### If Patch Works ✅
```
🎉 ALL TESTS PASSED!

✓ GPU Conv2d operations work correctly
✓ Forward and backward passes successful
✓ RDNA1 patch is working!
```

### If Patch Fails ❌
You might see:
```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
```

This means the patch didn't work as expected, and we'll need to:
1. Check if RDNA1 detection code is triggering
2. Verify the hipHostMallocNonCoherent flag is being used
3. Try alternative approaches

## Troubleshooting

### Check MIOpen Logs
Enable detailed logging:
```bash
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=4
python3 your_test.py 2>&1 | grep -i rdna
```

Look for: "RDNA1 GPU detected" message in the logs.

### Verify Correct Library is Loaded
```bash
ldd $(python3 -c "import torch; print(torch.__file__[:-11]+'lib/libtorch_hip.so')") | grep MIOpen
```

Should show: `/opt/rocm-miopen-rdna1/lib/libMIOpen.so`

### Test with Different Batch Sizes
```python
import torch
import torch.nn as nn

# Small batch
model = nn.Conv2d(1, 32, 3).cuda()
x = torch.randn(1, 1, 28, 28).cuda()
y = model(x)

# Large batch  
x2 = torch.randn(64, 1, 28, 28).cuda()
y2 = model(x2)
```

## Next Steps

If tests pass:
1. Update documentation with success
2. Test with real training workloads
3. Benchmark performance vs CPU
4. Consider submitting patch upstream

If tests fail:
1. Analyze failure mode
2. Check HIP API documentation
3. Try alternative memory allocation methods
4. Investigate deeper in ROCm stack
