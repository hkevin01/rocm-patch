#!/usr/bin/env python3
import os
import sys

# Try different combination to disable Find
os.environ['MIOPEN_FIND_ENFORCE'] = 'SEARCH_DB_UPDATE'  # Forces DB use only
os.environ['MIOPEN_FIND_MODE'] = '2'  # Hybrid mode
os.environ['MIOPEN_DEBUG_DISABLE_FIND_DB'] = '0'  # Enable DB
os.environ['MIOPEN_DEVICE_ARCH'] = 'gfx1030'  # Pretend to be supported arch
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm-miopen-rdna1/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

print("Trying to bypass Find by pretending to be gfx1030...")
print()

import torch
import torch.nn as nn

print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    model = nn.Conv2d(1, 2, 3, padding=1).cuda()
    x = torch.randn(1, 1, 8, 8).cuda()
    
    print("Running Conv2d...")
    y = model(x)
    
    print(f"✅ SUCCESS! Output: {y.shape}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)
