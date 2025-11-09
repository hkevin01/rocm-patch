#!/bin/bash
echo "🐳 Testing GPU with ROCm 5.7 in Docker..."
echo ""

docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v /home/kevin/Projects/rocm-patch:/workspace \
    rocm/pytorch:rocm5.7_ubuntu20.04_py3.9_pytorch_2.0.1 \
    python3 -c "
import torch
import torch.nn as nn

print('=' * 70)
print('ROCm 5.7 Docker Test - RDNA1 GPU')
print('=' * 70)

# Check GPU
print(f'GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch version: {torch.__version__}')
    print()
    
    # THE CRITICAL TEST - Conv2d that crashes on ROCm 7.0.2
    print('Testing Conv2d (crashes on ROCm 7.0.2)...')
    try:
        conv = nn.Conv2d(1, 32, (64, 1)).cuda()
        x = torch.randn(16, 1, 64, 256).cuda()
        print('  → Created layer and input')
        
        y = conv(x)
        print('  → Forward pass completed')
        
        y = y.squeeze(2)
        print('  → Squeeze completed')
        
        print()
        print('=' * 70)
        print('✅ SUCCESS! Conv2d works on ROCm 5.7 + RDNA1!')
        print('=' * 70)
        print()
        print('This means:')
        print('  • GPU training is possible')
        print('  • Full speed (not CPU fallback)')
        print('  • Docker provides working ROCm environment')
        
    except Exception as e:
        print(f'❌ FAILED: {e}')
        exit(1)
else:
    print('❌ No GPU available in container')
    print('Check:')
    print('  • --device=/dev/kfd --device=/dev/dri passed to docker')
    print('  • User in video group: sudo usermod -aG video \$USER')
    exit(1)
"
