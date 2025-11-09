from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# ROCm/HIP paths
rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
hip_include = os.path.join(rocm_path, 'include')
hip_lib = os.path.join(rocm_path, 'lib')

setup(
    name='rdna1_conv2d',
    ext_modules=[
        CppExtension(
            name='rdna1_conv2d',
            sources=['rdna1_conv2d.cpp'],
            include_dirs=[hip_include],
            library_dirs=[hip_lib],
            libraries=['amdhip64'],
            extra_compile_args={
                'cxx': ['-O3', '-D__HIP_PLATFORM_AMD__']
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
