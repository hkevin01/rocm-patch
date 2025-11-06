#!/usr/bin/env python3
"""
ROCm Patch Repository - Setup Configuration
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="rocm-patch",
    version="0.1.0",
    author="ROCm Patch Community",
    author_email="",
    description="Comprehensive patch collection for AMD ROCm platform issues",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/rocm-patch",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/rocm-patch/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/rocm-patch/docs",
        "Source Code": "https://github.com/YOUR_USERNAME/rocm-patch",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Hardware",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "Sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rocm-patch=rocm_patch.cli:main",
            "rocm-patch-manager=rocm_patch.patch_manager:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rocm_patch": [
            "patches/**/*",
            "configs/**/*",
        ],
    },
    zip_safe=False,
    keywords="rocm amd gpu patch fix hip cuda machine-learning hpc",
)
