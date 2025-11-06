#!/usr/bin/env python3
"""
System Information Detection Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detects ROCm version, GPU architecture, OS details, and system configuration.
"""

import os
import re
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System configuration information."""
    
    os_name: str
    os_version: str
    kernel_version: str
    rocm_version: Optional[str]
    rocm_path: Optional[Path]
    gpu_count: int
    gpu_architectures: List[str]
    gpu_names: List[str]
    has_rocm: bool
    has_hip: bool
    python_version: str
    
    @classmethod
    def detect(cls) -> 'SystemInfo':
        """
        Detect system configuration.
        
        Returns:
            SystemInfo: Detected system information
            
        Raises:
            RuntimeError: If critical detection fails
        """
        try:
            # Detect OS information
            os_name, os_version = cls._detect_os()
            kernel_version = cls._detect_kernel()
            
            # Detect ROCm installation
            rocm_version, rocm_path = cls._detect_rocm()
            has_rocm = rocm_version is not None
            has_hip = cls._detect_hip()
            
            # Detect GPU information
            gpu_count, gpu_architectures, gpu_names = cls._detect_gpus()
            
            # Python version
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            return cls(
                os_name=os_name,
                os_version=os_version,
                kernel_version=kernel_version,
                rocm_version=rocm_version,
                rocm_path=rocm_path,
                gpu_count=gpu_count,
                gpu_architectures=gpu_architectures,
                gpu_names=gpu_names,
                has_rocm=has_rocm,
                has_hip=has_hip,
                python_version=python_version,
            )
        except Exception as e:
            logger.error(f"Failed to detect system information: {e}")
            raise RuntimeError(f"System detection failed: {e}")
    
    @staticmethod
    def _detect_os() -> tuple[str, str]:
        """Detect OS name and version."""
        try:
            import distro
            return distro.name(), distro.version()
        except ImportError:
            # Fallback to reading /etc/os-release
            try:
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    os_info = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            os_info[key] = value.strip('"')
                    return os_info.get('NAME', 'Unknown'), os_info.get('VERSION_ID', 'Unknown')
            except:
                return 'Unknown', 'Unknown'
    
    @staticmethod
    def _detect_kernel() -> str:
        """Detect kernel version."""
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True, timeout=5)
            return result.stdout.strip()
        except:
            return 'Unknown'
    
    @staticmethod
    def _detect_rocm() -> tuple[Optional[str], Optional[Path]]:
        """Detect ROCm installation and version."""
        # Check common ROCm paths
        rocm_paths = [
            Path('/opt/rocm'),
            Path('/opt/rocm-7.1.0'),
            Path('/opt/rocm-7.0.0'),
            Path('/opt/rocm-6.4.0'),
            Path('/opt/rocm-5.7.0'),
        ]
        
        for path in rocm_paths:
            if path.exists():
                # Try to get version from .info directory
                version_file = path / '.info' / 'version'
                if version_file.exists():
                    try:
                        with open(version_file, 'r') as f:
                            version = f.read().strip()
                            return version, path
                    except:
                        pass
                
                # Try to extract from path name
                version_match = re.search(r'rocm-(\d+\.\d+\.\d+)', str(path))
                if version_match:
                    return version_match.group(1), path
                
                # If /opt/rocm (symlink), try to resolve
                if path == Path('/opt/rocm'):
                    try:
                        resolved = path.resolve()
                        version_match = re.search(r'rocm-(\d+\.\d+\.\d+)', str(resolved))
                        if version_match:
                            return version_match.group(1), path
                    except:
                        pass
                
                # Return with unknown version but valid path
                return 'Unknown', path
        
        return None, None
    
    @staticmethod
    def _detect_hip() -> bool:
        """Check if HIP is available."""
        try:
            result = subprocess.run(['hipconfig', '--version'], 
                                    capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def _detect_gpus() -> tuple[int, List[str], List[str]]:
        """
        Detect GPU information using rocminfo.
        
        Returns:
            Tuple of (count, architectures, names)
        """
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return 0, [], []
            
            output = result.stdout
            
            # Parse GPU information
            gpu_architectures = []
            gpu_names = []
            
            # Look for gfx architecture
            gfx_matches = re.findall(r'gfx(\d+)', output)
            for gfx in gfx_matches:
                arch = f"gfx{gfx}"
                if arch not in gpu_architectures:
                    gpu_architectures.append(arch)
            
            # Look for GPU names
            name_matches = re.findall(r'Name:\s+(.+)', output)
            for name in name_matches:
                name = name.strip()
                if 'GPU' in name.upper() or 'Radeon' in name or 'Instinct' in name:
                    if name not in gpu_names:
                        gpu_names.append(name)
            
            gpu_count = len(gpu_architectures)
            
            return gpu_count, gpu_architectures, gpu_names
            
        except Exception as e:
            logger.warning(f"Failed to detect GPUs: {e}")
            return 0, [], []
    
    def __str__(self) -> str:
        """String representation of system info."""
        lines = [
            "System Information",
            "=" * 50,
            f"OS: {self.os_name} {self.os_version}",
            f"Kernel: {self.kernel_version}",
            f"Python: {self.python_version}",
            "",
            "ROCm Information",
            "-" * 50,
            f"ROCm Installed: {self.has_rocm}",
            f"ROCm Version: {self.rocm_version or 'Not found'}",
            f"ROCm Path: {self.rocm_path or 'Not found'}",
            f"HIP Available: {self.has_hip}",
            "",
            "GPU Information",
            "-" * 50,
            f"GPU Count: {self.gpu_count}",
            f"Architectures: {', '.join(self.gpu_architectures) or 'None detected'}",
            f"GPU Names: {', '.join(self.gpu_names) or 'None detected'}",
        ]
        return '\n'.join(lines)


def main():
    """CLI entry point for system info detection."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        sys_info = SystemInfo.detect()
        print(sys_info)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
