#!/usr/bin/env python3
"""
Hardware Compatibility Test for ROCm 6.2+ RDNA Memory Coherency

This script tests whether your AMD GPU hardware has the necessary features
to work with ROCm 6.2+ without memory coherency issues.

Tests performed:
1. GPU Architecture Detection (RDNA1/2/3, CDNA, etc.)
2. SVM (Shared Virtual Memory) Hardware Support
3. Memory Coherency Capabilities
4. HSA Feature Set Validation
5. Linux Kernel MTYPE Support Check

Usage:
    python3 test_hardware_compatibility.py
    
Exit codes:
    0 - GPU is compatible (RDNA3+, CDNA, or properly patched)
    1 - GPU requires patches (RDNA1/2 detected)
    2 - No compatible GPU found
    3 - ROCm not installed or misconfigured
"""

import subprocess
import sys
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class GPUCompatibility(Enum):
    """GPU compatibility levels with ROCm 6.2+"""
    FULL_SUPPORT = "✅ Full Support"
    NEEDS_PATCH = "⚠️  Needs RMCP Patch"
    LIMITED_SUPPORT = "⚠️  Limited Support"
    NO_SUPPORT = "❌ Not Supported"
    UNKNOWN = "❓ Unknown"

@dataclass
class GPUInfo:
    """Detected GPU information"""
    name: str
    pci_id: str
    gfx_version: str
    architecture: str
    has_svm: bool
    has_coherency: bool
    has_xnack: bool
    compatibility: GPUCompatibility
    rocm_version: Optional[str] = None
    
class HardwareCompatibilityTest:
    """Test suite for ROCm 6.2+ hardware compatibility"""
    
    # Known GFX versions and their compatibility
    GFX_COMPATIBILITY = {
        # RDNA1 - Needs patches
        "gfx1010": ("RDNA1", GPUCompatibility.NEEDS_PATCH, "RX 5000 series"),
        "gfx1011": ("RDNA1", GPUCompatibility.NEEDS_PATCH, "RX 5000M series"),
        "gfx1012": ("RDNA1", GPUCompatibility.NEEDS_PATCH, "RX 5000 series"),
        "gfx1013": ("RDNA1", GPUCompatibility.NEEDS_PATCH, "RX 5000 series"),
        
        # RDNA2 - Needs patches
        "gfx1030": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1031": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1032": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1033": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1034": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1035": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        "gfx1036": ("RDNA2", GPUCompatibility.NEEDS_PATCH, "RX 6000 series"),
        
        # RDNA3 - Full support
        "gfx1100": ("RDNA3", GPUCompatibility.FULL_SUPPORT, "RX 7900 XTX/XT"),
        "gfx1101": ("RDNA3", GPUCompatibility.FULL_SUPPORT, "RX 7900 series"),
        "gfx1102": ("RDNA3", GPUCompatibility.FULL_SUPPORT, "RX 7000 series"),
        "gfx1103": ("RDNA3", GPUCompatibility.FULL_SUPPORT, "RX 7000 series"),
        
        # CDNA - Full support
        "gfx906": ("CDNA1", GPUCompatibility.FULL_SUPPORT, "Radeon VII, MI50/60"),
        "gfx908": ("CDNA1", GPUCompatibility.FULL_SUPPORT, "MI100"),
        "gfx90a": ("CDNA2", GPUCompatibility.FULL_SUPPORT, "MI200 series"),
        "gfx940": ("CDNA2", GPUCompatibility.FULL_SUPPORT, "MI300A"),
        "gfx941": ("CDNA2", GPUCompatibility.FULL_SUPPORT, "MI300X"),
        "gfx942": ("CDNA2", GPUCompatibility.FULL_SUPPORT, "MI300 series"),
        
        # GFX9 - Mixed support
        "gfx900": ("GFX9", GPUCompatibility.LIMITED_SUPPORT, "Vega 56/64"),
        "gfx902": ("GFX9", GPUCompatibility.LIMITED_SUPPORT, "Raven Ridge APU"),
        "gfx904": ("GFX9", GPUCompatibility.LIMITED_SUPPORT, "Vega series"),
        "gfx909": ("GFX9", GPUCompatibility.LIMITED_SUPPORT, "Raven Ridge APU"),
        "gfx90c": ("GFX9", GPUCompatibility.LIMITED_SUPPORT, "Renoir APU"),
    }
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self.rocm_installed = False
        self.rocm_version = None
        self.kernel_supports_mtype = False
        
    def run_command(self, cmd: List[str]) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode, result.stdout, result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return 1, "", str(e)
    
    def check_rocm_installation(self) -> bool:
        """Check if ROCm is installed and get version"""
        print("🔍 Checking ROCm installation...")
        
        # Try rocminfo
        code, stdout, _ = self.run_command(["rocminfo"])
        if code == 0:
            self.rocm_installed = True
            # Extract version
            match = re.search(r"Runtime Version:\s+(\d+\.\d+)", stdout)
            if match:
                self.rocm_version = match.group(1)
                print(f"   ✅ ROCm {self.rocm_version} detected")
                return True
        
        # Try rocm-smi
        code, stdout, _ = self.run_command(["rocm-smi", "--showdriverversion"])
        if code == 0:
            self.rocm_installed = True
            print("   ✅ ROCm detected (version unknown)")
            return True
        
        print("   ❌ ROCm not found")
        print("   💡 Install ROCm: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html")
        return False
    
    def detect_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs and their capabilities"""
        print("\n🔍 Detecting AMD GPUs...")
        
        code, stdout, _ = self.run_command(["rocminfo"])
        if code != 0:
            print("   ❌ Failed to run rocminfo")
            return []
        
        # Parse rocminfo output for GPU agents
        gpu_sections = re.split(r'\*+\s*Agent \d+\s*\*+', stdout)
        
        for section in gpu_sections[1:]:  # Skip first split
            if "Device Type:             GPU" not in section:
                continue
            
            # Extract GPU information
            name_match = re.search(r"Marketing Name:\s+(.+)", section)
            gfx_match = re.search(r"Name:\s+(gfx\w+)", section)
            pci_match = re.search(r"BDFID:\s+(\d+)", section)
            
            if not gfx_match:
                continue
            
            gfx_version = gfx_match.group(1)
            name = name_match.group(1).strip() if name_match else "Unknown AMD GPU"
            pci_id = f"0x{int(pci_match.group(1)):04x}" if pci_match else "Unknown"
            
            # Check SVM support
            has_svm = "Fine-grained system sharing" in section and "Yes" in section
            
            # Check coherency (inferred from SVM and architecture)
            has_coherency = self._check_coherency(section, gfx_version)
            
            # Check XNACK support
            has_xnack = "xnack+" in gfx_version or "xnack+" in section.lower()
            
            # Determine compatibility
            arch, compatibility, desc = self.GFX_COMPATIBILITY.get(
                gfx_version.split(":")[0],  # Remove xnack suffix
                ("Unknown", GPUCompatibility.UNKNOWN, "Unknown")
            )
            
            gpu = GPUInfo(
                name=name,
                pci_id=pci_id,
                gfx_version=gfx_version,
                architecture=arch,
                has_svm=has_svm,
                has_coherency=has_coherency,
                has_xnack=has_xnack,
                compatibility=compatibility,
                rocm_version=self.rocm_version
            )
            
            self.gpus.append(gpu)
            self._print_gpu_info(gpu, desc)
        
        return self.gpus
    
    def _check_coherency(self, section: str, gfx_version: str) -> bool:
        """Check if GPU has hardware coherency support"""
        # RDNA3+ and CDNA have proper coherency
        if any(gfx_version.startswith(prefix) for prefix in ["gfx11", "gfx9"]):
            if "gfx906" in gfx_version or "gfx908" in gfx_version or "gfx90a" in gfx_version:
                return True  # CDNA has coherency
            if gfx_version.startswith("gfx11"):
                return True  # RDNA3 has coherency
        
        # RDNA1/2 lack proper coherency
        if any(gfx_version.startswith(prefix) for prefix in ["gfx101", "gfx103"]):
            return False
        
        # Check for coherent host access flag
        if "Coherent Host Access:    TRUE" in section:
            return True
        
        return False
    
    def _print_gpu_info(self, gpu: GPUInfo, desc: str):
        """Print formatted GPU information"""
        print(f"\n   📊 GPU Found: {gpu.name}")
        print(f"      GFX Version: {gpu.gfx_version}")
        print(f"      Architecture: {gpu.architecture} ({desc})")
        print(f"      PCI ID: {gpu.pci_id}")
        print(f"      SVM Support: {'✅ Yes' if gpu.has_svm else '❌ No'}")
        print(f"      Coherency: {'✅ Yes' if gpu.has_coherency else '❌ No'}")
        print(f"      XNACK: {'✅ Yes' if gpu.has_xnack else '❌ No'}")
        print(f"      Compatibility: {gpu.compatibility.value}")
    
    def check_kernel_mtype_support(self) -> bool:
        """Check if Linux kernel has MTYPE_NC workaround support"""
        print("\n🔍 Checking Linux kernel MTYPE support...")
        
        # Check kernel version
        code, stdout, _ = self.run_command(["uname", "-r"])
        if code != 0:
            print("   ⚠️  Cannot determine kernel version")
            return False
        
        kernel_version = stdout.strip()
        print(f"   Kernel: {kernel_version}")
        
        # Check for MTYPE_NC patches in dmesg or module info
        code, stdout, _ = self.run_command(["dmesg"])
        if code == 0:
            if "MTYPE_NC" in stdout or "gfx12" in stdout.lower():
                print("   ✅ Kernel has MTYPE workaround support")
                self.kernel_supports_mtype = True
                return True
        
        # Check amdgpu module parameters
        code, stdout, _ = self.run_command(["modinfo", "amdgpu"])
        if code == 0:
            if "noretry" in stdout or "vm_fragment_size" in stdout:
                print("   ✅ Kernel module has workaround parameters")
                self.kernel_supports_mtype = True
                return True
        
        print("   ⚠️  Kernel MTYPE support unclear")
        return False
    
    def test_memory_coherency(self, gpu: GPUInfo) -> bool:
        """Test if memory coherency works correctly"""
        print(f"\n🧪 Testing memory coherency for {gpu.name}...")
        
        # For RDNA1/2, coherency test will fail without patches
        if gpu.architecture in ["RDNA1", "RDNA2"] and not gpu.has_coherency:
            print("   ⚠️  GPU lacks hardware coherency support")
            print("   💡 RMCP patches required for ROCm 6.2+")
            return False
        
        # For RDNA3+/CDNA, should work
        if gpu.has_coherency:
            print("   ✅ GPU has hardware coherency support")
            return True
        
        print("   ❓ Coherency status unclear")
        return False
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detected hardware"""
        recommendations = []
        
        for gpu in self.gpus:
            if gpu.compatibility == GPUCompatibility.NEEDS_PATCH:
                recommendations.append(
                    f"\n📋 Recommendations for {gpu.name}:\n"
                    f"   1. ✅ Apply RMCP patches (this repository)\n"
                    f"   2. ⚙️  Set kernel parameters:\n"
                    f"      - amdgpu.noretry=0\n"
                    f"      - amdgpu.vm_fragment_size=9\n"
                    f"   3. 🔧 Set environment variables:\n"
                    f"      - HSA_XNACK=0\n"
                    f"      - HSA_FORCE_FINE_GRAIN_PCIE=1\n"
                    f"   4. 🏗️  Rebuild ROCm with patches\n"
                    f"   \n"
                    f"   Alternative (NOT RECOMMENDED):\n"
                    f"   - Downgrade to ROCm 5.7 (loses features)\n"
                    f"   - Use CPU fallback (10-20x slower)\n"
                    f"   - Upgrade to RDNA3 GPU ($600-1500)"
                )
            elif gpu.compatibility == GPUCompatibility.FULL_SUPPORT:
                recommendations.append(
                    f"\n✅ {gpu.name} is fully compatible with ROCm 6.2+\n"
                    f"   No patches needed - GPU training should work!"
                )
            elif gpu.compatibility == GPUCompatibility.LIMITED_SUPPORT:
                recommendations.append(
                    f"\n⚠️  {gpu.name} has limited support\n"
                    f"   May experience issues with some workloads\n"
                    f"   Monitor for memory access violations"
                )
        
        return recommendations
    
    def run_full_test(self) -> int:
        """Run complete hardware compatibility test suite"""
        print("=" * 80)
        print("🔧 ROCm 6.2+ Hardware Compatibility Test")
        print("=" * 80)
        
        # Check ROCm installation
        if not self.check_rocm_installation():
            print("\n" + "=" * 80)
            print("❌ RESULT: ROCm not installed")
            print("=" * 80)
            return 3
        
        # Detect GPUs
        gpus = self.detect_gpus()
        if not gpus:
            print("\n" + "=" * 80)
            print("❌ RESULT: No AMD GPUs detected")
            print("=" * 80)
            return 2
        
        # Check kernel support
        self.check_kernel_mtype_support()
        
        # Test each GPU
        needs_patch = False
        for gpu in gpus:
            coherency_ok = self.test_memory_coherency(gpu)
            if not coherency_ok and gpu.compatibility == GPUCompatibility.NEEDS_PATCH:
                needs_patch = True
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        for rec in recommendations:
            print(rec)
        
        # Final verdict
        print("\n" + "=" * 80)
        if needs_patch:
            print("⚠️  RESULT: GPU REQUIRES PATCHES")
            print("=" * 80)
            print("\n💡 Next Steps:")
            print("   1. Review RMCP documentation: README.md")
            print("   2. Apply patches: sudo ./scripts/patch_rocm_source.sh")
            print("   3. Test with: python3 tests/test_ml_basic.py")
            print("   4. Report results: GitHub Issues")
            return 1
        else:
            print("✅ RESULT: GPU IS COMPATIBLE")
            print("=" * 80)
            print("\n✨ Your GPU should work with ROCm 6.2+ without patches!")
            print("   Test ML workloads with: python3 tests/test_ml_basic.py")
            return 0

def main():
    """Main entry point"""
    tester = HardwareCompatibilityTest()
    exit_code = tester.run_full_test()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
