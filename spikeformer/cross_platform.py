"""Cross-platform compatibility layer for neuromorphic computing systems."""

import os
import sys
import platform
import subprocess
import shutil
import tempfile
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
import json


class OperatingSystem(Enum):
    """Supported operating systems."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "darwin"
    FREEBSD = "freebsd"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


class PlatformCapability(Enum):
    """Platform-specific capabilities."""
    CUDA_GPU = "cuda_gpu"
    OPENCL = "opencl"
    NEURAL_PROCESSING_UNIT = "npu"
    MULTIPROCESSING = "multiprocessing"
    SHARED_MEMORY = "shared_memory"
    HIGH_RESOLUTION_TIMER = "high_res_timer"
    LARGE_PAGES = "large_pages"
    NUMA = "numa"
    AVX_INSTRUCTIONS = "avx"
    NEON_INSTRUCTIONS = "neon"


@dataclass
class PlatformInfo:
    """Comprehensive platform information."""
    os: OperatingSystem
    architecture: Architecture
    python_version: str
    cpu_cores: int
    memory_gb: float
    capabilities: List[PlatformCapability]
    environment_variables: Dict[str, str]
    library_versions: Dict[str, str]


class PlatformDetector:
    """Detects and analyzes current platform capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cached_info: Optional[PlatformInfo] = None
    
    def detect_platform(self) -> PlatformInfo:
        """Detect comprehensive platform information."""
        if self._cached_info is None:
            self._cached_info = self._analyze_platform()
        return self._cached_info
    
    def _analyze_platform(self) -> PlatformInfo:
        """Analyze current platform comprehensively."""
        # Detect OS
        system = platform.system().lower()
        if system == "windows":
            os_type = OperatingSystem.WINDOWS
        elif system == "linux":
            os_type = OperatingSystem.LINUX
        elif system == "darwin":
            os_type = OperatingSystem.MACOS
        elif system == "freebsd":
            os_type = OperatingSystem.FREEBSD
        else:
            os_type = OperatingSystem.UNKNOWN
        
        # Detect architecture
        machine = platform.machine().lower()
        if machine in ["x86_64", "amd64"]:
            arch = Architecture.X86_64
        elif machine in ["arm64", "aarch64"]:
            arch = Architecture.ARM64
        elif machine.startswith("arm"):
            arch = Architecture.ARM32
        else:
            arch = Architecture.UNKNOWN
        
        # Get system resources
        try:
            cpu_cores = os.cpu_count() or 1
        except:
            cpu_cores = 1
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback estimation
            memory_gb = 8.0  # Conservative estimate
        
        # Detect capabilities
        capabilities = self._detect_capabilities(os_type, arch)
        
        # Get environment variables relevant to ML/AI
        env_vars = self._get_relevant_env_vars()
        
        # Get library versions
        library_versions = self._get_library_versions()
        
        return PlatformInfo(
            os=os_type,
            architecture=arch,
            python_version=sys.version,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            capabilities=capabilities,
            environment_variables=env_vars,
            library_versions=library_versions
        )
    
    def _detect_capabilities(self, os_type: OperatingSystem, arch: Architecture) -> List[PlatformCapability]:
        """Detect platform-specific capabilities."""
        capabilities = []
        
        # CUDA GPU detection
        if self._has_cuda():
            capabilities.append(PlatformCapability.CUDA_GPU)
        
        # OpenCL detection
        if self._has_opencl():
            capabilities.append(PlatformCapability.OPENCL)
        
        # Multiprocessing (available on most platforms)
        capabilities.append(PlatformCapability.MULTIPROCESSING)
        
        # Shared memory
        if os_type != OperatingSystem.WINDOWS:  # More reliable on Unix-like systems
            capabilities.append(PlatformCapability.SHARED_MEMORY)
        
        # High-resolution timer
        if hasattr(os, 'times') or os_type == OperatingSystem.WINDOWS:
            capabilities.append(PlatformCapability.HIGH_RESOLUTION_TIMER)
        
        # Large pages support
        if os_type == OperatingSystem.LINUX:
            if self._check_large_pages_support():
                capabilities.append(PlatformCapability.LARGE_PAGES)
        
        # NUMA support
        if self._has_numa():
            capabilities.append(PlatformCapability.NUMA)
        
        # CPU instruction sets
        if arch == Architecture.X86_64:
            if self._has_avx():
                capabilities.append(PlatformCapability.AVX_INSTRUCTIONS)
        elif arch in [Architecture.ARM64, Architecture.ARM32]:
            if self._has_neon():
                capabilities.append(PlatformCapability.NEON_INSTRUCTIONS)
        
        return capabilities
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                # Check for nvidia-smi
                subprocess.run(['nvidia-smi'], capture_output=True, check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    def _has_opencl(self) -> bool:
        """Check if OpenCL is available."""
        try:
            import pyopencl
            platforms = pyopencl.get_platforms()
            return len(platforms) > 0
        except ImportError:
            return False
    
    def _check_large_pages_support(self) -> bool:
        """Check for large pages support on Linux."""
        try:
            with open('/proc/meminfo', 'r') as f:
                content = f.read()
                return 'HugePages_Total' in content
        except:
            return False
    
    def _has_numa(self) -> bool:
        """Check for NUMA support."""
        try:
            return shutil.which('numactl') is not None
        except:
            return False
    
    def _has_avx(self) -> bool:
        """Check for AVX instruction support."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            return 'avx' in info.get('flags', [])
        except ImportError:
            # Fallback: check /proc/cpuinfo on Linux
            if platform.system().lower() == 'linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        content = f.read()
                        return 'avx' in content
                except:
                    pass
            return False
    
    def _has_neon(self) -> bool:
        """Check for NEON instruction support."""
        if platform.system().lower() == 'linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    return 'neon' in content.lower()
            except:
                pass
        return False
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get ML/AI relevant environment variables."""
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'PYTORCH_CUDA_ALLOC_CONF',
            'TORCH_HOME',
            'HF_HOME',
            'TRANSFORMERS_CACHE',
            'NEUROMORPHIC_HARDWARE_PATH',
            'LOIHI_GEN',
            'SPINNAKER_HOST'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                env_vars[var] = value
        
        return env_vars
    
    def _get_library_versions(self) -> Dict[str, str]:
        """Get versions of important libraries."""
        libraries = {}
        
        # Core libraries
        try:
            import torch
            libraries['torch'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import numpy
            libraries['numpy'] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            libraries['scipy'] = scipy.__version__
        except ImportError:
            pass
        
        # Optional neuromorphic libraries
        try:
            import snntorch
            libraries['snntorch'] = snntorch.__version__
        except ImportError:
            pass
        
        try:
            import norse
            libraries['norse'] = norse.__version__
        except ImportError:
            pass
        
        return libraries


class PathManager:
    """Cross-platform path management."""
    
    @staticmethod
    def get_user_data_dir(app_name: str = "spikeformer") -> Path:
        """Get user data directory across platforms."""
        system = platform.system().lower()
        
        if system == "windows":
            base_dir = Path(os.environ.get('APPDATA', Path.home()))
        elif system == "darwin":  # macOS
            base_dir = Path.home() / "Library" / "Application Support"
        else:  # Linux and others
            base_dir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
        
        data_dir = base_dir / app_name
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @staticmethod
    def get_cache_dir(app_name: str = "spikeformer") -> Path:
        """Get cache directory across platforms."""
        system = platform.system().lower()
        
        if system == "windows":
            base_dir = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        elif system == "darwin":  # macOS
            base_dir = Path.home() / "Library" / "Caches"
        else:  # Linux and others
            base_dir = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
        
        cache_dir = base_dir / app_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    @staticmethod
    def get_temp_dir(app_name: str = "spikeformer") -> Path:
        """Get temporary directory."""
        temp_base = Path(tempfile.gettempdir())
        temp_dir = temp_base / app_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """Normalize path for current platform."""
        path = Path(path)
        return path.resolve()
    
    @staticmethod
    def make_executable(file_path: Path):
        """Make file executable on Unix-like systems."""
        if platform.system().lower() != "windows":
            try:
                current_mode = file_path.stat().st_mode
                file_path.chmod(current_mode | 0o755)
            except:
                pass


class ProcessManager:
    """Cross-platform process management."""
    
    @staticmethod
    def get_cpu_count() -> int:
        """Get number of CPU cores."""
        return os.cpu_count() or 1
    
    @staticmethod
    def set_thread_affinity(cpu_ids: List[int]) -> bool:
        """Set thread CPU affinity if supported."""
        try:
            if platform.system().lower() == "linux":
                import os
                pid = os.getpid()
                cpu_mask = sum(1 << cpu_id for cpu_id in cpu_ids)
                os.sched_setaffinity(pid, cpu_ids)
                return True
        except:
            pass
        
        return False
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except ImportError:
            return {
                'total_gb': 8.0,  # Conservative estimate
                'available_gb': 4.0,
                'used_gb': 4.0,
                'percent': 50.0
            }
    
    @staticmethod
    def run_command(cmd: List[str], timeout: Optional[int] = None, 
                   working_dir: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run command cross-platform."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", "Command not found"
        except Exception as e:
            return -1, "", str(e)


class LibraryCompatibility:
    """Handle library compatibility across platforms."""
    
    @staticmethod
    def import_with_fallback(primary: str, fallback: Optional[str] = None) -> Any:
        """Import library with fallback options."""
        try:
            return __import__(primary)
        except ImportError:
            if fallback:
                try:
                    return __import__(fallback)
                except ImportError:
                    pass
            return None
    
    @staticmethod
    def get_torch_device(prefer_gpu: bool = True) -> str:
        """Get appropriate PyTorch device."""
        try:
            import torch
            
            if prefer_gpu and torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon GPU
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    @staticmethod
    def optimize_for_platform(config: Dict[str, Any], platform_info: PlatformInfo) -> Dict[str, Any]:
        """Optimize configuration for current platform."""
        optimized_config = config.copy()
        
        # CPU optimization
        if PlatformCapability.AVX_INSTRUCTIONS in platform_info.capabilities:
            optimized_config['use_avx'] = True
        
        if PlatformCapability.NEON_INSTRUCTIONS in platform_info.capabilities:
            optimized_config['use_neon'] = True
        
        # Memory optimization
        if platform_info.memory_gb < 8:
            optimized_config['batch_size'] = min(optimized_config.get('batch_size', 32), 16)
            optimized_config['enable_gradient_checkpointing'] = True
        
        # Threading optimization
        max_threads = min(platform_info.cpu_cores, 16)  # Reasonable upper limit
        optimized_config['num_workers'] = max_threads // 2
        optimized_config['num_threads'] = max_threads
        
        # GPU optimization
        if PlatformCapability.CUDA_GPU in platform_info.capabilities:
            optimized_config['device'] = 'cuda'
            optimized_config['pin_memory'] = True
        else:
            optimized_config['device'] = 'cpu'
            optimized_config['pin_memory'] = False
        
        # Platform-specific optimizations
        if platform_info.os == OperatingSystem.WINDOWS:
            # Windows-specific optimizations
            optimized_config['persistent_workers'] = False  # Can cause issues on Windows
        elif platform_info.os == OperatingSystem.MACOS:
            # macOS-specific optimizations
            if platform_info.architecture == Architecture.ARM64:
                optimized_config['device'] = LibraryCompatibility.get_torch_device()
        
        return optimized_config


class CrossPlatformManager:
    """Main cross-platform compatibility manager."""
    
    def __init__(self):
        self.detector = PlatformDetector()
        self.platform_info = self.detector.detect_platform()
        self.logger = logging.getLogger(__name__)
        
        # Initialize platform-specific settings
        self._setup_platform()
    
    def _setup_platform(self):
        """Setup platform-specific configurations."""
        # Set appropriate number of threads for numerical libraries
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(min(self.platform_info.cpu_cores, 8))
        
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = str(min(self.platform_info.cpu_cores, 8))
        
        # Platform-specific memory optimizations
        if self.platform_info.os == OperatingSystem.LINUX:
            if PlatformCapability.LARGE_PAGES in self.platform_info.capabilities:
                # Could enable large pages if supported
                pass
        
        self.logger.info(f"Platform setup complete: {self.platform_info.os.value} {self.platform_info.architecture.value}")
    
    def get_platform_info(self) -> PlatformInfo:
        """Get platform information."""
        return self.platform_info
    
    def get_optimal_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get platform-optimized configuration."""
        return LibraryCompatibility.optimize_for_platform(base_config, self.platform_info)
    
    def check_hardware_compatibility(self, hardware_requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if platform meets hardware requirements."""
        issues = []
        
        # Check minimum memory
        min_memory = hardware_requirements.get('min_memory_gb', 4)
        if self.platform_info.memory_gb < min_memory:
            issues.append(f"Insufficient memory: {self.platform_info.memory_gb:.1f} GB < {min_memory} GB")
        
        # Check CPU cores
        min_cores = hardware_requirements.get('min_cpu_cores', 2)
        if self.platform_info.cpu_cores < min_cores:
            issues.append(f"Insufficient CPU cores: {self.platform_info.cpu_cores} < {min_cores}")
        
        # Check required capabilities
        required_capabilities = hardware_requirements.get('required_capabilities', [])
        for cap_name in required_capabilities:
            try:
                capability = PlatformCapability(cap_name)
                if capability not in self.platform_info.capabilities:
                    issues.append(f"Missing required capability: {cap_name}")
            except ValueError:
                issues.append(f"Unknown capability requirement: {cap_name}")
        
        return len(issues) == 0, issues
    
    def save_platform_report(self, output_path: Optional[Path] = None) -> Path:
        """Save detailed platform compatibility report."""
        if output_path is None:
            output_path = PathManager.get_user_data_dir() / "platform_report.json"
        
        report = {
            "timestamp": str(datetime.now()),
            "platform_info": {
                "os": self.platform_info.os.value,
                "architecture": self.platform_info.architecture.value,
                "python_version": self.platform_info.python_version,
                "cpu_cores": self.platform_info.cpu_cores,
                "memory_gb": self.platform_info.memory_gb,
                "capabilities": [cap.value for cap in self.platform_info.capabilities],
                "environment_variables": self.platform_info.environment_variables,
                "library_versions": self.platform_info.library_versions
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Platform report saved to: {output_path}")
        return output_path


# Global cross-platform manager instance
_platform_manager: Optional[CrossPlatformManager] = None
_manager_lock = threading.Lock()


def get_platform_manager() -> CrossPlatformManager:
    """Get global cross-platform manager instance."""
    global _platform_manager
    
    if _platform_manager is None:
        with _manager_lock:
            if _platform_manager is None:
                _platform_manager = CrossPlatformManager()
    
    return _platform_manager


def get_platform_info() -> PlatformInfo:
    """Get current platform information."""
    return get_platform_manager().get_platform_info()


def optimize_config_for_platform(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize configuration for current platform."""
    return get_platform_manager().get_optimal_config(config)


def check_platform_compatibility(requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check platform compatibility against requirements."""
    return get_platform_manager().check_hardware_compatibility(requirements)


# Import datetime for report timestamp
from datetime import datetime