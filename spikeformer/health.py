"""Health check endpoints and monitoring for SpikeFormer."""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """Comprehensive health monitoring for SpikeFormer services."""
    
    def __init__(self):
        self.start_time = time.time()
        self.check_registry = {}
        
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.check_registry[name] = check_func
        
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.check_registry:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check '{name}' not found",
                timestamp=time.time(),
                duration_ms=0
            )
        
        start_time = time.time()
        try:
            result = self.check_registry[name]()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.duration_ms = duration_ms
                return result
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    timestamp=time.time(),
                    duration_ms=duration_ms,
                    details=result if isinstance(result, dict) else None
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                timestamp=time.time(),
                duration_ms=duration_ms
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        for name in self.check_registry:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall health status based on all checks."""
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.UNHEALTHY
        
        unhealthy_count = sum(1 for check in results.values() 
                             if check.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for check in results.values() 
                            if check.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


# Default health checks
def check_system_resources() -> HealthCheck:
    """Check system resource availability."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine status based on resource usage
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = "Critical resource usage detected"
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
            status = HealthStatus.DEGRADED
            message = "High resource usage"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources OK"
            
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=time.time(),
            duration_ms=0,
            details=details
        )
    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status=HealthStatus.UNHEALTHY,
            message=f"Failed to check system resources: {e}",
            timestamp=time.time(),
            duration_ms=0
        )


def check_pytorch_cuda() -> HealthCheck:
    """Check PyTorch and CUDA availability."""
    try:
        details = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            details["cuda_version"] = torch.version.cuda
            details["cudnn_version"] = torch.backends.cudnn.version()
            
            # Check GPU memory
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i)
                details[f"gpu_{i}_memory_gb"] = gpu_memory.total_memory / (1024**3)
                details[f"gpu_{i}_name"] = gpu_memory.name
        
        return HealthCheck(
            name="pytorch_cuda",
            status=HealthStatus.HEALTHY,
            message="PyTorch environment OK",
            timestamp=time.time(),
            duration_ms=0,
            details=details
        )
    except Exception as e:
        return HealthCheck(
            name="pytorch_cuda",
            status=HealthStatus.UNHEALTHY,
            message=f"PyTorch check failed: {e}",
            timestamp=time.time(),
            duration_ms=0
        )


def check_neuromorphic_hardware() -> HealthCheck:
    """Check neuromorphic hardware availability."""
    try:
        hardware_status = {}
        
        # Check Loihi 2
        try:
            import nxsdk
            hardware_status["loihi2"] = {
                "available": True,
                "sdk_version": getattr(nxsdk, '__version__', 'unknown')
            }
        except ImportError:
            hardware_status["loihi2"] = {
                "available": False,
                "reason": "NxSDK not installed"
            }
        
        # Check SpiNNaker
        try:
            import spynnaker
            hardware_status["spinnaker"] = {
                "available": True,
                "version": getattr(spynnaker, '__version__', 'unknown')
            }
        except ImportError:
            hardware_status["spinnaker"] = {
                "available": False,
                "reason": "sPyNNaker not installed"
            }
        
        # Determine overall status
        available_hardware = [hw for hw, status in hardware_status.items() 
                             if status.get("available", False)]
        
        if len(available_hardware) >= 1:
            status = HealthStatus.HEALTHY
            message = f"Hardware available: {', '.join(available_hardware)}"
        else:
            status = HealthStatus.DEGRADED
            message = "No neuromorphic hardware available (simulation mode)"
        
        return HealthCheck(
            name="neuromorphic_hardware",
            status=status,
            message=message,
            timestamp=time.time(),
            duration_ms=0,
            details=hardware_status
        )
    except Exception as e:
        return HealthCheck(
            name="neuromorphic_hardware",
            status=HealthStatus.UNHEALTHY,
            message=f"Hardware check failed: {e}",
            timestamp=time.time(),
            duration_ms=0
        )


def check_model_health() -> HealthCheck:
    """Check model loading and basic inference capability."""
    try:
        # Simple model health check
        test_tensor = torch.randn(1, 10)
        simple_model = torch.nn.Linear(10, 5)
        
        # Test forward pass
        with torch.no_grad():
            output = simple_model(test_tensor)
        
        if output.shape == (1, 5):
            return HealthCheck(
                name="model_health",
                status=HealthStatus.HEALTHY,
                message="Model inference OK",
                timestamp=time.time(),
                duration_ms=0,
                details={"test_output_shape": list(output.shape)}
            )
        else:
            return HealthCheck(
                name="model_health",
                status=HealthStatus.UNHEALTHY,
                message="Model inference failed",
                timestamp=time.time(),
                duration_ms=0
            )
    except Exception as e:
        return HealthCheck(
            name="model_health",
            status=HealthStatus.UNHEALTHY,
            message=f"Model health check failed: {e}",
            timestamp=time.time(),
            duration_ms=0
        )


def check_dependencies() -> HealthCheck:
    """Check critical dependencies."""
    try:
        required_packages = [
            "torch", "numpy", "scipy", "matplotlib", "pandas",
            "tqdm", "transformers", "datasets"
        ]
        
        missing_packages = []
        package_versions = {}
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                package_versions[package] = version
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return HealthCheck(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Missing packages: {', '.join(missing_packages)}",
                timestamp=time.time(),
                duration_ms=0,
                details={"missing": missing_packages, "versions": package_versions}
            )
        else:
            return HealthCheck(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All dependencies available",
                timestamp=time.time(),
                duration_ms=0,
                details={"versions": package_versions}
            )
    except Exception as e:
        return HealthCheck(
            name="dependencies",
            status=HealthStatus.UNHEALTHY,
            message=f"Dependency check failed: {e}",
            timestamp=time.time(),
            duration_ms=0
        )


# Global health monitor instance
health_monitor = HealthMonitor()

# Register default checks
health_monitor.register_check("system_resources", check_system_resources)
health_monitor.register_check("pytorch_cuda", check_pytorch_cuda)
health_monitor.register_check("neuromorphic_hardware", check_neuromorphic_hardware)
health_monitor.register_check("model_health", check_model_health)
health_monitor.register_check("dependencies", check_dependencies)


def get_health_summary() -> Dict[str, Any]:
    """Get comprehensive health summary."""
    overall_status = health_monitor.get_overall_health()
    checks = health_monitor.run_all_checks()
    uptime = time.time() - health_monitor.start_time
    
    return {
        "status": overall_status.value,
        "uptime_seconds": uptime,
        "timestamp": time.time(),
        "checks": {
            name: {
                "status": check.status.value,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "details": check.details
            }
            for name, check in checks.items()
        }
    }


def get_readiness_status() -> bool:
    """Check if service is ready to accept requests."""
    critical_checks = ["dependencies", "pytorch_cuda"]
    
    for check_name in critical_checks:
        result = health_monitor.run_check(check_name)
        if result.status == HealthStatus.UNHEALTHY:
            return False
    
    return True


def get_liveness_status() -> bool:
    """Check if service is alive and responding."""
    try:
        # Simple liveness check - can we respond?
        return True
    except Exception:
        return False