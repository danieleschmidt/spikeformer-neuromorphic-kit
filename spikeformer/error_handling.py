"""Comprehensive error handling and recovery mechanisms for neuromorphic computing."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import traceback
from contextlib import contextmanager
import functools
import warnings
from enum import Enum
import time
import psutil
import gc


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SpikeformerError:
    """Structured error information."""
    error_type: str
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = None


class ErrorHandler:
    """Central error handling and logging system."""
    
    def __init__(self, log_level: int = logging.INFO, max_errors: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.max_errors = max_errors
        self.error_history: List[SpikeformerError] = []
        self.error_counts = {}
        
    def log_error(self, error: SpikeformerError) -> None:
        """Log an error with appropriate severity."""
        # Add to history
        self.error_history.append(error)
        if len(self.error_history) > self.max_errors:
            self.error_history.pop(0)
        
        # Update counts
        error_key = f"{error.error_type}:{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log based on severity
        log_message = f"[{error.error_type}] {error.message}"
        if error.context:
            log_message += f" Context: {error.context}"
        
        if error.severity == ErrorSeverity.INFO:
            self.logger.info(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
            if error.stack_trace:
                self.logger.error(f"Stack trace: {error.stack_trace}")
        elif error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            if error.stack_trace:
                self.logger.critical(f"Stack trace: {error.stack_trace}")
    
    def create_error(self, error_type: str, severity: ErrorSeverity, 
                    message: str, context: Dict[str, Any] = None,
                    include_stack: bool = True) -> SpikeformerError:
        """Create a structured error."""
        return SpikeformerError(
            error_type=error_type,
            severity=severity,
            message=message,
            context=context or {},
            timestamp=time.time(),
            stack_trace=traceback.format_exc() if include_stack else None,
            recovery_suggestions=self._get_recovery_suggestions(error_type)
        )
    
    def _get_recovery_suggestions(self, error_type: str) -> List[str]:
        """Get recovery suggestions for common errors."""
        suggestions = {
            "memory_overflow": [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use smaller model architecture",
                "Clear GPU cache with torch.cuda.empty_cache()"
            ],
            "dimension_mismatch": [
                "Check input tensor shapes",
                "Verify model configuration",
                "Ensure timesteps parameter consistency"
            ],
            "convergence_failure": [
                "Reduce learning rate",
                "Check gradient clipping",
                "Verify loss function implementation",
                "Increase regularization"
            ],
            "hardware_compatibility": [
                "Check hardware specifications",
                "Update drivers",
                "Verify backend installation",
                "Use simulation mode for testing"
            ]
        }
        return suggestions.get(error_type, [])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error patterns."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }


# Global error handler
error_handler = ErrorHandler()


def safe_execute(operation_name: str, error_type: str = "general",
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                default_return: Any = None):
    """Decorator for safe execution with error handling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "args_types": [type(arg).__name__ for arg in args],
                    "kwargs_keys": list(kwargs.keys()),
                    "exception_type": type(e).__name__
                }
                
                error = error_handler.create_error(
                    error_type=error_type,
                    severity=severity,
                    message=f"{operation_name} failed: {str(e)}",
                    context=context
                )
                error_handler.log_error(error)
                
                if severity == ErrorSeverity.CRITICAL:
                    raise
                return default_return
        return wrapper
    return decorator


@contextmanager
def error_context(operation_name: str, error_type: str = "general"):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        context = {
            "operation": operation_name,
            "exception_type": type(e).__name__
        }
        
        error = error_handler.create_error(
            error_type=error_type,
            severity=ErrorSeverity.ERROR,
            message=f"{operation_name} failed: {str(e)}",
            context=context
        )
        error_handler.log_error(error)
        raise


class ResourceMonitor:
    """Monitor system resources and handle resource-related errors."""
    
    def __init__(self, memory_threshold: float = 0.9, cpu_threshold: float = 0.95):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.logger = logging.getLogger(__name__)
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        resources = {
            "memory_percent": memory_info.percent / 100.0,
            "memory_available": memory_info.available,
            "cpu_percent": cpu_percent / 100.0,
            "gpu_memory": self._get_gpu_memory() if torch.cuda.is_available() else None
        }
        
        return resources
    
    def _get_gpu_memory(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "percent_used": allocated / total
        }
    
    @contextmanager
    def resource_guard(self):
        """Context manager for resource monitoring."""
        initial_resources = self.check_resources()
        
        try:
            yield initial_resources
        finally:
            final_resources = self.check_resources()
            self._check_resource_leaks(initial_resources, final_resources)
    
    def _check_resource_leaks(self, initial: Dict, final: Dict):
        """Check for potential resource leaks."""
        if final["memory_percent"] - initial["memory_percent"] > 0.1:
            error = error_handler.create_error(
                error_type="memory_leak",
                severity=ErrorSeverity.WARNING,
                message=f"Memory usage increased significantly: {initial['memory_percent']:.2f} -> {final['memory_percent']:.2f}",
                context={"initial": initial, "final": final}
            )
            error_handler.log_error(error)
    
    def auto_cleanup(self):
        """Automatic cleanup when resources are constrained."""
        resources = self.check_resources()
        
        if resources["memory_percent"] > self.memory_threshold:
            self.logger.warning(f"High memory usage ({resources['memory_percent']:.1%}), triggering cleanup")
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if resources["cpu_percent"] > self.cpu_threshold:
            self.logger.warning(f"High CPU usage ({resources['cpu_percent']:.1%})")


class GradientValidator:
    """Validate gradients during training to detect issues early."""
    
    def __init__(self, nan_threshold: float = 1e-8, explosion_threshold: float = 1e8):
        self.nan_threshold = nan_threshold
        self.explosion_threshold = explosion_threshold
    
    def validate_gradients(self, model: nn.Module) -> List[SpikeformerError]:
        """Validate gradients for common issues."""
        errors = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Check for NaN gradients
                if torch.isnan(grad).any():
                    error = error_handler.create_error(
                        error_type="gradient_nan",
                        severity=ErrorSeverity.ERROR,
                        message=f"NaN gradients detected in parameter {name}",
                        context={"parameter": name, "shape": grad.shape}
                    )
                    errors.append(error)
                
                # Check for infinite gradients
                if torch.isinf(grad).any():
                    error = error_handler.create_error(
                        error_type="gradient_inf",
                        severity=ErrorSeverity.ERROR,
                        message=f"Infinite gradients detected in parameter {name}",
                        context={"parameter": name, "shape": grad.shape}
                    )
                    errors.append(error)
                
                # Check for gradient explosion
                grad_norm = grad.norm().item()
                if grad_norm > self.explosion_threshold:
                    error = error_handler.create_error(
                        error_type="gradient_explosion",
                        severity=ErrorSeverity.WARNING,
                        message=f"Large gradient norm in parameter {name}: {grad_norm}",
                        context={"parameter": name, "gradient_norm": grad_norm}
                    )
                    errors.append(error)
                
                # Check for vanishing gradients
                elif grad_norm < self.nan_threshold:
                    error = error_handler.create_error(
                        error_type="gradient_vanishing",
                        severity=ErrorSeverity.WARNING,
                        message=f"Very small gradient norm in parameter {name}: {grad_norm}",
                        context={"parameter": name, "gradient_norm": grad_norm}
                    )
                    errors.append(error)
        
        return errors


class RobustSpikingLayer(nn.Module):
    """Robust spiking layer with built-in error handling."""
    
    def __init__(self, base_layer: nn.Module, max_retries: int = 3,
                 fallback_threshold: float = 0.5):
        super().__init__()
        self.base_layer = base_layer
        self.max_retries = max_retries
        self.fallback_threshold = fallback_threshold
        self.error_count = 0
    
    @safe_execute("spiking_forward", "layer_computation")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with error recovery."""
        for attempt in range(self.max_retries):
            try:
                return self.base_layer(x)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle GPU memory issues
                    torch.cuda.empty_cache()
                    if attempt < self.max_retries - 1:
                        continue
                
                if attempt == self.max_retries - 1:
                    # Final fallback: return zeros with same shape
                    self.error_count += 1
                    return torch.zeros_like(x)
                
        return torch.zeros_like(x)


class ModelCheckpointing:
    """Robust model checkpointing with error recovery."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)
        
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    @safe_execute("save_checkpoint", "checkpointing")
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metadata: Dict = None) -> str:
        """Save model checkpoint with error handling."""
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }
        
        # Save with temporary file to avoid corruption
        temp_path = checkpoint_path + ".tmp"
        torch.save(checkpoint, temp_path)
        
        # Atomic move
        import os
        os.rename(temp_path, checkpoint_path)
        
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    @safe_execute("load_checkpoint", "checkpointing")
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: torch.optim.Optimizer = None) -> Dict:
        """Load model checkpoint with validation."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'epoch', 'loss']
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Checkpoint missing required key: {key}")
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return checkpoint
            
        except Exception as e:
            error = error_handler.create_error(
                error_type="checkpoint_load_failure",
                severity=ErrorSeverity.ERROR,
                message=f"Failed to load checkpoint {checkpoint_path}: {str(e)}",
                context={"checkpoint_path": checkpoint_path}
            )
            error_handler.log_error(error)
            raise
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        import os
        import glob
        
        checkpoints = glob.glob(f"{self.checkpoint_dir}/checkpoint_epoch_*.pt")
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=os.path.getmtime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                try:
                    os.remove(checkpoint)
                    self.logger.info(f"Removed old checkpoint: {checkpoint}")
                except OSError:
                    pass


def setup_error_handling(log_level: int = logging.INFO,
                        enable_resource_monitoring: bool = True) -> Tuple[ErrorHandler, ResourceMonitor]:
    """Setup comprehensive error handling system."""
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('spikeformer_errors.log'),
            logging.StreamHandler()
        ]
    )
    
    # Setup error handler
    global error_handler
    error_handler = ErrorHandler(log_level=log_level)
    
    # Setup resource monitor
    resource_monitor = None
    if enable_resource_monitoring:
        resource_monitor = ResourceMonitor()
    
    return error_handler, resource_monitor


# Exception classes for specific errors
class NeuromorphicError(Exception):
    """Base exception for neuromorphic computing errors."""
    pass


class HardwareCompatibilityError(NeuromorphicError):
    """Raised when hardware compatibility issues are detected."""
    pass


class ConversionError(NeuromorphicError):
    """Raised when model conversion fails."""
    pass


class SpikingDynamicsError(NeuromorphicError):
    """Raised when spiking dynamics are invalid."""
    pass


class EnergyProfilingError(NeuromorphicError):
    """Raised when energy profiling fails."""
    pass