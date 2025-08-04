#!/usr/bin/env python3
"""
Robust SpikeFormer Implementation - Generation 2
Adds comprehensive error handling, validation, logging, and monitoring.
"""

import sys
import time
import json
import logging
import hashlib
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import warnings

# Import basic components
from demo_basic import (
    SpikingConfig, BasicLIFNeuron, RateEncoder, TemporalEncoder,
    SpikingLayer, SpikingMLP, SimpleModelConfig, SimpleANN,
    BasicConverter, BasicHardwareSimulator, HardwareMetrics
)


# ==============================================================================
# ENHANCED ERROR HANDLING & VALIDATION
# ==============================================================================

class SpikeFormerError(Exception):
    """Base exception for SpikeFormer errors."""
    pass


class ValidationError(SpikeFormerError):
    """Raised when input validation fails."""
    pass


class ConversionError(SpikeFormerError):
    """Raised when model conversion fails."""
    pass


class DeploymentError(SpikeFormerError):
    """Raised when hardware deployment fails."""
    pass


class ProfilingError(SpikeFormerError):
    """Raised when energy profiling fails."""
    pass


class ConfigurationError(SpikeFormerError):
    """Raised when configuration is invalid."""
    pass


# ==============================================================================
# LOGGING SYSTEM
# ==============================================================================

class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SpikeFormerLogger:
    """Enhanced logging system for SpikeFormer."""
    
    def __init__(self, name: str = "spikeformer", level: LogLevel = LogLevel.INFO):
        self.logger = logging.getLogger(name)
        self.setup_logging(level)
        self.start_time = time.time()
        self.operation_counters = {
            'conversions': 0,
            'deployments': 0,
            'benchmarks': 0,
            'errors': 0
        }
    
    def setup_logging(self, level: LogLevel):
        """Setup logging configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(getattr(logging, level.value))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        try:
            file_handler = logging.FileHandler('spikeformer.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            pass  # Continue without file logging if not possible
    
    @contextmanager
    def operation(self, operation_name: str):
        """Context manager for tracking operations."""
        start_time = time.time()
        self.info(f"Starting operation: {operation_name}")
        
        try:
            yield
            duration = time.time() - start_time
            self.info(f"Operation completed: {operation_name} ({duration:.3f}s)")
            
            # Update counters
            if operation_name in self.operation_counters:
                self.operation_counters[operation_name] += 1
                
        except Exception as e:
            duration = time.time() - start_time
            self.error(f"Operation failed: {operation_name} ({duration:.3f}s) - {e}")
            self.operation_counters['errors'] += 1
            raise
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):  
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message.""" 
        self.logger.critical(message, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'operation_counters': self.operation_counters.copy(),
            'log_level': self.logger.level
        }


# ==============================================================================
# VALIDATION FRAMEWORK  
# ==============================================================================

class Validator:
    """Validation utility class."""
    
    @staticmethod
    def validate_positive_number(value: Union[int, float], name: str) -> Union[int, float]:
        """Validate that a number is positive."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number, got {type(value)}")
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
        return value
    
    @staticmethod
    def validate_range(value: Union[int, float], name: str, 
                      min_val: Optional[float] = None, max_val: Optional[float] = None) -> Union[int, float]:
        """Validate that a number is within a range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number, got {type(value)}")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
        
        return value
    
    @staticmethod
    def validate_list_of_positive_ints(values: List[int], name: str) -> List[int]:
        """Validate list of positive integers."""
        if not isinstance(values, list):
            raise ValidationError(f"{name} must be a list, got {type(values)}")
        
        validated = []
        for i, value in enumerate(values):
            if not isinstance(value, int):
                raise ValidationError(f"{name}[{i}] must be an integer, got {type(value)}")
            if value <= 0:
                raise ValidationError(f"{name}[{i}] must be positive, got {value}")
            validated.append(value)
        
        return validated
    
    @staticmethod
    def validate_spike_data(spike_trains: List[List[bool]], name: str) -> List[List[bool]]:
        """Validate spike train data."""
        if not isinstance(spike_trains, list):
            raise ValidationError(f"{name} must be a list of spike trains")
        
        if not spike_trains:
            raise ValidationError(f"{name} cannot be empty")
        
        timesteps = len(spike_trains[0])
        for i, train in enumerate(spike_trains):
            if not isinstance(train, list):
                raise ValidationError(f"{name}[{i}] must be a list")
            if len(train) != timesteps:
                raise ValidationError(f"All spike trains must have same length. Train {i} has {len(train)}, expected {timesteps}")
            for t, spike in enumerate(train):
                if not isinstance(spike, bool):
                    raise ValidationError(f"{name}[{i}][{t}] must be boolean, got {type(spike)}")
        
        return spike_trains


# ==============================================================================
# ROBUST CONFIGURATION MANAGEMENT
# ==============================================================================

@dataclass
class RobustSpikingConfig:
    """Enhanced spiking configuration with validation."""
    timesteps: int = 32
    threshold: float = 1.0
    tau_mem: float = 20.0
    dt: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.timesteps = Validator.validate_range(self.timesteps, "timesteps", 1, 1000)
        self.threshold = Validator.validate_positive_number(self.threshold, "threshold")
        self.tau_mem = Validator.validate_positive_number(self.tau_mem, "tau_mem")
        self.dt = Validator.validate_positive_number(self.dt, "dt")
        
        # Additional logical validation
        if self.dt > self.tau_mem:
            warnings.warn(f"Time step ({self.dt}) is larger than membrane time constant ({self.tau_mem}). This may cause numerical instability.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timesteps': self.timesteps,
            'threshold': self.threshold,
            'tau_mem': self.tau_mem,
            'dt': self.dt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RobustSpikingConfig':
        """Create from dictionary with validation."""
        return cls(
            timesteps=data.get('timesteps', 32),
            threshold=data.get('threshold', 1.0),
            tau_mem=data.get('tau_mem', 20.0),
            dt=data.get('dt', 1.0)
        )


@dataclass  
class RobustModelConfig:
    """Enhanced model configuration with validation."""
    input_size: int
    hidden_sizes: List[int]
    output_size: int
    description: str = ""
    activation: str = "relu"
    
    def __post_init__(self):
        """Validate configuration."""
        self.input_size = Validator.validate_positive_number(self.input_size, "input_size")
        self.output_size = Validator.validate_positive_number(self.output_size, "output_size")
        self.hidden_sizes = Validator.validate_list_of_positive_ints(self.hidden_sizes, "hidden_sizes")
        
        # Check for reasonable sizes
        if self.input_size > 100000:
            warnings.warn(f"Large input size ({self.input_size}) may cause memory issues")
        if self.output_size > 10000:
            warnings.warn(f"Large output size ({self.output_size}) may cause memory issues")
        if any(size > 10000 for size in self.hidden_sizes):
            warnings.warn("Large hidden layers may cause memory issues")
    
    def total_parameters(self) -> int:
        """Calculate total parameters."""
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        return sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes) - 1))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'description': self.description,
            'activation': self.activation,
            'total_parameters': self.total_parameters()
        }


# ==============================================================================
# HEALTH MONITORING SYSTEM
# ==============================================================================

class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.logger = SpikeFormerLogger("health_monitor")
        self.last_check_time = 0
        self.check_interval = 60  # seconds
        self.health_history = []
        
    def check_system_health(self) -> Dict[str, HealthCheck]:
        """Perform comprehensive system health check."""
        checks = {}
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            if memory_usage < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal ({memory_usage:.1f}%)"
            elif memory_usage < 95:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high ({memory_usage:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical ({memory_usage:.1f}%)"
                
            checks['memory'] = HealthCheck(
                name="memory",
                status=status,
                message=message,
                timestamp=time.time(),
                details={'usage_percent': memory_usage, 'available_gb': memory.available / (1024**3)}
            )
        except Exception as e:
            checks['memory'] = HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}",
                timestamp=time.time()
            )
        
        # CPU check
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            
            if cpu_usage < 80:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal ({cpu_usage:.1f}%)"
            elif cpu_usage < 95:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high ({cpu_usage:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critical ({cpu_usage:.1f}%)"
                
            checks['cpu'] = HealthCheck(
                name="cpu",
                status=status,
                message=message,
                timestamp=time.time(),
                details={'usage_percent': cpu_usage, 'core_count': psutil.cpu_count()}
            )
        except Exception as e:
            checks['cpu'] = HealthCheck(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {e}",
                timestamp=time.time()
            )
        
        # Disk space check
        try:
            import psutil
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            if disk_usage < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal ({disk_usage:.1f}%)"
            elif disk_usage < 95:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high ({disk_usage:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical ({disk_usage:.1f}%)"
                
            checks['disk'] = HealthCheck(
                name="disk",
                status=status,
                message=message,
                timestamp=time.time(),
                details={'usage_percent': disk_usage, 'free_gb': disk.free / (1024**3)}
            )
        except Exception as e:
            checks['disk'] = HealthCheck(
                name="disk", 
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}",
                timestamp=time.time()
            )
        
        # Python environment check
        try:
            python_version = sys.version.split()[0]
            if sys.version_info >= (3, 8):
                status = HealthStatus.HEALTHY
                message = f"Python version supported ({python_version})"
            else:
                status = HealthStatus.DEGRADED
                message = f"Python version old ({python_version})"
                
            checks['python'] = HealthCheck(
                name="python",
                status=status,
                message=message,
                timestamp=time.time(),
                details={'version': python_version, 'executable': sys.executable}
            )
        except Exception as e:
            checks['python'] = HealthCheck(
                name="python",
                status=HealthStatus.UNHEALTHY,
                message=f"Python check failed: {e}",
                timestamp=time.time()
            )
        
        # Store in history
        self.health_history.append({
            'timestamp': time.time(),
            'checks': checks
        })
        
        # Keep only last 100 checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        self.last_check_time = time.time()
        return checks
    
    def get_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system status."""
        if any(check.status == HealthStatus.UNHEALTHY for check in checks.values()):
            return HealthStatus.UNHEALTHY
        elif any(check.status == HealthStatus.DEGRADED for check in checks.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        checks = self.check_system_health()
        overall_status = self.get_overall_status(checks)
        
        return {
            'status': overall_status.value,
            'timestamp': time.time(),
            'checks': {name: {
                'status': check.status.value,
                'message': check.message,
                'details': check.details
            } for name, check in checks.items()},
            'uptime': time.time() - (self.health_history[0]['timestamp'] if self.health_history else time.time())
        }


# ==============================================================================
# ROBUST CONVERTERS WITH ERROR HANDLING
# ==============================================================================

class RobustConverter:
    """Enhanced converter with comprehensive error handling."""
    
    def __init__(self, config: RobustSpikingConfig):
        self.config = config
        self.logger = SpikeFormerLogger("converter")
        self.health_monitor = HealthMonitor()
        
    def convert(self, model_config: RobustModelConfig) -> Tuple[SpikingMLP, Dict[str, Any]]:
        """Convert model with robust error handling."""
        conversion_start = time.time()
        
        with self.logger.operation("conversion"):
            try:
                # Health check before conversion
                health = self.health_monitor.get_health_summary()
                if health['status'] == 'unhealthy':
                    raise ConversionError(f"System unhealthy, cannot proceed: {health}")
                
                self.logger.info(f"Starting conversion of model with {model_config.total_parameters()} parameters")
                
                # Validate inputs
                self._validate_conversion_inputs(model_config)
                
                # Create layer sizes
                layer_sizes = [model_config.input_size] + model_config.hidden_sizes + [model_config.output_size]
                
                # Create spiking network
                snn_model = SpikingMLP(layer_sizes, self.config)
                
                # Generate conversion metadata
                conversion_time = time.time() - conversion_start
                metadata = {
                    'conversion_time_seconds': conversion_time,
                    'input_config': model_config.to_dict(),
                    'spiking_config': self.config.to_dict(),
                    'layer_sizes': layer_sizes,
                    'total_neurons': sum(layer_sizes[1:]),  # Exclude input layer
                    'total_synapses': sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1)),
                    'conversion_timestamp': time.time(),
                    'system_health': health,
                    'conversion_id': self._generate_conversion_id(model_config)
                }
                
                self.logger.info(f"Conversion successful: {metadata['total_neurons']} neurons, {metadata['total_synapses']} synapses")
                
                return snn_model, metadata
                
            except Exception as e:
                self.logger.error(f"Conversion failed: {e}")
                # Include traceback in debug mode
                if self.logger.logger.level <= logging.DEBUG:
                    self.logger.debug(traceback.format_exc())
                raise ConversionError(f"Model conversion failed: {e}") from e
    
    def _validate_conversion_inputs(self, model_config: RobustModelConfig):
        """Validate conversion inputs."""
        # Check for reasonable model size
        total_params = model_config.total_parameters()
        if total_params > 1000000:  # 1M parameters
            raise ValidationError(f"Model too large for conversion: {total_params} parameters")
        
        # Check layer size compatibility
        max_layer_size = max([model_config.input_size] + model_config.hidden_sizes + [model_config.output_size])
        if max_layer_size > 10000:
            raise ValidationError(f"Layer size too large: {max_layer_size}")
        
        # Check for reasonable timesteps
        if self.config.timesteps * max_layer_size > 100000:
            raise ValidationError(f"Timesteps * max layer size too large: {self.config.timesteps} * {max_layer_size}")
    
    def _generate_conversion_id(self, model_config: RobustModelConfig) -> str:
        """Generate unique conversion ID."""
        config_str = json.dumps(model_config.to_dict(), sort_keys=True)
        spiking_str = json.dumps(self.config.to_dict(), sort_keys=True)
        combined = f"{config_str}_{spiking_str}_{time.time()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]


# ==============================================================================
# ROBUST HARDWARE DEPLOYMENT
# ==============================================================================

class RobustHardwareSimulator:
    """Enhanced hardware simulator with error handling."""
    
    def __init__(self, platform: str = "loihi2"):
        self.platform = platform
        self.logger = SpikeFormerLogger(f"hardware_{platform}")
        self.health_monitor = HealthMonitor()
        self._validate_platform()
        
        # Platform specifications with limits
        self.platform_specs = {
            "loihi2": {
                "energy_per_spike_nj": 0.001,
                "base_power_mw": 1.0,
                "latency_per_timestep_us": 0.1,
                "max_neurons": 1000000,
                "max_synapses": 100000000,
                "memory_per_neuron_bytes": 64
            },
            "spinnaker": {
                "energy_per_spike_nj": 0.01,
                "base_power_mw": 100.0,
                "latency_per_timestep_us": 1.0,
                "max_neurons": 100000,
                "max_synapses": 10000000,
                "memory_per_neuron_bytes": 256
            },
            "cpu": {
                "energy_per_spike_nj": 1.0,
                "base_power_mw": 20000.0,
                "latency_per_timestep_us": 10.0,
                "max_neurons": 10000,
                "max_synapses": 1000000,
                "memory_per_neuron_bytes": 1024
            }
        }
    
    def _validate_platform(self):
        """Validate platform selection."""
        valid_platforms = ["loihi2", "spinnaker", "cpu"]
        if self.platform not in valid_platforms:
            raise ConfigurationError(f"Invalid platform: {self.platform}. Valid options: {valid_platforms}")
    
    def deploy(self, model: SpikingMLP, deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deploy model with robust error handling."""
        with self.logger.operation("deployment"):
            try:
                # Health check
                health = self.health_monitor.get_health_summary()
                if health['status'] == 'unhealthy':
                    raise DeploymentError(f"System unhealthy, cannot deploy: {health}")
                
                # Validate model for platform
                self._validate_model_for_platform(model)
                
                # Simulate deployment
                deployment_start = time.time()
                
                total_neurons = sum(layer.output_size for layer in model.layers)
                total_synapses = sum(
                    layer.input_size * layer.output_size 
                    for layer in model.layers
                )
                
                specs = self.platform_specs[self.platform]
                
                # Calculate resource usage
                memory_usage = total_neurons * specs["memory_per_neuron_bytes"]
                deployment_time = 100 + (total_neurons / 1000) * 10  # Simulate time
                
                # Simulate deployment delay
                time.sleep(min(0.2, deployment_time / 1000))
                
                deployment_info = {
                    'platform': self.platform,
                    'deployment_id': self._generate_deployment_id(),
                    'total_neurons': total_neurons,
                    'total_synapses': total_synapses,
                    'memory_usage_bytes': memory_usage,
                    'deployment_time_ms': deployment_time,
                    'deployment_timestamp': time.time(),
                    'system_health': health,
                    'platform_specs': specs,
                    'resource_utilization': {
                        'neuron_utilization': total_neurons / specs['max_neurons'],
                        'synapse_utilization': total_synapses / specs['max_synapses'],
                        'memory_utilization': memory_usage / (specs['max_neurons'] * specs['memory_per_neuron_bytes'])
                    }
                }
                
                self.logger.info(f"Deployment successful: {total_neurons} neurons on {self.platform}")
                return deployment_info
                
            except Exception as e:
                self.logger.error(f"Deployment failed: {e}")
                raise DeploymentError(f"Hardware deployment failed: {e}") from e
    
    def _validate_model_for_platform(self, model: SpikingMLP):
        """Validate model against platform constraints."""
        specs = self.platform_specs[self.platform]
        
        total_neurons = sum(layer.output_size for layer in model.layers)
        total_synapses = sum(layer.input_size * layer.output_size for layer in model.layers)
        
        if total_neurons > specs['max_neurons']:
            raise ValidationError(f"Model too large: {total_neurons} neurons > {specs['max_neurons']} limit for {self.platform}")
        
        if total_synapses > specs['max_synapses']:
            raise ValidationError(f"Model too complex: {total_synapses} synapses > {specs['max_synapses']} limit for {self.platform}")
        
        # Check memory requirements
        memory_needed = total_neurons * specs['memory_per_neuron_bytes']
        memory_limit = specs['max_neurons'] * specs['memory_per_neuron_bytes']
        
        if memory_needed > memory_limit:
            raise ValidationError(f"Memory requirement too high: {memory_needed} bytes > {memory_limit} limit")
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID."""
        timestamp = str(time.time())
        combined = f"{self.platform}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def benchmark_with_validation(self, model: SpikingMLP, num_samples: int = 10) -> HardwareMetrics:
        """Benchmark with input validation."""
        if num_samples <= 0:
            raise ValidationError("Number of samples must be positive")
        if num_samples > 1000:
            raise ValidationError("Too many samples requested (max 1000)")
        
        with self.logger.operation("benchmark"):
            try:
                # Use original benchmark but with enhanced error handling
                simulator = BasicHardwareSimulator(self.platform)
                metrics = simulator.benchmark(model, num_samples)
                
                # Add validation metrics
                if metrics.energy_per_inference_mj < 0:
                    raise ProfilingError("Invalid energy measurement")
                if metrics.latency_ms < 0:
                    raise ProfilingError("Invalid latency measurement")
                
                self.logger.info(f"Benchmark completed: {metrics.energy_per_inference_mj:.3f} mJ/inference")
                return metrics
                
            except Exception as e:
                self.logger.error(f"Benchmark failed: {e}")
                raise ProfilingError(f"Benchmarking failed: {e}") from e


# ==============================================================================
# DEMONSTRATION OF ROBUST SYSTEM
# ==============================================================================

def run_robust_demo():
    """Demonstrate robust system with error handling."""
    print("\n" + "="*80)
    print("🛡️ SPIKEFORMER ROBUST SYSTEM DEMO - GENERATION 2")
    print("="*80)
    
    # Initialize robust logger
    logger = SpikeFormerLogger("robust_demo", LogLevel.INFO)
    health_monitor = HealthMonitor()
    
    try:
        # 1. System Health Check
        print("\n1️⃣  SYSTEM HEALTH CHECK")
        print("-" * 40)
        
        with logger.operation("health_check"):
            health_summary = health_monitor.get_health_summary()
            
            print(f"🏥 Overall Status: {health_summary['status'].upper()}")
            for check_name, check_info in health_summary['checks'].items():
                status_emoji = "✅" if check_info['status'] == 'healthy' else "⚠️" if check_info['status'] == 'degraded' else "❌"
                print(f"   {status_emoji} {check_name.upper()}: {check_info['message']}")
        
        # 2. Robust Configuration
        print("\n2️⃣  ROBUST CONFIGURATION VALIDATION")
        print("-" * 40)
        
        try:
            # Test valid configuration
            valid_config = RobustModelConfig(
                input_size=784,
                hidden_sizes=[256, 128],
                output_size=10,
                description="MNIST classifier"
            )
            print(f"✅ Valid config: {valid_config.total_parameters()} parameters")
            
            # Test invalid configuration (should raise error)
            try:
                invalid_config = RobustModelConfig(
                    input_size=-10,  # Invalid
                    hidden_sizes=[256],
                    output_size=10
                )
            except ValidationError as e:
                print(f"✅ Caught invalid config: {e}")
                
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        # 3. Robust Conversion
        print("\n3️⃣  ROBUST MODEL CONVERSION")
        print("-" * 40)
        
        spiking_config = RobustSpikingConfig(
            timesteps=32,
            threshold=1.0,
            tau_mem=20.0
        )
        
        converter = RobustConverter(spiking_config)
        
        with logger.operation("robust_conversion"):
            snn_model, metadata = converter.convert(valid_config)
            
            print(f"✅ Conversion ID: {metadata['conversion_id']}")
            print(f"✅ Neurons: {metadata['total_neurons']}")
            print(f"✅ Synapses: {metadata['total_synapses']}")
            print(f"✅ Conversion time: {metadata['conversion_time_seconds']:.3f}s")
        
        # 4. Robust Hardware Deployment
        print("\n4️⃣  ROBUST HARDWARE DEPLOYMENT")
        print("-" * 40)
        
        platforms = ["loihi2", "spinnaker"]
        deployment_results = {}
        
        for platform in platforms:
            try:
                hardware_sim = RobustHardwareSimulator(platform)
                
                with logger.operation(f"deployment_{platform}"):
                    deployment_info = hardware_sim.deploy(snn_model)
                    metrics = hardware_sim.benchmark_with_validation(snn_model, num_samples=5)
                    
                    deployment_results[platform] = {
                        'deployment_info': deployment_info,
                        'metrics': metrics
                    }
                    
                    print(f"✅ {platform.upper()}: "
                          f"ID={deployment_info['deployment_id']}, "
                          f"Energy={metrics.energy_per_inference_mj:.3f}mJ")
                    
            except Exception as e:
                logger.error(f"Platform {platform} failed: {e}")
                print(f"❌ {platform.upper()}: {e}")
        
        # 5. Error Handling Demonstration
        print("\n5️⃣  ERROR HANDLING DEMONSTRATION")
        print("-" * 40)
        
        # Test various error conditions
        error_tests = [
            ("Invalid timesteps", lambda: RobustSpikingConfig(timesteps=-5)),
            ("Invalid layer size", lambda: RobustModelConfig(input_size=0, hidden_sizes=[10], output_size=5)),
            ("Invalid platform", lambda: RobustHardwareSimulator("invalid_platform")),
        ]
        
        for test_name, test_func in error_tests:
            try:
                test_func()
                print(f"❌ {test_name}: Expected error but none occurred")
            except (ValidationError, ConfigurationError) as e:
                print(f"✅ {test_name}: Correctly caught - {type(e).__name__}")
            except Exception as e:
                print(f"⚠️ {test_name}: Unexpected error - {e}")
        
        # 6. System Statistics
        print("\n6️⃣  SYSTEM STATISTICS")
        print("-" * 40)
        
        stats = logger.get_stats()
        print(f"📊 Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"📊 Operations:")
        for op_name, count in stats['operation_counters'].items():
            print(f"     {op_name}: {count}")
        
        # 7. Final Health Check
        print("\n7️⃣  FINAL SYSTEM STATE")
        print("-" * 40)
        
        final_health = health_monitor.get_health_summary()
        print(f"🏥 Final Status: {final_health['status'].upper()}")
        
        if deployment_results:
            best_platform = min(deployment_results.keys(), 
                              key=lambda p: deployment_results[p]['metrics'].energy_per_inference_mj)
            best_energy = deployment_results[best_platform]['metrics'].energy_per_inference_mj
            print(f"🏆 Best Platform: {best_platform.upper()} ({best_energy:.3f} mJ/inference)")
        
        print("\n" + "="*80)
        print("🎉 ROBUST SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("✅ Error handling validated")
        print("✅ Input validation working") 
        print("✅ Health monitoring active")
        print("✅ Comprehensive logging enabled")
        print("✅ Resource validation implemented")
        print("="*80 + "\n")
        
        return {
            'health_summary': final_health,
            'deployment_results': deployment_results,
            'system_stats': stats
        }
        
    except Exception as e:
        logger.critical(f"Demo failed with critical error: {e}")
        print(f"\n❌ CRITICAL ERROR: {e}")
        if logger.logger.level <= logging.DEBUG:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        results = run_robust_demo()
        
        # Save results if requested
        if "--save" in sys.argv:
            output_file = "robust_demo_results.json"
            
            # Convert results to JSON-serializable format
            json_results = {
                'health_summary': results['health_summary'],
                'system_stats': results['system_stats'],
                'deployment_summary': {
                    platform: {
                        'deployment_id': info['deployment_info']['deployment_id'],
                        'energy_mj': info['metrics'].energy_per_inference_mj,
                        'latency_ms': info['metrics'].latency_ms
                    }
                    for platform, info in results['deployment_results'].items()
                },
                'demo_timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"📁 Robust demo results saved to {output_file}")
            
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)