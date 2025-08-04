"""Energy profiling and performance monitoring for neuromorphic systems."""

import torch
import torch.nn as nn
import time
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union, ContextManager
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from collections import defaultdict, deque
import json
import logging
from pathlib import Path


@dataclass
class EnergyMeasurement:
    """Single energy measurement record."""
    timestamp: float
    energy_joules: float
    power_watts: float
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerProfile:
    """Power consumption profile over time."""
    timestamps: List[float]
    power_watts: List[float]
    component: str
    total_energy_joules: float
    avg_power_watts: float
    peak_power_watts: float
    duration_seconds: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    throughput_samples_per_sec: float
    latency_ms: float
    energy_per_inference_mj: float
    accuracy: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    spike_rate: float
    energy_efficiency_ratio: float  # vs baseline
    
    
class EnergyMonitor(ABC):
    """Abstract base class for energy monitoring."""
    
    @abstractmethod
    def start_measurement(self) -> None:
        """Start energy measurement."""
        pass
        
    @abstractmethod
    def stop_measurement(self) -> EnergyMeasurement:
        """Stop measurement and return results."""
        pass
        
    @abstractmethod
    def get_current_power(self) -> float:
        """Get instantaneous power consumption."""
        pass


class CPUEnergyMonitor(EnergyMonitor):
    """CPU energy monitor using system metrics."""
    
    def __init__(self, sampling_rate_hz: float = 10.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.start_time = None
        self.measurements = []
        self.logger = logging.getLogger(__name__)
        
    def start_measurement(self) -> None:
        """Start CPU energy measurement."""
        self.start_time = time.time()
        self.measurements = []
        
    def stop_measurement(self) -> EnergyMeasurement:
        """Stop measurement and calculate energy consumption."""
        if self.start_time is None:
            raise RuntimeError("Measurement not started")
            
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Estimate CPU power consumption
        cpu_percent = psutil.cpu_percent(interval=0.1)
        estimated_power = self._estimate_cpu_power(cpu_percent)
        estimated_energy = estimated_power * duration
        
        measurement = EnergyMeasurement(
            timestamp=end_time,
            energy_joules=estimated_energy,
            power_watts=estimated_power,
            component="cpu",
            metadata={
                "duration_seconds": duration,
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count()
            }
        )
        
        self.start_time = None
        return measurement
        
    def get_current_power(self) -> float:
        """Get current CPU power consumption estimate."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return self._estimate_cpu_power(cpu_percent)
        
    def _estimate_cpu_power(self, cpu_percent: float) -> float:
        """Estimate CPU power consumption based on utilization."""
        # Simplified model: assume 100W max power for high-end CPU
        base_power = 20.0  # Idle power
        max_additional_power = 80.0  # Additional power at 100% load
        return base_power + (cpu_percent / 100.0) * max_additional_power


class GPUEnergyMonitor(EnergyMonitor):
    """GPU energy monitor."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.start_time = None
        self.start_energy = None
        self.logger = logging.getLogger(__name__)
        
    def start_measurement(self) -> None:
        """Start GPU energy measurement."""
        self.start_time = time.time()
        self.start_energy = self._get_gpu_energy()
        
    def stop_measurement(self) -> EnergyMeasurement:
        """Stop GPU measurement."""
        if self.start_time is None:
            raise RuntimeError("Measurement not started")
            
        end_time = time.time()
        end_energy = self._get_gpu_energy()
        
        duration = end_time - self.start_time
        energy_consumed = end_energy - self.start_energy
        avg_power = energy_consumed / duration if duration > 0 else 0
        
        measurement = EnergyMeasurement(
            timestamp=end_time,
            energy_joules=energy_consumed,
            power_watts=avg_power,
            component="gpu",
            metadata={
                "device_id": self.device_id,
                "duration_seconds": duration,
                "gpu_memory_allocated": torch.cuda.memory_allocated(self.device_id) if torch.cuda.is_available() else 0
            }
        )
        
        self.start_time = None
        self.start_energy = None
        return measurement
        
    def get_current_power(self) -> float:
        """Get current GPU power consumption."""
        if not torch.cuda.is_available():
            return 0.0
            
        # Simplified estimation based on memory usage
        memory_allocated = torch.cuda.memory_allocated(self.device_id)
        memory_total = torch.cuda.get_device_properties(self.device_id).total_memory
        utilization = memory_allocated / memory_total if memory_total > 0 else 0
        
        # Estimate power: 50-300W range for typical GPUs
        base_power = 50.0
        max_additional_power = 250.0
        return base_power + utilization * max_additional_power
        
    def _get_gpu_energy(self) -> float:
        """Get cumulative GPU energy consumption."""
        # Simplified: integrate power over time
        power = self.get_current_power()
        return power * 0.1  # Assume 0.1 second measurement interval


class NeuromorphicEnergyMonitor(EnergyMonitor):
    """Energy monitor for neuromorphic hardware."""
    
    def __init__(self, hardware_type: str = "loihi2"):
        self.hardware_type = hardware_type
        self.start_time = None
        self.measurements = []
        
    def start_measurement(self) -> None:
        """Start neuromorphic hardware measurement."""
        self.start_time = time.time()
        self.measurements = []
        
    def stop_measurement(self) -> EnergyMeasurement:
        """Stop measurement."""
        if self.start_time is None:
            raise RuntimeError("Measurement not started")
            
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Simulate neuromorphic power consumption
        if self.hardware_type.lower() == "loihi2":
            power = 0.001  # 1mW typical for Loihi 2
        elif self.hardware_type.lower() == "spinnaker":
            power = 0.1    # 100mW typical for SpiNNaker
        else:
            power = 0.01   # 10mW for generic neuromorphic
            
        energy = power * duration
        
        measurement = EnergyMeasurement(
            timestamp=end_time,
            energy_joules=energy,
            power_watts=power,
            component=self.hardware_type,
            metadata={
                "duration_seconds": duration,
                "hardware_type": self.hardware_type
            }
        )
        
        self.start_time = None
        return measurement
        
    def get_current_power(self) -> float:
        """Get current neuromorphic hardware power."""
        if self.hardware_type.lower() == "loihi2":
            return 0.001  # 1mW
        elif self.hardware_type.lower() == "spinnaker":
            return 0.1    # 100mW
        else:
            return 0.01   # 10mW


class EnergyProfiler:
    """Comprehensive energy profiling system."""
    
    def __init__(self, monitors: Optional[List[EnergyMonitor]] = None):
        self.monitors = monitors or [CPUEnergyMonitor()]
        self.measurements = []
        self.logger = logging.getLogger(__name__)
        self._context_measurements = {}
        
    def __enter__(self) -> 'EnergyProfiler':
        """Enter profiling context."""
        self.start_profiling()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit profiling context."""
        measurements = self.stop_profiling()
        self.measurements.extend(measurements)
        
    def start_profiling(self) -> None:
        """Start energy profiling on all monitors."""
        for monitor in self.monitors:
            monitor.start_measurement()
            
    def stop_profiling(self) -> List[EnergyMeasurement]:
        """Stop profiling and return measurements."""
        measurements = []
        for monitor in self.monitors:
            try:
                measurement = monitor.stop_measurement()
                measurements.append(measurement)
            except Exception as e:
                self.logger.warning(f"Monitor measurement failed: {e}")
                
        return measurements
        
    def profile_inference(self, model: nn.Module, input_data: torch.Tensor, 
                         num_runs: int = 10) -> Dict[str, Any]:
        """Profile model inference energy consumption."""
        self.logger.info(f"Profiling inference for {num_runs} runs")
        
        inference_measurements = []
        latencies = []
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_data)
                
        # Profiled runs
        for run in range(num_runs):
            with self:
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_data)
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # ms
                
            # Get measurements from this run
            run_measurements = self.measurements[-len(self.monitors):]
            inference_measurements.extend(run_measurements)
            
        # Calculate statistics
        total_energy = sum(m.energy_joules for m in inference_measurements)
        avg_energy_per_inference = total_energy / num_runs
        avg_latency = np.mean(latencies)
        
        return {
            "avg_energy_per_inference_mj": avg_energy_per_inference * 1000,
            "total_energy_joules": total_energy,
            "avg_latency_ms": avg_latency,
            "num_runs": num_runs,
            "energy_per_component": self._group_energy_by_component(inference_measurements),
            "measurements": inference_measurements
        }
        
    def profile_training(self, model: nn.Module, train_loader, 
                        num_epochs: int = 1) -> Dict[str, Any]:
        """Profile training energy consumption."""
        self.logger.info(f"Profiling training for {num_epochs} epochs")
        
        training_measurements = []
        epoch_energies = []
        
        for epoch in range(num_epochs):
            with self:
                epoch_start = time.time()
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    # Simulate training step
                    outputs = model(data)
                    # Loss and backprop would go here
                    
                    if batch_idx >= 10:  # Limit profiling to first 10 batches
                        break
                        
                epoch_time = time.time() - epoch_start
                
            # Get measurements from this epoch
            epoch_measurements = self.measurements[-len(self.monitors):]
            training_measurements.extend(epoch_measurements)
            
            epoch_energy = sum(m.energy_joules for m in epoch_measurements)
            epoch_energies.append(epoch_energy)
            
        return {
            "total_training_energy_joules": sum(epoch_energies),
            "avg_epoch_energy_joules": np.mean(epoch_energies),
            "energy_per_component": self._group_energy_by_component(training_measurements),
            "measurements": training_measurements
        }
        
    def compare_models(self, models: Dict[str, nn.Module], 
                      test_data: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Compare energy consumption across multiple models."""
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"Profiling model: {model_name}")
            model_results = self.profile_inference(model, test_data)
            results[model_name] = model_results
            
        return results
        
    def _group_energy_by_component(self, measurements: List[EnergyMeasurement]) -> Dict[str, float]:
        """Group energy measurements by component."""
        component_energy = defaultdict(float)
        
        for measurement in measurements:
            component_energy[measurement.component] += measurement.energy_joules
            
        return dict(component_energy)
        
    def generate_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive energy report."""
        if not self.measurements:
            return {"error": "No measurements available"}
            
        report = {
            "summary": {
                "total_measurements": len(self.measurements),
                "total_energy_joules": sum(m.energy_joules for m in self.measurements),
                "components": list(set(m.component for m in self.measurements))
            },
            "by_component": self._group_energy_by_component(self.measurements),
            "timeline": [
                {
                    "timestamp": m.timestamp,
                    "component": m.component,
                    "energy_joules": m.energy_joules,
                    "power_watts": m.power_watts
                }
                for m in self.measurements
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report


class PowerMonitor:
    """Real-time power monitoring system."""
    
    def __init__(self, sampling_rate_hz: float = 1.0, buffer_size: int = 1000):
        self.sampling_rate_hz = sampling_rate_hz
        self.buffer_size = buffer_size
        self.power_readings = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.monitors = [CPUEnergyMonitor(), GPUEnergyMonitor()]
        self._monitoring = False
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """Start continuous power monitoring."""
        self._monitoring = True
        self.logger.info("Starting power monitoring")
        
        import threading
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop power monitoring."""
        self._monitoring = False
        self.logger.info("Stopping power monitoring")
        
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop."""
        while self._monitoring:
            timestamp = time.time()
            total_power = sum(monitor.get_current_power() for monitor in self.monitors)
            
            self.power_readings.append(total_power)
            self.timestamps.append(timestamp)
            
            time.sleep(1.0 / self.sampling_rate_hz)
            
    def get_current_power(self) -> float:
        """Get latest power reading."""
        return self.power_readings[-1] if self.power_readings else 0.0
        
    def get_power_statistics(self) -> Dict[str, float]:
        """Get power consumption statistics."""
        if not self.power_readings:
            return {}
            
        readings = list(self.power_readings)
        return {
            "current_power_watts": readings[-1],
            "avg_power_watts": np.mean(readings),
            "max_power_watts": np.max(readings),
            "min_power_watts": np.min(readings),
            "std_power_watts": np.std(readings),
            "num_readings": len(readings)
        }
        
    def export_power_profile(self, filename: str) -> None:
        """Export power profile to file."""
        profile_data = {
            "timestamps": list(self.timestamps),
            "power_watts": list(self.power_readings),
            "sampling_rate_hz": self.sampling_rate_hz,
            "statistics": self.get_power_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(profile_data, f, indent=2)


class EnergyComparison:
    """Compare energy efficiency across different implementations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def compare_ann_vs_snn(self, ann_model: nn.Module, snn_model: nn.Module,
                          test_data: torch.Tensor) -> Dict[str, Any]:
        """Compare ANN vs SNN energy consumption."""
        profiler = EnergyProfiler()
        
        # Profile ANN
        ann_results = profiler.profile_inference(ann_model, test_data)
        
        # Profile SNN  
        snn_results = profiler.profile_inference(snn_model, test_data)
        
        # Calculate improvement
        energy_reduction = (ann_results["avg_energy_per_inference_mj"] - 
                          snn_results["avg_energy_per_inference_mj"]) / ann_results["avg_energy_per_inference_mj"]
        
        return {
            "ann_results": ann_results,
            "snn_results": snn_results,
            "energy_reduction_ratio": energy_reduction,
            "energy_improvement_factor": ann_results["avg_energy_per_inference_mj"] / snn_results["avg_energy_per_inference_mj"],
            "latency_comparison": {
                "ann_latency_ms": ann_results["avg_latency_ms"],
                "snn_latency_ms": snn_results["avg_latency_ms"],
                "latency_ratio": snn_results["avg_latency_ms"] / ann_results["avg_latency_ms"]
            }
        }
        
    def benchmark_hardware_backends(self, model: nn.Module, 
                                   test_data: torch.Tensor) -> Dict[str, Any]:
        """Benchmark model across different hardware backends."""
        from .hardware import Loihi2Backend, SpiNNakerBackend, HardwareConfig
        
        results = {}
        
        # CPU baseline
        profiler = EnergyProfiler([CPUEnergyMonitor()])
        cpu_results = profiler.profile_inference(model, test_data)
        results["cpu"] = cpu_results
        
        # GPU baseline
        if torch.cuda.is_available():
            gpu_profiler = EnergyProfiler([GPUEnergyMonitor()])
            gpu_results = gpu_profiler.profile_inference(model, test_data)
            results["gpu"] = gpu_results
            
        # Neuromorphic backends
        for backend_name in ["loihi2", "spinnaker"]:
            neuromorphic_profiler = EnergyProfiler([NeuromorphicEnergyMonitor(backend_name)])
            neuro_results = neuromorphic_profiler.profile_inference(model, test_data)
            results[backend_name] = neuro_results
            
        return results


# Convenience functions
def profile_model_energy(model: nn.Module, input_data: torch.Tensor,
                        num_runs: int = 10) -> Dict[str, Any]:
    """Quick energy profiling for a model."""
    profiler = EnergyProfiler()
    return profiler.profile_inference(model, input_data, num_runs)


def compare_energy_efficiency(models: Dict[str, nn.Module], 
                             test_data: torch.Tensor) -> Dict[str, Any]:
    """Compare energy efficiency across multiple models."""
    profiler = EnergyProfiler()
    return profiler.compare_models(models, test_data)