"""Mock objects and utilities for neuromorphic testing."""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from unittest.mock import Mock, MagicMock, patch
import time
import random


class MockNeuromorphicHardware:
    """Base class for mocking neuromorphic hardware."""
    
    def __init__(self, hardware_type: str, latency_ms: float = 50.0, energy_uj: float = 100.0):
        self.hardware_type = hardware_type
        self.latency_ms = latency_ms
        self.energy_uj = energy_uj
        self.is_connected = True
        self.compilation_time = 1.0  # seconds
        
    def compile(self, model, config: Optional[Dict] = None):
        """Mock model compilation."""
        time.sleep(self.compilation_time)  # Simulate compilation time
        
        compiled_model = Mock()
        compiled_model.hardware_type = self.hardware_type
        compiled_model.config = config or {}
        compiled_model.num_neurons = getattr(model, 'num_neurons', 1000)
        compiled_model.num_synapses = getattr(model, 'num_synapses', 10000)
        
        return compiled_model
    
    def deploy(self, compiled_model):
        """Mock model deployment."""
        if not self.is_connected:
            raise RuntimeError(f"{self.hardware_type} hardware not connected")
        
        deployed_model = Mock()
        deployed_model.hardware_type = self.hardware_type
        deployed_model.compiled_model = compiled_model
        deployed_model.status = 'deployed'
        
        return deployed_model
    
    def execute(self, deployed_model, input_data):
        """Mock model execution."""
        # Simulate execution time
        time.sleep(self.latency_ms / 1000.0)
        
        # Generate mock results
        batch_size = input_data.shape[0] if hasattr(input_data, 'shape') else 1
        output = torch.randn(batch_size, 10)  # Mock classification output
        
        metrics = {
            'latency_ms': self.latency_ms + random.uniform(-5, 5),
            'energy_uj': self.energy_uj + random.uniform(-10, 10),
            'power_mw': self.energy_uj / self.latency_ms * 1000,
            'throughput_samples_per_sec': 1000 / self.latency_ms * batch_size,
            'accuracy': random.uniform(0.8, 0.95)
        }
        
        return {
            'output': output,
            'metrics': metrics,
            'status': 'success'
        }
    
    def disconnect(self):
        """Mock hardware disconnection."""
        self.is_connected = False
    
    def reconnect(self):
        """Mock hardware reconnection."""
        self.is_connected = True


class MockLoihi2Hardware(MockNeuromorphicHardware):
    """Mock Intel Loihi 2 hardware."""
    
    def __init__(self):
        super().__init__('loihi2', latency_ms=20.0, energy_uj=50.0)
        self.num_chips = 2
        self.cores_per_chip = 128
        self.neurons_per_core = 1024
        self.synapses_per_core = 64 * 1024
        
    def compile(self, model, config: Optional[Dict] = None):
        """Mock Loihi 2 specific compilation."""
        compiled_model = super().compile(model, config)
        
        # Add Loihi 2 specific attributes
        compiled_model.chip_allocation = {
            'num_chips_used': min(self.num_chips, (compiled_model.num_neurons // (self.cores_per_chip * self.neurons_per_core)) + 1),
            'cores_used': min(self.cores_per_chip, (compiled_model.num_neurons // self.neurons_per_core) + 1),
            'utilization': min(1.0, compiled_model.num_neurons / (self.num_chips * self.cores_per_chip * self.neurons_per_core))
        }
        
        compiled_model.routing_table = Mock()
        compiled_model.weight_precision = config.get('precision', 8) if config else 8
        
        return compiled_model


class MockSpiNNakerHardware(MockNeuromorphicHardware):
    """Mock SpiNNaker hardware."""
    
    def __init__(self):
        super().__init__('spinnaker', latency_ms=80.0, energy_uj=120.0)
        self.num_boards = 1
        self.chips_per_board = 48
        self.cores_per_chip = 18
        self.time_scale_factor = 1000
        
    def compile(self, model, config: Optional[Dict] = None):
        """Mock SpiNNaker specific compilation."""
        compiled_model = super().compile(model, config)
        
        # Add SpiNNaker specific attributes
        compiled_model.board_allocation = {
            'num_boards_used': self.num_boards,
            'chips_used': min(self.chips_per_board, (compiled_model.num_neurons // 256) + 1),
            'routing_algorithm': config.get('routing_algorithm', 'neighbor_aware') if config else 'neighbor_aware'
        }
        
        compiled_model.time_scale_factor = config.get('time_scale_factor', self.time_scale_factor) if config else self.time_scale_factor
        compiled_model.packet_ordering = config.get('packet_ordering', True) if config else True
        
        return compiled_model


class MockEnergyProfiler:
    """Mock energy profiler for testing."""
    
    def __init__(self, baseline_energy: float = 1000.0):
        self.baseline_energy = baseline_energy
        self.current_energy = 0.0
        self.power_samples = []
        self.is_measuring = False
        self.start_time = None
        
    def __enter__(self):
        """Start energy measurement."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop energy measurement."""
        self.stop()
    
    def start(self):
        """Start energy measurement."""
        self.is_measuring = True
        self.start_time = time.time()
        self.current_energy = 0.0
        self.power_samples = []
        
    def stop(self):
        """Stop energy measurement."""
        if not self.is_measuring:
            return
            
        self.is_measuring = False
        duration = time.time() - self.start_time
        
        # Simulate energy consumption based on duration
        self.current_energy = duration * random.uniform(50, 200)  # mJ
        
        # Generate power samples
        num_samples = max(1, int(duration * 10))  # 10 Hz sampling
        for _ in range(num_samples):
            power = random.uniform(100, 300)  # mW
            self.power_samples.append(power)
    
    @property
    def energy_mJ(self) -> float:
        """Get energy consumption in mJ."""
        return self.current_energy
    
    @property
    def power_mW(self) -> float:
        """Get average power consumption in mW."""
        return np.mean(self.power_samples) if self.power_samples else 0.0
    
    @property
    def gpu_baseline_ratio(self) -> float:
        """Get energy efficiency ratio vs GPU baseline."""
        return self.baseline_energy / max(self.current_energy, 0.1)
    
    def get_power_timeline(self) -> List[float]:
        """Get power consumption timeline."""
        return self.power_samples.copy()


class MockModelConverter:
    """Mock model converter for testing."""
    
    def __init__(self, conversion_time: float = 2.0):
        self.conversion_time = conversion_time
        self.calibration_data = None
        
    def convert(self, model, config: Optional[Dict] = None):
        """Mock model conversion."""
        # Simulate conversion time
        time.sleep(self.conversion_time)
        
        config = config or {}
        
        # Create mock converted model
        converted_model = Mock()
        converted_model.original_model = model
        converted_model.config = config
        converted_model.timesteps = config.get('timesteps', 32)
        converted_model.threshold = config.get('threshold', 1.0)
        converted_model.neuron_model = config.get('neuron_model', 'LIF')
        
        # Mock forward pass
        def mock_forward(x):
            if len(x.shape) == 4:  # Image input
                batch_size = x.shape[0]
                return torch.randn(batch_size, 10)  # Mock classification
            elif len(x.shape) == 3:  # Sequence input
                batch_size, seq_len = x.shape[:2]
                return torch.randn(batch_size, seq_len, 10)
            else:
                return torch.randn_like(x)
        
        converted_model.forward = mock_forward
        converted_model.__call__ = mock_forward
        
        # Add evaluation methods
        def # eval() removed for security):
            converted_model.training = False
            return converted_model
        
        def train():
            converted_model.training = True
            return converted_model
        
        converted_model.eval = eval
        converted_model.train = train
        converted_model.training = False
        
        return converted_model
    
    def calibrate(self, model, calibration_data):
        """Mock calibration process."""
        self.calibration_data = calibration_data
        # Simulate calibration time
        time.sleep(0.5)
        
        calibration_stats = {
            'num_samples': len(calibration_data) if hasattr(calibration_data, '__len__') else 100,
            'activation_ranges': {
                f'layer_{i}': {'min': random.uniform(-2, 0), 'max': random.uniform(1, 3)}
                for i in range(5)
            },
            'optimal_thresholds': {
                f'layer_{i}': random.uniform(0.5, 2.0)
                for i in range(5)
            }
        }
        
        return calibration_stats


class MockBenchmarkRunner:
    """Mock benchmark runner for testing."""
    
    def __init__(self):
        self.results_history = []
        
    def run_benchmark(
        self,
        model,
        dataset,
        metrics: List[str] = None,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Mock benchmark execution."""
        metrics = metrics or ['accuracy', 'latency', 'energy']
        
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                # Simulate accuracy with some variance
                base_acc = 0.85
                results[metric] = [base_acc + random.uniform(-0.05, 0.05) for _ in range(num_runs)]
            elif metric == 'latency':
                # Simulate latency in ms
                base_latency = 50.0
                results[metric] = [base_latency + random.uniform(-10, 10) for _ in range(num_runs)]
            elif metric == 'energy':
                # Simulate energy in mJ
                base_energy = 100.0
                results[metric] = [base_energy + random.uniform(-20, 20) for _ in range(num_runs)]
            elif metric == 'throughput':
                # Simulate throughput in samples/sec
                base_throughput = 1000.0
                results[metric] = [base_throughput + random.uniform(-100, 100) for _ in range(num_runs)]
        
        # Calculate statistics
        stats = {}
        for metric, values in results.items():
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        benchmark_result = {
            'raw_results': results,
            'statistics': stats,
            'num_runs': num_runs,
            'timestamp': time.time()
        }
        
        self.results_history.append(benchmark_result)
        return benchmark_result
    
    def compare_models(self, model_results: List[Dict]) -> Dict[str, Any]:
        """Mock model comparison."""
        comparison = {
            'models': [],
            'metrics': {},
            'rankings': {}
        }
        
        for i, result in enumerate(model_results):
            model_name = f'model_{i}'
            comparison['models'].append(model_name)
            
            for metric in result['statistics']:
                if metric not in comparison['metrics']:
                    comparison['metrics'][metric] = {}
                comparison['metrics'][metric][model_name] = result['statistics'][metric]['mean']
        
        # Create rankings
        for metric in comparison['metrics']:
            sorted_models = sorted(
                comparison['metrics'][metric].items(),
                key=lambda x: x[1],
                reverse=metric in ['accuracy', 'throughput']  # Higher is better for these
            )
            comparison['rankings'][metric] = [model for model, _ in sorted_models]
        
        return comparison


class MockHardwareMonitor:
    """Mock hardware monitoring for testing."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        
    def start_monitoring(self, hardware_device):
        """Start monitoring hardware."""
        self.monitoring = True
        self.hardware_device = hardware_device
        
    def stop_monitoring(self):
        """Stop monitoring hardware."""
        self.monitoring = False
        
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current hardware metrics."""
        if not self.monitoring:
            return {}
        
        metrics = {
            'temperature_c': random.uniform(25, 45),
            'power_w': random.uniform(5, 15),
            'utilization_percent': random.uniform(60, 95),
            'memory_usage_mb': random.uniform(100, 500),
            'core_utilization_percent': random.uniform(70, 100)
        }
        
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
        
        return metrics
    
    def get_metrics_history(self) -> List[Dict]:
        """Get historical metrics."""
        return self.metrics_history.copy()
    
    def detect_anomalies(self) -> List[Dict]:
        """Mock anomaly detection."""
        anomalies = []
        
        # Simulate occasional anomalies
        if random.random() < 0.1:  # 10% chance
            anomalies.append({
                'type': 'high_temperature',
                'severity': 'warning',
                'value': 55.0,
                'threshold': 50.0,
                'timestamp': time.time()
            })
        
        if random.random() < 0.05:  # 5% chance
            anomalies.append({
                'type': 'power_spike',
                'severity': 'critical',
                'value': 18.0,
                'threshold': 15.0,
                'timestamp': time.time()
            })
        
        return anomalies


# Context managers for mocking
class MockHardwareContext:
    """Context manager for mocking hardware during tests."""
    
    def __init__(self, hardware_type: str = 'loihi2'):
        self.hardware_type = hardware_type
        self.patches = []
        
    def __enter__(self):
        if self.hardware_type == 'loihi2':
            self.hardware = MockLoihi2Hardware()
            hardware_patch = patch('spikeformer.hardware.loihi2.Loihi2Device', return_value=self.hardware)
        elif self.hardware_type == 'spinnaker':
            self.hardware = MockSpiNNakerHardware()
            hardware_patch = patch('spikeformer.hardware.spinnaker.SpiNNakerDevice', return_value=self.hardware)
        else:
            raise ValueError(f"Unknown hardware type: {self.hardware_type}")
        
        self.patches.append(hardware_patch)
        
        for patch_obj in self.patches:
            patch_obj.start()
        
        return self.hardware
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


class MockConversionContext:
    """Context manager for mocking model conversion during tests."""
    
    def __init__(self, conversion_time: float = 0.1):
        self.converter = MockModelConverter(conversion_time)
        self.patches = []
        
    def __enter__(self):
        converter_patch = patch('spikeformer.conversion.SpikeformerConverter', return_value=self.converter)
        self.patches.append(converter_patch)
        
        for patch_obj in self.patches:
            patch_obj.start()
        
        return self.converter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch_obj in self.patches:
            patch_obj.stop()


# Utility functions for creating mocks
def create_mock_dataset(num_samples: int = 100, input_shape: tuple = (3, 32, 32), num_classes: int = 10):
    """Create a mock dataset for testing."""
    class MockDataset:
        def __init__(self):
            self.data = [
                (torch.randn(*input_shape), torch.randint(0, num_classes, (1,)).item())
                for _ in range(num_samples)
            ]
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MockDataset()


def create_mock_model(input_size: int = 784, output_size: int = 10):
    """Create a mock model for testing."""
    model = Mock()
    model.num_neurons = 1000
    model.num_synapses = 10000
    
    def forward(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, output_size)
    
    model.forward = forward
    model.__call__ = forward
    model.eval = Mock(return_value=model)
    model.train = Mock(return_value=model)
    model.to = Mock(return_value=model)
    
    return model