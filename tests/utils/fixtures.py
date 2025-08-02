"""Additional test fixtures for neuromorphic testing."""

import pytest
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .generators import SpikeTrainGenerator, ModelGenerator, DatasetGenerator
from .mocks import MockLoihi2Hardware, MockSpiNNakerHardware, MockEnergyProfiler


# Hardware fixtures
@pytest.fixture
def mock_loihi2_device():
    """Mock Loihi 2 device for testing."""
    return MockLoihi2Hardware()


@pytest.fixture
def mock_spinnaker_device():
    """Mock SpiNNaker device for testing."""
    return MockSpiNNakerHardware()


@pytest.fixture
def mock_energy_profiler():
    """Mock energy profiler for testing."""
    return MockEnergyProfiler()


@pytest.fixture(params=['loihi2', 'spinnaker'])
def mock_hardware_device(request):
    """Parametrized mock hardware device."""
    if request.param == 'loihi2':
        return MockLoihi2Hardware()
    else:
        return MockSpiNNakerHardware()


# Model fixtures
@pytest.fixture
def simple_mlp():
    """Simple MLP model for testing."""
    return ModelGenerator.simple_mlp(
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10
    )


@pytest.fixture
def mini_transformer():
    """Mini transformer model for testing."""
    return ModelGenerator.mini_transformer(
        vocab_size=1000,
        d_model=64,
        nhead=4,
        num_layers=2
    )


@pytest.fixture
def conv_net():
    """Convolutional network for testing."""
    return ModelGenerator.conv_net(
        input_channels=3,
        num_classes=10,
        architecture='simple'
    )


@pytest.fixture(params=['simple_mlp', 'conv_net'])
def various_models(request):
    """Parametrized various model architectures."""
    if request.param == 'simple_mlp':
        return ModelGenerator.simple_mlp()
    else:
        return ModelGenerator.conv_net()


# Data generation fixtures
@pytest.fixture
def spike_generator():
    """Spike train generator for testing."""
    return SpikeTrainGenerator(seed=42)


@pytest.fixture
def poisson_spikes(spike_generator, device):
    """Poisson spike trains for testing."""
    return spike_generator.poisson_spikes(
        shape=(2, 50, 10),
        rate=0.2,
        device=device
    )


@pytest.fixture
def regular_spikes(spike_generator, device):
    """Regular spike trains for testing."""
    return spike_generator.regular_spikes(
        shape=(2, 50, 10),
        interval=10,
        device=device
    )


@pytest.fixture
def burst_spikes(spike_generator, device):
    """Burst spike trains for testing."""
    return spike_generator.burst_spikes(
        shape=(2, 50, 10),
        burst_length=5,
        burst_interval=20,
        device=device
    )


@pytest.fixture(params=['poisson', 'regular', 'burst'])
def various_spike_patterns(request, spike_generator, device):
    """Parametrized various spike patterns."""
    shape = (2, 50, 10)
    
    if request.param == 'poisson':
        return spike_generator.poisson_spikes(shape, rate=0.2, device=device)
    elif request.param == 'regular':
        return spike_generator.regular_spikes(shape, interval=10, device=device)
    else:  # burst
        return spike_generator.burst_spikes(shape, burst_length=5, burst_interval=20, device=device)


# Dataset fixtures
@pytest.fixture
def synthetic_classification_data():
    """Synthetic classification dataset."""
    return DatasetGenerator.synthetic_classification(
        num_samples=200,
        num_features=20,
        num_classes=5
    )


@pytest.fixture
def temporal_pattern_data():
    """Temporal pattern recognition dataset."""
    return DatasetGenerator.temporal_pattern_dataset(
        num_samples=100,
        sequence_length=50,
        num_features=10,
        num_patterns=3
    )


@pytest.fixture
def spike_pattern_data():
    """Spike pattern classification dataset."""
    return DatasetGenerator.spike_pattern_dataset(
        num_samples=60,
        timesteps=30,
        num_neurons=15
    )


# Configuration fixtures
@pytest.fixture
def basic_conversion_config():
    """Basic conversion configuration."""
    return {
        'timesteps': 32,
        'threshold': 1.0,
        'neuron_model': 'LIF',
        'spike_encoding': 'rate',
        'surrogate_gradient': 'fast_sigmoid'
    }


@pytest.fixture
def advanced_conversion_config():
    """Advanced conversion configuration."""
    return {
        'timesteps': 100,
        'threshold': 0.8,
        'neuron_model': 'AdLIF',
        'spike_encoding': 'temporal',
        'surrogate_gradient': 'super_spike',
        'adaptive_threshold': True,
        'adaptation_rate': 0.1,
        'noise_level': 0.01,
        'dropout_rate': 0.1
    }


@pytest.fixture(params=['basic', 'advanced'])
def various_conversion_configs(request, basic_conversion_config, advanced_conversion_config):
    """Parametrized conversion configurations."""
    if request.param == 'basic':
        return basic_conversion_config
    else:
        return advanced_conversion_config


@pytest.fixture
def loihi2_hardware_config():
    """Loihi 2 hardware configuration."""
    return {
        'hardware': 'loihi2',
        'num_chips': 2,
        'core_allocation': 'automatic',
        'precision': 8,
        'max_fanin': 64,
        'max_fanout': 128,
        'routing_algorithm': 'shortest_path'
    }


@pytest.fixture
def spinnaker_hardware_config():
    """SpiNNaker hardware configuration."""
    return {
        'hardware': 'spinnaker',
        'num_boards': 1,
        'routing_algorithm': 'neighbor_aware',
        'time_scale_factor': 1000,
        'packet_ordering': True,
        'live_output': False
    }


@pytest.fixture(params=['loihi2', 'spinnaker'])
def various_hardware_configs(request, loihi2_hardware_config, spinnaker_hardware_config):
    """Parametrized hardware configurations."""
    if request.param == 'loihi2':
        return loihi2_hardware_config
    else:
        return spinnaker_hardware_config


# File and directory fixtures
@pytest.fixture
def test_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "test_data"
        data_dir.mkdir(exist_ok=True)
        yield data_dir


@pytest.fixture
def test_model_file(test_data_dir):
    """Temporary model file for testing."""
    model_file = test_data_dir / "test_model.pth"
    
    # Create a simple model and save it
    model = ModelGenerator.simple_mlp()
    torch.save(model.state_dict(), model_file)
    
    return model_file


@pytest.fixture
def test_config_file(test_data_dir):
    """Temporary configuration file for testing."""
    import json
    
    config_file = test_data_dir / "test_config.json"
    config = {
        'timesteps': 50,
        'threshold': 1.2,
        'neuron_model': 'LIF',
        'hardware_target': 'loihi2'
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f)
    
    return config_file


# Performance and monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Performance monitoring utility."""
    import time
    import psutil
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.peak_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
        def stop(self):
            self.end_time = time.time()
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory or 0, current_memory)
        
        @property
        def duration_seconds(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        @property
        def memory_increase_mb(self):
            if self.start_memory and self.peak_memory:
                return self.peak_memory - self.start_memory
            return None
    
    return PerformanceMonitor()


@pytest.fixture
def gpu_memory_monitor():
    """GPU memory monitoring utility."""
    class GPUMemoryMonitor:
        def __init__(self):
            self.start_memory = 0
            self.peak_memory = 0
            self.available = torch.cuda.is_available()
        
        def start(self):
            if self.available:
                torch.cuda.empty_cache()
                self.start_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        def update_peak(self):
            if self.available:
                current = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.peak_memory = max(self.peak_memory, current)
        
        @property
        def memory_increase_mb(self):
            if self.available:
                return self.peak_memory - self.start_memory
            return 0
    
    return GPUMemoryMonitor()


# Benchmark fixtures
@pytest.fixture
def benchmark_suite():
    """Benchmark suite configuration."""
    return {
        'models': ['vit-tiny', 'vit-small'],
        'hardware': ['cpu', 'loihi2'],
        'metrics': ['accuracy', 'latency', 'energy'],
        'batch_sizes': [1, 8],
        'num_runs': 3,
        'warmup_runs': 1
    }


@pytest.fixture
def accuracy_tolerance():
    """Accuracy tolerance configuration for tests."""
    return {
        'high_precision': 0.01,
        'normal': 0.05,
        'relaxed': 0.1
    }


@pytest.fixture
def timing_tolerances():
    """Timing tolerance configuration for tests."""
    return {
        'strict': 0.1,   # 10% tolerance
        'normal': 0.2,   # 20% tolerance
        'relaxed': 0.5   # 50% tolerance
    }


# Integration test fixtures
@pytest.fixture
def end_to_end_pipeline():
    """End-to-end pipeline configuration."""
    return {
        'input_model': 'vit-tiny',
        'conversion_config': {
            'timesteps': 32,
            'threshold': 1.0,
            'neuron_model': 'LIF'
        },
        'hardware_config': {
            'hardware': 'loihi2',
            'num_chips': 1
        },
        'evaluation_config': {
            'dataset': 'cifar10_subset',
            'num_samples': 100,
            'metrics': ['accuracy', 'energy']
        }
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_torch_cache():
    """Automatically clean up PyTorch cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    import numpy as np
    import random
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    yield


# Parametrized device fixtures
@pytest.fixture(params=["cpu"])
def available_devices(request):
    """Parametrize tests only across actually available devices."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)


@pytest.fixture(params=["float32", "float16"])
def precision_modes(request):
    """Parametrize tests across different precision modes."""
    precision = request.param
    if precision == "float16" and not torch.cuda.is_available():
        pytest.skip("float16 requires CUDA")
    return getattr(torch, precision)