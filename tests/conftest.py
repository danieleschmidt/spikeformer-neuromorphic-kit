"""Pytest configuration and shared fixtures for SpikeFormer tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

# Test configuration
pytest_plugins = ["pytest_benchmark"]

# Hardware availability markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )
    config.addinivalue_line(
        "markers", "loihi2: mark test as requiring Loihi 2 hardware"
    )
    config.addinivalue_line(
        "markers", "spinnaker: mark test as requiring SpiNNaker hardware"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark integration tests
        if "integration" in str(item.fspath) or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark hardware tests
        if "hardware" in str(item.fspath):
            item.add_marker(pytest.mark.hardware)
            
        # Skip hardware tests if hardware not available
        if item.get_closest_marker("loihi2"):
            if not _is_loihi2_available():
                item.add_marker(pytest.mark.skip(reason="Loihi 2 hardware not available"))
                
        if item.get_closest_marker("spinnaker"):
            if not _is_spinnaker_available():
                item.add_marker(pytest.mark.skip(reason="SpiNNaker hardware not available"))
                
        if item.get_closest_marker("gpu"):
            if not torch.cuda.is_available():
                item.add_marker(pytest.mark.skip(reason="GPU not available"))

def _is_loihi2_available() -> bool:
    """Check if Loihi 2 hardware is available."""
    try:
        import nxsdk
        # Additional checks for actual hardware availability
        return os.environ.get("LOIHI2_AVAILABLE", "false").lower() == "true"
    except ImportError:
        return False

def _is_spinnaker_available() -> bool:
    """Check if SpiNNaker hardware is available."""
    try:
        import spynnaker
        # Additional checks for actual hardware availability
        return os.environ.get("SPINNAKER_AVAILABLE", "false").lower() == "true"
    except ImportError:
        return False

# Base fixtures

@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "timesteps": 32,
        "threshold": 1.0,
        "neuron_model": "LIF",
        "spike_encoding": "rate",
        "surrogate_gradient": "fast_sigmoid",
        "hardware_constraints": {
            "max_fanin": 64,
            "max_fanout": 128,
            "memory_limit_mb": 16
        }
    }

# Model fixtures

@pytest.fixture
def small_vit_config() -> Dict[str, Any]:
    """Small ViT configuration for testing."""
    return {
        "image_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "dim": 64,
        "depth": 2,
        "heads": 4,
        "mlp_dim": 128,
        "dropout": 0.1,
        "emb_dropout": 0.1
    }

@pytest.fixture
def sample_transformer_model(device):
    """Create a small transformer model for testing."""
    from transformers import ViTConfig, ViTModel
    
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        num_labels=10
    )
    
    model = ViTModel(config)
    model.to(device)
    model.eval()
    return model

@pytest.fixture
def sample_input_tensor(device):
    """Create a sample input tensor for testing."""
    return torch.randn(2, 3, 32, 32, device=device)

@pytest.fixture
def sample_spike_tensor(device):
    """Create a sample spike tensor for testing."""
    batch_size, timesteps, channels, height, width = 2, 32, 3, 32, 32
    # Create sparse spike tensor (mostly zeros)
    spikes = torch.zeros(batch_size, timesteps, channels, height, width, device=device)
    # Add some random spikes
    spike_indices = torch.rand_like(spikes) < 0.1  # 10% spike rate
    spikes[spike_indices] = 1.0
    return spikes

# Hardware mock fixtures

@pytest.fixture
def mock_loihi2_backend():
    """Mock Loihi 2 backend for testing."""
    mock_backend = Mock()
    mock_backend.compile.return_value = Mock()
    mock_backend.deploy.return_value = Mock()
    mock_backend.execute.return_value = Mock(
        energy_uj=100.0,
        latency_ms=50.0,
        accuracy=0.85
    )
    return mock_backend

@pytest.fixture
def mock_spinnaker_backend():
    """Mock SpiNNaker backend for testing."""
    mock_backend = Mock()
    mock_backend.compile.return_value = Mock()
    mock_backend.deploy.return_value = Mock()
    mock_backend.execute.return_value = Mock(
        energy_uj=150.0,
        latency_ms=80.0,
        accuracy=0.83
    )
    return mock_backend

@pytest.fixture
def mock_energy_profiler():
    """Mock energy profiler for testing."""
    mock_profiler = Mock()
    mock_profiler.energy_mJ = 10.5
    mock_profiler.power_mW = 250.0
    mock_profiler.gpu_baseline_ratio = 12.0
    return mock_profiler

# Data fixtures

@pytest.fixture
def sample_dataset():
    """Create a small dataset for testing."""
    class MockDataset:
        def __init__(self):
            self.data = [
                (torch.randn(3, 32, 32), torch.tensor(i % 10))
                for i in range(100)
            ]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MockDataset()

@pytest.fixture
def sample_dataloader(sample_dataset):
    """Create a DataLoader for testing."""
    from torch.utils.data import DataLoader
    return DataLoader(sample_dataset, batch_size=8, shuffle=False)

# Benchmark fixtures

@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 3,
        "max_time": 10.0,
        "warmup": True,
        "warmup_iterations": 2
    }

# Environment fixtures

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("NEUROMORPHIC_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("TORCH_HOME", "/tmp/torch_test")
    monkeypatch.setenv("HF_HOME", "/tmp/hf_test")
    
    # Disable external services in tests
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")

# Performance monitoring fixtures

@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    import psutil
    process = psutil.Process()
    
    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def get_memory_usage(self):
            return process.memory_info().rss / 1024 / 1024  # MB
        
        def get_memory_increase(self):
            return self.get_memory_usage() - self.initial_memory
    
    return MemoryMonitor()

# Parametrization helpers

@pytest.fixture(params=["cpu", "cuda"])
def all_devices(request):
    """Parametrize tests across available devices."""
    device_name = request.param
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(device_name)

@pytest.fixture(params=["LIF", "PLIF", "AdLIF"])
def neuron_models(request):
    """Parametrize tests across different neuron models."""
    return request.param

@pytest.fixture(params=["rate", "temporal", "poisson"])
def spike_encodings(request):
    """Parametrize tests across different spike encodings."""
    return request.param

# Utility functions for tests

def assert_spike_tensor_valid(spike_tensor: torch.Tensor):
    """Assert that a tensor is a valid spike tensor."""
    assert spike_tensor.dtype == torch.float32
    assert torch.all(spike_tensor >= 0.0)
    assert torch.all(spike_tensor <= 1.0)
    assert len(spike_tensor.shape) == 5  # [batch, time, channels, height, width]

def assert_energy_reduction(snn_energy: float, ann_energy: float, min_reduction: float = 5.0):
    """Assert that SNN achieves energy reduction over ANN."""
    reduction_factor = ann_energy / snn_energy
    assert reduction_factor >= min_reduction, f"Energy reduction {reduction_factor:.2f}x is below minimum {min_reduction}x"

def assert_accuracy_retention(snn_accuracy: float, ann_accuracy: float, min_retention: float = 0.95):
    """Assert that SNN retains accuracy compared to ANN."""
    retention = snn_accuracy / ann_accuracy
    assert retention >= min_retention, f"Accuracy retention {retention:.3f} is below minimum {min_retention}"