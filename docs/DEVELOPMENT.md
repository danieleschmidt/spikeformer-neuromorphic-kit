# SpikeFormer Development Guide

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- (Optional) Intel NxSDK for Loihi 2 development
- (Optional) sPyNNaker for SpiNNaker development

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/spikeformer-neuromorphic-kit
cd spikeformer-neuromorphic-kit

# Setup development environment
make install-dev

# Start development services
make run-dev
```

### Development Environment Options

#### Option 1: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run tests
make test
```

#### Option 2: Docker Development

```bash
# Build and run development container
make build-docker-dev
make run-dev

# Access Jupyter Lab at http://localhost:8888
```

#### Option 3: DevContainer (VS Code)

1. Install VS Code and the Remote-Containers extension
2. Open the repository in VS Code
3. Click "Reopen in Container" when prompted
4. The development environment will be automatically configured

## Project Structure

```
spikeformer-neuromorphic-kit/
├── spikeformer/           # Main package
│   ├── conversion/        # ANN to SNN conversion
│   ├── training/          # Training frameworks
│   ├── hardware/          # Hardware backends
│   ├── models/           # Model architectures
│   ├── monitoring/       # Metrics and monitoring
│   └── cli/              # Command-line interface
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── hardware/        # Hardware-specific tests
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── monitoring/          # Monitoring configurations
└── examples/            # Usage examples
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards:
- Use type hints for all functions
- Add docstrings for public APIs
- Write tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make typecheck

# Run tests
make test

# Run all checks
make ci-all
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add spike rate optimization for ViT models"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use isort for import organization
- Use type hints consistently

### Example Code Structure

```python
"""Module docstring explaining the purpose."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


class SpikingLayer(nn.Module):
    """A spiking neural network layer.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        threshold: Firing threshold
        
    Example:
        >>> layer = SpikingLayer(128, 64, threshold=1.0)
        >>> spikes = layer(input_tensor)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        threshold: float = 1.0
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking layer."""
        membrane_potential = self.linear(x)
        spikes = (membrane_potential >= self.threshold).float()
        return spikes
```

### Documentation Standards

- Use Google-style docstrings
- Include type annotations
- Provide usage examples
- Document hardware-specific considerations

## Testing Guidelines

### Test Organization

```python
# tests/unit/test_conversion.py
import pytest
import torch
from spikeformer.conversion import SpikeformerConverter


class TestSpikeformerConverter:
    """Test suite for SpikeformerConverter."""
    
    def test_basic_conversion(self, sample_transformer_model):
        """Test basic model conversion."""
        converter = SpikeformerConverter(timesteps=32)
        snn_model = converter.convert(sample_transformer_model)
        
        assert snn_model is not None
        # Add specific assertions
        
    @pytest.mark.hardware
    def test_hardware_deployment(self, mock_loihi2_backend):
        """Test deployment to hardware backend."""
        # Hardware-specific tests
        pass
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-hardware  # Requires hardware

# Run with coverage
make test-coverage

# Run performance benchmarks
make benchmark
```

## Hardware Development

### Loihi 2 Development

```python
from spikeformer.hardware.loihi2 import Loihi2Backend

# Initialize backend
backend = Loihi2Backend(num_chips=2)

# Compile model for Loihi 2
compiled_model = backend.compile(snn_model)

# Deploy and run
deployed_model = backend.deploy(compiled_model)
results = deployed_model.run(input_spikes)
```

### SpiNNaker Development

```python
from spikeformer.hardware.spinnaker import SpiNNakerBackend

# Initialize backend
backend = SpiNNakerBackend(board_config="spin2-48chip")

# Deploy model
deployed_model = backend.deploy(snn_model)
results = deployed_model.run(input_data)
```

## Adding New Features

### 1. Adding a New Neuron Model

```python
# spikeformer/models/neurons.py
class NewNeuronModel(nn.Module):
    """Custom neuron model implementation."""
    
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement neuron dynamics
        return spikes
```

### 2. Adding a New Hardware Backend

```python
# spikeformer/hardware/new_backend.py
from .base import NeuromorphicBackend

class NewHardwareBackend(NeuromorphicBackend):
    """Backend for new neuromorphic hardware."""
    
    def compile(self, model):
        # Implement compilation
        pass
        
    def deploy(self, compiled_model):
        # Implement deployment
        pass
```

### 3. Adding New CLI Commands

```python
# spikeformer/cli/new_command.py
import click

@click.command()
@click.option('--input', required=True, help='Input model path')
def new_command(input):
    """New CLI command description."""
    # Implement command logic
    pass
```

## Debugging

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # Reduce batch size
   batch_size = 8  # Instead of 32
   ```

2. **Hardware Connection Issues**
   ```bash
   # Check hardware availability
   python -c "import nxsdk; print('Loihi 2 available')"
   python -c "import spynnaker; print('SpiNNaker available')"
   ```

3. **Performance Issues**
   ```bash
   # Profile code
   make profile
   
   # Check memory usage
   python -m memory_profiler your_script.py
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with profiling
python -m cProfile -o profile.stats your_script.py
```

## Performance Optimization

### Memory Optimization

- Use gradient checkpointing for large models
- Implement model parallelism for multi-GPU setups
- Use mixed precision training where appropriate

### Compute Optimization

- Leverage sparse operations for spike tensors
- Use hardware-specific optimizations
- Profile and optimize hot paths

## Release Process

### Version Bumping

```bash
# Bump version
bump2version patch  # or minor, major

# Build and test
make build
make test-all

# Tag release
git tag v0.1.1
git push origin v0.1.1
```

### Documentation Updates

- Update CHANGELOG.md
- Update version in documentation
- Verify all examples work with new version

## Contributing Guidelines

1. **Fork the repository**
2. **Create feature branch** from `develop`
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run quality checks** locally
7. **Submit pull request** with clear description
8. **Address review feedback**

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Performance impact considered
- [ ] Hardware compatibility verified

## Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Community**: Join our Discord server
- **Email**: developers@your-org.com

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Neuromorphic Computing Primer](https://docs.your-org.com/primer)
- [Intel Loihi Documentation](https://intel-ncl.atlassian.net/)
- [SpiNNaker Documentation](http://spinnakermanchester.github.io/)