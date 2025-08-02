# Developer Contributing Guide

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized development)
- Access to target hardware (optional, for hardware testing)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/spikeformer-neuromorphic-kit
cd spikeformer-neuromorphic-kit

# Set up development environment
npm run setup  # Installs dependencies and sets up pre-commit hooks

# Or manually with pip
pip install -e ".[dev]"
pre-commit install
```

### Using UV (Recommended)

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project
uv sync --dev
pre-commit install
```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/feature-name`: Individual feature branches
- `hotfix/issue-name`: Critical bug fixes
- `release/version`: Release preparation

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and quality checks**
   ```bash
   npm run test           # Run all tests
   npm run lint           # Lint code
   npm run typecheck      # Type checking
   npm run format         # Format code
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
   
   We use [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test additions/modifications
   - `refactor:` Code refactoring
   - `perf:` Performance improvements
   - `ci:` CI/CD changes

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

### Python Style Guide

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort (profile: black)
- **Linting**: Ruff
- **Type checking**: MyPy
- **Docstrings**: Google style

### Example Code Structure

```python
"""Module for spiking neural network conversion.

This module provides utilities for converting traditional neural networks
to spiking neural networks with configurable parameters.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class SpikeConverter:
    """Converts traditional neural networks to spiking neural networks.
    
    Args:
        timesteps: Number of simulation timesteps.
        threshold: Firing threshold for neurons.
        reset_mode: How neurons reset after firing ('zero' or 'subtract').
        
    Example:
        >>> converter = SpikeConverter(timesteps=100, threshold=1.0)
        >>> spiking_model = converter.convert(original_model)
    """
    
    def __init__(
        self,
        timesteps: int = 100,
        threshold: float = 1.0,
        reset_mode: str = "subtract"
    ) -> None:
        self.timesteps = timesteps
        self.threshold = threshold
        self.reset_mode = reset_mode
    
    def convert(self, model: nn.Module) -> nn.Module:
        """Convert a model to its spiking equivalent.
        
        Args:
            model: The PyTorch model to convert.
            
        Returns:
            Converted spiking neural network.
            
        Raises:
            ValueError: If model contains unsupported layers.
        """
        # Implementation here
        pass
```

### Testing Standards

- **Framework**: pytest
- **Coverage**: Minimum 80% coverage required
- **Test Structure**: Arrange-Act-Assert pattern
- **Test Categories**:
  - Unit tests: `tests/unit/`
  - Integration tests: `tests/integration/`
  - Hardware tests: `tests/hardware/`
  - Performance tests: `tests/performance/`

### Example Test

```python
"""Tests for spike conversion functionality."""

import pytest
import torch
import torch.nn as nn
from spikeformer.conversion import SpikeConverter


class TestSpikeConverter:
    """Test suite for SpikeConverter class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return SpikeConverter(timesteps=100, threshold=1.0)
    
    def test_converter_initialization(self, converter):
        """Test converter initializes with correct parameters."""
        assert converter.timesteps == 100
        assert converter.threshold == 1.0
        assert converter.reset_mode == "subtract"
    
    def test_model_conversion(self, converter, simple_model):
        """Test basic model conversion functionality."""
        spiking_model = converter.convert(simple_model)
        
        # Test that conversion produces a valid model
        assert isinstance(spiking_model, nn.Module)
        
        # Test forward pass works
        test_input = torch.randn(1, 10)
        output = spiking_model(test_input)
        assert output.shape == (1, 1)
    
    @pytest.mark.parametrize("timesteps,threshold", [
        (50, 0.5),
        (200, 2.0),
        (1000, 0.1),
    ])
    def test_different_parameters(self, timesteps, threshold, simple_model):
        """Test conversion with different parameter combinations."""
        converter = SpikeConverter(timesteps=timesteps, threshold=threshold)
        spiking_model = converter.convert(simple_model)
        
        test_input = torch.randn(1, 10)
        output = spiking_model(test_input)
        assert output is not None
```

## Hardware Testing

### Loihi 2 Testing

```bash
# Run hardware-specific tests (requires Loihi 2 access)
pytest tests/hardware/test_loihi2.py --loihi2

# Skip hardware tests in regular CI
pytest tests/ -m "not hardware"
```

### SpiNNaker Testing

```bash
# Run SpiNNaker tests (requires SpiNNaker access)
pytest tests/hardware/test_spinnaker.py --spinnaker
```

## Documentation

### API Documentation

- Use Google-style docstrings
- Include type hints
- Provide examples in docstrings
- Auto-generated with Sphinx

### User Documentation

- Markdown format in `docs/guides/`
- Include code examples
- Keep explanations clear and concise
- Test all code examples

## Performance Considerations

### Benchmarking

```python
# Add performance tests for new features
import pytest_benchmark

def test_conversion_performance(benchmark, converter, large_model):
    """Benchmark conversion performance."""
    result = benchmark(converter.convert, large_model)
    assert result is not None
```

### Memory Management

- Use torch.no_grad() for inference
- Clear GPU cache when appropriate
- Monitor memory usage in tests
- Profile memory-intensive operations

## Security Guidelines

### Code Security

- Never commit secrets or API keys
- Use `.env` files for local configuration
- Sanitize all inputs
- Follow security best practices for dependencies

### Hardware Security

- Validate hardware configurations
- Implement secure communication protocols
- Monitor for hardware tampering
- Follow hardware vendor security guidelines

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release branch
5. Tag release
6. Deploy to PyPI (automated)
7. Update documentation

## Getting Help

### Code Review Process

- All changes require at least one review
- Automated checks must pass
- Consider performance impact
- Maintain backward compatibility when possible

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Design discussions and questions
- **Discord**: Real-time developer chat
- **Email**: Private security concerns

### Mentorship

New contributors are paired with experienced developers for:
- Code review guidance
- Architecture discussions
- Best practice training
- Hardware access coordination