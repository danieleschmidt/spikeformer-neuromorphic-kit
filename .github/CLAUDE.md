# Claude Integration Configuration for Spikeformer

This file provides context and instructions for Claude AI assistant when working with the Spikeformer neuromorphic toolkit repository.

## Project Overview

**Spikeformer** is a comprehensive toolkit for training and deploying spiking transformer networks on neuromorphic hardware platforms including Intel Loihi 2 and SpiNNaker.

### Key Technologies
- **Python 3.9+** with PyTorch and neuromorphic libraries
- **Neuromorphic Hardware**: Loihi 2, SpiNNaker, edge deployment
- **Containerization**: Multi-stage Docker builds for different hardware targets
- **CI/CD**: GitHub Actions with comprehensive testing and deployment
- **Monitoring**: Prometheus, Grafana, OpenTelemetry, Loki stack

## Development Commands

### Testing
```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
coverage run -m pytest tests/
coverage report
coverage html

# Run neuromorphic hardware tests (requires hardware)
pytest tests/hardware/ --hardware=loihi2
pytest tests/hardware/ --hardware=spinnaker
```

### Code Quality
```bash
# Format code
black spikeformer/ tests/ examples/
ruff format spikeformer/ tests/ examples/

# Lint code
ruff check spikeformer/ tests/ examples/
pylint spikeformer/

# Type checking
mypy spikeformer/
```

### Building
```bash
# Build Python package
python -m build

# Build Docker images
docker build --target development -t spikeformer:dev .
docker build --target production-cpu -t spikeformer:prod-cpu .
docker build --target loihi2 -t spikeformer:loihi2 .
```

### Security Scanning
```bash
# Run security checks
./scripts/build-and-scan.sh --security-only

# Check dependencies
safety check
bandit -r spikeformer/

# Scan for secrets
detect-secrets scan --all-files
```

### Metrics Collection
```bash
# Collect all metrics
python scripts/collect-metrics.py --category=all --verbose

# Generate metrics report
python scripts/generate-metrics-report.py \
  --input=metrics/collected-metrics.json \
  --output=metrics/report.md

# Check thresholds
python scripts/check-metric-thresholds.py \
  --metrics=metrics/collected-metrics.json \
  --config=.github/project-metrics.json
```

## Repository Structure

```
spikeformer-neuromorphic-kit/
├── spikeformer/           # Main package code
│   ├── core/              # Core spiking neural network implementations
│   ├── models/            # Pre-built model architectures
│   ├── hardware/          # Hardware-specific adapters
│   ├── conversion/        # Model conversion utilities
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── hardware/          # Hardware-specific tests
│   └── utils/             # Test utilities and fixtures
├── docs/                  # Documentation
│   ├── guides/            # User and developer guides
│   ├── api/               # API documentation
│   └── workflows/         # Workflow examples and templates
├── examples/              # Usage examples and tutorials
├── scripts/               # Automation and utility scripts
├── monitoring/            # Monitoring and observability configs
├── .github/               # GitHub workflows and configurations
└── docker/                # Docker configurations
```

## Key Features

### Neuromorphic Model Conversion
- Convert standard transformer models to spiking neural networks
- Optimize for energy efficiency while maintaining accuracy
- Support for multiple hardware targets

### Hardware Integration
- **Intel Loihi 2**: Native support with NXSDK integration
- **SpiNNaker**: PyNN-based implementation
- **Edge Deployment**: Optimized for resource-constrained environments
- **Simulation**: High-fidelity simulation for development and testing

### Performance Optimization
- Automatic spike rate optimization
- Energy consumption tracking and minimization
- Hardware utilization monitoring
- Model compression and quantization

## Development Guidelines

### Code Style
- Use **Black** for consistent formatting
- Follow **PEP 8** guidelines
- Use **type hints** for all functions
- Maximum line length: 88 characters
- Docstrings in **Google style**

### Testing Strategy
- **Unit Tests**: Individual function/class testing
- **Integration Tests**: End-to-end workflow validation
- **Hardware Tests**: Real hardware validation (when available)
- **Performance Tests**: Benchmarking and regression detection
- **Energy Tests**: Power consumption validation

### Commit Message Format
```
type(scope): brief description

Longer description explaining the change in detail.
Include context about why the change was made.

- List specific changes if needed
- Reference issues: Fixes #123
- Breaking changes: BREAKING CHANGE: description
```

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for new features
- `feature/*`: Individual feature development
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation

## AI Assistant Guidelines

### When Working with Neuromorphic Code
1. **Understand Context**: Neuromorphic computing involves spiking neural networks that process information as discrete events (spikes) over time
2. **Energy Efficiency**: Always consider power consumption when making changes
3. **Hardware Constraints**: Different neuromorphic chips have different capabilities and limitations
4. **Temporal Dynamics**: Code often involves time-dependent behavior and spike timing

### Common Tasks
1. **Model Conversion**: Converting traditional ANNs to SNNs while preserving functionality
2. **Hardware Optimization**: Adapting models for specific neuromorphic hardware
3. **Performance Analysis**: Measuring energy efficiency, accuracy retention, and inference speed
4. **Testing**: Validating functionality across simulation and real hardware

### Best Practices
1. **Preserve Existing Patterns**: Follow established code organization and naming conventions
2. **Add Comprehensive Tests**: Include unit, integration, and hardware tests
3. **Update Documentation**: Keep docs current with code changes
4. **Consider Backward Compatibility**: Maintain API stability when possible
5. **Performance Impact**: Measure and document performance implications

### Debugging Neuromorphic Code
1. **Spike Visualization**: Use built-in visualization tools to understand spike patterns
2. **Energy Profiling**: Check power consumption during development
3. **Hardware Logs**: Monitor hardware-specific logs for issues
4. **Simulation vs Reality**: Validate that simulation results match hardware behavior

## Monitoring and Observability

### Key Metrics to Monitor
- **Model Performance**: Accuracy, inference latency, throughput
- **Energy Efficiency**: Power consumption, energy per inference
- **Hardware Utilization**: Chip utilization, memory usage
- **System Health**: Error rates, availability, response times
- **Development Metrics**: Test coverage, build times, deployment frequency

### Available Dashboards
- **Grafana**: System performance and energy metrics
- **Prometheus**: Time-series metrics and alerting
- **Jaeger**: Distributed tracing for complex workflows
- **Loki**: Centralized log aggregation and analysis

## Troubleshooting

### Common Issues
1. **Hardware Not Detected**: Check driver installations and hardware connections
2. **Model Conversion Failures**: Verify model compatibility and conversion parameters
3. **Energy Efficiency Below Target**: Review spike rates and hardware utilization
4. **Test Failures**: Check hardware availability and test environment setup

### Getting Help
1. **Documentation**: Check `docs/` directory for detailed guides
2. **Examples**: Review `examples/` for working code samples
3. **Issues**: Create GitHub issues for bugs or feature requests
4. **Discussions**: Use GitHub Discussions for questions and ideas

## Security Considerations

### Neuromorphic-Specific Security
1. **Model Protection**: Secure model weights and architectures
2. **Hardware Security**: Protect access to neuromorphic hardware
3. **Data Privacy**: Ensure training data confidentiality
4. **Supply Chain**: Verify integrity of hardware drivers and dependencies

### Development Security
1. **Secrets**: Never commit API keys, passwords, or certificates
2. **Dependencies**: Regularly update and scan dependencies
3. **Access Control**: Use principle of least privilege
4. **Code Review**: All changes require peer review

## Performance Baselines

### Expected Performance
- **Model Conversion**: < 5 minutes for standard transformer models
- **Energy Efficiency**: 5-25x improvement over traditional implementations
- **Accuracy Retention**: > 90% of original model accuracy
- **Build Time**: < 15 minutes for full CI/CD pipeline
- **Test Coverage**: > 85% code coverage

### Hardware-Specific Targets
- **Loihi 2**: Ultra-low power operation (< 1W typical)
- **SpiNNaker**: High throughput parallel processing
- **Edge**: < 10MB memory footprint, < 100ms inference

## Environment Variables

```bash
# Hardware configuration
NXSDK_ROOT=/opt/nxsdk              # Loihi 2 SDK path
SPYNNAKER_IP=192.168.1.100         # SpiNNaker board IP

# Development settings
SPIKEFORMER_LOG_LEVEL=INFO         # Logging level
SPIKEFORMER_CACHE_DIR=/tmp/cache   # Cache directory
SPIKEFORMER_HARDWARE=simulation    # Default hardware mode

# Testing configuration
PYTEST_HARDWARE_TESTS=false       # Enable hardware tests
PYTEST_TIMEOUT=300                 # Test timeout in seconds

# Monitoring
PROMETHEUS_GATEWAY=localhost:9091  # Metrics gateway
OTEL_EXPORTER_OTLP_ENDPOINT=...   # OpenTelemetry endpoint
```

This configuration file helps Claude understand the project context, available commands, and best practices when working with the Spikeformer neuromorphic toolkit.