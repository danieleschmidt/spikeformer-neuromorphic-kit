# Testing Strategy for Spikeformer Neuromorphic Kit

## Overview

Our comprehensive testing strategy ensures reliability, performance, and hardware compatibility across all supported neuromorphic platforms.

## Testing Pyramid

```
    /\     E2E Tests (Hardware Integration)
   /  \    
  /____\   Integration Tests (Component Integration)
 /      \  
/________\  Unit Tests (Individual Functions/Classes)
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions and classes in isolation

**Coverage Target**: 95%+ for core functionality

**Key Areas**:
- Conversion algorithms
- Neuron models (LIF, AdLIF, PLIF)
- Layer implementations
- Utility functions
- Configuration validation

**Example**:
```python
def test_lif_neuron_spike_generation():
    """Test LIF neuron generates spikes correctly."""
    neuron = LIFNeuron(threshold=1.0, reset_potential=0.0)
    
    # Input exceeds threshold - should spike
    membrane_potential = neuron.forward(torch.tensor([1.5]))
    assert membrane_potential < neuron.threshold  # Reset after spike
    assert neuron.spike_count == 1
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and data flow

**Focus Areas**:
- Model conversion end-to-end
- Training pipeline integration
- Data loading and preprocessing
- Export/import functionality

**Example**:
```python
def test_vit_conversion_pipeline():
    """Test complete ViT conversion pipeline."""
    # Load original model
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
    
    # Convert to spiking
    converter = SpikeformerConverter(timesteps=100)
    spiking_vit = converter.convert(vit)
    
    # Test accuracy retention
    accuracy = evaluate_model(spiking_vit, test_dataset)
    assert accuracy > 0.8  # At least 80% of original accuracy
```

### 3. Hardware Tests (`tests/hardware/`)

**Purpose**: Validate functionality on neuromorphic hardware

**Platforms Tested**:
- Intel Loihi 2
- SpiNNaker2
- BrainScaleS (future)
- Edge neuromorphic chips

**Test Execution**:
```bash
# Run with specific hardware
pytest tests/hardware/test_loihi2.py --loihi2
pytest tests/hardware/test_spinnaker.py --spinnaker

# Skip hardware tests in CI
pytest -m "not hardware"
```

### 4. Performance Tests (`tests/performance/`)

**Purpose**: Benchmark performance and resource usage

**Metrics Tracked**:
- Conversion time
- Inference latency
- Memory usage
- Energy consumption
- Throughput

**Benchmarking Framework**:
```python
@pytest.mark.benchmark
def test_conversion_performance(benchmark):
    """Benchmark model conversion performance."""
    model = create_test_model()
    converter = SpikeformerConverter()
    
    result = benchmark(converter.convert, model)
    
    # Performance assertions
    assert benchmark.stats.mean < 60.0  # Under 60 seconds
    assert benchmark.stats.max < 120.0  # Never over 2 minutes
```

### 5. Load Tests (`tests/load/`)

**Purpose**: Test system behavior under high load

**Scenarios**:
- Concurrent model conversions
- Batch processing
- Memory pressure
- Hardware resource contention

**Load Testing with Locust**:
```python
from locust import HttpUser, task

class NeuromorphicUser(HttpUser):
    @task
    def convert_model(self):
        """Simulate model conversion load."""
        model_data = generate_test_model()
        self.client.post("/api/convert", json=model_data)
    
    @task(3)
    def inference_request(self):
        """Simulate inference requests (higher frequency)."""
        input_data = generate_test_input()
        self.client.post("/api/inference", json=input_data)
```

## Test Infrastructure

### Continuous Integration

**GitHub Actions Workflow**:
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -e ".[test]"
      - name: Run unit tests
        run: pytest tests/unit/ --cov=spikeformer
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run integration tests
        run: pytest tests/integration/
  
  hardware-tests:
    runs-on: self-hosted  # Hardware-equipped runners
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - name: Test Loihi 2
        run: pytest tests/hardware/test_loihi2.py --loihi2
```

### Test Data Management

**Fixtures and Test Data**:
```python
@pytest.fixture(scope="session")
def large_test_model():
    """Create large model for performance testing."""
    model = create_vit_model(layers=24, hidden_size=1024)
    return model

@pytest.fixture
def calibration_data():
    """Generate calibration dataset."""
    return torch.randn(1000, 3, 224, 224)

@pytest.fixture
def hardware_config():
    """Hardware configuration for testing."""
    return {
        "loihi2": {"num_chips": 2, "power_budget": 1000},
        "spinnaker": {"boards": 1, "routing": "manual"}
    }
```

### Mock Hardware Testing

**For CI/CD where hardware isn't available**:
```python
class MockLoihi2Device:
    """Mock Loihi 2 device for testing."""
    
    def compile_model(self, model):
        """Simulate model compilation."""
        return MockCompiledModel(model)
    
    def run_inference(self, input_data):
        """Simulate inference execution."""
        time.sleep(0.1)  # Simulate processing time
        return generate_mock_output(input_data.shape)

@pytest.fixture
def mock_loihi2():
    """Provide mock Loihi 2 device."""
    with patch('spikeformer.hardware.loihi2.Loihi2Device', MockLoihi2Device):
        yield MockLoihi2Device()
```

## Quality Gates

### Coverage Requirements

- **Unit Tests**: 95% minimum coverage
- **Integration Tests**: 85% minimum coverage
- **Hardware Tests**: 70% minimum coverage (hardware-dependent)

### Performance Benchmarks

**Conversion Performance**:
- Small models (< 10M params): < 30 seconds
- Medium models (10-100M params): < 5 minutes
- Large models (> 100M params): < 30 minutes

**Inference Performance**:
- Energy efficiency: > 10x improvement over GPU
- Accuracy retention: > 90% of original model
- Latency: Competitive with original model

### Regression Testing

**Automated Regression Detection**:
```python
def test_accuracy_regression():
    """Detect accuracy regressions in converted models."""
    baseline_results = load_baseline_results()
    current_results = run_benchmark_suite()
    
    for model_name, current_acc in current_results.items():
        baseline_acc = baseline_results[model_name]
        regression = (baseline_acc - current_acc) / baseline_acc
        
        assert regression < 0.05, f"Accuracy regression detected: {regression:.2%}"
```

## Test Execution

### Local Development

```bash
# Run all tests
npm run test

# Run specific test categories
npm run test:unit
npm run test:integration
npm run test:performance

# Run with coverage
npm run test:coverage

# Run hardware tests (requires hardware)
npm run test:hardware
```

### CI/CD Pipeline

```bash
# Fast feedback loop
pytest tests/unit/ -x --tb=short

# Full test suite
pytest tests/ --cov=spikeformer --cov-report=html

# Performance regression testing
pytest tests/performance/ --benchmark-only
```

## Test Reporting

### Coverage Reports

Generated automatically and uploaded to:
- Codecov for trend analysis
- GitHub Actions artifacts
- Local HTML reports in `htmlcov/`

### Performance Reports

```python
# Automated performance report generation
def generate_performance_report():
    """Generate comprehensive performance report."""
    results = {
        'conversion_times': benchmark_conversion_times(),
        'memory_usage': profile_memory_usage(),
        'energy_consumption': measure_energy_consumption(),
        'accuracy_metrics': evaluate_accuracy_retention()
    }
    
    generate_html_report(results, 'performance_report.html')
    upload_to_monitoring_system(results)
```

### Hardware Test Reports

```yaml
# Hardware test summary
hardware_results:
  loihi2:
    status: "passed"
    energy_efficiency: "15.2x improvement"
    accuracy_retention: "92.5%"
    latency: "0.8ms"
  spinnaker:
    status: "passed"
    power_consumption: "12.3mW"
    throughput: "1000 samples/sec"
```

## Best Practices

### Test Development

1. **Write tests first** (TDD approach)
2. **Keep tests independent** and isolated
3. **Use descriptive test names** that explain behavior
4. **Mock external dependencies** appropriately
5. **Test both happy path and edge cases**

### Hardware Testing

1. **Isolate hardware-specific code** for easier mocking
2. **Use hardware simulators** when possible
3. **Implement graceful degradation** for hardware failures
4. **Monitor hardware health** during testing
5. **Document hardware setup requirements**

### Performance Testing

1. **Establish baseline metrics** early
2. **Test with realistic data** sizes and patterns
3. **Monitor resource usage** (CPU, memory, GPU)
4. **Include energy profiling** where possible
5. **Test across different hardware configurations**

## Troubleshooting

### Common Test Failures

**Hardware Connection Issues**:
```bash
# Check hardware availability
python -c "from spikeformer.hardware import detect_hardware; print(detect_hardware())"

# Test with mock hardware
pytest tests/hardware/ --mock-hardware
```

**Memory Issues in Large Model Tests**:
```python
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
def test_large_model_conversion():
    """Test conversion of large models."""
    with torch.cuda.device(1):  # Use secondary GPU
        model = create_large_model()
        # Test implementation
```

**Flaky Performance Tests**:
```python
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_performance_benchmark():
    """Performance test with retry logic."""
    # Warm up
    run_warmup_iterations(5)
    
    # Actual benchmark
    results = benchmark_function()
    assert results.mean < threshold
```