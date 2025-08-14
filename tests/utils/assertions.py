"""Custom assertions for neuromorphic testing."""

import torch
import numpy as np
from typing import Dict, Any, Optional, Union


def assert_spike_tensor_properties(
    spike_tensor: torch.Tensor,
    expected_shape: Optional[tuple] = None,
    expected_dtype: torch.dtype = torch.float32,
    spike_rate_range: tuple = (0.0, 1.0),
    temporal_dimension: int = 1
) -> None:
    """Assert spike tensor has correct properties.
    
    Args:
        spike_tensor: The spike tensor to validate
        expected_shape: Expected tensor shape (optional)
        expected_dtype: Expected data type
        spike_rate_range: Expected range of spike rates (min, max)
        temporal_dimension: Which dimension represents time
    """
    # Basic tensor properties
    assert isinstance(spike_tensor, torch.Tensor), "Input must be torch.Tensor"
    assert spike_tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {spike_tensor.dtype}"
    
    if expected_shape is not None:
        assert spike_tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {spike_tensor.shape}"
    
    # Spike values should be binary (0 or 1) or in [0, 1] range
    assert torch.all(spike_tensor >= 0.0), "Spike values must be non-negative"
    assert torch.all(spike_tensor <= 1.0), "Spike values must not exceed 1.0"
    
    # Check spike rate
    spike_rate = torch.mean(spike_tensor.float())
    assert spike_rate_range[0] <= spike_rate <= spike_rate_range[1], \
        f"Spike rate {spike_rate:.3f} outside expected range {spike_rate_range}"


def assert_energy_efficiency(
    snn_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    min_energy_reduction: float = 5.0,
    max_accuracy_loss: float = 0.05
) -> None:
    """Assert SNN achieves energy efficiency targets.
    
    Args:
        snn_metrics: SNN performance metrics
        baseline_metrics: Baseline (ANN) performance metrics
        min_energy_reduction: Minimum energy reduction factor
        max_accuracy_loss: Maximum acceptable accuracy loss
    """
    # Energy efficiency check
    if 'energy' in snn_metrics and 'energy' in baseline_metrics:
        energy_reduction = baseline_metrics['energy'] / snn_metrics['energy']
        assert energy_reduction >= min_energy_reduction, \
            f"Energy reduction {energy_reduction:.2f}x below minimum {min_energy_reduction}x"
    
    # Accuracy retention check
    if 'accuracy' in snn_metrics and 'accuracy' in baseline_metrics:
        accuracy_loss = baseline_metrics['accuracy'] - snn_metrics['accuracy']
        assert accuracy_loss <= max_accuracy_loss, \
            f"Accuracy loss {accuracy_loss:.3f} exceeds maximum {max_accuracy_loss}"


def assert_hardware_deployment_success(
    deployment_result: Dict[str, Any],
    expected_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """Assert hardware deployment completed successfully.
    
    Args:
        deployment_result: Result from hardware deployment
        expected_metrics: Expected performance metrics (optional)
    """
    # Basic deployment checks
    assert deployment_result.get('status') == 'success', \
        f"Deployment failed: {deployment_result.get('error', 'Unknown error')}"
    
    assert 'compiled_model' in deployment_result, "Missing compiled model"
    assert 'deployment_info' in deployment_result, "Missing deployment info"
    
    # Performance metrics checks
    if expected_metrics:
        metrics = deployment_result.get('metrics', {})
        
        for metric, expected_value in expected_metrics.items():
            actual_value = metrics.get(metric)
            assert actual_value is not None, f"Missing metric: {metric}"
            
            if isinstance(expected_value, dict):
                # Handle range expectations
                if 'min' in expected_value:
                    assert actual_value >= expected_value['min'], \
                        f"{metric} {actual_value} below minimum {expected_value['min']}"
                if 'max' in expected_value:
                    assert actual_value <= expected_value['max'], \
                        f"{metric} {actual_value} above maximum {expected_value['max']}"
            else:
                # Handle exact value expectations
                tolerance = expected_value * 0.1  # 10% tolerance
                assert abs(actual_value - expected_value) <= tolerance, \
                    f"{metric} {actual_value} differs from expected {expected_value} by more than 10%"


def assert_conversion_preserves_semantics(
    original_model: torch.nn.Module,
    converted_model: torch.nn.Module,
    test_inputs: torch.Tensor,
    tolerance: float = 0.1
) -> None:
    """Assert conversion preserves model semantics.
    
    Args:
        original_model: Original ANN model
        converted_model: Converted SNN model
        test_inputs: Test input tensors
        tolerance: Acceptable difference in outputs
    """
    original_model.# eval() removed for security)
    converted_model.# eval() removed for security)
    
    with torch.no_grad():
        original_output = original_model(test_inputs)
        converted_output = converted_model(test_inputs)
        
        # Handle different output formats
        if isinstance(original_output, dict) and isinstance(converted_output, dict):
            for key in original_output.keys():
                if key in converted_output:
                    diff = torch.mean(torch.abs(original_output[key] - converted_output[key]))
                    assert diff <= tolerance, f"Output {key} differs by {diff:.4f}, tolerance {tolerance}"
        else:
            # Handle tensor outputs
            if hasattr(original_output, 'logits'):
                original_output = original_output.logits
            if hasattr(converted_output, 'logits'):
                converted_output = converted_output.logits
                
            diff = torch.mean(torch.abs(original_output - converted_output))
            assert diff <= tolerance, f"Output differs by {diff:.4f}, tolerance {tolerance}"


def assert_spike_timing_properties(
    spike_tensor: torch.Tensor,
    expected_temporal_patterns: Optional[Dict[str, Any]] = None
) -> None:
    """Assert spike timing follows expected patterns.
    
    Args:
        spike_tensor: Spike tensor with temporal dimension
        expected_temporal_patterns: Expected timing patterns
    """
    # Ensure temporal dimension exists
    assert spike_tensor.dim() >= 3, "Spike tensor must have temporal dimension"
    
    # Check for temporal sparsity
    temporal_activity = torch.sum(spike_tensor, dim=tuple(range(2, spike_tensor.dim())))
    non_zero_timesteps = torch.sum(temporal_activity > 0, dim=1).float()
    total_timesteps = spike_tensor.shape[1]
    
    sparsity = non_zero_timesteps / total_timesteps
    assert torch.all(sparsity > 0.0), "No temporal activity detected"
    assert torch.all(sparsity < 1.0), "No temporal sparsity - all timesteps active"
    
    if expected_temporal_patterns:
        # Check specific patterns
        if 'max_burst_length' in expected_temporal_patterns:
            max_burst = expected_temporal_patterns['max_burst_length']
            # Simplified burst detection
            for batch_idx in range(spike_tensor.shape[0]):
                activity = temporal_activity[batch_idx]
                burst_lengths = []
                current_burst = 0
                
                for timestep in range(len(activity)):
                    if activity[timestep] > 0:
                        current_burst += 1
                    else:
                        if current_burst > 0:
                            burst_lengths.append(current_burst)
                            current_burst = 0
                
                if burst_lengths:
                    assert max(burst_lengths) <= max_burst, \
                        f"Burst length {max(burst_lengths)} exceeds maximum {max_burst}"


def assert_model_compatibility(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device = torch.device('cpu')
) -> None:
    """Assert model is compatible with expected inputs and device.
    
    Args:
        model: Model to test
        input_shape: Expected input shape
        device: Target device
    """
    model.to(device)
    model.# eval() removed for security)
    
    # Create test input
    test_input = torch.randn(*input_shape, device=device)
    
    try:
        with torch.no_grad():
            output = model(test_input)
        
        # Basic output checks
        assert output is not None, "Model produced no output"
        assert isinstance(output, (torch.Tensor, dict)), "Invalid output type"
        
        if isinstance(output, torch.Tensor):
            assert output.device == device, f"Output device {output.device} doesn't match expected {device}"
            assert not torch.isnan(output).any(), "Model output contains NaN values"
            assert not torch.isinf(output).any(), "Model output contains infinite values"
            
    except Exception as e:
        raise AssertionError(f"Model compatibility test failed: {str(e)}")


def assert_memory_efficiency(
    operation_func,
    max_memory_mb: float,
    *args,
    **kwargs
) -> None:
    """Assert operation stays within memory limits.
    
    Args:
        operation_func: Function to test
        max_memory_mb: Maximum memory usage in MB
        *args: Arguments for operation_func
        **kwargs: Keyword arguments for operation_func
    """
    import psutil
    import gc
    
    # Clear memory before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Measure initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    try:
        # Execute operation
        result = operation_func(*args, **kwargs)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_increase = peak_gpu_memory - initial_gpu_memory
            
            assert gpu_memory_increase <= max_memory_mb, \
                f"GPU memory usage {gpu_memory_increase:.1f}MB exceeds limit {max_memory_mb}MB"
        
        assert memory_increase <= max_memory_mb, \
            f"Memory usage {memory_increase:.1f}MB exceeds limit {max_memory_mb}MB"
        
        return result
        
    finally:
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assert_reproducible_results(
    operation_func,
    num_runs: int = 3,
    tolerance: float = 1e-6,
    *args,
    **kwargs
) -> None:
    """Assert operation produces reproducible results.
    
    Args:
        operation_func: Function to test
        num_runs: Number of runs to compare
        tolerance: Acceptable difference between runs
        *args: Arguments for operation_func
        **kwargs: Keyword arguments for operation_func
    """
    results = []
    
    for _ in range(num_runs):
        # Set deterministic behavior
        torch.manual_seed(42)
        np.random.seed(42)
        
        result = operation_func(*args, **kwargs)
        results.append(result)
    
    # Compare results
    reference = results[0]
    
    for i, result in enumerate(results[1:], 1):
        if isinstance(reference, torch.Tensor) and isinstance(result, torch.Tensor):
            diff = torch.max(torch.abs(reference - result)).item()
            assert diff <= tolerance, \
                f"Run {i} differs from reference by {diff:.2e}, tolerance {tolerance:.2e}"
        elif isinstance(reference, dict) and isinstance(result, dict):
            for key in reference.keys():
                if key in result:
                    if isinstance(reference[key], torch.Tensor):
                        diff = torch.max(torch.abs(reference[key] - result[key])).item()
                        assert diff <= tolerance, \
                            f"Run {i} key {key} differs by {diff:.2e}, tolerance {tolerance:.2e}"