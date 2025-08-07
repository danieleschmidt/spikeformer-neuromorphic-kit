"""Tests for validation and error handling."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from spikeformer.validation import (
    ComprehensiveValidator, ModelArchitectureValidator, InputDataValidator,
    ValidationResult, ValidationLevel, ValidationError
)
from spikeformer.models import SpikingTransformer
from spikeformer.neurons import LifNeuron


class TestModelArchitectureValidator:
    """Test model architecture validation."""
    
    def test_spiking_neurons_detection(self):
        """Test detection of spiking neurons in model."""
        validator = ModelArchitectureValidator()
        
        # Create model with spiking neurons
        model = SpikingTransformer(
            vocab_size=1000, hidden_size=256, num_layers=2,
            num_heads=8, intermediate_size=512, timesteps=32
        )
        
        results = validator.validate(model)
        
        # Should find spiking neurons
        spiking_results = [r for r in results if r.check_name == "spiking_neurons"]
        assert len(spiking_results) == 1
        assert spiking_results[0].passed
        assert spiking_results[0].level == ValidationLevel.INFO
    
    def test_non_spiking_model(self):
        """Test validation of non-spiking model."""
        validator = ModelArchitectureValidator()
        
        # Standard PyTorch model without spiking components
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        results = validator.validate(model)
        
        # Should warn about missing spiking neurons
        spiking_results = [r for r in results if r.check_name == "spiking_neurons"]
        assert len(spiking_results) == 1
        assert not spiking_results[0].passed
        assert spiking_results[0].level == ValidationLevel.WARNING
    
    def test_large_model_warning(self):
        """Test warning for very large models."""
        validator = ModelArchitectureValidator()
        
        # Create artificially large model
        model = nn.Linear(50000, 50000)  # ~2.5B parameters
        
        results = validator.validate(model)
        
        # Should warn about model size
        size_results = [r for r in results if r.check_name == "model_size"]
        assert len(size_results) == 1
        assert size_results[0].level == ValidationLevel.WARNING
    
    def test_threshold_validation(self):
        """Test threshold validation."""
        validator = ModelArchitectureValidator()
        
        # Create neuron with invalid threshold
        class BadNeuron(LifNeuron):
            def __init__(self):
                super().__init__(threshold=-1.0)  # Invalid negative threshold
        
        model = BadNeuron()
        results = validator.validate(model)
        
        # Should error on negative threshold
        threshold_results = [r for r in results if r.check_name == "threshold_positive"]
        assert len(threshold_results) == 1
        assert not threshold_results[0].passed
        assert threshold_results[0].level == ValidationLevel.ERROR


class TestInputDataValidator:
    """Test input data validation."""
    
    def test_valid_input_data(self):
        """Test validation of valid input data."""
        validator = InputDataValidator()
        model = Mock()
        
        # Valid floating point tensor
        data = torch.randn(32, 128, 256)  # batch, seq, features
        
        results = validator.validate(data, model)
        
        # Should pass basic checks
        critical_errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(critical_errors) == 0
    
    def test_invalid_data_type(self):
        """Test validation of invalid data types."""
        validator = InputDataValidator()
        model = Mock()
        
        # Integer tensor instead of float
        data = torch.randint(0, 100, (32, 128))
        
        results = validator.validate(data, model)
        
        # Should error on integer data type
        dtype_results = [r for r in results if r.check_name == "data_type"]
        assert len(dtype_results) == 1
        assert not dtype_results[0].passed
        assert dtype_results[0].level == ValidationLevel.ERROR
    
    def test_nan_values(self):
        """Test detection of NaN values."""
        validator = InputDataValidator()
        model = Mock()
        
        # Data with NaN values
        data = torch.randn(10, 10)
        data[0, 0] = float('nan')
        
        results = validator.validate(data, model)
        
        # Should error on NaN values
        nan_results = [r for r in results if r.check_name == "nan_values"]
        assert len(nan_results) == 1
        assert not nan_results[0].passed
        assert nan_results[0].level == ValidationLevel.ERROR
    
    def test_infinite_values(self):
        """Test detection of infinite values."""
        validator = InputDataValidator()
        model = Mock()
        
        # Data with infinite values
        data = torch.randn(10, 10)
        data[0, 0] = float('inf')
        
        results = validator.validate(data, model)
        
        # Should error on infinite values
        inf_results = [r for r in results if r.check_name == "inf_values"]
        assert len(inf_results) == 1
        assert not inf_results[0].passed
        assert inf_results[0].level == ValidationLevel.ERROR
    
    def test_large_data_range(self):
        """Test warning for large data range."""
        validator = InputDataValidator()
        model = Mock()
        
        # Data with very large range
        data = torch.tensor([[100.0, -100.0]])
        
        results = validator.validate(data, model)
        
        # Should warn about large range
        range_results = [r for r in results if r.check_name == "data_range"]
        assert len(range_results) == 1
        assert range_results[0].level == ValidationLevel.WARNING


class TestComprehensiveValidator:
    """Test comprehensive validation system."""
    
    def test_full_validation(self):
        """Test complete validation pipeline."""
        validator = ComprehensiveValidator()
        
        # Create valid spiking model
        model = SpikingTransformer(
            vocab_size=1000, hidden_size=128, num_layers=2,
            num_heads=4, intermediate_size=256, timesteps=16
        )
        
        # Valid input data
        input_data = torch.randn(2, 10, 128)
        
        results = validator.validate_model(
            model=model,
            input_data=input_data,
            validation_level=ValidationLevel.INFO
        )
        
        # Should have results from multiple validators
        check_names = {r.check_name for r in results}
        assert "spiking_neurons" in check_names
        assert "data_type" in check_names or len([r for r in results if "data" in r.check_name]) > 0
    
    def test_validation_error_raising(self):
        """Test validation error raising for critical issues."""
        validator = ComprehensiveValidator()
        
        # Create model that will fail validation
        model = nn.Linear(10, 10)
        
        # Invalid input data
        input_data = torch.randint(0, 10, (5, 10))  # Integer data
        
        # Should raise ValidationError for critical issues
        with pytest.raises(ValidationError):
            validator.validate_and_raise(
                model=model,
                input_data=input_data
            )
    
    def test_validation_report_generation(self):
        """Test validation report generation."""
        validator = ComprehensiveValidator()
        
        # Create some mock validation results
        results = [
            ValidationResult(
                check_name="test_pass",
                level=ValidationLevel.INFO,
                passed=True,
                message="Test passed"
            ),
            ValidationResult(
                check_name="test_fail",
                level=ValidationLevel.ERROR,
                passed=False,
                message="Test failed",
                suggested_fix="Fix the test"
            )
        ]
        
        report = validator.generate_report(results)
        
        # Check report structure
        assert "summary" in report
        assert "issues" in report
        assert "recommendations" in report
        
        assert report["summary"]["total_checks"] == 2
        assert report["summary"]["passed"] == 1
        assert report["summary"]["failed"] == 1
        assert report["summary"]["pass_rate"] == 0.5
        
        # Check issues are properly formatted
        assert len(report["issues"]) == 1
        assert report["issues"][0]["check"] == "test_fail"
        assert report["issues"][0]["level"] == "error"
        assert not report["issues"][0]["passed"]
    
    def test_validation_level_filtering(self):
        """Test validation level filtering."""
        validator = ComprehensiveValidator()
        
        # Mock model to avoid complex setup
        with patch.object(validator.validators['architecture'], 'validate') as mock_arch:
            mock_arch.return_value = [
                ValidationResult("info_check", ValidationLevel.INFO, True, "Info message"),
                ValidationResult("warn_check", ValidationLevel.WARNING, False, "Warning message"),
                ValidationResult("error_check", ValidationLevel.ERROR, False, "Error message")
            ]
            
            # Test filtering at WARNING level
            results = validator.validate_model(
                model=Mock(),
                validation_level=ValidationLevel.WARNING
            )
            
            # Should only include WARNING and ERROR results
            levels = {r.level for r in results}
            assert ValidationLevel.INFO not in levels
            assert ValidationLevel.WARNING in levels or ValidationLevel.ERROR in levels


class TestValidationUtilities:
    """Test utility functions."""
    
    def test_quick_validation(self):
        """Test quick validation function."""
        from spikeformer.validation import validate_model_quick
        
        # Valid model should pass
        model = SpikingTransformer(
            vocab_size=100, hidden_size=64, num_layers=1,
            num_heads=2, intermediate_size=128, timesteps=8
        )
        
        with patch('spikeformer.validation.ComprehensiveValidator') as mock_validator:
            mock_instance = Mock()
            mock_validator.return_value = mock_instance
            mock_instance.validate_model.return_value = [
                ValidationResult("test", ValidationLevel.INFO, True, "Passed")
            ]
            
            result = validate_model_quick(model)
            assert result is True
    
    def test_detailed_validation(self):
        """Test detailed validation function."""
        from spikeformer.validation import validate_model_detailed
        
        model = Mock()
        
        with patch('spikeformer.validation.ComprehensiveValidator') as mock_validator:
            mock_instance = Mock()
            mock_validator.return_value = mock_instance
            mock_instance.validate_model.return_value = []
            mock_instance.generate_report.return_value = {"summary": {"total_checks": 0}}
            
            report = validate_model_detailed(model)
            assert "summary" in report


if __name__ == "__main__":
    pytest.main([__file__])