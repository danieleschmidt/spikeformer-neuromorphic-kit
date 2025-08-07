"""Comprehensive validation and error handling for neuromorphic computing."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import warnings
from pathlib import Path
import json
import traceback
from abc import ABC, abstractmethod
from enum import Enum

from .models import SpikingTransformer, SpikingViT, SpikingBERT
from .neurons import SpikingNeuron
from .conversion import ConversionConfig, ConversionResult


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None


class ValidationError(Exception):
    """Custom exception for validation failures."""
    
    def __init__(self, message: str, validation_results: List[ValidationResult]):
        super().__init__(message)
        self.validation_results = validation_results


class Validator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> List[ValidationResult]:
        """Perform validation and return results."""
        pass


class ModelArchitectureValidator(Validator):
    """Validate spiking neural network model architecture."""
    
    def validate(self, model: nn.Module) -> List[ValidationResult]:
        """Validate model architecture for neuromorphic compatibility."""
        results = []
        
        # Check for spiking neurons
        has_spiking_neurons = False
        spiking_layers = 0
        
        for name, module in model.named_modules():
            if isinstance(module, SpikingNeuron):
                has_spiking_neurons = True
                spiking_layers += 1
            elif hasattr(module, 'neuron') and isinstance(module.neuron, SpikingNeuron):
                has_spiking_neurons = True
                spiking_layers += 1
        
        if not has_spiking_neurons:
            results.append(ValidationResult(
                check_name="spiking_neurons",
                level=ValidationLevel.WARNING,
                passed=False,
                message="No spiking neurons detected in model",
                suggested_fix="Convert standard neurons to spiking neurons using ConversionPipeline"
            ))
        else:
            results.append(ValidationResult(
                check_name="spiking_neurons",
                level=ValidationLevel.INFO,
                passed=True,
                message=f"Found {spiking_layers} spiking layers",
                details={"spiking_layers": spiking_layers}
            ))
        
        # Check model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if total_params > 1e9:  # > 1B parameters
            results.append(ValidationResult(
                check_name="model_size",
                level=ValidationLevel.WARNING,
                passed=True,
                message=f"Large model detected: {total_params:,} parameters",
                details={"total_params": total_params, "trainable_params": trainable_params},
                suggested_fix="Consider model pruning or quantization for hardware deployment"
            ))
        
        # Check for temporal dimensions
        has_temporal_processing = self._check_temporal_processing(model)
        if not has_temporal_processing:
            results.append(ValidationResult(
                check_name="temporal_processing",
                level=ValidationLevel.WARNING,
                passed=False,
                message="No temporal processing layers detected",
                suggested_fix="Ensure model processes temporal spike sequences"
            ))
        
        # Check for proper reset mechanisms
        reset_issues = self._check_reset_mechanisms(model)
        results.extend(reset_issues)
        
        # Check threshold configurations
        threshold_issues = self._check_thresholds(model)
        results.extend(threshold_issues)
        
        return results
    
    def _check_temporal_processing(self, model: nn.Module) -> bool:
        """Check if model has temporal processing capabilities."""
        for name, module in model.named_modules():
            if hasattr(module, 'timesteps') and module.timesteps > 1:
                return True
            if 'temporal' in name.lower() or 'spiking' in name.lower():
                return True
        return False
    
    def _check_reset_mechanisms(self, model: nn.Module) -> List[ValidationResult]:
        """Check for proper reset mechanisms in spiking neurons."""
        results = []
        
        for name, module in model.named_modules():
            if isinstance(module, SpikingNeuron):
                if not hasattr(module, 'reset_state'):
                    results.append(ValidationResult(
                        check_name="reset_mechanism",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Spiking neuron {name} missing reset_state method",
                        suggested_fix="Implement reset_state method for proper state management"
                    ))
                elif hasattr(module, 'reset') and module.reset not in ['subtract', 'zero', 'custom']:
                    results.append(ValidationResult(
                        check_name="reset_type",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Unknown reset type '{module.reset}' in {name}",
                        suggested_fix="Use 'subtract', 'zero', or 'custom' reset types"
                    ))
        
        return results
    
    def _check_thresholds(self, model: nn.Module) -> List[ValidationResult]:
        """Check threshold configurations."""
        results = []
        threshold_values = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'threshold'):
                if isinstance(module.threshold, (int, float)):
                    threshold_values.append(module.threshold)
                elif hasattr(module.threshold, 'data'):
                    threshold_values.extend(module.threshold.data.flatten().tolist())
        
        if threshold_values:
            min_thresh = min(threshold_values)
            max_thresh = max(threshold_values)
            
            if min_thresh <= 0:
                results.append(ValidationResult(
                    check_name="threshold_positive",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Non-positive threshold detected: {min_thresh}",
                    suggested_fix="Ensure all thresholds are positive values"
                ))
            
            if max_thresh > 10:
                results.append(ValidationResult(
                    check_name="threshold_range",
                    level=ValidationLevel.WARNING,
                    passed=True,
                    message=f"Very high threshold detected: {max_thresh}",
                    details={"min_threshold": min_thresh, "max_threshold": max_thresh},
                    suggested_fix="Consider lowering thresholds for better spike activity"
                ))
        
        return results


class InputDataValidator(Validator):
    """Validate input data for neuromorphic models."""
    
    def validate(self, data: torch.Tensor, model: nn.Module) -> List[ValidationResult]:
        """Validate input data compatibility."""
        results = []
        
        # Check data type
        if not data.dtype.is_floating_point:
            results.append(ValidationResult(
                check_name="data_type",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Input data type {data.dtype} is not floating point",
                suggested_fix="Convert input to float32 or float16"
            ))
        
        # Check for NaN or inf values
        if torch.isnan(data).any():
            results.append(ValidationResult(
                check_name="nan_values",
                level=ValidationLevel.ERROR,
                passed=False,
                message="Input contains NaN values",
                suggested_fix="Clean input data to remove NaN values"
            ))
        
        if torch.isinf(data).any():
            results.append(ValidationResult(
                check_name="inf_values",
                level=ValidationLevel.ERROR,
                passed=False,
                message="Input contains infinite values",
                suggested_fix="Clip or normalize input values"
            ))
        
        # Check data range
        data_min = data.min().item()
        data_max = data.max().item()
        
        if data_min < -10 or data_max > 10:
            results.append(ValidationResult(
                check_name="data_range",
                level=ValidationLevel.WARNING,
                passed=True,
                message=f"Input data range [{data_min:.3f}, {data_max:.3f}] is large",
                details={"min": data_min, "max": data_max},
                suggested_fix="Consider normalizing input data"
            ))
        
        # Check temporal dimension for spiking models
        if hasattr(model, 'timesteps') or any('spiking' in name.lower() for name, _ in model.named_modules()):
            if data.dim() < 3:
                results.append(ValidationResult(
                    check_name="temporal_dimension",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Input has {data.dim()} dimensions, spiking models typically need temporal dimension",
                    suggested_fix="Add temporal dimension or use spike encoding"
                ))
        
        # Check batch size
        batch_size = data.shape[0] if data.dim() > 0 else 1
        if batch_size > 1000:
            results.append(ValidationResult(
                check_name="batch_size",
                level=ValidationLevel.WARNING,
                passed=True,
                message=f"Large batch size: {batch_size}",
                suggested_fix="Consider reducing batch size for memory efficiency"
            ))
        
        return results


class ConversionValidator(Validator):
    """Validate ANN-to-SNN conversion results."""
    
    def validate(self, original_model: nn.Module, converted_model: nn.Module,
                 conversion_result: ConversionResult) -> List[ValidationResult]:
        """Validate conversion quality and correctness."""
        results = []
        
        # Check accuracy retention
        if conversion_result.accuracy_retention < 0.8:
            results.append(ValidationResult(
                check_name="accuracy_retention",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Low accuracy retention: {conversion_result.accuracy_retention:.1%}",
                details={"accuracy_retention": conversion_result.accuracy_retention},
                suggested_fix="Adjust conversion parameters or use fine-tuning"
            ))
        elif conversion_result.accuracy_retention < 0.5:
            results.append(ValidationResult(
                check_name="accuracy_retention",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Very low accuracy retention: {conversion_result.accuracy_retention:.1%}",
                suggested_fix="Review conversion process and calibration data"
            ))
        
        # Check spike sparsity
        if conversion_result.spike_sparsity < 0.5:
            results.append(ValidationResult(
                check_name="spike_sparsity",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Low spike sparsity: {conversion_result.spike_sparsity:.1%}",
                details={"spike_sparsity": conversion_result.spike_sparsity},
                suggested_fix="Adjust thresholds to increase sparsity"
            ))
        
        # Check parameter count consistency
        orig_params = sum(p.numel() for p in original_model.parameters())
        conv_params = sum(p.numel() for p in converted_model.parameters())
        
        param_ratio = conv_params / orig_params if orig_params > 0 else float('inf')
        
        if param_ratio > 1.5:
            results.append(ValidationResult(
                check_name="parameter_inflation",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Parameter count increased by {param_ratio:.1f}x during conversion",
                details={"original_params": orig_params, "converted_params": conv_params},
                suggested_fix="Review conversion process for parameter efficiency"
            ))
        
        # Check conversion time
        if conversion_result.conversion_time > 3600:  # 1 hour
            results.append(ValidationResult(
                check_name="conversion_time",
                level=ValidationLevel.WARNING,
                passed=True,
                message=f"Long conversion time: {conversion_result.conversion_time:.1f}s",
                suggested_fix="Consider parallel processing or optimized conversion algorithms"
            ))
        
        # Validate converted model architecture
        arch_results = ModelArchitectureValidator().validate(converted_model)
        results.extend(arch_results)
        
        return results


class HardwareCompatibilityValidator(Validator):
    """Validate model compatibility with neuromorphic hardware."""
    
    def validate(self, model: nn.Module, hardware_target: str) -> List[ValidationResult]:
        """Validate hardware compatibility."""
        results = []
        
        # Hardware-specific constraints
        hardware_constraints = {
            'loihi2': {
                'max_fanin': 64,
                'max_fanout': 128,
                'max_synapses_per_core': 1024,
                'synapse_precision': 8,
                'neuron_precision': 24
            },
            'spinnaker': {
                'max_neurons_per_core': 256,
                'max_synapses_per_neuron': 1024,
                'packet_based': True
            },
            'brainsales': {
                'analog_neurons': True,
                'continuous_time': True
            }
        }
        
        if hardware_target not in hardware_constraints:
            results.append(ValidationResult(
                check_name="hardware_support",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Unknown hardware target: {hardware_target}",
                suggested_fix="Use supported hardware targets: loihi2, spinnaker, brainsales"
            ))
            return results
        
        constraints = hardware_constraints[hardware_target]
        
        # Check connectivity constraints
        if 'max_fanin' in constraints or 'max_fanout' in constraints:
            connectivity_issues = self._check_connectivity(model, constraints)
            results.extend(connectivity_issues)
        
        # Check precision requirements
        if 'synapse_precision' in constraints:
            precision_issues = self._check_precision(model, constraints)
            results.extend(precision_issues)
        
        # Check neuron type compatibility
        neuron_issues = self._check_neuron_compatibility(model, hardware_target)
        results.extend(neuron_issues)
        
        return results
    
    def _check_connectivity(self, model: nn.Module, constraints: Dict) -> List[ValidationResult]:
        """Check connectivity constraints."""
        results = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                fanin = module.in_features
                fanout = module.out_features
                
                if 'max_fanin' in constraints and fanin > constraints['max_fanin']:
                    results.append(ValidationResult(
                        check_name="fanin_constraint",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Layer {name} exceeds max fanin: {fanin} > {constraints['max_fanin']}",
                        suggested_fix="Reduce layer width or use sparse connectivity"
                    ))
                
                if 'max_fanout' in constraints and fanout > constraints['max_fanout']:
                    results.append(ValidationResult(
                        check_name="fanout_constraint",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Layer {name} exceeds max fanout: {fanout} > {constraints['max_fanout']}",
                        suggested_fix="Reduce layer width or use multiple cores"
                    ))
        
        return results
    
    def _check_precision(self, model: nn.Module, constraints: Dict) -> List[ValidationResult]:
        """Check precision requirements."""
        results = []
        
        for name, param in model.named_parameters():
            if param.dtype == torch.float64:
                results.append(ValidationResult(
                    check_name="precision_compatibility",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Parameter {name} uses float64, hardware may not support high precision",
                    suggested_fix="Convert to float32 or use quantization"
                ))
        
        return results
    
    def _check_neuron_compatibility(self, model: nn.Module, hardware_target: str) -> List[ValidationResult]:
        """Check neuron type compatibility."""
        results = []
        
        compatible_neurons = {
            'loihi2': ['LIF', 'ADLIF'],
            'spinnaker': ['LIF', 'IZHIKEVICH'],
            'brainsales': ['LIF', 'ADLIF', 'IZHIKEVICH']
        }
        
        for name, module in model.named_modules():
            if hasattr(module, 'neuron_type'):
                neuron_type = getattr(module, 'neuron_type', 'UNKNOWN')
                if neuron_type not in compatible_neurons.get(hardware_target, []):
                    results.append(ValidationResult(
                        check_name="neuron_compatibility",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Neuron type {neuron_type} in {name} not supported by {hardware_target}",
                        suggested_fix=f"Use compatible neuron types: {compatible_neurons.get(hardware_target, [])}"
                    ))
        
        return results


class SecurityValidator(Validator):
    """Validate security aspects of neuromorphic models."""
    
    def validate(self, model: nn.Module, input_data: Optional[torch.Tensor] = None) -> List[ValidationResult]:
        """Validate model security."""
        results = []
        
        # Check for potential backdoors in model weights
        backdoor_results = self._check_backdoors(model)
        results.extend(backdoor_results)
        
        # Check for adversarial robustness
        if input_data is not None:
            robustness_results = self._check_adversarial_robustness(model, input_data)
            results.extend(robustness_results)
        
        # Check for information leakage
        leakage_results = self._check_information_leakage(model)
        results.extend(leakage_results)
        
        # Check for model extraction vulnerability
        extraction_results = self._check_model_extraction_risk(model)
        results.extend(extraction_results)
        
        return results
    
    def _check_backdoors(self, model: nn.Module) -> List[ValidationResult]:
        """Check for potential backdoors in model weights."""
        results = []
        
        # Simple statistical check for unusual weight distributions
        for name, param in model.named_parameters():
            if param.numel() > 100:  # Only check large parameter tensors
                weights = param.data.flatten()
                
                # Check for unusual spikes in weight distribution
                mean_weight = weights.mean().item()
                std_weight = weights.std().item()
                
                outliers = torch.abs(weights - mean_weight) > 5 * std_weight
                outlier_ratio = outliers.float().mean().item()
                
                if outlier_ratio > 0.05:  # More than 5% outliers
                    results.append(ValidationResult(
                        check_name="weight_outliers",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Parameter {name} has {outlier_ratio:.1%} weight outliers",
                        details={"outlier_ratio": outlier_ratio},
                        suggested_fix="Review model source and training process"
                    ))
        
        return results
    
    def _check_adversarial_robustness(self, model: nn.Module, input_data: torch.Tensor) -> List[ValidationResult]:
        """Check adversarial robustness."""
        results = []
        
        # Simple adversarial test with small perturbations
        try:
            model.eval()
            original_output = model(input_data[:1])  # Single sample
            
            # Add small perturbation
            epsilon = 0.01
            perturbation = torch.randn_like(input_data[:1]) * epsilon
            perturbed_input = input_data[:1] + perturbation
            perturbed_output = model(perturbed_input)
            
            # Check output stability
            if hasattr(original_output, 'shape') and hasattr(perturbed_output, 'shape'):
                if original_output.shape == perturbed_output.shape:
                    output_diff = torch.norm(original_output - perturbed_output).item()
                    input_norm = torch.norm(perturbation).item()
                    
                    sensitivity = output_diff / max(input_norm, 1e-8)
                    
                    if sensitivity > 100:  # High sensitivity
                        results.append(ValidationResult(
                            check_name="adversarial_robustness",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"High adversarial sensitivity: {sensitivity:.1f}",
                            details={"sensitivity": sensitivity},
                            suggested_fix="Consider adversarial training or input preprocessing"
                        ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="adversarial_test",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Could not test adversarial robustness: {str(e)}",
                suggested_fix="Ensure model supports single-sample inference"
            ))
        
        return results
    
    def _check_information_leakage(self, model: nn.Module) -> List[ValidationResult]:
        """Check for potential information leakage."""
        results = []
        
        # Check for unencrypted sensitive data in model
        sensitive_patterns = ['password', 'key', 'token', 'secret', 'private']
        
        for name, param in model.named_parameters():
            param_str = str(param.data.cpu().numpy())
            for pattern in sensitive_patterns:
                if pattern in param_str.lower():
                    results.append(ValidationResult(
                        check_name="sensitive_data",
                        level=ValidationLevel.CRITICAL,
                        passed=False,
                        message=f"Potential sensitive data pattern '{pattern}' found in {name}",
                        suggested_fix="Remove sensitive data from model parameters"
                    ))
        
        return results
    
    def _check_model_extraction_risk(self, model: nn.Module) -> List[ValidationResult]:
        """Check model extraction vulnerability."""
        results = []
        
        # Count total parameters - large models are higher risk
        total_params = sum(p.numel() for p in model.parameters())
        
        if total_params > 1e8:  # > 100M parameters
            results.append(ValidationResult(
                check_name="model_extraction_risk",
                level=ValidationLevel.INFO,
                passed=True,
                message=f"Large model ({total_params:,} params) may be vulnerable to extraction attacks",
                details={"total_parameters": total_params},
                suggested_fix="Consider model obfuscation or watermarking for deployment"
            ))
        
        return results


class ComprehensiveValidator:
    """Main validator that runs all validation checks."""
    
    def __init__(self):
        self.validators = {
            'architecture': ModelArchitectureValidator(),
            'input_data': InputDataValidator(),
            'conversion': ConversionValidator(),
            'hardware': HardwareCompatibilityValidator(),
            'security': SecurityValidator(),
        }
        
        self.logger = logging.getLogger(__name__)
    
    def validate_model(self, model: nn.Module, 
                      input_data: Optional[torch.Tensor] = None,
                      hardware_target: Optional[str] = None,
                      conversion_result: Optional[ConversionResult] = None,
                      original_model: Optional[nn.Module] = None,
                      validation_level: ValidationLevel = ValidationLevel.WARNING) -> List[ValidationResult]:
        """Run comprehensive validation checks."""
        
        all_results = []
        
        # Always validate architecture
        try:
            arch_results = self.validators['architecture'].validate(model)
            all_results.extend(arch_results)
        except Exception as e:
            self.logger.error(f"Architecture validation failed: {e}")
            all_results.append(ValidationResult(
                check_name="architecture_validation",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Architecture validation failed: {str(e)}",
                suggested_fix="Check model structure and implementation"
            ))
        
        # Validate input data if provided
        if input_data is not None:
            try:
                input_results = self.validators['input_data'].validate(input_data, model)
                all_results.extend(input_results)
            except Exception as e:
                self.logger.error(f"Input validation failed: {e}")
        
        # Validate hardware compatibility if specified
        if hardware_target is not None:
            try:
                hw_results = self.validators['hardware'].validate(model, hardware_target)
                all_results.extend(hw_results)
            except Exception as e:
                self.logger.error(f"Hardware validation failed: {e}")
        
        # Validate conversion if results provided
        if conversion_result is not None and original_model is not None:
            try:
                conv_results = self.validators['conversion'].validate(
                    original_model, model, conversion_result
                )
                all_results.extend(conv_results)
            except Exception as e:
                self.logger.error(f"Conversion validation failed: {e}")
        
        # Always run security validation
        try:
            security_results = self.validators['security'].validate(model, input_data)
            all_results.extend(security_results)
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
        
        # Filter by validation level
        filtered_results = [r for r in all_results if self._should_include_result(r, validation_level)]
        
        return filtered_results
    
    def _should_include_result(self, result: ValidationResult, min_level: ValidationLevel) -> bool:
        """Check if result should be included based on validation level."""
        level_order = {
            ValidationLevel.INFO: 0,
            ValidationLevel.WARNING: 1, 
            ValidationLevel.ERROR: 2,
            ValidationLevel.CRITICAL: 3
        }
        
        return level_order[result.level] >= level_order[min_level]
    
    def validate_and_raise(self, *args, **kwargs):
        """Run validation and raise exception if critical issues found."""
        results = self.validate_model(*args, **kwargs)
        
        critical_results = [r for r in results if r.level == ValidationLevel.CRITICAL]
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        
        if critical_results or error_results:
            critical_msg = f"Found {len(critical_results)} critical and {len(error_results)} error validation issues"
            raise ValidationError(critical_msg, results)
    
    def generate_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation report."""
        passed = len([r for r in results if r.passed])
        failed = len([r for r in results if not r.passed])
        
        by_level = {}
        for level in ValidationLevel:
            by_level[level.value] = len([r for r in results if r.level == level])
        
        report = {
            "summary": {
                "total_checks": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / len(results) if results else 0.0,
                "by_level": by_level
            },
            "issues": [
                {
                    "check": r.check_name,
                    "level": r.level.value,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "suggested_fix": r.suggested_fix
                }
                for r in results if not r.passed
            ],
            "recommendations": self._generate_recommendations(results)
        }
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations from validation results."""
        recommendations = []
        
        # Collect suggested fixes
        fixes = [r.suggested_fix for r in results if r.suggested_fix and not r.passed]
        
        # Remove duplicates while preserving order
        unique_fixes = []
        for fix in fixes:
            if fix not in unique_fixes:
                unique_fixes.append(fix)
                
        # Add priority recommendations
        critical_issues = [r for r in results if r.level == ValidationLevel.CRITICAL and not r.passed]
        if critical_issues:
            recommendations.append("URGENT: Address critical security or safety issues immediately")
        
        error_issues = [r for r in results if r.level == ValidationLevel.ERROR and not r.passed]
        if error_issues:
            recommendations.append("Address error-level issues before deployment")
        
        recommendations.extend(unique_fixes[:5])  # Top 5 fixes
        
        return recommendations


# Utility functions for easy validation
def validate_model_quick(model: nn.Module) -> bool:
    """Quick validation check - returns True if basic validation passes."""
    validator = ComprehensiveValidator()
    try:
        results = validator.validate_model(model, validation_level=ValidationLevel.ERROR)
        return all(r.passed for r in results)
    except:
        return False


def validate_model_detailed(model: nn.Module, **kwargs) -> Dict[str, Any]:
    """Detailed validation with full report."""
    validator = ComprehensiveValidator()
    results = validator.validate_model(model, **kwargs)
    return validator.generate_report(results)


if __name__ == "__main__":
    # Example usage
    from .models import SpikingTransformer
    
    model = SpikingTransformer(
        vocab_size=1000, hidden_size=256, num_layers=6,
        num_heads=8, intermediate_size=1024, timesteps=32
    )
    
    # Quick validation
    is_valid = validate_model_quick(model)
    print(f"Model validation passed: {is_valid}")
    
    # Detailed validation
    report = validate_model_detailed(model)
    print(f"Validation report: {report['summary']}")