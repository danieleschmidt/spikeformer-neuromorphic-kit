"""Robustness enhancements for neuromorphic computing systems."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager
from abc import ABC, abstractmethod

from .error_handling import safe_execute, ErrorSeverity, error_handler
from .validation import Validator, ValidationResult, ValidationLevel
from .neurons import LifNeuron, SpikingNeuron
from .security import SecurityConfig, NeuromorphicSecurityManager
from .monitoring import HealthMonitor, PerformanceProfiler


@dataclass
class RobustnessConfig:
    """Configuration for robustness features."""
    enable_gradient_clipping: bool = True
    gradient_clip_value: float = 1.0
    enable_weight_decay: bool = True
    weight_decay: float = 1e-4
    enable_dropout: bool = True
    dropout_rate: float = 0.1
    enable_batch_normalization: bool = True
    enable_early_stopping: bool = True
    patience: int = 10
    enable_lr_scheduling: bool = True


class AdversarialRobustness:
    """Adversarial robustness testing for spiking neural networks."""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1, num_steps: int = 10):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.logger = logging.getLogger(__name__)
    
    def fgsm_attack(self, inputs: torch.Tensor, labels: torch.Tensor, 
                    criterion: nn.Module) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        inputs.requires_grad_(True)
        
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial examples
        gradient_sign = inputs.grad.sign()
        adversarial_inputs = inputs + self.epsilon * gradient_sign
        
        return adversarial_inputs.detach()
    
    def pgd_attack(self, inputs: torch.Tensor, labels: torch.Tensor,
                   criterion: nn.Module, alpha: float = 0.01) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        original_inputs = inputs.clone().detach()
        adversarial_inputs = inputs.clone().detach()
        
        for step in range(self.num_steps):
            adversarial_inputs.requires_grad_(True)
            
            outputs = self.model(adversarial_inputs)
            loss = criterion(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial inputs
            gradient = adversarial_inputs.grad
            adversarial_inputs = adversarial_inputs + alpha * gradient.sign()
            
            # Project to epsilon ball
            perturbation = adversarial_inputs - original_inputs
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            adversarial_inputs = original_inputs + perturbation
            adversarial_inputs = adversarial_inputs.detach()
        
        return adversarial_inputs
    
    def evaluate_robustness(self, dataloader, criterion) -> Dict[str, float]:
        """Evaluate model robustness against adversarial attacks."""
        self.model.eval()
        
        clean_accuracy = 0.0
        fgsm_accuracy = 0.0
        pgd_accuracy = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Clean accuracy
                clean_outputs = self.model(inputs)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_accuracy += (clean_pred == labels).sum().item()
                
                # FGSM adversarial accuracy
                fgsm_inputs = self.fgsm_attack(inputs, labels, criterion)
                fgsm_outputs = self.model(fgsm_inputs)
                fgsm_pred = fgsm_outputs.argmax(dim=1)
                fgsm_accuracy += (fgsm_pred == labels).sum().item()
                
                # PGD adversarial accuracy
                pgd_inputs = self.pgd_attack(inputs, labels, criterion)
                pgd_outputs = self.model(pgd_inputs)
                pgd_pred = pgd_outputs.argmax(dim=1)
                pgd_accuracy += (pgd_pred == labels).sum().item()
        
        return {
            'clean_accuracy': clean_accuracy / total_samples,
            'fgsm_accuracy': fgsm_accuracy / total_samples,
            'pgd_accuracy': pgd_accuracy / total_samples,
            'robustness_score': (fgsm_accuracy + pgd_accuracy) / (2 * total_samples)
        }


class NoiseRobustness:
    """Robustness testing against various noise types."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.noise_types = {
            'gaussian': self._gaussian_noise,
            'salt_pepper': self._salt_pepper_noise,
            'poisson': self._poisson_noise,
            'speckle': self._speckle_noise
        }
    
    def _gaussian_noise(self, inputs: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(inputs) * noise_level
        return inputs + noise
    
    def _salt_pepper_noise(self, inputs: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add salt and pepper noise."""
        noisy_inputs = inputs.clone()
        mask = torch.rand_like(inputs) < noise_level
        noisy_inputs[mask] = torch.randint_like(inputs[mask], 0, 2).float()
        return noisy_inputs
    
    def _poisson_noise(self, inputs: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Poisson noise."""
        # Scale inputs to be non-negative for Poisson
        scaled_inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        noisy_inputs = torch.poisson(scaled_inputs * noise_level) / noise_level
        return noisy_inputs * (inputs.max() - inputs.min()) + inputs.min()
    
    def _speckle_noise(self, inputs: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add speckle noise."""
        noise = torch.randn_like(inputs)
        return inputs + inputs * noise * noise_level
    
    def evaluate_noise_robustness(self, dataloader, noise_levels: List[float] = None,
                                noise_types: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness against different noise types and levels."""
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.5]
        if noise_types is None:
            noise_types = ['gaussian', 'salt_pepper']
        
        self.model.eval()
        results = {}
        
        for noise_type in noise_types:
            results[noise_type] = {}
            
            for noise_level in noise_levels:
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in dataloader:
                        # Add noise
                        noisy_inputs = self.noise_types[noise_type](inputs, noise_level)
                        
                        # Forward pass
                        outputs = self.model(noisy_inputs)
                        pred = outputs.argmax(dim=1)
                        
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                
                results[noise_type][str(noise_level)] = correct / total
        
        return results


class QuantizationRobustness:
    """Test robustness to quantization for deployment."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def apply_weight_quantization(self, bits: int = 8) -> nn.Module:
        """Apply weight quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def apply_activation_quantization(self, inputs: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Apply activation quantization."""
        # Simple linear quantization
        input_min, input_max = inputs.min(), inputs.max()
        scale = (input_max - input_min) / (2**bits - 1)
        zero_point = input_min
        
        quantized = torch.round((inputs - zero_point) / scale)
        quantized = torch.clamp(quantized, 0, 2**bits - 1)
        
        # Dequantize
        return quantized * scale + zero_point
    
    def evaluate_quantization_robustness(self, dataloader, 
                                       quantization_bits: List[int] = None) -> Dict[str, float]:
        """Evaluate robustness to different quantization levels."""
        if quantization_bits is None:
            quantization_bits = [8, 4, 2]
        
        results = {}
        
        # Baseline accuracy
        self.model.eval()
        baseline_accuracy = self._compute_accuracy(dataloader)
        results['baseline'] = baseline_accuracy
        
        for bits in quantization_bits:
            try:
                # Weight quantization
                quantized_model = self.apply_weight_quantization(bits)
                quantized_accuracy = self._compute_accuracy(dataloader, quantized_model)
                results[f'{bits}_bit_weights'] = quantized_accuracy
                
                # Activation quantization
                activation_accuracy = self._compute_accuracy_with_activation_quantization(
                    dataloader, bits
                )
                results[f'{bits}_bit_activations'] = activation_accuracy
                
            except Exception as e:
                results[f'{bits}_bit_error'] = str(e)
        
        return results
    
    def _compute_accuracy(self, dataloader, model: nn.Module = None) -> float:
        """Compute model accuracy."""
        if model is None:
            model = self.model
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def _compute_accuracy_with_activation_quantization(self, dataloader, bits: int) -> float:
        """Compute accuracy with activation quantization."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Quantize inputs
                quantized_inputs = self.apply_activation_quantization(inputs, bits)
                
                outputs = self.model(quantized_inputs)
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return correct / total


class RobustTraining:
    """Robust training procedures with error recovery."""
    
    def __init__(self, config: RobustnessConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.learning_rates = []
    
    @safe_execute("robust_training_step", "training")
    def training_step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                     criterion: nn.Module, inputs: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """Robust training step with error handling."""
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Add regularization
        if self.config.enable_weight_decay:
            l2_reg = sum(param.pow(2).sum() for param in model.parameters())
            loss += self.config.weight_decay * l2_reg
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.enable_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.gradient_clip_value
            )
        
        # Check for gradient anomalies
        self._check_gradients(model)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            accuracy = (pred == labels).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'lr': optimizer.param_groups[0]['lr']
        }
        
        return loss.item(), metrics
    
    def _check_gradients(self, model: nn.Module):
        """Check for gradient anomalies."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                if torch.isnan(grad).any():
                    error_handler.log_error(error_handler.create_error(
                        error_type="gradient_nan",
                        severity=ErrorSeverity.ERROR,
                        message=f"NaN gradients in {name}",
                        context={"parameter": name}
                    ))
                
                if torch.isinf(grad).any():
                    error_handler.log_error(error_handler.create_error(
                        error_type="gradient_inf",
                        severity=ErrorSeverity.ERROR,
                        message=f"Infinite gradients in {name}",
                        context={"parameter": name}
                    ))
    
    def should_early_stop(self, validation_loss: float) -> bool:
        """Check if training should stop early."""
        if not self.config.enable_early_stopping:
            return False
        
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                return True
        
        return False
    
    def update_learning_rate(self, scheduler: torch.optim.lr_scheduler._LRScheduler,
                           validation_loss: float):
        """Update learning rate with scheduling."""
        if self.config.enable_lr_scheduling:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            else:
                scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)


class FaultTolerantInference:
    """Fault-tolerant inference with redundancy and error correction."""
    
    def __init__(self, model: nn.Module, num_replicas: int = 3,
                 consensus_threshold: float = 0.6):
        self.model = model
        self.num_replicas = num_replicas
        self.consensus_threshold = consensus_threshold
        self.replicas = [self._create_replica() for _ in range(num_replicas)]
    
    def _create_replica(self) -> nn.Module:
        """Create a replica of the model with slight variations."""
        replica = type(self.model)(**self.model.init_args if hasattr(self.model, 'init_args') else {})
        
        # Copy weights with small noise for diversity
        for orig_param, replica_param in zip(self.model.parameters(), replica.parameters()):
            noise = torch.randn_like(orig_param) * 0.01  # 1% noise
            replica_param.data = orig_param.data + noise
        
        return replica
    
    @safe_execute("fault_tolerant_inference", "inference")
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fault-tolerant forward pass with voting."""
        outputs = []
        
        # Run inference on all replicas
        for replica in self.replicas:
            try:
                replica.# eval() removed for security)
                with torch.no_grad():
                    output = replica(inputs)
                outputs.append(output)
            except Exception as e:
                self.logger.warning(f"Replica failed: {e}")
                continue
        
        if not outputs:
            raise RuntimeError("All replicas failed")
        
        # Majority voting or averaging
        if len(outputs) == 1:
            return outputs[0]
        
        # Ensemble averaging
        ensemble_output = torch.stack(outputs).mean(dim=0)
        
        # Confidence-based selection
        confidences = [F.softmax(output, dim=-1).max(dim=-1)[0].mean() for output in outputs]
        best_replica_idx = max(range(len(outputs)), key=lambda i: confidences[i])
        
        # Use ensemble if confidence is low, otherwise use best replica
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < self.consensus_threshold:
            return ensemble_output
        else:
            return outputs[best_replica_idx]


class HealthMonitor:
    """Monitor model health during training and inference."""
    
    def __init__(self, model: nn.Module, check_interval: int = 100):
        self.model = model
        self.check_interval = check_interval
        self.step_count = 0
        self.health_metrics = {
            'weight_norms': [],
            'activation_stats': [],
            'gradient_norms': [],
            'loss_history': []
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.step_count += 1
        
        health_report = {
            'step': self.step_count,
            'timestamp': time.time(),
            'issues': []
        }
        
        # Check weight statistics
        weight_stats = self._check_weights()
        health_report['weight_stats'] = weight_stats
        
        # Check for dead neurons
        dead_neurons = self._check_dead_neurons()
        if dead_neurons:
            health_report['issues'].append(f"Dead neurons detected: {dead_neurons}")
        
        # Check model capacity utilization
        capacity_util = self._check_capacity_utilization()
        health_report['capacity_utilization'] = capacity_util
        
        return health_report
    
    def _check_weights(self) -> Dict[str, float]:
        """Check weight statistics."""
        weight_norms = []
        zero_weights = 0
        total_weights = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                weight_norms.append(param.norm().item())
                zero_weights += (param == 0).sum().item()
                total_weights += param.numel()
        
        return {
            'mean_norm': np.mean(weight_norms),
            'max_norm': np.max(weight_norms),
            'min_norm': np.min(weight_norms),
            'zero_ratio': zero_weights / total_weights
        }
    
    def _check_dead_neurons(self) -> int:
        """Check for dead neurons (never activate)."""
        # This is a simplified check - would need activation tracking in practice
        dead_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    # Check for weights that are effectively zero
                    dead_weights = (module.weight.abs() < 1e-8).all(dim=-1).sum()
                    dead_count += dead_weights.item()
        
        return dead_count
    
    def _check_capacity_utilization(self) -> float:
        """Check model capacity utilization."""
        active_params = 0
        total_params = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                active_params += (param.abs() > 1e-8).sum().item()
                total_params += param.numel()
        
        return active_params / total_params if total_params > 0 else 0.0


# ==============================================================================
# ADVANCED ROBUSTNESS SYSTEMS - GENERATION 2 ENHANCEMENTS
# ==============================================================================

class NeuromorphicFailsafeSystem:
    """Advanced failsafe system for neuromorphic hardware deployment."""
    
    def __init__(self, model: nn.Module, backup_models: Optional[List[nn.Module]] = None):
        self.primary_model = model
        self.backup_models = backup_models or []
        self.current_model_index = 0
        self.failure_count = 0
        self.max_failures = 3
        self.health_monitor = HealthMonitor(model)
        self.security_manager = NeuromorphicSecurityManager(SecurityConfig())
        self.performance_profiler = PerformanceProfiler()
        
    @safe_execute("failsafe_inference", "critical")
    def resilient_inference(self, inputs: torch.Tensor, 
                          enable_fallback: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform inference with automatic failover capabilities."""
        
        execution_report = {
            'primary_model_used': True,
            'fallback_triggered': False,
            'performance_metrics': {},
            'security_checks': {},
            'health_status': {}
        }
        
        # Security pre-checks
        security_result = self.security_manager.validate_input(inputs)
        execution_report['security_checks'] = security_result
        
        if not security_result.get('valid', True):
            raise SecurityError(f"Input validation failed: {security_result.get('issues', [])}")
        
        # Health check
        health_status = self.health_monitor.check_health()
        execution_report['health_status'] = health_status
        
        try:
            # Primary model inference with performance monitoring
            with self.performance_profiler.profile_context("primary_inference"):
                current_model = self._get_current_model()
                outputs = current_model(inputs)
                
                # Validate outputs
                if self._validate_outputs(outputs):
                    execution_report['performance_metrics'] = self.performance_profiler.get_latest_metrics()
                    return outputs, execution_report
                else:
                    raise ValidationError("Output validation failed")
                    
        except Exception as e:
            self.failure_count += 1
            execution_report['primary_model_used'] = False
            execution_report['failure_reason'] = str(e)
            
            if enable_fallback and self.failure_count < self.max_failures:
                # Attempt fallback to backup model
                fallback_result = self._attempt_fallback(inputs)
                execution_report['fallback_triggered'] = True
                execution_report['fallback_result'] = fallback_result
                return fallback_result['outputs'], execution_report
            else:
                # Critical failure - trigger emergency protocols
                emergency_result = self._emergency_response(inputs, e)
                execution_report['emergency_response'] = emergency_result
                return emergency_result['outputs'], execution_report
    
    def _get_current_model(self) -> nn.Module:
        """Get the currently active model."""
        if self.current_model_index == 0:
            return self.primary_model
        elif self.current_model_index - 1 < len(self.backup_models):
            return self.backup_models[self.current_model_index - 1]
        else:
            raise RuntimeError("No available models")
    
    def _validate_outputs(self, outputs: torch.Tensor) -> bool:
        """Validate model outputs for anomalies."""
        try:
            # Check for NaN or infinite values
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                return False
            
            # Check output ranges
            if outputs.abs().max() > 1e6:  # Unreasonably large values
                return False
            
            # Check for zero gradients (if requires_grad)
            if outputs.requires_grad and outputs.grad is not None:
                if outputs.grad.abs().max() < 1e-10:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _attempt_fallback(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """Attempt fallback to backup model."""
        fallback_result = {
            'success': False,
            'model_used': None,
            'outputs': None,
            'performance': {}
        }
        
        for i, backup_model in enumerate(self.backup_models):
            try:
                self.current_model_index = i + 1
                
                with self.performance_profiler.profile_context(f"backup_inference_{i}"):
                    outputs = backup_model(inputs)
                    
                    if self._validate_outputs(outputs):
                        fallback_result.update({
                            'success': True,
                            'model_used': f'backup_{i}',
                            'outputs': outputs,
                            'performance': self.performance_profiler.get_latest_metrics()
                        })
                        return fallback_result
                        
            except Exception as e:
                continue
        
        raise RuntimeError("All backup models failed")
    
    def _emergency_response(self, inputs: torch.Tensor, original_error: Exception) -> Dict[str, Any]:
        """Emergency response when all models fail."""
        emergency_result = {
            'mode': 'emergency',
            'original_error': str(original_error),
            'outputs': None,
            'safe_outputs_generated': False
        }
        
        try:
            # Generate safe default outputs
            batch_size = inputs.shape[0]
            output_shape = self._infer_output_shape(inputs)
            
            # Create safe neutral outputs (all zeros or uniform distribution)
            safe_outputs = torch.zeros(batch_size, *output_shape)
            emergency_result['outputs'] = safe_outputs
            emergency_result['safe_outputs_generated'] = True
            
        except Exception as e:
            emergency_result['emergency_error'] = str(e)
            # Final fallback - return minimal tensor
            emergency_result['outputs'] = torch.zeros(1, 1)
        
        return emergency_result
    
    def _infer_output_shape(self, inputs: torch.Tensor) -> Tuple[int, ...]:
        """Infer expected output shape from input."""
        # This is model-specific and would need to be configured
        # For now, assume classification output
        return (10,)  # 10 classes


class AdaptiveSecurityLayer:
    """Adaptive security layer that learns from attack patterns."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.attack_history = []
        self.defense_strategies = {}
        self.threat_level = 0.0
        
    def adaptive_defense(self, inputs: torch.Tensor, 
                        attack_indicators: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Apply adaptive defenses based on threat assessment."""
        
        # Threat assessment
        current_threat = self._assess_threat_level(inputs, attack_indicators)
        self.threat_level = 0.9 * self.threat_level + 0.1 * current_threat
        
        # Apply defenses based on threat level
        if self.threat_level > 0.7:
            # High threat - apply strong defenses
            defended_inputs = self._apply_strong_defense(inputs)
        elif self.threat_level > 0.3:
            # Medium threat - apply moderate defenses
            defended_inputs = self._apply_moderate_defense(inputs)
        else:
            # Low threat - minimal defenses
            defended_inputs = self._apply_light_defense(inputs)
        
        return defended_inputs
    
    def _assess_threat_level(self, inputs: torch.Tensor, 
                           indicators: Optional[Dict[str, float]]) -> float:
        """Assess current threat level based on inputs and indicators."""
        threat_score = 0.0
        
        # Statistical anomaly detection
        input_stats = {
            'mean': inputs.mean().item(),
            'std': inputs.std().item(), 
            'min': inputs.min().item(),
            'max': inputs.max().item()
        }
        
        # Compare with expected ranges (simplified)
        if abs(input_stats['mean']) > 2.0:
            threat_score += 0.3
        if input_stats['std'] > 5.0:
            threat_score += 0.2
        if input_stats['max'] - input_stats['min'] > 20.0:
            threat_score += 0.2
        
        # External indicators
        if indicators:
            threat_score += indicators.get('adversarial_score', 0.0) * 0.4
            threat_score += indicators.get('anomaly_score', 0.0) * 0.3
        
        return min(threat_score, 1.0)
    
    def _apply_strong_defense(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply strong defensive measures."""
        # Gaussian noise injection
        noise_level = 0.1
        noise = torch.randn_like(inputs) * noise_level
        defended = inputs + noise
        
        # Input smoothing
        defended = F.avg_pool2d(defended.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        
        # Gradient masking (if requires_grad)
        if defended.requires_grad:
            defended = defended + torch.randn_like(defended) * 0.01
        
        return defended
    
    def _apply_moderate_defense(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply moderate defensive measures."""
        # Light noise injection
        noise = torch.randn_like(inputs) * 0.05
        defended = inputs + noise
        
        # Slight input perturbation
        defended = defended + torch.randn_like(defended) * 0.02
        
        return defended
    
    def _apply_light_defense(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply light defensive measures."""
        # Minimal noise injection for robustness
        noise = torch.randn_like(inputs) * 0.01
        return inputs + noise
    
    def learn_from_attack(self, attack_pattern: Dict[str, Any]):
        """Learn from observed attack patterns to improve defenses."""
        self.attack_history.append(attack_pattern)
        
        # Update defense strategies based on attack patterns
        attack_type = attack_pattern.get('type', 'unknown')
        
        if attack_type not in self.defense_strategies:
            self.defense_strategies[attack_type] = {
                'frequency': 0,
                'severity': 0.0,
                'countermeasures': []
            }
        
        strategy = self.defense_strategies[attack_type]
        strategy['frequency'] += 1
        strategy['severity'] = max(strategy['severity'], attack_pattern.get('severity', 0.0))
        
        # Adaptive learning of countermeasures
        if attack_pattern.get('severity', 0.0) > 0.5:
            new_countermeasure = self._derive_countermeasure(attack_pattern)
            if new_countermeasure not in strategy['countermeasures']:
                strategy['countermeasures'].append(new_countermeasure)


class SelfHealingNeuromorphicSystem:
    """Self-healing system that can recover from hardware and software failures."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.baseline_weights = self._save_baseline_weights()
        self.performance_baseline = None
        self.healing_strategies = {
            'weight_corruption': self._heal_weight_corruption,
            'dead_neurons': self._heal_dead_neurons,
            'gradient_explosion': self._heal_gradient_explosion,
            'memory_leak': self._heal_memory_leak
        }
        
    def _save_baseline_weights(self) -> Dict[str, torch.Tensor]:
        """Save baseline weights for recovery."""
        baseline = {}
        for name, param in self.model.named_parameters():
            baseline[name] = param.data.clone()
        return baseline
    
    def diagnose_and_heal(self) -> Dict[str, Any]:
        """Diagnose issues and apply appropriate healing strategies."""
        
        diagnosis = self._diagnose_system_health()
        healing_report = {
            'diagnosis': diagnosis,
            'healing_actions': [],
            'success': True
        }
        
        for issue_type, severity in diagnosis['issues'].items():
            if severity > 0.5:  # Significant issue
                try:
                    healing_strategy = self.healing_strategies.get(issue_type)
                    if healing_strategy:
                        result = healing_strategy(severity)
                        healing_report['healing_actions'].append({
                            'issue': issue_type,
                            'severity': severity,
                            'action': result,
                            'success': result.get('success', False)
                        })
                except Exception as e:
                    healing_report['healing_actions'].append({
                        'issue': issue_type,
                        'severity': severity,
                        'error': str(e),
                        'success': False
                    })
                    healing_report['success'] = False
        
        return healing_report
    
    def _diagnose_system_health(self) -> Dict[str, Any]:
        """Diagnose system health issues."""
        diagnosis = {
            'overall_health': 1.0,
            'issues': {},
            'metrics': {}
        }
        
        # Check for weight corruption
        weight_corruption = self._check_weight_corruption()
        diagnosis['issues']['weight_corruption'] = weight_corruption
        
        # Check for dead neurons
        dead_neuron_ratio = self._check_dead_neurons()
        diagnosis['issues']['dead_neurons'] = dead_neuron_ratio
        
        # Check for gradient issues
        gradient_health = self._check_gradient_health()
        diagnosis['issues']['gradient_explosion'] = gradient_health
        
        # Check memory usage
        memory_health = self._check_memory_health()
        diagnosis['issues']['memory_leak'] = memory_health
        
        # Calculate overall health
        issue_scores = list(diagnosis['issues'].values())
        diagnosis['overall_health'] = 1.0 - (sum(issue_scores) / len(issue_scores))
        
        return diagnosis
    
    def _check_weight_corruption(self) -> float:
        """Check for weight corruption compared to baseline."""
        corruption_score = 0.0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if name in self.baseline_weights:
                baseline = self.baseline_weights[name]
                
                # Check for NaN or infinite values
                if torch.isnan(param).any() or torch.isinf(param).any():
                    corruption_score += param.numel()
                
                # Check for extreme deviations
                deviation = (param - baseline).abs()
                extreme_deviations = (deviation > baseline.abs() * 10).sum()
                corruption_score += extreme_deviations.item()
                
                total_params += param.numel()
        
        return corruption_score / max(total_params, 1)
    
    def _check_dead_neurons(self) -> float:
        """Check for dead neurons (always zero output)."""
        dead_neurons = 0
        total_neurons = 0
        
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Check for zero weights
                zero_weights = (module.weight.abs() < 1e-8).all(dim=1).sum()
                dead_neurons += zero_weights.item()
                total_neurons += module.weight.shape[0]
        
        return dead_neurons / max(total_neurons, 1)
    
    def _check_gradient_health(self) -> float:
        """Check for gradient explosion or vanishing."""
        gradient_norms = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_norms.append(param.grad.norm().item())
        
        if not gradient_norms:
            return 0.0
        
        max_norm = max(gradient_norms)
        min_norm = min(gradient_norms)
        
        # Score based on extreme gradients
        explosion_score = min(max_norm / 10.0, 1.0)  # >10 is considered explosion
        vanishing_score = max(0.0, (1e-8 - min_norm) / 1e-8)  # <1e-8 is vanishing
        
        return max(explosion_score, vanishing_score)
    
    def _check_memory_health(self) -> float:
        """Check for memory leaks or excessive usage."""
        try:
            import psutil
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Score based on memory usage (>80% is concerning)
            return max(0.0, (memory_percent - 80) / 20.0)
            
        except ImportError:
            return 0.0  # Can't check without psutil
    
    def _heal_weight_corruption(self, severity: float) -> Dict[str, Any]:
        """Heal weight corruption by restoring from baseline."""
        restoration_count = 0
        
        for name, param in self.model.named_parameters():
            if name in self.baseline_weights:
                baseline = self.baseline_weights[name]
                
                # Detect corrupted weights
                corrupted_mask = torch.isnan(param) | torch.isinf(param)
                extreme_deviation = (param - baseline).abs() > baseline.abs() * 10
                
                corruption_mask = corrupted_mask | extreme_deviation
                
                if corruption_mask.any():
                    # Restore corrupted weights
                    param.data[corruption_mask] = baseline[corruption_mask]
                    restoration_count += corruption_mask.sum().item()
        
        return {
            'success': True,
            'weights_restored': restoration_count,
            'method': 'baseline_restoration'
        }
    
    def _heal_dead_neurons(self, severity: float) -> Dict[str, Any]:
        """Heal dead neurons by reinitializing weights."""
        reinitialized_neurons = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Find dead neurons (zero weights)
                dead_mask = (module.weight.abs() < 1e-8).all(dim=1)
                
                if dead_mask.any():
                    # Reinitialize dead neurons
                    dead_indices = dead_mask.nonzero().squeeze()
                    if dead_indices.numel() > 0:
                        nn.init.xavier_normal_(module.weight[dead_indices])
                        reinitialized_neurons += dead_indices.numel()
        
        return {
            'success': True,
            'neurons_reinitialized': reinitialized_neurons,
            'method': 'xavier_reinitialization'
        }
    
    def _heal_gradient_explosion(self, severity: float) -> Dict[str, Any]:
        """Heal gradient explosion by clipping."""
        clipped_params = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > 10.0:  # Explosion threshold
                    # Clip gradients
                    param.grad.data = param.grad.data / grad_norm * 1.0
                    clipped_params += 1
        
        return {
            'success': True,
            'parameters_clipped': clipped_params,
            'method': 'gradient_clipping'
        }
    
    def _heal_memory_leak(self, severity: float) -> Dict[str, Any]:
        """Heal memory leaks by forcing cleanup."""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'objects_collected': collected,
            'method': 'garbage_collection_and_cache_clear'
        }


class SecurityError(Exception):
    """Custom security exception."""
    pass


class ValidationError(Exception):
    """Custom validation exception."""
    pass


# Example usage and testing
if __name__ == "__main__":
    print("🛡️ Advanced Neuromorphic Robustness Suite")
    print("=" * 60)
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Test failsafe system
    print("\n1. Testing Neuromorphic Failsafe System:")
    failsafe = NeuromorphicFailsafeSystem(test_model)
    test_input = torch.randn(32, 784)
    
    try:
        outputs, report = failsafe.resilient_inference(test_input)
        print(f"   ✅ Inference successful: {outputs.shape}")
        print(f"   📊 Primary model used: {report['primary_model_used']}")
        print(f"   🔒 Security checks passed: {report['security_checks'].get('valid', True)}")
    except Exception as e:
        print(f"   ❌ Failsafe test failed: {e}")
    
    # Test adaptive security
    print("\n2. Testing Adaptive Security Layer:")
    security_layer = AdaptiveSecurityLayer(test_model)
    
    # Simulate different threat levels
    for threat_name, threat_input in [
        ("Normal", torch.randn(32, 784) * 0.5),
        ("Suspicious", torch.randn(32, 784) * 2.0),
        ("Adversarial", torch.randn(32, 784) * 5.0)
    ]:
        defended_input = security_layer.adaptive_defense(threat_input)
        print(f"   🔍 {threat_name} input: threat level {security_layer.threat_level:.3f}")
    
    # Test self-healing system
    print("\n3. Testing Self-Healing System:")
    healing_system = SelfHealingNeuromorphicSystem(test_model)
    
    # Simulate some corruption
    with torch.no_grad():
        for param in test_model.parameters():
            param[0] = float('nan')  # Introduce corruption
            break
    
    healing_report = healing_system.diagnose_and_heal()
    print(f"   🔧 Overall health: {healing_report['diagnosis']['overall_health']:.3f}")
    print(f"   🩹 Healing actions taken: {len(healing_report['healing_actions'])}")
    print(f"   ✅ Healing successful: {healing_report['success']}")
    
    print("\n🛡️ Advanced Robustness Suite Complete!")