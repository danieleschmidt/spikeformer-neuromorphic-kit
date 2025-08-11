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
                replica.eval()
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