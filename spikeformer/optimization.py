"""Advanced optimization algorithms for spiking neural networks."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
from collections import defaultdict, deque

from .neurons import SpikingNeuron, LifNeuron
from .models import SpikingTransformer, SpikingViT


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    
    # Spike-specific parameters
    spike_regularization: float = 0.01
    threshold_adaptation_rate: float = 0.001
    membrane_regularization: float = 0.001
    temporal_consistency_weight: float = 0.1
    
    # STDP parameters
    stdp_learning_rate: float = 1e-5
    stdp_window_size: int = 20
    stdp_tau_pre: float = 20.0
    stdp_tau_post: float = 20.0


class SpikeRegularizer(nn.Module):
    """Regularization techniques for spiking neural networks."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        self.spike_history = defaultdict(list)
        
    def spike_rate_regularization(self, spikes: torch.Tensor, target_rate: float = 0.1) -> torch.Tensor:
        """Regularize spike rate to target value."""
        current_rate = spikes.mean()
        return self.config.spike_regularization * (current_rate - target_rate) ** 2
    
    def temporal_consistency_regularization(self, spikes: torch.Tensor) -> torch.Tensor:
        """Encourage temporal consistency in spike patterns."""
        if spikes.dim() < 3:  # Need time dimension
            return torch.tensor(0.0, device=spikes.device)
            
        # Calculate temporal variance
        temporal_var = spikes.var(dim=1).mean()
        return self.config.temporal_consistency_weight * temporal_var
    
    def membrane_potential_regularization(self, membrane_potentials: List[torch.Tensor]) -> torch.Tensor:
        """Regularize membrane potentials to prevent saturation."""
        if not membrane_potentials:
            return torch.tensor(0.0)
            
        reg_loss = torch.tensor(0.0, device=membrane_potentials[0].device)
        
        for membrane in membrane_potentials:
            # Penalize extreme membrane potentials
            extreme_penalty = torch.mean(torch.clamp(torch.abs(membrane) - 2.0, min=0.0))
            reg_loss += self.config.membrane_regularization * extreme_penalty
            
        return reg_loss / len(membrane_potentials)
    
    def synaptic_weight_regularization(self, model: nn.Module) -> torch.Tensor:
        """Regularize synaptic weights to biological ranges."""
        reg_loss = torch.tensor(0.0)
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                # Encourage sparse connectivity
                l1_penalty = torch.abs(param).mean()
                
                # Prevent extreme weights
                l2_penalty = (param ** 2).mean()
                
                reg_loss += 0.001 * l1_penalty + 0.0001 * l2_penalty
                
        return reg_loss
    
    def forward(self, model: nn.Module, spikes: torch.Tensor, 
               membrane_potentials: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Apply all regularization terms."""
        total_reg = torch.tensor(0.0, device=spikes.device)
        
        # Spike rate regularization
        total_reg += self.spike_rate_regularization(spikes)
        
        # Temporal consistency
        total_reg += self.temporal_consistency_regularization(spikes)
        
        # Membrane potential regularization
        if membrane_potentials:
            total_reg += self.membrane_potential_regularization(membrane_potentials)
        
        # Synaptic weight regularization
        total_reg += self.synaptic_weight_regularization(model)
        
        return total_reg


class AdaptiveThresholdOptimizer:
    """Adaptive threshold optimization for spiking neurons."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.threshold_history = defaultdict(list)
        self.optimal_thresholds = {}
        
    def collect_statistics(self, layer_name: str, spikes: torch.Tensor, 
                          membrane_potential: torch.Tensor):
        """Collect spike statistics for threshold adaptation."""
        spike_rate = spikes.mean().item()
        membrane_mean = membrane_potential.mean().item()
        membrane_std = membrane_potential.std().item()
        
        stats = {
            'spike_rate': spike_rate,
            'membrane_mean': membrane_mean,
            'membrane_std': membrane_std,
            'timestamp': len(self.threshold_history[layer_name])
        }
        
        self.threshold_history[layer_name].append(stats)
    
    def adapt_thresholds(self, target_spike_rate: float = 0.1):
        """Adapt thresholds based on collected statistics."""
        for name, module in self.model.named_modules():
            if isinstance(module, SpikingNeuron):
                if name in self.threshold_history and self.threshold_history[name]:
                    recent_stats = self.threshold_history[name][-10:]  # Last 10 measurements
                    
                    avg_spike_rate = np.mean([s['spike_rate'] for s in recent_stats])
                    avg_membrane_mean = np.mean([s['membrane_mean'] for s in recent_stats])
                    
                    # Adjust threshold based on spike rate
                    if avg_spike_rate > target_spike_rate * 1.2:
                        # Too many spikes - increase threshold
                        new_threshold = module.threshold * 1.05
                    elif avg_spike_rate < target_spike_rate * 0.8:
                        # Too few spikes - decrease threshold
                        new_threshold = module.threshold * 0.95
                    else:
                        new_threshold = module.threshold
                    
                    # Clamp threshold to reasonable range
                    new_threshold = np.clip(new_threshold, 0.1, 5.0)
                    
                    # Apply gradual adaptation
                    module.threshold = (module.threshold * (1 - self.config.threshold_adaptation_rate) + 
                                      new_threshold * self.config.threshold_adaptation_rate)
                    
                    self.optimal_thresholds[name] = module.threshold
    
    def get_threshold_report(self) -> Dict[str, Any]:
        """Generate report on threshold adaptation."""
        report = {}
        
        for name, history in self.threshold_history.items():
            if history:
                recent_stats = history[-5:]
                report[name] = {
                    'current_threshold': self.optimal_thresholds.get(name, 1.0),
                    'avg_spike_rate': np.mean([s['spike_rate'] for s in recent_stats]),
                    'spike_rate_std': np.std([s['spike_rate'] for s in recent_stats]),
                    'avg_membrane_potential': np.mean([s['membrane_mean'] for s in recent_stats]),
                    'num_adaptations': len(history)
                }
        
        return report


class STDPOptimizer:
    """Spike-Timing Dependent Plasticity optimizer."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.spike_traces = defaultdict(lambda: deque(maxlen=config.stdp_window_size))
        self.weight_updates = defaultdict(float)
        
    def update_spike_traces(self, layer_name: str, pre_spikes: torch.Tensor, 
                           post_spikes: torch.Tensor):
        """Update spike traces for STDP."""
        self.spike_traces[f"{layer_name}_pre"].append(pre_spikes.clone())
        self.spike_traces[f"{layer_name}_post"].append(post_spikes.clone())
    
    def compute_stdp_updates(self, layer_name: str, weights: torch.Tensor) -> torch.Tensor:
        """Compute STDP weight updates."""
        pre_key = f"{layer_name}_pre"
        post_key = f"{layer_name}_post"
        
        if len(self.spike_traces[pre_key]) < 2 or len(self.spike_traces[post_key]) < 2:
            return torch.zeros_like(weights)
        
        # Get recent spike traces
        pre_traces = list(self.spike_traces[pre_key])
        post_traces = list(self.spike_traces[post_key])
        
        weight_update = torch.zeros_like(weights)
        
        # STDP calculation
        for dt in range(1, min(len(pre_traces), len(post_traces))):
            # Pre-before-post: potentiation
            if dt < len(pre_traces):
                pre_earlier = pre_traces[-dt-1]  # Earlier pre-synaptic spike
                post_later = post_traces[-1]     # Later post-synaptic spike
                
                ltp_update = (pre_earlier.unsqueeze(-1) @ post_later.unsqueeze(0)) * \
                           math.exp(-dt / self.config.stdp_tau_pre)
                
                if ltp_update.shape == weights.shape:
                    weight_update += self.config.stdp_learning_rate * ltp_update
            
            # Post-before-pre: depression
            if dt < len(post_traces):
                post_earlier = post_traces[-dt-1]  # Earlier post-synaptic spike
                pre_later = pre_traces[-1]         # Later pre-synaptic spike
                
                ltd_update = (pre_later.unsqueeze(-1) @ post_earlier.unsqueeze(0)) * \
                           math.exp(-dt / self.config.stdp_tau_post)
                
                if ltd_update.shape == weights.shape:
                    weight_update -= self.config.stdp_learning_rate * ltd_update
        
        return weight_update
    
    def apply_stdp_updates(self):
        """Apply STDP updates to model weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if f"{name}_pre" in self.spike_traces and f"{name}_post" in self.spike_traces:
                    weight_update = self.compute_stdp_updates(name, module.weight)
                    
                    # Apply update with momentum
                    self.weight_updates[name] = (0.9 * self.weight_updates[name] + 
                                               0.1 * weight_update)
                    
                    # Update weights
                    with torch.no_grad():
                        module.weight += self.weight_updates[name]
                        
                        # Clip weights to prevent instability
                        module.weight.clamp_(-2.0, 2.0)


class SpikingAdam(optim.Optimizer):
    """Adam optimizer adapted for spiking neural networks."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, spike_regularization=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, spike_regularization=spike_regularization)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                
                # Add spike regularization for spiking layers
                if 'spiking' in str(type(p)):
                    spike_reg = group['spike_regularization'] * torch.sign(p.data)
                    grad.add_(spike_reg)
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


class NeuromorphicScheduler:
    """Learning rate scheduler optimized for neuromorphic training."""
    
    def __init__(self, optimizer, mode='adaptive', patience=10, factor=0.5, 
                 min_lr=1e-8, spike_rate_target=0.1):
        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.spike_rate_target = spike_rate_target
        
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.last_lr_update = 0
        self.spike_rate_history = deque(maxlen=10)
        
    def step(self, loss: float, spike_rate: Optional[float] = None):
        """Update learning rate based on loss and spike statistics."""
        if self.mode == 'adaptive':
            self._adaptive_step(loss, spike_rate)
        elif self.mode == 'cosine_annealing':
            self._cosine_annealing_step()
        elif self.mode == 'spike_aware':
            self._spike_aware_step(loss, spike_rate)
    
    def _adaptive_step(self, loss: float, spike_rate: Optional[float] = None):
        """Adaptive learning rate based on loss plateau."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _spike_aware_step(self, loss: float, spike_rate: Optional[float] = None):
        """Learning rate adaptation based on spike rate and loss."""
        if spike_rate is not None:
            self.spike_rate_history.append(spike_rate)
            
            # If spike rate is too far from target, adjust learning rate
            avg_spike_rate = np.mean(self.spike_rate_history)
            
            if abs(avg_spike_rate - self.spike_rate_target) > 0.05:
                # Spike rate is not stable - reduce learning rate
                self._reduce_lr()
            elif loss < self.best_loss:
                # Good progress - can maintain or increase learning rate
                self.best_loss = loss
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                if self.num_bad_epochs >= self.patience:
                    self._reduce_lr()
                    self.num_bad_epochs = 0
    
    def _cosine_annealing_step(self):
        """Cosine annealing schedule."""
        # Implementation would depend on epoch tracking
        pass
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}")


class GradientOptimizer:
    """Advanced gradient optimization for spiking networks."""
    
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        self.gradient_history = defaultdict(list)
        
    def clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients to prevent instability."""
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        return total_norm
    
    def normalize_gradients(self):
        """Normalize gradients across layers."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                if grad_norm > 0:
                    param.grad = param.grad / grad_norm
    
    def accumulate_gradients(self, steps: int = 1):
        """Gradient accumulation for effective larger batch sizes."""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad / steps
    
    def apply_gradient_smoothing(self, alpha: float = 0.9):
        """Apply exponential smoothing to gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.gradient_history:
                    self.gradient_history[name] = param.grad.clone()
                else:
                    self.gradient_history[name] = (alpha * self.gradient_history[name] + 
                                                 (1 - alpha) * param.grad)
                    param.grad = self.gradient_history[name].clone()
    
    def detect_gradient_anomalies(self) -> Dict[str, bool]:
        """Detect gradient anomalies (NaN, inf, extreme values)."""
        anomalies = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                has_nan = torch.isnan(param.grad).any()
                has_inf = torch.isinf(param.grad).any()
                has_extreme = (param.grad.abs() > 100).any()
                
                anomalies[name] = has_nan or has_inf or has_extreme
        
        return anomalies


class ModelPruner:
    """Pruning techniques for spiking neural networks."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pruning_masks = {}
        
    def magnitude_pruning(self, sparsity: float = 0.5):
        """Prune weights based on magnitude."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Calculate threshold for pruning
                flat_weights = weight.view(-1)
                threshold = torch.quantile(flat_weights.abs(), sparsity)
                
                # Create pruning mask
                mask = (weight.abs() > threshold).float()
                self.pruning_masks[name] = mask
                
                # Apply pruning
                module.weight.data *= mask
    
    def structured_pruning(self, prune_ratio: float = 0.3):
        """Structured pruning by removing entire neurons/channels."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores (L2 norm of weights)
                weight_norms = module.weight.norm(dim=1)
                
                # Determine neurons to prune
                num_to_prune = int(prune_ratio * weight_norms.size(0))
                _, indices_to_prune = torch.topk(weight_norms, num_to_prune, largest=False)
                
                # Create structured mask
                mask = torch.ones_like(module.weight)
                mask[indices_to_prune, :] = 0
                
                self.pruning_masks[name] = mask
                module.weight.data *= mask
    
    def activity_based_pruning(self, activity_threshold: float = 0.01):
        """Prune based on neuron activity levels."""
        # This would require activity monitoring during training
        # Implementation would track neuron firing rates and prune inactive neurons
        pass
    
    def apply_masks(self):
        """Apply stored pruning masks to ensure weights stay pruned."""
        for name, module in self.model.named_modules():
            if name in self.pruning_masks:
                module.weight.data *= self.pruning_masks[name]
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics for each layer."""
        stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                total_params = module.weight.numel()
                zero_params = (module.weight == 0).sum().item()
                sparsity = zero_params / total_params
                stats[name] = sparsity
        
        return stats


class QuantizationOptimizer:
    """Quantization techniques for efficient deployment."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantization_schemes = {}
        
    def uniform_quantization(self, bits: int = 8):
        """Apply uniform quantization to weights."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                
                # Calculate quantization parameters
                w_min, w_max = weight.min(), weight.max()
                scale = (w_max - w_min) / (2**bits - 1)
                zero_point = -w_min / scale
                
                # Quantize
                quantized = torch.round(weight / scale + zero_point)
                quantized = torch.clamp(quantized, 0, 2**bits - 1)
                
                # Dequantize
                dequantized = (quantized - zero_point) * scale
                
                # Store quantization scheme
                self.quantization_schemes[name] = {
                    'scale': scale,
                    'zero_point': zero_point,
                    'bits': bits
                }
                
                module.weight.data = dequantized
    
    def dynamic_quantization(self):
        """Apply dynamic quantization using PyTorch's built-in functionality."""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def post_training_quantization(self, calibration_loader):
        """Post-training quantization with calibration."""
        # This would implement full post-training quantization
        # with calibration dataset for better accuracy
        pass


# Factory function for creating optimizers
def create_optimizer(model: nn.Module, optimizer_type: str = "spiking_adam", 
                    config: Optional[OptimizationConfig] = None) -> optim.Optimizer:
    """Factory function for creating optimizers."""
    if config is None:
        config = OptimizationConfig()
    
    if optimizer_type == "spiking_adam":
        return SpikingAdam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
            spike_regularization=config.spike_regularization
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str = "neuromorphic",
                    config: Optional[OptimizationConfig] = None) -> Union[NeuromorphicScheduler, optim.lr_scheduler._LRScheduler]:
    """Factory function for creating learning rate schedulers."""
    if config is None:
        config = OptimizationConfig()
    
    if scheduler_type == "neuromorphic":
        return NeuromorphicScheduler(optimizer, mode='adaptive')
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif scheduler_type == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")