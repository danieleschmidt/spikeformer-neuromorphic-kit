"""Spiking neuron models for neuromorphic computing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class SurrogateGradient(nn.Module):
    """Surrogate gradient functions for backpropagation through spikes."""
    
    @staticmethod
    def fast_sigmoid(x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        """Fast sigmoid surrogate gradient."""
        return 1.0 / (1.0 + torch.abs(beta * x))
    
    @staticmethod
    def straight_through_estimator(x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator."""
        return torch.ones_like(x)
    
    @staticmethod
    def triangular(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Triangular surrogate gradient."""
        return torch.clamp(1.0 - alpha * torch.abs(x), min=0.0)


class SpikingFunction(torch.autograd.Function):
    """Differentiable spiking function with surrogate gradients."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float, surrogate_type: str = "fast_sigmoid") -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.surrogate_type = surrogate_type
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        input, = ctx.saved_tensors
        
        if ctx.surrogate_type == "fast_sigmoid":
            surrogate_grad = SurrogateGradient.fast_sigmoid(input - ctx.threshold)
        elif ctx.surrogate_type == "straight_through":
            surrogate_grad = SurrogateGradient.straight_through_estimator(input - ctx.threshold)
        elif ctx.surrogate_type == "triangular":
            surrogate_grad = SurrogateGradient.triangular(input - ctx.threshold)
        else:
            surrogate_grad = SurrogateGradient.fast_sigmoid(input - ctx.threshold)
            
        return grad_output * surrogate_grad, None, None


class SpikingNeuron(nn.Module, ABC):
    """Abstract base class for spiking neurons."""
    
    def __init__(self, threshold: float = 1.0, reset: str = "subtract", surrogate_type: str = "fast_sigmoid"):
        super().__init__()
        self.threshold = threshold
        self.reset = reset  # "subtract" or "zero"
        self.surrogate_type = surrogate_type
        self.register_buffer('membrane_potential', None)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuron."""
        pass
        
    def reset_state(self):
        """Reset neuron state between sequences."""
        self.membrane_potential = None
        
    def spike_function(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        """Apply spiking function with surrogate gradients."""
        return SpikingFunction.apply(membrane_potential, self.threshold, self.surrogate_type)


class LifNeuron(SpikingNeuron):
    """Leaky Integrate-and-Fire (LIF) neuron model."""
    
    def __init__(self, 
                 threshold: float = 1.0, 
                 tau_mem: float = 20.0,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.tau_mem = tau_mem
        self.alpha = torch.exp(torch.tensor(-1.0 / tau_mem))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LIF neuron."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize membrane potential if needed
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            
        spikes = []
        
        for t in range(seq_len):
            # Leaky integration
            self.membrane_potential = self.alpha * self.membrane_potential + x[:, t]
            
            # Generate spikes
            spike = self.spike_function(self.membrane_potential)
            spikes.append(spike)
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)


class AdLifNeuron(SpikingNeuron):
    """Adaptive Leaky Integrate-and-Fire (AdLIF) neuron with threshold adaptation."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 20.0,
                 tau_adp: float = 200.0,
                 beta: float = 0.1,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.tau_mem = tau_mem
        self.tau_adp = tau_adp
        self.beta = beta
        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.alpha_adp = torch.exp(torch.tensor(-1.0 / tau_adp))
        self.register_buffer('adaptive_threshold', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AdLIF neuron."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize states
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            self.adaptive_threshold = torch.zeros_like(self.membrane_potential)
            
        spikes = []
        
        for t in range(seq_len):
            # Leaky integration
            self.membrane_potential = self.alpha_mem * self.membrane_potential + x[:, t]
            
            # Adaptive threshold
            current_threshold = self.threshold + self.beta * self.adaptive_threshold
            
            # Generate spikes
            spike = self.spike_function(self.membrane_potential - current_threshold)
            spikes.append(spike)
            
            # Update adaptive threshold
            self.adaptive_threshold = self.alpha_adp * self.adaptive_threshold + spike
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)
        
    def reset_state(self):
        """Reset neuron state including adaptive threshold."""
        super().reset_state()
        self.adaptive_threshold = None


class IzhikevichNeuron(SpikingNeuron):
    """Izhikevich neuron model with rich dynamics."""
    
    def __init__(self, 
                 a: float = 0.02,
                 b: float = 0.2, 
                 c: float = -65.0,
                 d: float = 8.0,
                 threshold: float = 30.0,
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, "custom", surrogate_type)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.register_buffer('recovery_variable', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Izhikevich neuron."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        dt = 0.1  # Time step
        
        # Initialize states
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.full((batch_size, *x.shape[2:]), self.c, device=device)
            self.recovery_variable = torch.full_like(self.membrane_potential, self.b * self.c)
            
        spikes = []
        
        for t in range(seq_len):
            # Izhikevich dynamics
            v = self.membrane_potential
            u = self.recovery_variable
            
            # Update membrane potential
            dv = (0.04 * v * v + 5 * v + 140 - u + x[:, t]) * dt
            self.membrane_potential = v + dv
            
            # Update recovery variable
            du = self.a * (self.b * v - u) * dt
            self.recovery_variable = u + du
            
            # Generate spikes
            spike = self.spike_function(self.membrane_potential)
            spikes.append(spike)
            
            # Reset after spike
            self.membrane_potential = torch.where(spike > 0, self.c, self.membrane_potential)
            self.recovery_variable = torch.where(spike > 0, self.recovery_variable + self.d, self.recovery_variable)
            
        return torch.stack(spikes, dim=1)
        
    def reset_state(self):
        """Reset neuron state including recovery variable."""
        super().reset_state()
        self.recovery_variable = None


class SpikingLayer(nn.Module):
    """Spiking layer that combines linear transformation with neuron dynamics."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 neuron_type: str = "LIF",
                 threshold: float = 1.0,
                 bias: bool = True,
                 **neuron_kwargs):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Create neuron based on type
        if neuron_type.upper() == "LIF":
            self.neuron = LifNeuron(threshold=threshold, **neuron_kwargs)
        elif neuron_type.upper() == "ADLIF":
            self.neuron = AdLifNeuron(threshold=threshold, **neuron_kwargs)
        elif neuron_type.upper() == "IZHIKEVICH":
            self.neuron = IzhikevichNeuron(threshold=threshold, **neuron_kwargs)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking layer."""
        # Apply linear transformation
        if x.dim() == 4:  # (batch, time, seq, features)
            batch_size, time_steps, seq_len, features = x.shape
            x = x.view(batch_size * time_steps, seq_len, features)
            linear_output = self.linear(x)
            linear_output = linear_output.view(batch_size, time_steps, seq_len, -1)
        else:  # (batch, time, features)
            linear_output = self.linear(x)
            
        # Apply neuron dynamics
        return self.neuron(linear_output)
        
    def reset_state(self):
        """Reset neuron state."""
        self.neuron.reset_state()


class TemporalBatchNorm(nn.Module):
    """Batch normalization adapted for temporal spike data."""
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.register_parameter('weight', nn.Parameter(torch.ones(num_features)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_features)))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal batch normalization."""
        if self.training:
            # Calculate statistics across batch and time dimensions
            mean = x.mean(dim=(0, 1), keepdim=True)
            var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            
            # Normalize
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Use running statistics
            mean = self.running_mean.view(1, 1, -1)
            var = self.running_var.view(1, 1, -1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            
        # Apply learnable parameters
        return x_norm * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)


class AttentionOptimizedNeuron(SpikingNeuron):
    """Transformer-optimized neuron model with attention-aware dynamics."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 20.0,
                 tau_attention: float = 50.0,
                 attention_gain: float = 0.2,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.tau_mem = tau_mem
        self.tau_attention = tau_attention
        self.attention_gain = attention_gain
        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.alpha_att = torch.exp(torch.tensor(-1.0 / tau_attention))
        self.register_buffer('attention_trace', None)
        
    def forward(self, x: torch.Tensor, attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention-aware dynamics."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize states
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            self.attention_trace = torch.zeros_like(self.membrane_potential)
            
        spikes = []
        
        for t in range(seq_len):
            # Attention modulation
            if attention_weights is not None:
                att_weight = attention_weights[:, t] if attention_weights.dim() == 2 else attention_weights[:, t, :]
                self.attention_trace = self.alpha_att * self.attention_trace + att_weight * self.attention_gain
            
            # Attention-modulated integration
            effective_input = x[:, t] * (1 + self.attention_trace)
            self.membrane_potential = self.alpha_mem * self.membrane_potential + effective_input
            
            # Generate spikes
            spike = self.spike_function(self.membrane_potential)
            spikes.append(spike)
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)
        
    def reset_state(self):
        """Reset neuron state including attention trace."""
        super().reset_state()
        self.attention_trace = None


class StochasticNeuron(SpikingNeuron):
    """Stochastic spiking neuron with biological noise for improved robustness."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 20.0,
                 noise_std: float = 0.1,
                 dropout_p: float = 0.05,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.tau_mem = tau_mem
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.alpha = torch.exp(torch.tensor(-1.0 / tau_mem))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic dynamics."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize membrane potential
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            
        spikes = []
        
        for t in range(seq_len):
            # Add membrane noise
            if self.training:
                noise = torch.randn_like(self.membrane_potential) * self.noise_std
                dropout_mask = (torch.rand_like(self.membrane_potential) > self.dropout_p).float()
            else:
                noise = 0
                dropout_mask = 1
                
            # Leaky integration with noise and dropout
            self.membrane_potential = (self.alpha * self.membrane_potential + 
                                     x[:, t] * dropout_mask + noise)
            
            # Stochastic spiking (temperature-based)
            if self.training:
                # Temperature-scaled sigmoid for stochastic spiking
                temp = 1.0 + 0.5 * torch.randn_like(self.membrane_potential)
                spike_prob = torch.sigmoid((self.membrane_potential - self.threshold) / temp)
                spike = torch.bernoulli(spike_prob)
            else:
                # Deterministic for inference
                spike = self.spike_function(self.membrane_potential)
                
            spikes.append(spike)
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)


class MultiCompartmentNeuron(SpikingNeuron):
    """Multi-compartment neuron with dendritic processing for heterogeneous populations."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 num_compartments: int = 3,
                 tau_soma: float = 20.0,
                 tau_dendrite: float = 40.0,
                 coupling_strength: float = 0.3,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.num_compartments = num_compartments
        self.tau_soma = tau_soma
        self.tau_dendrite = tau_dendrite
        self.coupling_strength = coupling_strength
        
        self.alpha_soma = torch.exp(torch.tensor(-1.0 / tau_soma))
        self.alpha_dend = torch.exp(torch.tensor(-1.0 / tau_dendrite))
        
        self.register_buffer('dendritic_potentials', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-compartment dynamics."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize compartment potentials
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)  # Soma
            self.dendritic_potentials = torch.zeros(batch_size, self.num_compartments, 
                                                   *x.shape[2:], device=device)
            
        spikes = []
        
        for t in range(seq_len):
            # Split input across dendritic compartments
            input_per_comp = x[:, t].unsqueeze(1) / self.num_compartments
            
            # Update dendritic compartments
            self.dendritic_potentials = (self.alpha_dend * self.dendritic_potentials + 
                                       input_per_comp)
            
            # Dendritic coupling to soma
            dendritic_sum = torch.sum(self.dendritic_potentials, dim=1) * self.coupling_strength
            
            # Update somatic potential
            self.membrane_potential = self.alpha_soma * self.membrane_potential + dendritic_sum
            
            # Generate spikes at soma
            spike = self.spike_function(self.membrane_potential)
            spikes.append(spike)
            
            # Reset mechanism affects all compartments
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
                # Partial reset in dendrites
                reset_factor = spike.unsqueeze(1) * 0.5
                self.dendritic_potentials = self.dendritic_potentials - reset_factor * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                reset_mask = (1 - spike).unsqueeze(1)
                self.dendritic_potentials = self.dendritic_potentials * reset_mask
                
        return torch.stack(spikes, dim=1)
        
    def reset_state(self):
        """Reset all compartment states."""
        super().reset_state()
        self.dendritic_potentials = None


class AdaptiveSparsityNeuron(SpikingNeuron):
    """Neuron with adaptive sparsity control for energy optimization."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 20.0,
                 target_sparsity: float = 0.1,
                 adaptation_rate: float = 0.01,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        self.tau_mem = tau_mem
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.alpha = torch.exp(torch.tensor(-1.0 / tau_mem))
        
        self.register_buffer('dynamic_threshold', None)
        self.register_buffer('spike_history', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive sparsity control."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Initialize states
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            self.dynamic_threshold = torch.full_like(self.membrane_potential, self.threshold)
            self.spike_history = torch.zeros(batch_size, 10, *x.shape[2:], device=device)  # Last 10 steps
            
        spikes = []
        
        for t in range(seq_len):
            # Leaky integration
            self.membrane_potential = self.alpha * self.membrane_potential + x[:, t]
            
            # Generate spikes with dynamic threshold
            spike = self.spike_function(self.membrane_potential - self.dynamic_threshold)
            spikes.append(spike)
            
            # Update spike history (rolling window)
            self.spike_history = torch.roll(self.spike_history, -1, dims=1)
            self.spike_history[:, -1] = spike
            
            # Adapt threshold based on recent activity
            if t > 9:  # Have enough history
                recent_sparsity = torch.mean(self.spike_history, dim=1)
                sparsity_error = recent_sparsity - self.target_sparsity
                
                # Adaptive threshold update
                self.dynamic_threshold += self.adaptation_rate * sparsity_error
                self.dynamic_threshold = torch.clamp(self.dynamic_threshold, 
                                                   min=0.1 * self.threshold,
                                                   max=3.0 * self.threshold)
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.dynamic_threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)
        
    def reset_state(self):
        """Reset neuron state including adaptive components."""
        super().reset_state()
        self.dynamic_threshold = None
        self.spike_history = None


class PLIFNeuron(SpikingNeuron):
    """Parametric Leaky Integrate-and-Fire neuron with learnable time constants."""
    
    def __init__(self, 
                 threshold: float = 1.0,
                 tau_mem: float = 20.0,
                 learnable_tau: bool = True,
                 reset: str = "subtract",
                 surrogate_type: str = "fast_sigmoid"):
        super().__init__(threshold, reset, surrogate_type)
        
        if learnable_tau:
            self.log_tau_mem = nn.Parameter(torch.log(torch.tensor(tau_mem)))
        else:
            self.register_buffer('log_tau_mem', torch.log(torch.tensor(tau_mem)))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learnable dynamics."""
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # Compute decay factor from learnable parameter
        tau_mem = torch.exp(self.log_tau_mem)
        alpha = torch.exp(-1.0 / tau_mem)
        
        # Initialize membrane potential
        if self.membrane_potential is None or self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, *x.shape[2:], device=device)
            
        spikes = []
        
        for t in range(seq_len):
            # Leaky integration with learnable decay
            self.membrane_potential = alpha * self.membrane_potential + x[:, t]
            
            # Generate spikes
            spike = self.spike_function(self.membrane_potential)
            spikes.append(spike)
            
            # Reset mechanism
            if self.reset == "subtract":
                self.membrane_potential = self.membrane_potential - spike * self.threshold
            elif self.reset == "zero":
                self.membrane_potential = self.membrane_potential * (1 - spike)
                
        return torch.stack(spikes, dim=1)


# Enhanced factory function for creating neurons
def create_neuron(neuron_type: str, **kwargs) -> SpikingNeuron:
    """Enhanced factory function to create neurons by type."""
    neuron_types = {
        "LIF": LifNeuron,
        "ADLIF": AdLifNeuron,
        "IZHIKEVICH": IzhikevichNeuron,
        "ATTENTION": AttentionOptimizedNeuron,
        "STOCHASTIC": StochasticNeuron,
        "MULTICOMP": MultiCompartmentNeuron,
        "ADAPTIVE": AdaptiveSparsityNeuron,
        "PLIF": PLIFNeuron,
    }
    
    if neuron_type.upper() not in neuron_types:
        raise ValueError(f"Unknown neuron type: {neuron_type}. Available: {list(neuron_types.keys())}")
        
    return neuron_types[neuron_type.upper()](**kwargs)