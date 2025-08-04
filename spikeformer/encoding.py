"""Spike encoding strategies for converting continuous data to spike trains."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod
import math


class SpikeEncoder(nn.Module, ABC):
    """Abstract base class for spike encoders."""
    
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to spike trains."""
        pass


class RateCoding(SpikeEncoder):
    """Rate coding: spike frequency proportional to input intensity."""
    
    def __init__(self, timesteps: int, max_rate: float = 1.0, gain: float = 1.0):
        super().__init__(timesteps)
        self.max_rate = max_rate
        self.gain = gain
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to rate-coded spikes."""
        # Normalize input to [0, 1] range
        x_norm = torch.clamp(torch.abs(x) * self.gain, 0, 1)
        
        # Calculate spike probabilities
        spike_prob = x_norm * self.max_rate
        
        # Generate spikes for each timestep
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        for t in range(self.timesteps):
            # Bernoulli sampling for spikes
            random_values = torch.rand_like(spike_prob)
            spikes[:, t] = (random_values < spike_prob).float()
            
        return spikes


class TemporalCoding(SpikeEncoder):
    """Temporal coding: spike timing encodes information."""
    
    def __init__(self, timesteps: int, min_delay: int = 0):
        super().__init__(timesteps)
        self.min_delay = min_delay
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to temporally-coded spikes."""
        # Normalize input to [0, 1]
        x_norm = torch.clamp(x, 0, 1)
        
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        # Calculate spike times (earlier for higher intensities)
        spike_times = (self.min_delay + (1.0 - x_norm) * (self.timesteps - self.min_delay - 1)).long()
        spike_times = torch.clamp(spike_times, 0, self.timesteps - 1)
        
        # Create spike trains
        for b in range(batch_size):
            for idx in torch.ndindex(input_shape):
                if x_norm[b][idx] > 0:  # Only create spikes for non-zero inputs
                    t = spike_times[b][idx].item()
                    spikes[b, t][idx] = 1.0
                    
        return spikes


class PoissonCoding(SpikeEncoder):
    """Poisson coding: spikes follow Poisson distribution."""
    
    def __init__(self, timesteps: int, max_rate: float = 100.0, dt: float = 1e-3):
        super().__init__(timesteps)
        self.max_rate = max_rate
        self.dt = dt
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to Poisson-coded spikes."""
        # Normalize input to positive values
        x_norm = torch.clamp(torch.abs(x), 0, 1)
        
        # Calculate firing rates
        rates = x_norm * self.max_rate
        
        # Generate Poisson spikes
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        for t in range(self.timesteps):
            # Poisson process: P(spike) = rate * dt for small dt
            spike_prob = rates * self.dt
            spike_prob = torch.clamp(spike_prob, 0, 1)  # Ensure valid probabilities
            
            random_values = torch.rand_like(spike_prob)
            spikes[:, t] = (random_values < spike_prob).float()
            
        return spikes


class PopulationCoding(SpikeEncoder):
    """Population coding: use multiple neurons with different tuning curves."""
    
    def __init__(self, timesteps: int, num_neurons: int, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(timesteps)
        self.num_neurons = num_neurons
        self.min_val = min_val
        self.max_val = max_val
        
        # Create Gaussian receptive fields
        centers = torch.linspace(min_val, max_val, num_neurons)
        self.register_buffer('centers', centers)
        self.sigma = (max_val - min_val) / (2 * num_neurons)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to population-coded spikes."""
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        # Expand input for population coding
        x_expanded = x.unsqueeze(-1)  # Add neuron dimension
        centers_expanded = self.centers.view(1, *[1] * len(input_shape), self.num_neurons)
        
        # Calculate Gaussian responses
        responses = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * self.sigma ** 2))
        
        # Generate spikes using rate coding for each neuron
        output_shape = (*input_shape, self.num_neurons)
        spikes = torch.zeros(batch_size, self.timesteps, *output_shape, device=x.device)
        
        for t in range(self.timesteps):
            random_values = torch.rand_like(responses)
            spikes[:, t] = (random_values < responses).float()
            
        return spikes


class DeltaCoding(SpikeEncoder):
    """Delta coding: spikes represent changes in input."""
    
    def __init__(self, timesteps: int, threshold: float = 0.1):
        super().__init__(timesteps)
        self.threshold = threshold
        self.register_buffer('previous_input', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input changes to spikes."""
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        if self.previous_input is None:
            self.previous_input = torch.zeros_like(x)
            
        current_input = x
        
        for t in range(self.timesteps):
            # Calculate change from previous input
            delta = current_input - self.previous_input
            
            # Generate spikes for positive and negative changes
            pos_spikes = (delta > self.threshold).float()
            neg_spikes = (delta < -self.threshold).float()
            
            # Combine positive and negative spikes (could use separate channels)
            spikes[:, t] = pos_spikes - neg_spikes  # +1 for increase, -1 for decrease
            
            # Update previous input where spikes occurred
            self.previous_input = torch.where(
                torch.abs(delta) > self.threshold,
                current_input,
                self.previous_input
            )
            
        return spikes
        
    def reset_state(self):
        """Reset encoder state."""
        self.previous_input = None


class LatencyCoding(SpikeEncoder):
    """Latency coding: first spike time encodes information."""
    
    def __init__(self, timesteps: int, max_latency: Optional[int] = None):
        super().__init__(timesteps)
        self.max_latency = max_latency or timesteps - 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to latency-coded spikes."""
        # Normalize input to [0, 1]
        x_norm = torch.clamp(x, 0, 1)
        
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        # Calculate latencies (shorter for higher intensities)
        latencies = ((1.0 - x_norm) * self.max_latency).long()
        latencies = torch.clamp(latencies, 0, self.timesteps - 1)
        
        # Create first-spike-time encoding
        for b in range(batch_size):
            for idx in torch.ndindex(input_shape):
                if x_norm[b][idx] > 0:  # Only create spikes for non-zero inputs
                    t = latencies[b][idx].item()
                    spikes[b, t][idx] = 1.0
                    
        return spikes


class BurstCoding(SpikeEncoder):
    """Burst coding: bursts of spikes encode information."""
    
    def __init__(self, timesteps: int, burst_length: int = 3, interburst_interval: int = 5):
        super().__init__(timesteps)
        self.burst_length = burst_length
        self.interburst_interval = interburst_interval
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to burst-coded spikes."""
        # Normalize input to [0, 1]
        x_norm = torch.clamp(x, 0, 1)
        
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        # Calculate number of bursts based on input intensity
        max_bursts = self.timesteps // (self.burst_length + self.interburst_interval)
        num_bursts = (x_norm * max_bursts).long()
        
        for b in range(batch_size):
            for idx in torch.ndindex(input_shape):
                n_bursts = num_bursts[b][idx].item()
                
                for burst_idx in range(n_bursts):
                    start_time = burst_idx * (self.burst_length + self.interburst_interval)
                    end_time = min(start_time + self.burst_length, self.timesteps)
                    
                    spikes[b, start_time:end_time][idx] = 1.0
                    
        return spikes


class AdaptiveThresholdCoding(SpikeEncoder):
    """Adaptive threshold coding with dynamic thresholds."""
    
    def __init__(self, timesteps: int, initial_threshold: float = 0.5, adaptation_rate: float = 0.1):
        super().__init__(timesteps)
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.register_buffer('threshold', None)
        self.register_buffer('accumulated_input', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input using adaptive thresholds."""
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        if self.threshold is None:
            self.threshold = torch.full((batch_size, *input_shape), 
                                      self.initial_threshold, device=x.device)
            self.accumulated_input = torch.zeros_like(self.threshold)
            
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device)
        
        for t in range(self.timesteps):
            # Accumulate input
            self.accumulated_input += x
            
            # Generate spikes when threshold is exceeded
            spike_mask = self.accumulated_input >= self.threshold
            spikes[:, t] = spike_mask.float()
            
            # Reset accumulated input where spikes occurred
            self.accumulated_input = torch.where(spike_mask, 
                                               torch.zeros_like(self.accumulated_input),
                                               self.accumulated_input)
            
            # Adapt threshold based on recent activity
            self.threshold = self.threshold * (1 - self.adaptation_rate) + \
                           self.adaptation_rate * torch.abs(x)
            
        return spikes
        
    def reset_state(self):
        """Reset encoder state."""
        self.threshold = None
        self.accumulated_input = None


# Factory function for creating encoders
def create_encoder(encoding_type: str, timesteps: int, **kwargs) -> SpikeEncoder:
    """Factory function to create encoders by type."""
    encoder_types = {
        "RATE": RateCoding,
        "TEMPORAL": TemporalCoding,
        "POISSON": PoissonCoding,
        "POPULATION": PopulationCoding,
        "DELTA": DeltaCoding,
        "LATENCY": LatencyCoding,
        "BURST": BurstCoding,
        "ADAPTIVE": AdaptiveThresholdCoding
    }
    
    if encoding_type.upper() not in encoder_types:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Available: {list(encoder_types.keys())}")
        
    return encoder_types[encoding_type.upper()](timesteps=timesteps, **kwargs)


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder combining different encoding strategies."""
    
    def __init__(self, encoders: Dict[str, SpikeEncoder], combination_method: str = "concatenate"):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.combination_method = combination_method
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using multiple strategies and combine."""
        encoded_outputs = []
        
        for name, encoder in self.encoders.items():
            encoded = encoder(x)
            encoded_outputs.append(encoded)
            
        if self.combination_method == "concatenate":
            # Concatenate along feature dimension
            return torch.cat(encoded_outputs, dim=-1)
        elif self.combination_method == "average":
            # Average across encoders
            return torch.stack(encoded_outputs, dim=0).mean(dim=0)
        elif self.combination_method == "weighted_sum":
            # Learnable weighted combination (weights would need to be parameters)
            return torch.stack(encoded_outputs, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
            
    def reset_state(self):
        """Reset all encoder states."""
        for encoder in self.encoders.values():
            if hasattr(encoder, 'reset_state'):
                encoder.reset_state()