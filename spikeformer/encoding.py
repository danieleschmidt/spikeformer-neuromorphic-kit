"""Spike encoding strategies for converting continuous data to spike trains."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
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
    """Rate coding: spike frequency proportional to input intensity.
    
    Args:
        timesteps: Number of time steps for encoding
        max_rate: Maximum spike rate (probability per timestep)
        gain: Scaling factor for input values
        
    Raises:
        ValueError: If parameters are invalid
    """
    
    def __init__(self, timesteps: int, max_rate: float = 1.0, gain: float = 1.0):
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if max_rate <= 0 or max_rate > 1:
            raise ValueError(f"max_rate must be in (0, 1], got {max_rate}")
        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain}")
            
        super().__init__(timesteps)
        self.max_rate = max_rate
        self.gain = gain
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to rate-coded spikes.
        
        Args:
            x: Input tensor of shape (batch, ...)
            
        Returns:
            Spike tensor of shape (batch, timesteps, ...)
            
        Raises:
            ValueError: If input tensor is invalid
        """
        if x.dim() < 1:
            raise ValueError(f"Input must have at least 1 dimension, got {x.dim()}")
            
        # Normalize input to [0, 1] range
        x_norm = torch.clamp(torch.abs(x) * self.gain, 0, 1)
        
        # Calculate spike probabilities
        spike_prob = x_norm * self.max_rate
        
        # Generate spikes for each timestep
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device, dtype=x.dtype)
        
        # More efficient spike generation
        random_tensor = torch.rand(batch_size, self.timesteps, *input_shape, device=x.device, dtype=x.dtype)
        spike_prob_expanded = spike_prob.unsqueeze(1).expand_as(random_tensor)
        spikes = (random_tensor < spike_prob_expanded).float()
            
        return spikes


class TemporalCoding(SpikeEncoder):
    """Temporal coding: spike timing encodes information.
    
    Higher input values result in earlier spike times.
    
    Args:
        timesteps: Number of time steps for encoding
        min_delay: Minimum delay (earliest possible spike time)
        
    Raises:
        ValueError: If parameters are invalid
    """
    
    def __init__(self, timesteps: int, min_delay: int = 0):
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if min_delay < 0 or min_delay >= timesteps:
            raise ValueError(f"min_delay must be in [0, timesteps), got {min_delay}")
            
        super().__init__(timesteps)
        self.min_delay = min_delay
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to temporally-coded spikes.
        
        Args:
            x: Input tensor of shape (batch, ...)
            
        Returns:
            Spike tensor of shape (batch, timesteps, ...)
        """
        if x.dim() < 1:
            raise ValueError(f"Input must have at least 1 dimension, got {x.dim()}")
            
        # Normalize input to [0, 1]
        x_norm = torch.clamp(x, 0, 1)
        
        batch_size = x.shape[0]
        input_shape = x.shape[1:]
        
        spikes = torch.zeros(batch_size, self.timesteps, *input_shape, device=x.device, dtype=x.dtype)
        
        # Calculate spike times (earlier for higher intensities)
        available_time_range = self.timesteps - self.min_delay - 1
        spike_times = (self.min_delay + (1.0 - x_norm) * available_time_range).long()
        spike_times = torch.clamp(spike_times, self.min_delay, self.timesteps - 1)
        
        # More efficient spike placement using advanced indexing
        mask = x_norm > 0  # Only create spikes for non-zero inputs
        if mask.any():
            # Get indices where spikes should be placed
            batch_indices = torch.arange(batch_size, device=x.device).view(-1, *([1] * len(input_shape)))
            batch_indices = batch_indices.expand_as(x_norm)[mask]
            
            time_indices = spike_times[mask]
            
            # Handle multi-dimensional indexing
            if len(input_shape) > 0:
                spatial_indices = torch.stack(torch.meshgrid(
                    *[torch.arange(s, device=x.device) for s in input_shape], indexing='ij'
                ), dim=0).view(len(input_shape), -1).T
                spatial_indices = spatial_indices.repeat(batch_size, 1)[mask.flatten()]
                
                # Place spikes
                for i, (b_idx, t_idx) in enumerate(zip(batch_indices, time_indices)):
                    spatial_idx = tuple(spatial_indices[i])
                    spikes[b_idx, t_idx][spatial_idx] = 1.0
            else:
                # Scalar case
                spikes[batch_indices, time_indices] = 1.0
                    
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
        if hasattr(self, 'previous_input'):
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
    """Adaptive threshold coding with dynamic thresholds.
    
    Maintains adaptive thresholds that change based on input statistics
    to maintain consistent spike rates.
    
    Args:
        timesteps: Number of time steps for encoding
        initial_threshold: Initial threshold value
        adaptation_rate: Rate of threshold adaptation
        
    Raises:
        ValueError: If parameters are invalid
    """
    
    def __init__(self, timesteps: int, initial_threshold: float = 0.5, adaptation_rate: float = 0.1):
        if timesteps <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if initial_threshold <= 0:
            raise ValueError(f"initial_threshold must be positive, got {initial_threshold}")
        if not 0 < adaptation_rate <= 1:
            raise ValueError(f"adaptation_rate must be in (0, 1], got {adaptation_rate}")
            
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
        if hasattr(self, 'threshold'):
            self.threshold = None
        if hasattr(self, 'accumulated_input'):
            self.accumulated_input = None


# Factory function for creating encoders
def create_encoder(encoding_type: str, timesteps: int, **kwargs) -> SpikeEncoder:
    """Factory function to create encoders by type.
    
    Args:
        encoding_type: Type of encoding to create (case-insensitive)
        timesteps: Number of timesteps for encoding
        **kwargs: Additional arguments for encoder
        
    Returns:
        Initialized encoder instance
        
    Raises:
        ValueError: If encoding_type is not supported or parameters are invalid
        TypeError: If invalid arguments are provided
        
    Available encoding types:
        - RATE: Rate coding (frequency proportional to intensity)
        - TEMPORAL: Temporal coding (timing encodes information)
        - POISSON: Poisson coding (Poisson spike generation)
        - POPULATION: Population coding (multiple tuned neurons)
        - DELTA: Delta coding (spikes represent changes)
        - LATENCY: Latency coding (first spike time)
        - BURST: Burst coding (bursts of spikes)
        - ADAPTIVE: Adaptive threshold coding
    """
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
    
    encoding_type_upper = encoding_type.upper()
    
    if encoding_type_upper not in encoder_types:
        available_types = ", ".join(encoder_types.keys())
        raise ValueError(
            f"Unknown encoding type: '{encoding_type}'. "
            f"Available types: {available_types}"
        )
    
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    
    try:
        encoder_class = encoder_types[encoding_type_upper]
        return encoder_class(timesteps=timesteps, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid arguments for {encoding_type} encoder: {e}. "
            f"Please check the encoder's __init__ method for required arguments."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to create {encoding_type} encoder: {e}"
        ) from e


def get_available_encoding_types() -> List[str]:
    """Get list of available encoding types.
    
    Returns:
        List of supported encoding type names
    """
    return [
        "RATE", "TEMPORAL", "POISSON", "POPULATION",
        "DELTA", "LATENCY", "BURST", "ADAPTIVE"
    ]


def validate_encoding_config(encoding_type: str, timesteps: int, **kwargs) -> Dict[str, Any]:
    """Validate encoding configuration parameters.
    
    Args:
        encoding_type: Type of encoding
        timesteps: Number of timesteps
        **kwargs: Configuration parameters
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValueError: If configuration is invalid
    """
    import inspect
    
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
    
    encoding_type_upper = encoding_type.upper()
    if encoding_type_upper not in encoder_types:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
        
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
        
    encoder_class = encoder_types[encoding_type_upper]
    sig = inspect.signature(encoder_class.__init__)
    
    # Validate parameters
    valid_params = {"timesteps": timesteps}
    for param_name, param_value in kwargs.items():
        if param_name in sig.parameters:
            valid_params[param_name] = param_value
        else:
            raise ValueError(
                f"Parameter '{param_name}' is not valid for {encoding_type} encoder. "
                f"Valid parameters: {list(sig.parameters.keys())[1:]}"  # Skip 'self'
            )
    
    return valid_params


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder combining different encoding strategies.
    
    Args:
        encoders: Dictionary of encoder name to SpikeEncoder instance
        combination_method: How to combine encoder outputs (concatenate, average, weighted_sum)
        weights: Optional weights for weighted_sum combination
        
    Raises:
        ValueError: If combination_method is not supported
    """
    
    def __init__(self, encoders: Dict[str, SpikeEncoder], combination_method: str = "concatenate", 
                 weights: Optional[Dict[str, float]] = None):
        if not encoders:
            raise ValueError("At least one encoder must be provided")
            
        valid_methods = ["concatenate", "average", "weighted_sum"]
        if combination_method not in valid_methods:
            raise ValueError(f"combination_method must be one of {valid_methods}, got {combination_method}")
            
        if combination_method == "weighted_sum":
            if weights is None:
                weights = {name: 1.0 for name in encoders.keys()}
            elif set(weights.keys()) != set(encoders.keys()):
                raise ValueError("Weight keys must match encoder keys")
                
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.combination_method = combination_method
        
        if weights is not None:
            self.register_buffer('weights', torch.tensor(list(weights.values())))
            self.weight_names = list(weights.keys())
        else:
            self.weights = None
            self.weight_names = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using multiple strategies and combine.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined encoded tensor
        """
        if not self.encoders:
            raise RuntimeError("No encoders available")
            
        encoded_outputs = []
        
        for name, encoder in self.encoders.items():
            try:
                encoded = encoder(x)
                encoded_outputs.append(encoded)
            except Exception as e:
                raise RuntimeError(f"Encoder '{name}' failed: {e}") from e
            
        if self.combination_method == "concatenate":
            # Concatenate along feature dimension
            return torch.cat(encoded_outputs, dim=-1)
        elif self.combination_method == "average":
            # Average across encoders
            return torch.stack(encoded_outputs, dim=0).mean(dim=0)
        elif self.combination_method == "weighted_sum":
            # Weighted combination
            if self.weights is None:
                return torch.stack(encoded_outputs, dim=0).sum(dim=0)
            else:
                weighted_outputs = []
                for i, output in enumerate(encoded_outputs):
                    weight = self.weights[i]
                    weighted_outputs.append(output * weight)
                return torch.stack(weighted_outputs, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
            
    def reset_state(self):
        """Reset all encoder states."""
        for name, encoder in self.encoders.items():
            if hasattr(encoder, 'reset_state'):
                try:
                    encoder.reset_state()
                except Exception as e:
                    # Log warning but don't fail
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to reset state for encoder '{name}': {e}")