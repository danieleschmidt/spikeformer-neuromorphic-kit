"""
Input encoding for spiking neural networks.

Converts continuous-valued inputs into spike trains for SNN processing.

Encoders:
- RateEncoder:    Poisson rate coding — spike probability ∝ value
- LatencyEncoder: First-to-fire latency coding — higher value fires sooner
- DirectEncoder:  Pass-through (no temporal coding; input repeated T times)
"""

import torch
import torch.nn as nn


class RateEncoder(nn.Module):
    """
    Poisson rate coding.

    Each input value v ∈ [0,1] is treated as a firing probability per timestep.
    Over T timesteps, avg spike rate ≈ v.

    Args:
        timesteps: number of simulation timesteps
        normalize: if True, linearly scale input to [0, 1] before encoding
    """

    def __init__(self, timesteps: int = 16, normalize: bool = True):
        super().__init__()
        self.timesteps = timesteps
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, *features) — continuous values

        Returns:
            spikes: (batch, timesteps, *features) — binary
        """
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            x = x.clamp(0.0, 1.0)

        # Expand over time and sample Bernoulli
        x_expanded = x.unsqueeze(1).expand(-1, self.timesteps, *[-1] * (x.dim() - 1))
        return torch.bernoulli(x_expanded)


class LatencyEncoder(nn.Module):
    """
    Latency (time-to-first-spike) coding.

    Neurons with higher input values fire earlier.  Each neuron fires exactly
    once — whichever timestep corresponds to its latency.

    Args:
        timesteps: total number of timesteps
    """

    def __init__(self, timesteps: int = 16):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, *features) — values in [0, 1]

        Returns:
            spikes: (batch, timesteps, *features) — one spike per neuron
        """
        # Normalize to [0,1], then map to fire at timestep t = (1-v) * T
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x_norm = x_norm.clamp(0.0, 1.0)

        # latency step for each element (higher value → earlier fire)
        latency = ((1.0 - x_norm) * (self.timesteps - 1)).long()  # (batch, *features)

        # Build spike tensor
        T = self.timesteps
        spikes = torch.zeros(x.shape[0], T, *x.shape[1:], device=x.device)
        t_idx = latency.unsqueeze(1)  # (batch, 1, *features)
        spikes.scatter_(1, t_idx, 1.0)

        return spikes


class DirectEncoder(nn.Module):
    """
    Direct / trivial encoding — simply repeat input T times along a new
    time axis.  Useful for models that handle temporal dynamics internally
    (e.g. via LIF neurons).  No binarization.

    Args:
        timesteps: number of simulation timesteps
    """

    def __init__(self, timesteps: int = 16):
        super().__init__()
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, *features)

        Returns:
            out: (batch, timesteps, *features)
        """
        return x.unsqueeze(1).expand(-1, self.timesteps, *[-1] * (x.dim() - 1))
