"""SpikeFormer Neuromorphic Kit — clean spiking transformer in pure PyTorch."""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from .neurons import LIFNeuron, SpikeRateTracker, spike_fn
from .models import SpikeFormer, SpikeFormerBlock, SpikeAttention, SpikeMLP
from .encoding import RateEncoder, LatencyEncoder, DirectEncoder

__all__ = [
    # Neurons
    "LIFNeuron",
    "SpikeRateTracker",
    "spike_fn",
    # Models
    "SpikeFormer",
    "SpikeFormerBlock",
    "SpikeAttention",
    "SpikeMLP",
    # Encoding
    "RateEncoder",
    "LatencyEncoder",
    "DirectEncoder",
]
