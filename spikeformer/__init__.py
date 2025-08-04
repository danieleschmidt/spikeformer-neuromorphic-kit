"""SpikeFormer Neuromorphic Kit - Complete toolkit for spiking neural networks."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@your-org.com"

# Core imports
from .conversion import SpikeformerConverter, ConversionPipeline
from .models import SpikingTransformer, SpikingAttention, SpikingMLP
from .neurons import LifNeuron, AdLifNeuron
from .encoding import RateCoding, TemporalCoding, PoissonCoding
from .profiling import EnergyProfiler, PowerMonitor
from .hardware import NeuromorphicDeployer

# Monitoring imports (already exist)
from .monitoring import metrics, start_monitoring, stop_monitoring
from .health import health_monitor, get_health_summary

__all__ = [
    # Core functionality
    "SpikeformerConverter",
    "ConversionPipeline", 
    "SpikingTransformer",
    "SpikingAttention",
    "SpikingMLP",
    "LifNeuron",
    "AdLifNeuron",
    "RateCoding",
    "TemporalCoding", 
    "PoissonCoding",
    "EnergyProfiler",
    "PowerMonitor",
    "NeuromorphicDeployer",
    
    # Monitoring
    "metrics",
    "start_monitoring",
    "stop_monitoring",
    "health_monitor",
    "get_health_summary",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]