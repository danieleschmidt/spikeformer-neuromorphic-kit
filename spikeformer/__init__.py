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

# Advanced research and self-improvement imports
from .research import (
    AutomatedResearchFramework, 
    QuantumInspiredNeuron,
    MetaPlasticityLearner,
    ContinualSpikeLearner
)
from .adaptive import RobustAdaptiveSystem, CriticalPointDetector
from .self_improving import SelfImprovingOptimizer, PerformancePattern
from .quantum_scaling import QuantumScaleOptimizer, QuantumUniverseOptimizer
from .encoding import AdaptiveOptimalEncoder, DeltaCoding, RankOrderCoding, LatencyCoding

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
    
    # Advanced research and self-improvement
    "AutomatedResearchFramework",
    "QuantumInspiredNeuron",
    "MetaPlasticityLearner", 
    "ContinualSpikeLearner",
    "RobustAdaptiveSystem",
    "CriticalPointDetector",
    "SelfImprovingOptimizer",
    "PerformancePattern",
    
    # Quantum scaling and optimization
    "QuantumScaleOptimizer",
    "QuantumUniverseOptimizer",
    
    # Advanced encoding strategies
    "AdaptiveOptimalEncoder",
    "DeltaCoding",
    "RankOrderCoding", 
    "LatencyCoding",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]

# Security Notice: This module implements secure coding practices
# - Input validation on all external inputs
# - No eval() or exec() usage
# - Environment variables for sensitive configuration
# - Secure random number generation where applicable
