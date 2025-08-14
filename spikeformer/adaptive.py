"""Adaptive learning algorithms for neuromorphic computing with online optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import time
from collections import deque
import math

from .neurons import LifNeuron, SpikingLayer


@dataclass
class AdaptationConfig:
    """Configuration for adaptive learning mechanisms."""
    adaptation_rate: float = 0.01
    momentum: float = 0.9
    adaptation_window: int = 1000
    stability_threshold: float = 0.01
    performance_metric: str = "accuracy"
    adaptation_frequency: int = 100
    min_adaptation_rate: float = 1e-6
    max_adaptation_rate: float = 1.0


class AdaptiveMechanism(ABC):
    """Abstract base class for adaptive mechanisms."""
    
    @abstractmethod
    def adapt(self, performance_metrics: Dict[str, float], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model parameters based on performance metrics."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset adaptation state."""
        pass


class AdaptiveThresholdController(AdaptiveMechanism):
    """Adaptive threshold control for spiking neurons with homeostasis."""
    
    def __init__(self, config: AdaptationConfig, target_spike_rate: float = 0.1):
        self.config = config
        self.target_spike_rate = target_spike_rate
        self.spike_history = deque(maxlen=config.adaptation_window)
        self.threshold_history = deque(maxlen=config.adaptation_window)
        self.adaptation_momentum = 0.0
        self.logger = logging.getLogger(__name__)
        
    def adapt(self, performance_metrics: Dict[str, float], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt firing thresholds based on spike statistics and performance."""
        
        current_spike_rate = performance_metrics.get("spike_rate", 0.0)
        current_accuracy = performance_metrics.get("accuracy", 0.0)
        
        # Track history
        self.spike_history.append(current_spike_rate)
        
        # Calculate adaptation signal
        spike_rate_error = self.target_spike_rate - current_spike_rate
        
        # Multi-objective optimization: balance spike rate and accuracy
        if len(self.spike_history) >= 2:
            accuracy_trend = current_accuracy - performance_metrics.get("previous_accuracy", current_accuracy)
            
            # If accuracy is improving, be more conservative with threshold changes
            adaptation_scale = 1.0 if accuracy_trend <= 0 else max(0.1, 1.0 - abs(accuracy_trend))
        else:
            adaptation_scale = 1.0
        
        # Adaptive learning rate based on stability
        if len(self.spike_history) >= 10:
            spike_stability = np.std(list(self.spike_history)[-10:])
            adaptive_rate = self.config.adaptation_rate * (1.0 + spike_stability)
            adaptive_rate = np.clip(adaptive_rate, self.config.min_adaptation_rate, self.config.max_adaptation_rate)
        else:
            adaptive_rate = self.config.adaptation_rate
        
        # Calculate threshold adjustment with momentum
        threshold_delta = adaptive_rate * spike_rate_error * adaptation_scale
        self.adaptation_momentum = (self.config.momentum * self.adaptation_momentum + 
                                   (1 - self.config.momentum) * threshold_delta)
        
        # Apply threshold updates to model
        adaptations = {}
        for name, param in model_state.items():
            if "threshold" in name.lower() and hasattr(param, "data"):
                old_threshold = param.data.clone()
                
                # Adaptive threshold with bounds checking
                new_threshold = old_threshold + self.adaptation_momentum
                new_threshold = torch.clamp(new_threshold, 0.1, 5.0)  # Reasonable bounds
                
                adaptations[name] = {
                    "old_value": old_threshold.mean().item(),
                    "new_value": new_threshold.mean().item(),
                    "delta": self.adaptation_momentum
                }
                
                param.data.copy_(new_threshold)
                
        self.threshold_history.append(self.adaptation_momentum)
        
        return adaptations
    
    def reset(self):
        """Reset adaptation state."""
        self.spike_history.clear()
        self.threshold_history.clear()
        self.adaptation_momentum = 0.0


class AdaptiveSynapticPlasticity(AdaptiveMechanism):
    """STDP-inspired adaptive synaptic plasticity for online learning."""
    
    def __init__(self, config: AdaptationConfig, tau_plus: float = 20.0, tau_minus: float = 20.0):
        self.config = config
        self.tau_plus = tau_plus  # LTP time constant
        self.tau_minus = tau_minus  # LTD time constant
        self.spike_traces = {}
        self.weight_changes = {}
        self.plasticity_history = deque(maxlen=config.adaptation_window)
        
    def adapt(self, performance_metrics: Dict[str, float], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt synaptic weights using spike-timing dependent plasticity."""
        
        adaptations = {}
        current_performance = performance_metrics.get(self.config.performance_metric, 0.0)
        
        # Performance-modulated plasticity
        if len(self.plasticity_history) > 0:
            performance_trend = current_performance - np.mean(self.plasticity_history)
            plasticity_modulation = torch.sigmoid(torch.tensor(performance_trend * 10.0)).item()
        else:
            plasticity_modulation = 0.5
        
        # Apply STDP to linear layers
        for name, param in model_state.items():
            if "weight" in name.lower() and param.dim() == 2:  # Linear layer weights
                
                # Hebbian-style adaptation based on activation correlation
                if name in self.spike_traces:
                    pre_trace = self.spike_traces[name]["pre"]
                    post_trace = self.spike_traces[name]["post"]
                    
                    # STDP weight update rule
                    ltp = torch.outer(post_trace, pre_trace) * plasticity_modulation
                    ltd = torch.outer(pre_trace, post_trace) * plasticity_modulation
                    
                    weight_delta = self.config.adaptation_rate * (ltp - ltd)
                    
                    # Apply weight decay and bounds
                    weight_decay = 0.001 * param.data
                    new_weights = param.data + weight_delta - weight_decay
                    new_weights = torch.clamp(new_weights, -10.0, 10.0)
                    
                    adaptations[name] = {
                        "plasticity_strength": weight_delta.abs().mean().item(),
                        "weight_change": (new_weights - param.data).abs().mean().item(),
                        "ltp_strength": ltp.abs().mean().item(),
                        "ltd_strength": ltd.abs().mean().item()
                    }
                    
                    param.data.copy_(new_weights)
                else:
                    # Initialize spike traces for this layer
                    self.spike_traces[name] = {
                        "pre": torch.zeros(param.shape[1]),
                        "post": torch.zeros(param.shape[0])
                    }
        
        self.plasticity_history.append(current_performance)
        return adaptations
    
    def update_spike_traces(self, layer_name: str, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update spike traces for STDP calculation."""
        if layer_name not in self.spike_traces:
            self.spike_traces[layer_name] = {
                "pre": torch.zeros_like(pre_spikes[0]),
                "post": torch.zeros_like(post_spikes[0])
            }
        
        # Exponential decay of traces
        decay_pre = torch.exp(torch.tensor(-1.0 / self.tau_plus))
        decay_post = torch.exp(torch.tensor(-1.0 / self.tau_minus))
        
        self.spike_traces[layer_name]["pre"] *= decay_pre
        self.spike_traces[layer_name]["post"] *= decay_post
        
        # Add current spikes
        self.spike_traces[layer_name]["pre"] += pre_spikes.mean(dim=0)
        self.spike_traces[layer_name]["post"] += post_spikes.mean(dim=0)
    
    def reset(self):
        """Reset plasticity state."""
        self.spike_traces.clear()
        self.weight_changes.clear()
        self.plasticity_history.clear()


class AdaptiveArchitectureController(AdaptiveMechanism):
    """Dynamic architecture adaptation based on performance."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.adaptation_window)
        self.architecture_changes = []
        self.adaptation_cycle = 0
        
    def adapt(self, performance_metrics: Dict[str, float], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt model architecture based on performance trends."""
        
        current_performance = performance_metrics.get(self.config.performance_metric, 0.0)
        self.performance_history.append(current_performance)
        
        adaptations = {}
        
        # Only adapt architecture periodically
        self.adaptation_cycle += 1
        if self.adaptation_cycle % (self.config.adaptation_frequency * 10) != 0:
            return adaptations
        
        if len(self.performance_history) < 50:  # Need sufficient history
            return adaptations
        
        # Analyze performance trend
        recent_performance = np.mean(list(self.performance_history)[-20:])
        older_performance = np.mean(list(self.performance_history)[-50:-20])
        
        performance_trend = recent_performance - older_performance
        performance_variance = np.var(list(self.performance_history)[-20:])
        
        # Architecture adaptation decisions
        if performance_trend < -0.05 and performance_variance > 0.01:
            # Performance declining and unstable - simplify architecture
            adaptation_decision = "simplify"
        elif performance_trend > 0.02 and performance_variance < 0.005:
            # Performance improving and stable - can complexify
            adaptation_decision = "complexify"
        else:
            # Status quo
            adaptation_decision = "maintain"
        
        adaptations["architecture_decision"] = adaptation_decision
        adaptations["performance_trend"] = performance_trend
        adaptations["performance_variance"] = performance_variance
        
        # Record architecture change
        self.architecture_changes.append({
            "cycle": self.adaptation_cycle,
            "decision": adaptation_decision,
            "performance": current_performance,
            "trend": performance_trend
        })
        
        return adaptations
    
    def reset(self):
        """Reset architecture adaptation state."""
        self.performance_history.clear()
        self.architecture_changes.clear()
        self.adaptation_cycle = 0


class MetaLearningController(AdaptiveMechanism):
    """Meta-learning for adaptation parameter optimization."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.meta_parameters = {
            "adaptation_rate": config.adaptation_rate,
            "momentum": config.momentum,
            "stability_threshold": config.stability_threshold
        }
        self.meta_gradients = {k: 0.0 for k in self.meta_parameters.keys()}
        self.performance_history = deque(maxlen=1000)
        self.meta_learning_rate = 0.001
        
    def adapt(self, performance_metrics: Dict[str, float], model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt meta-parameters based on learning efficiency."""
        
        current_performance = performance_metrics.get(self.config.performance_metric, 0.0)
        self.performance_history.append(current_performance)
        
        adaptations = {}
        
        if len(self.performance_history) >= 100:
            # Calculate learning efficiency
            recent_improvement = self._calculate_learning_efficiency()
            
            # Update meta-parameters based on efficiency
            for param_name in self.meta_parameters.keys():
                # Simple gradient-based meta-parameter update
                if recent_improvement > 0:
                    self.meta_gradients[param_name] = 0.9 * self.meta_gradients[param_name] + 0.1
                else:
                    self.meta_gradients[param_name] = 0.9 * self.meta_gradients[param_name] - 0.1
                
                # Update meta-parameter
                old_value = self.meta_parameters[param_name]
                new_value = old_value + self.meta_learning_rate * self.meta_gradients[param_name]
                
                # Apply bounds
                if param_name == "adaptation_rate":
                    new_value = np.clip(new_value, 1e-6, 1.0)
                elif param_name == "momentum":
                    new_value = np.clip(new_value, 0.0, 0.99)
                elif param_name == "stability_threshold":
                    new_value = np.clip(new_value, 1e-4, 1.0)
                
                self.meta_parameters[param_name] = new_value
                
                adaptations[f"meta_{param_name}"] = {
                    "old_value": old_value,
                    "new_value": new_value,
                    "gradient": self.meta_gradients[param_name]
                }
        
        return adaptations
    
    def _calculate_learning_efficiency(self) -> float:
        """Calculate recent learning efficiency."""
        if len(self.performance_history) < 50:
            return 0.0
        
        # Compare recent performance to earlier performance
        recent_window = list(self.performance_history)[-25:]
        earlier_window = list(self.performance_history)[-50:-25]
        
        recent_mean = np.mean(recent_window)
        earlier_mean = np.mean(earlier_window)
        
        return recent_mean - earlier_mean
    
    def get_meta_parameters(self) -> Dict[str, float]:
        """Get current meta-parameters for other adaptation mechanisms."""
        return self.meta_parameters.copy()
    
    def reset(self):
        """Reset meta-learning state."""
        self.meta_gradients = {k: 0.0 for k in self.meta_parameters.keys()}
        self.performance_history.clear()


class AdaptiveSpikingModel(nn.Module):
    """Spiking model with integrated adaptive learning mechanisms."""
    
    def __init__(self, base_model: nn.Module, adaptation_config: AdaptationConfig = None):
        super().__init__()
        self.base_model = base_model
        self.config = adaptation_config or AdaptationConfig()
        
        # Initialize adaptation mechanisms
        self.threshold_controller = AdaptiveThresholdController(self.config)
        self.plasticity_controller = AdaptiveSynapticPlasticity(self.config)
        self.architecture_controller = AdaptiveArchitectureController(self.config)
        self.meta_controller = MetaLearningController(self.config)
        
        # Adaptation tracking
        self.adaptation_step = 0
        self.performance_buffer = deque(maxlen=100)
        self.adaptation_history = []
        
        # Performance tracking
        self.register_buffer("running_accuracy", torch.tensor(0.0))
        self.register_buffer("running_spike_rate", torch.tensor(0.0))
        self.register_buffer("adaptation_momentum", torch.tensor(0.9))
        
        self.logger = logging.getLogger(__name__)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spike monitoring for adaptation."""
        
        # Standard forward pass
        output = self.base_model(x)
        
        # Monitor spikes if in training mode
        if self.training:
            self._monitor_spikes()
            
        return output
    
    def _monitor_spikes(self):
        """Monitor spike activity for adaptation."""
        total_spikes = 0
        total_activations = 0
        
        # Hook-based spike monitoring
        def spike_hook(module, input, output):
            nonlocal total_spikes, total_activations
            if hasattr(output, 'sum'):  # Spike tensor
                total_spikes += output.sum().item()
                total_activations += output.numel()
        
        # Register temporary hooks
        hooks = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, (LifNeuron, SpikingLayer)):
                hook = module.register_forward_hook(spike_hook)
                hooks.append(hook)
        
        # Clean up hooks (they were already executed during forward pass)
        for hook in hooks:
            hook.remove()
        
        # Update running spike rate
        if total_activations > 0:
            current_spike_rate = total_spikes / total_activations
            self.running_spike_rate = (0.95 * self.running_spike_rate + 
                                     0.05 * current_spike_rate)
    
    def adapt_online(self, performance_metrics: Dict[str, float]):
        """Perform online adaptation based on performance metrics."""
        
        # Update performance tracking
        if "accuracy" in performance_metrics:
            accuracy = performance_metrics["accuracy"]
            self.running_accuracy = 0.95 * self.running_accuracy + 0.05 * accuracy
            
        # Add spike rate to metrics
        performance_metrics["spike_rate"] = self.running_spike_rate.item()
        performance_metrics["previous_accuracy"] = self.running_accuracy.item()
        
        # Get current model state
        model_state = {name: param for name, param in self.named_parameters()}
        
        # Update meta-parameters
        meta_adaptations = self.meta_controller.adapt(performance_metrics, model_state)
        current_meta_params = self.meta_controller.get_meta_parameters()
        
        # Update adaptation mechanisms with new meta-parameters
        self.threshold_controller.config.adaptation_rate = current_meta_params["adaptation_rate"]
        self.threshold_controller.config.momentum = current_meta_params["momentum"]
        self.plasticity_controller.config.adaptation_rate = current_meta_params["adaptation_rate"]
        
        # Apply adaptation mechanisms
        adaptations = {}
        
        # Only adapt every N steps
        self.adaptation_step += 1
        if self.adaptation_step % self.config.adaptation_frequency == 0:
            
            # Threshold adaptation
            threshold_adaptations = self.threshold_controller.adapt(performance_metrics, model_state)
            adaptations["thresholds"] = threshold_adaptations
            
            # Synaptic plasticity
            plasticity_adaptations = self.plasticity_controller.adapt(performance_metrics, model_state)
            adaptations["plasticity"] = plasticity_adaptations
            
            # Architecture adaptation (less frequent)
            arch_adaptations = self.architecture_controller.adapt(performance_metrics, model_state)
            adaptations["architecture"] = arch_adaptations
            
        # Add meta-learning adaptations
        adaptations["meta_learning"] = meta_adaptations
        
        # Store adaptation history
        self.adaptation_history.append({
            "step": self.adaptation_step,
            "performance": performance_metrics,
            "adaptations": adaptations,
            "timestamp": time.time()
        })
        
        # Limit history size
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]
        
        return adaptations
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation performance."""
        if not self.adaptation_history:
            return {"status": "no_adaptations"}
        
        recent_history = self.adaptation_history[-50:]
        
        summary = {
            "total_adaptations": len(self.adaptation_history),
            "recent_performance": {
                "mean_accuracy": np.mean([h["performance"].get("accuracy", 0) for h in recent_history]),
                "mean_spike_rate": np.mean([h["performance"].get("spike_rate", 0) for h in recent_history]),
            },
            "adaptation_frequency": {
                "threshold_adaptations": sum(1 for h in recent_history if h["adaptations"].get("thresholds")),
                "plasticity_adaptations": sum(1 for h in recent_history if h["adaptations"].get("plasticity")),
                "architecture_adaptations": sum(1 for h in recent_history if h["adaptations"].get("architecture")),
            },
            "meta_parameters": self.meta_controller.get_meta_parameters(),
            "adaptation_trend": self._calculate_adaptation_trend()
        }
        
        return summary
    
    def _calculate_adaptation_trend(self) -> str:
        """Calculate overall adaptation trend."""
        if len(self.adaptation_history) < 20:
            return "insufficient_data"
        
        recent_performance = [h["performance"].get("accuracy", 0) for h in self.adaptation_history[-10:]]
        older_performance = [h["performance"].get("accuracy", 0) for h in self.adaptation_history[-20:-10]]
        
        recent_mean = np.mean(recent_performance)
        older_mean = np.mean(older_performance)
        
        if recent_mean > older_mean + 0.01:
            return "improving"
        elif recent_mean < older_mean - 0.01:
            return "declining"
        else:
            return "stable"
    
    def reset_adaptation_state(self):
        """Reset all adaptation mechanisms."""
        self.threshold_controller.reset()
        self.plasticity_controller.reset()
        self.architecture_controller.reset()
        self.meta_controller.reset()
        
        self.adaptation_step = 0
        self.adaptation_history.clear()
        self.running_accuracy.fill_(0.0)
        self.running_spike_rate.fill_(0.0)
    
    def save_adaptation_state(self, filepath: str):
        """Save adaptation state for later restoration."""
        state = {
            "adaptation_step": self.adaptation_step,
            "adaptation_history": self.adaptation_history,
            "running_accuracy": self.running_accuracy.item(),
            "running_spike_rate": self.running_spike_rate.item(),
            "meta_parameters": self.meta_controller.get_meta_parameters(),
            "config": self.config
        }
        
        torch.save(state, filepath)
        self.logger.info(f"Adaptation state saved to {filepath}")
    
    def load_adaptation_state(self, filepath: str):
        """Load adaptation state from file."""
        state = torch.load(filepath)
        
        self.adaptation_step = state["adaptation_step"]
        self.adaptation_history = state["adaptation_history"]
        self.running_accuracy.fill_(state["running_accuracy"])
        self.running_spike_rate.fill_(state["running_spike_rate"])
        
        # Restore meta-parameters
        self.meta_controller.meta_parameters = state["meta_parameters"]
        
        self.logger.info(f"Adaptation state loaded from {filepath}")


def create_adaptive_model(base_model: nn.Module, 
                         adaptation_config: Optional[AdaptationConfig] = None) -> AdaptiveSpikingModel:
    """Factory function to create adaptive spiking model."""
    
    if adaptation_config is None:
        adaptation_config = AdaptationConfig(
            adaptation_rate=0.01,
            momentum=0.9,
            adaptation_window=1000,
            stability_threshold=0.01,
            performance_metric="accuracy",
            adaptation_frequency=50
        )
    
    return AdaptiveSpikingModel(base_model, adaptation_config)


# Example usage and testing
if __name__ == "__main__":
    # Example of creating and using an adaptive spiking model
    from .models import SpikingTransformer
    
    # Create base model
    base_model = SpikingTransformer(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        intermediate_size=1024,
        timesteps=32
    )
    
    # Create adaptive model
    adaptive_model = create_adaptive_model(base_model)
    
    print(f"Created adaptive model with {sum(p.numel() for p in adaptive_model.parameters())} parameters")
    print(f"Adaptation frequency: {adaptive_model.config.adaptation_frequency}")
    print(f"Initial adaptation rate: {adaptive_model.config.adaptation_rate}")
    
    # Simulate online learning
    for step in range(100):
        # Simulate performance metrics
        performance = {
            "accuracy": 0.5 + 0.3 * np.sin(step * 0.1) + 0.1 * np.random.randn(),
            "loss": 1.0 - 0.3 * np.sin(step * 0.1) + 0.1 * np.random.randn()
        }
        
        # Perform adaptation
        adaptations = adaptive_model.adapt_online(performance)
        
        if step % 20 == 0:
            summary = adaptive_model.get_adaptation_summary()
            print(f"Step {step}: Accuracy={performance['accuracy']:.3f}, "
                  f"Adaptation trend={summary['adaptation_trend']}")


class RobustAdaptiveSystem:
    """Robust adaptive system with comprehensive error handling and self-recovery."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.adaptation_mechanisms = {
            "threshold": AdaptiveThresholdController(config),
            "plasticity": AdaptiveSynapticPlasticity(config),
            "architecture": AdaptiveArchitectureController(config)
        }
        self.health_monitor = AdaptationHealthMonitor()
        self.recovery_manager = SelfRecoveryManager()
        self.logger = logging.getLogger(__name__)
        
    def robust_adapt(self, performance_metrics: Dict[str, float], 
                    model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform robust adaptation with error handling and recovery."""
        
        all_adaptations = {}
        
        for mechanism_name, mechanism in self.adaptation_mechanisms.items():
            try:
                # Monitor health before adaptation
                health_status = self.health_monitor.check_mechanism_health(mechanism_name, performance_metrics)
                
                if health_status["healthy"]:
                    adaptations = mechanism.adapt(performance_metrics, model_state)
                    all_adaptations[mechanism_name] = adaptations
                    
                    # Validate adaptations
                    if self._validate_adaptations(adaptations):
                        self.logger.info(f"✅ {mechanism_name} adaptation successful")
                    else:
                        self.logger.warning(f"⚠️ {mechanism_name} adaptation validation failed")
                        all_adaptations[mechanism_name] = self._get_safe_adaptations(mechanism_name)
                        
                else:
                    self.logger.warning(f"🏥 {mechanism_name} unhealthy, initiating recovery")
                    recovery_actions = self.recovery_manager.recover_mechanism(mechanism_name, health_status)
                    all_adaptations[mechanism_name] = recovery_actions
                    
            except Exception as e:
                self.logger.error(f"❌ Error in {mechanism_name} adaptation: {e}")
                # Graceful degradation
                all_adaptations[mechanism_name] = self._get_emergency_adaptations(mechanism_name)
                
        return all_adaptations
    
    def _validate_adaptations(self, adaptations: Dict[str, Any]) -> bool:
        """Validate that adaptations are within safe bounds."""
        for param_name, adaptation in adaptations.items():
            if isinstance(adaptation, dict) and "delta" in adaptation:
                delta = adaptation["delta"]
                if hasattr(delta, "item"):
                    delta = delta.item()
                if abs(delta) > 1.0:  # Safety threshold
                    return False
        return True
    
    def _get_safe_adaptations(self, mechanism_name: str) -> Dict[str, Any]:
        """Get conservative safe adaptations."""
        return {"status": "safe_mode", "delta": 0.0}
    
    def _get_emergency_adaptations(self, mechanism_name: str) -> Dict[str, Any]:
        """Get emergency fallback adaptations."""
        return {"status": "emergency_mode", "action": "reset_to_default"}


class AdaptationHealthMonitor:
    """Monitors health of adaptation mechanisms."""
    
    def __init__(self):
        self.health_history = {}
        
    def check_mechanism_health(self, mechanism_name: str, 
                             performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check health status of adaptation mechanism."""
        
        health_score = 1.0
        issues = []
        
        # Check for performance degradation
        current_accuracy = performance_metrics.get("accuracy", 0.8)
        if current_accuracy < 0.7:
            health_score *= 0.3
            issues.append("performance_degradation")
            
        # Check for instability
        if mechanism_name not in self.health_history:
            self.health_history[mechanism_name] = []
            
        self.health_history[mechanism_name].append(current_accuracy)
        
        if len(self.health_history[mechanism_name]) >= 5:
            recent_variance = np.var(self.health_history[mechanism_name][-5:])
            if recent_variance > 0.1:
                health_score *= 0.5
                issues.append("high_variance")
                
        return {
            "healthy": health_score > 0.5,
            "score": health_score,
            "issues": issues
        }


class SelfRecoveryManager:
    """Manages self-recovery of adaptation mechanisms."""
    
    def recover_mechanism(self, mechanism_name: str, health_status: Dict[str, Any]) -> Dict[str, Any]:
        """Implement recovery actions for unhealthy mechanisms."""
        
        recovery_actions = {
            "status": "recovering",
            "actions": []
        }
        
        for issue in health_status["issues"]:
            if issue == "performance_degradation":
                recovery_actions["actions"].append({
                    "type": "reset",
                    "method": "restore_checkpoint"
                })
            elif issue == "high_variance":
                recovery_actions["actions"].append({
                    "type": "stabilize",
                    "method": "reduce_adaptation_rate"
                })
                
        return recovery_actions


class CriticalPointDetector:
    """Detects critical points in neuromorphic system behavior."""
    
    def __init__(self, window_size: int = 100):
        self.performance_history = deque(maxlen=window_size)
        self.energy_history = deque(maxlen=window_size)
        self.spike_history = deque(maxlen=window_size)
        
    def detect_critical_transitions(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect phase transitions and critical points."""
        
        self.performance_history.append(metrics.get("accuracy", 0.8))
        self.energy_history.append(metrics.get("energy_mj", 50.0))
        self.spike_history.append(metrics.get("spike_rate", 0.1))
        
        critical_points = {
            "performance_cliff": False,
            "energy_spike": False,
            "neural_avalanche": False,
            "system_instability": False
        }
        
        if len(self.performance_history) >= 10:
            # Detect performance cliff
            recent_perf = list(self.performance_history)[-5:]
            if max(recent_perf) - min(recent_perf) > 0.2:
                critical_points["performance_cliff"] = True
                
            # Detect energy spike
            recent_energy = list(self.energy_history)[-5:]
            if len(self.energy_history) > 10:
                avg_energy = np.mean(list(self.energy_history)[:-5])
                if max(recent_energy) > avg_energy * 2:
                    critical_points["energy_spike"] = True
                
            # Detect neural avalanche (spike rate explosion)
            recent_spikes = list(self.spike_history)[-5:]
            if max(recent_spikes) > 0.8:  # Very high activity
                critical_points["neural_avalanche"] = True
                
        return critical_points