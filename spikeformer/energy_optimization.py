"""Real-time energy optimization for neuromorphic computing systems."""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import deque
import threading
import queue
from pathlib import Path
import json

from .neurons import create_neuron
from .models import SpikingTransformer
from .profiling import EnergyProfiler, PowerMonitor


@dataclass
class EnergyConstraint:
    """Energy constraint specification."""
    max_power_mw: Optional[float] = None
    max_energy_per_sample_mj: Optional[float] = None
    target_efficiency_samples_per_joule: Optional[float] = None
    thermal_limit_celsius: Optional[float] = None
    battery_level_percentage: Optional[float] = None


@dataclass
class OptimizationState:
    """Current state of the energy optimization system."""
    current_power_mw: float
    average_power_mw: float
    energy_per_sample_mj: float
    efficiency_samples_per_joule: float
    temperature_celsius: float
    battery_percentage: float
    constraint_violations: List[str]
    optimization_actions: List[str]


class EnergyOptimizer(ABC):
    """Abstract base class for energy optimization strategies."""
    
    @abstractmethod
    def optimize(self, model: nn.Module, constraints: EnergyConstraint, 
                current_state: OptimizationState) -> Dict[str, Any]:
        """Apply optimization strategy and return configuration changes."""
        pass


class AdaptiveThresholdOptimizer(EnergyOptimizer):
    """Optimize energy consumption by adapting neuron firing thresholds."""
    
    def __init__(self, adaptation_rate: float = 0.01, min_threshold: float = 0.1, 
                 max_threshold: float = 5.0):
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
    def optimize(self, model: nn.Module, constraints: EnergyConstraint, 
                current_state: OptimizationState) -> Dict[str, Any]:
        """Adapt thresholds based on energy constraints."""
        changes = {}
        
        # Calculate target adjustment
        power_violation = 0.0
        if constraints.max_power_mw:
            power_violation = max(0, current_state.current_power_mw - constraints.max_power_mw)
        
        energy_violation = 0.0
        if constraints.max_energy_per_sample_mj:
            energy_violation = max(0, current_state.energy_per_sample_mj - constraints.max_energy_per_sample_mj)
        
        # Adjust thresholds to reduce power consumption
        if power_violation > 0 or energy_violation > 0:
            # Increase thresholds to reduce spike activity
            threshold_increase = self.adaptation_rate * (power_violation / 1000 + energy_violation)
            
            for name, module in model.named_modules():
                if hasattr(module, 'threshold') and hasattr(module.threshold, 'data'):
                    old_threshold = module.threshold.data.clone()
                    new_threshold = torch.clamp(
                        old_threshold + threshold_increase,
                        self.min_threshold, self.max_threshold
                    )
                    module.threshold.data = new_threshold
                    changes[f"threshold_{name}"] = {
                        "old": float(old_threshold.mean()),
                        "new": float(new_threshold.mean())
                    }
        
        # Lower thresholds if we have energy headroom and poor performance
        elif (constraints.max_power_mw and current_state.current_power_mw < constraints.max_power_mw * 0.8):
            threshold_decrease = self.adaptation_rate * 0.5
            
            for name, module in model.named_modules():
                if hasattr(module, 'threshold') and hasattr(module.threshold, 'data'):
                    old_threshold = module.threshold.data.clone()
                    new_threshold = torch.clamp(
                        old_threshold - threshold_decrease,
                        self.min_threshold, self.max_threshold
                    )
                    module.threshold.data = new_threshold
                    changes[f"threshold_{name}"] = {
                        "old": float(old_threshold.mean()),
                        "new": float(new_threshold.mean())
                    }
        
        return changes


class SparsityOptimizer(EnergyOptimizer):
    """Optimize energy by controlling network sparsity."""
    
    def __init__(self, target_sparsity_range: Tuple[float, float] = (0.8, 0.95)):
        self.min_sparsity, self.max_sparsity = target_sparsity_range
        
    def optimize(self, model: nn.Module, constraints: EnergyConstraint,
                current_state: OptimizationState) -> Dict[str, Any]:
        """Adjust model sparsity for energy efficiency."""
        changes = {}
        
        # Estimate current sparsity from spike statistics
        current_sparsity = self._estimate_sparsity(model)
        
        # Adjust sparsity based on power constraints
        if constraints.max_power_mw and current_state.current_power_mw > constraints.max_power_mw:
            # Increase sparsity
            target_sparsity = min(self.max_sparsity, current_sparsity + 0.05)
            changes.update(self._adjust_sparsity(model, target_sparsity))
            
        elif constraints.max_power_mw and current_state.current_power_mw < constraints.max_power_mw * 0.7:
            # Decrease sparsity for better performance
            target_sparsity = max(self.min_sparsity, current_sparsity - 0.02)
            changes.update(self._adjust_sparsity(model, target_sparsity))
        
        return changes
    
    def _estimate_sparsity(self, model: nn.Module) -> float:
        """Estimate current network sparsity."""
        # This would analyze actual spike patterns
        # For now, return a placeholder
        return 0.85
    
    def _adjust_sparsity(self, model: nn.Module, target_sparsity: float) -> Dict[str, Any]:
        """Adjust model parameters to achieve target sparsity."""
        changes = {}
        
        # Adjust adaptive sparsity neurons
        for name, module in model.named_modules():
            if hasattr(module, 'target_sparsity'):
                old_target = module.target_sparsity
                module.target_sparsity = target_sparsity
                changes[f"sparsity_{name}"] = {
                    "old": float(old_target),
                    "new": float(target_sparsity)
                }
        
        return changes


class FrequencyScalingOptimizer(EnergyOptimizer):
    """Optimize energy through dynamic frequency scaling."""
    
    def __init__(self, min_timesteps: int = 16, max_timesteps: int = 64):
        self.min_timesteps = min_timesteps
        self.max_timesteps = max_timesteps
        
    def optimize(self, model: nn.Module, constraints: EnergyConstraint,
                current_state: OptimizationState) -> Dict[str, Any]:
        """Adjust temporal resolution based on energy constraints."""
        changes = {}
        
        # Get current timesteps
        current_timesteps = getattr(model, 'timesteps', 32)
        
        # Adjust timesteps based on power situation
        if constraints.max_power_mw and current_state.current_power_mw > constraints.max_power_mw:
            # Reduce temporal resolution
            new_timesteps = max(self.min_timesteps, current_timesteps - 4)
        elif (constraints.max_power_mw and 
              current_state.current_power_mw < constraints.max_power_mw * 0.6):
            # Increase temporal resolution for better accuracy
            new_timesteps = min(self.max_timesteps, current_timesteps + 2)
        else:
            new_timesteps = current_timesteps
            
        if new_timesteps != current_timesteps:
            if hasattr(model, 'timesteps'):
                model.timesteps = new_timesteps
                changes["timesteps"] = {
                    "old": current_timesteps,
                    "new": new_timesteps
                }
        
        return changes


class ThermalThrottlingOptimizer(EnergyOptimizer):
    """Optimize performance based on thermal constraints."""
    
    def __init__(self, throttle_temperature: float = 80.0, shutdown_temperature: float = 90.0):
        self.throttle_temperature = throttle_temperature
        self.shutdown_temperature = shutdown_temperature
        
    def optimize(self, model: nn.Module, constraints: EnergyConstraint,
                current_state: OptimizationState) -> Dict[str, Any]:
        """Apply thermal throttling optimizations."""
        changes = {}
        
        if current_state.temperature_celsius > self.shutdown_temperature:
            # Emergency shutdown protection
            changes["emergency_throttle"] = True
            changes["performance_reduction"] = 0.9  # Reduce to 10% performance
            
        elif current_state.temperature_celsius > self.throttle_temperature:
            # Gradual throttling
            throttle_factor = (current_state.temperature_celsius - self.throttle_temperature) / 20.0
            throttle_factor = min(throttle_factor, 0.8)  # Max 80% reduction
            
            changes["thermal_throttle"] = True
            changes["performance_reduction"] = throttle_factor
            
            # Increase thresholds to reduce activity
            threshold_increase = throttle_factor * 0.5
            for name, module in model.named_modules():
                if hasattr(module, 'threshold') and hasattr(module.threshold, 'data'):
                    old_threshold = module.threshold.data.clone()
                    new_threshold = old_threshold * (1 + threshold_increase)
                    module.threshold.data = new_threshold
                    changes[f"thermal_threshold_{name}"] = {
                        "old": float(old_threshold.mean()),
                        "new": float(new_threshold.mean())
                    }
        
        return changes


class BatteryAwareOptimizer(EnergyOptimizer):
    """Optimize based on battery level and power management."""
    
    def __init__(self, low_battery_threshold: float = 20.0, critical_battery_threshold: float = 5.0):
        self.low_battery_threshold = low_battery_threshold
        self.critical_battery_threshold = critical_battery_threshold
        
    def optimize(self, model: nn.Module, constraints: EnergyConstraint,
                current_state: OptimizationState) -> Dict[str, Any]:
        """Apply battery-aware optimizations."""
        changes = {}
        
        battery_level = current_state.battery_percentage
        
        if battery_level < self.critical_battery_threshold:
            # Critical battery - aggressive power saving
            changes["power_saving_mode"] = "critical"
            changes["performance_reduction"] = 0.8
            
            # Minimize timesteps
            if hasattr(model, 'timesteps'):
                model.timesteps = max(8, model.timesteps // 4)
                changes["timesteps_critical"] = model.timesteps
                
        elif battery_level < self.low_battery_threshold:
            # Low battery - moderate power saving
            changes["power_saving_mode"] = "low_battery"
            changes["performance_reduction"] = 0.4
            
            # Reduce temporal resolution
            if hasattr(model, 'timesteps'):
                model.timesteps = max(16, model.timesteps // 2)
                changes["timesteps_low"] = model.timesteps
        
        return changes


class RealTimeEnergyManager:
    """Real-time energy management system for neuromorphic computing."""
    
    def __init__(self, model: nn.Module, constraints: EnergyConstraint,
                 optimization_interval: float = 1.0, history_length: int = 100):
        self.model = model
        self.constraints = constraints
        self.optimization_interval = optimization_interval
        self.history_length = history_length
        
        # Initialize optimizers
        self.optimizers = [
            AdaptiveThresholdOptimizer(),
            SparsityOptimizer(),
            FrequencyScalingOptimizer(),
            ThermalThrottlingOptimizer(),
            BatteryAwareOptimizer(),
        ]
        
        # Energy monitoring
        self.energy_profiler = EnergyProfiler()
        self.power_monitor = PowerMonitor()
        
        # State tracking
        self.power_history = deque(maxlen=history_length)
        self.energy_history = deque(maxlen=history_length)
        self.optimization_history = deque(maxlen=history_length)
        
        # Control flags
        self.running = False
        self.optimization_thread = None
        self.monitoring_thread = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start real-time energy management."""
        if self.running:
            self.logger.warning("Energy manager already running")
            return
            
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(target=self._optimization_loop)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
        self.logger.info("Real-time energy management started")
    
    def stop(self):
        """Stop real-time energy management."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        if self.optimization_thread:
            self.optimization_thread.join(timeout=2.0)
            
        self.logger.info("Real-time energy management stopped")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.running:
            try:
                # Collect power metrics
                power_metrics = self.power_monitor.get_current_metrics()
                self.power_history.append({
                    "timestamp": time.time(),
                    "power_mw": power_metrics.get("power_mw", 0),
                    "temperature": power_metrics.get("temperature", 25),
                    "battery_percentage": power_metrics.get("battery_percentage", 100)
                })
                
                time.sleep(0.1)  # 10Hz monitoring
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _optimization_loop(self):
        """Continuous optimization loop."""
        while self.running:
            try:
                # Get current state
                current_state = self._get_current_state()
                
                # Check for constraint violations
                violations = self._check_constraints(current_state)
                
                if violations:
                    self.logger.info(f"Constraint violations detected: {violations}")
                    
                    # Apply optimizations
                    total_changes = {}
                    for optimizer in self.optimizers:
                        try:
                            changes = optimizer.optimize(self.model, self.constraints, current_state)
                            total_changes.update(changes)
                        except Exception as e:
                            self.logger.error(f"Optimizer {optimizer.__class__.__name__} failed: {e}")
                    
                    if total_changes:
                        self.optimization_history.append({
                            "timestamp": time.time(),
                            "state": current_state,
                            "changes": total_changes
                        })
                        self.logger.info(f"Applied optimizations: {list(total_changes.keys())}")
                
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                time.sleep(self.optimization_interval)
    
    def _get_current_state(self) -> OptimizationState:
        """Get current system state."""
        if not self.power_history:
            return OptimizationState(
                current_power_mw=0, average_power_mw=0, energy_per_sample_mj=0,
                efficiency_samples_per_joule=0, temperature_celsius=25,
                battery_percentage=100, constraint_violations=[], optimization_actions=[]
            )
        
        recent_power = [entry["power_mw"] for entry in list(self.power_history)[-10:]]
        recent_temp = [entry["temperature"] for entry in list(self.power_history)[-10:]]
        recent_battery = [entry["battery_percentage"] for entry in list(self.power_history)[-10:]]
        
        current_power = recent_power[-1] if recent_power else 0
        avg_power = np.mean(recent_power) if recent_power else 0
        
        # Estimate energy per sample (simplified)
        energy_per_sample = avg_power * 0.001  # Placeholder calculation
        efficiency = 1000.0 / max(energy_per_sample, 1e-6) if energy_per_sample > 0 else 0
        
        return OptimizationState(
            current_power_mw=current_power,
            average_power_mw=avg_power,
            energy_per_sample_mj=energy_per_sample,
            efficiency_samples_per_joule=efficiency,
            temperature_celsius=np.mean(recent_temp) if recent_temp else 25,
            battery_percentage=recent_battery[-1] if recent_battery else 100,
            constraint_violations=[],
            optimization_actions=[]
        )
    
    def _check_constraints(self, state: OptimizationState) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        if self.constraints.max_power_mw and state.current_power_mw > self.constraints.max_power_mw:
            violations.append(f"Power limit exceeded: {state.current_power_mw:.1f}mW > {self.constraints.max_power_mw}mW")
        
        if (self.constraints.max_energy_per_sample_mj and 
            state.energy_per_sample_mj > self.constraints.max_energy_per_sample_mj):
            violations.append(f"Energy per sample exceeded: {state.energy_per_sample_mj:.3f}mJ > {self.constraints.max_energy_per_sample_mj}mJ")
        
        if (self.constraints.target_efficiency_samples_per_joule and 
            state.efficiency_samples_per_joule < self.constraints.target_efficiency_samples_per_joule):
            violations.append(f"Efficiency below target: {state.efficiency_samples_per_joule:.1f} < {self.constraints.target_efficiency_samples_per_joule}")
        
        if (self.constraints.thermal_limit_celsius and 
            state.temperature_celsius > self.constraints.thermal_limit_celsius):
            violations.append(f"Temperature limit exceeded: {state.temperature_celsius:.1f}°C > {self.constraints.thermal_limit_celsius}°C")
        
        if (self.constraints.battery_level_percentage and 
            state.battery_percentage < self.constraints.battery_level_percentage):
            violations.append(f"Battery level low: {state.battery_percentage:.1f}% < {self.constraints.battery_level_percentage}%")
        
        return violations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current energy management statistics."""
        if not self.power_history:
            return {"error": "No data collected yet"}
        
        power_data = [entry["power_mw"] for entry in self.power_history]
        temp_data = [entry["temperature"] for entry in self.power_history]
        
        return {
            "monitoring": {
                "samples_collected": len(self.power_history),
                "monitoring_duration_minutes": (time.time() - self.power_history[0]["timestamp"]) / 60,
                "current_power_mw": power_data[-1] if power_data else 0,
                "average_power_mw": np.mean(power_data),
                "power_std_mw": np.std(power_data),
                "max_power_mw": np.max(power_data),
                "min_power_mw": np.min(power_data),
                "current_temperature": temp_data[-1] if temp_data else 25,
                "max_temperature": np.max(temp_data) if temp_data else 25,
            },
            "optimization": {
                "optimizations_applied": len(self.optimization_history),
                "recent_optimizations": list(self.optimization_history)[-5:] if self.optimization_history else [],
            },
            "constraints": {
                "max_power_mw": self.constraints.max_power_mw,
                "max_energy_per_sample_mj": self.constraints.max_energy_per_sample_mj,
                "thermal_limit_celsius": self.constraints.thermal_limit_celsius,
                "battery_threshold_percentage": self.constraints.battery_level_percentage,
            }
        }
    
    def save_log(self, filepath: Path):
        """Save monitoring and optimization log to file."""
        log_data = {
            "power_history": list(self.power_history),
            "optimization_history": list(self.optimization_history),
            "constraints": {
                "max_power_mw": self.constraints.max_power_mw,
                "max_energy_per_sample_mj": self.constraints.max_energy_per_sample_mj,
                "target_efficiency_samples_per_joule": self.constraints.target_efficiency_samples_per_joule,
                "thermal_limit_celsius": self.constraints.thermal_limit_celsius,
                "battery_level_percentage": self.constraints.battery_level_percentage,
            },
            "statistics": self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"Energy management log saved to {filepath}")


class EnergyAwareInference:
    """Energy-aware inference wrapper for neuromorphic models."""
    
    def __init__(self, model: nn.Module, constraints: EnergyConstraint,
                 enable_realtime_optimization: bool = True):
        self.model = model
        self.constraints = constraints
        self.enable_realtime = enable_realtime_optimization
        
        if enable_realtime_optimization:
            self.energy_manager = RealTimeEnergyManager(model, constraints)
            self.energy_manager.start()
        else:
            self.energy_manager = None
            
        self.inference_count = 0
        self.total_energy = 0.0
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Energy-aware forward pass."""
        if self.energy_manager:
            # Check if we need to throttle based on current state
            current_state = self.energy_manager._get_current_state()
            
            if current_state.temperature_celsius > 85:
                # Thermal throttling
                time.sleep(0.1)  # Add delay to reduce heat generation
        
        # Regular inference
        with torch.no_grad():
            output = self.model(x)
            
        self.inference_count += 1
        
        # Estimate energy consumption
        estimated_energy = self._estimate_energy_consumption()
        self.total_energy += estimated_energy
        
        return output
    
    def _estimate_energy_consumption(self) -> float:
        """Estimate energy consumption for the last inference."""
        # This would integrate with actual power monitoring
        # For now, return a simplified estimate
        return 1.0  # 1mJ per inference (placeholder)
    
    def get_energy_stats(self) -> Dict[str, float]:
        """Get energy consumption statistics."""
        avg_energy = self.total_energy / max(self.inference_count, 1)
        
        return {
            "total_inferences": self.inference_count,
            "total_energy_mj": self.total_energy,
            "average_energy_per_inference_mj": avg_energy,
            "efficiency_inferences_per_joule": 1000.0 / max(avg_energy, 1e-6)
        }
    
    def __del__(self):
        """Cleanup energy manager."""
        if self.energy_manager:
            self.energy_manager.stop()


# Example usage for research and deployment
def create_energy_optimized_model(base_model: nn.Module, 
                                power_budget_mw: float = 1000) -> EnergyAwareInference:
    """Create an energy-optimized version of a neuromorphic model."""
    
    constraints = EnergyConstraint(
        max_power_mw=power_budget_mw,
        max_energy_per_sample_mj=2.0,
        target_efficiency_samples_per_joule=500,
        thermal_limit_celsius=80.0,
        battery_level_percentage=10.0
    )
    
    return EnergyAwareInference(
        model=base_model,
        constraints=constraints,
        enable_realtime_optimization=True
    )


if __name__ == "__main__":
    # Example: Create energy-optimized spiking transformer
    from .models import SpikingTransformer
    
    model = SpikingTransformer(
        vocab_size=30522, hidden_size=768, num_layers=12,
        num_heads=12, intermediate_size=3072, timesteps=32
    )
    
    # Wrap with energy optimization
    energy_optimized_model = create_energy_optimized_model(model, power_budget_mw=500)
    
    print("Energy optimization system initialized")
    print("Real-time power monitoring and optimization active")