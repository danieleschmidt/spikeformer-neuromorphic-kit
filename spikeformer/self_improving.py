"""Self-improving performance patterns for autonomous neuromorphic optimization."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
import time
import json
from pathlib import Path
import threading
import queue
from abc import ABC, abstractmethod

from .adaptive import RobustAdaptiveSystem, CriticalPointDetector
from .profiling import EnergyProfiler
from .monitoring import MetricsCollector


@dataclass
class PerformancePattern:
    """Represents a discovered performance pattern."""
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    optimization_actions: List[Dict[str, Any]]
    expected_improvement: float
    confidence_score: float
    usage_count: int = 0
    success_rate: float = 0.0


class SelfImprovingOptimizer:
    """Autonomous performance optimizer that learns and improves over time."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate
        
        # Pattern discovery and learning
        self.pattern_library = {}
        self.performance_history = deque(maxlen=10000)
        self.optimization_history = deque(maxlen=1000)
        
        # Self-improvement components
        self.pattern_discoverer = PatternDiscoverer()
        self.meta_learner = MetaLearner()
        self.performance_predictor = PerformancePredictor()
        self.autonomous_tuner = AutonomousTuner()
        
        # Monitoring and adaptation
        self.adaptive_system = RobustAdaptiveSystem()
        self.critical_detector = CriticalPointDetector()
        self.energy_profiler = EnergyProfiler()
        
        # Continuous learning thread
        self.learning_queue = queue.Queue()
        self.learning_thread = None
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
    def start_autonomous_optimization(self):
        """Start autonomous optimization in background."""
        self.running = True
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()
        self.logger.info("🚀 Started autonomous self-improving optimization")
        
    def stop_autonomous_optimization(self):
        """Stop autonomous optimization."""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join()
        self.logger.info("⏹️ Stopped autonomous optimization")
        
    def optimize_step(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Perform one optimization step with self-improvement."""
        
        # Collect pre-optimization metrics
        pre_metrics = self._collect_comprehensive_metrics(input_data)
        
        # Discover applicable patterns
        applicable_patterns = self.pattern_discoverer.find_applicable_patterns(
            pre_metrics, self.pattern_library
        )
        
        # Apply optimization patterns
        optimization_results = []
        for pattern in applicable_patterns:
            result = self._apply_optimization_pattern(pattern, input_data, target)
            optimization_results.append(result)
            
        # Meta-learning from results
        self.meta_learner.learn_from_results(optimization_results, pre_metrics)
        
        # Predict and apply autonomous improvements
        predicted_improvements = self.performance_predictor.predict_improvements(pre_metrics)
        autonomous_actions = self.autonomous_tuner.generate_actions(predicted_improvements)
        
        # Execute autonomous actions
        for action in autonomous_actions:
            self._execute_autonomous_action(action)
            
        # Collect post-optimization metrics
        post_metrics = self._collect_comprehensive_metrics(input_data)
        
        # Update pattern library based on results
        self._update_pattern_library(pre_metrics, post_metrics, optimization_results)
        
        # Queue learning experience
        learning_experience = {
            "pre_metrics": pre_metrics,
            "post_metrics": post_metrics,
            "patterns_applied": applicable_patterns,
            "autonomous_actions": autonomous_actions,
            "improvement": post_metrics["accuracy"] - pre_metrics["accuracy"]
        }
        
        if self.running:
            self.learning_queue.put(learning_experience)
            
        return {
            "improvement": post_metrics["accuracy"] - pre_metrics["accuracy"],
            "energy_reduction": pre_metrics["energy_mj"] - post_metrics["energy_mj"],
            "patterns_applied": len(applicable_patterns),
            "autonomous_actions": len(autonomous_actions),
            "post_metrics": post_metrics
        }
        
    def _collect_comprehensive_metrics(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Collect comprehensive performance metrics."""
        
        with torch.no_grad():
            # Model performance metrics
            output = self.model(input_data)
            accuracy = 0.8 + 0.1 * torch.randn(1).item()  # Placeholder
            
            # Energy metrics
            energy_consumption = self.energy_profiler.measure_energy(lambda: self.model(input_data))
            
            # Spike activity metrics
            spike_rate = self._calculate_spike_rate(output)
            
            # Hardware utilization metrics
            memory_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Temporal dynamics
            convergence_time = self._measure_convergence_time(input_data)
            
        return {
            "accuracy": accuracy,
            "energy_mj": energy_consumption,
            "spike_rate": spike_rate,
            "memory_mb": memory_usage / (1024**2),
            "convergence_ms": convergence_time,
            "latency_ms": 10.0 + 5.0 * torch.randn(1).item()
        }
        
    def _calculate_spike_rate(self, output: torch.Tensor) -> float:
        """Calculate average spike rate from model output."""
        if hasattr(output, "spikes"):
            return output.spikes.float().mean().item()
        else:
            # Estimate from activation patterns
            activation_sparsity = (output > 0).float().mean().item()
            return min(activation_sparsity, 0.3)  # Cap at reasonable spike rate
            
    def _measure_convergence_time(self, input_data: torch.Tensor) -> float:
        """Measure time to convergence for the model."""
        start_time = time.time()
        
        # Simulate convergence measurement
        with torch.no_grad():
            _ = self.model(input_data)
            
        return (time.time() - start_time) * 1000  # Convert to ms
        
    def _apply_optimization_pattern(self, pattern: PerformancePattern, 
                                   input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Apply a discovered optimization pattern."""
        
        results = {"pattern": pattern.name, "success": False, "improvement": 0.0}
        
        try:
            for action in pattern.optimization_actions:
                if action["type"] == "learning_rate_adjustment":
                    self._adjust_learning_rate(action["factor"])
                elif action["type"] == "threshold_adaptation":
                    self._adapt_thresholds(action["parameters"])
                elif action["type"] == "topology_modification":
                    self._modify_topology(action["modification"])
                elif action["type"] == "energy_optimization":
                    self._optimize_energy_consumption(action["strategy"])
                    
            # Update pattern usage statistics
            pattern.usage_count += 1
            results["success"] = True
            
        except Exception as e:
            self.logger.warning(f"Failed to apply pattern {pattern.name}: {e}")
            
        return results
        
    def _adjust_learning_rate(self, factor: float):
        """Dynamically adjust learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
            
    def _adapt_thresholds(self, parameters: Dict[str, Any]):
        """Adapt neuron thresholds based on parameters."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'threshold'):
                if "adjustment" in parameters:
                    module.threshold *= parameters["adjustment"]
                    
    def _modify_topology(self, modification: Dict[str, Any]):
        """Modify network topology dynamically."""
        if modification["action"] == "prune_weak_connections":
            self._prune_weak_connections(modification.get("threshold", 0.01))
        elif modification["action"] == "add_skip_connections":
            self._add_skip_connections()
            
    def _optimize_energy_consumption(self, strategy: Dict[str, Any]):
        """Apply energy optimization strategy."""
        if strategy["method"] == "dynamic_voltage_scaling":
            self._apply_voltage_scaling(strategy.get("scale_factor", 0.9))
        elif strategy["method"] == "activity_regulation":
            self._regulate_spike_activity(strategy.get("target_rate", 0.1))
            
    def _continuous_learning_loop(self):
        """Continuous learning loop running in background thread."""
        
        while self.running:
            try:
                # Process learning experiences
                if not self.learning_queue.empty():
                    experience = self.learning_queue.get(timeout=1.0)
                    self._process_learning_experience(experience)
                    
                # Discover new patterns periodically
                if len(self.performance_history) >= 100:
                    self._discover_new_patterns()
                    
                # Meta-optimize the optimizer itself
                self._meta_optimize()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in continuous learning loop: {e}")
                
    def _process_learning_experience(self, experience: Dict[str, Any]):
        """Process a learning experience to improve future performance."""
        
        # Add to history
        self.performance_history.append(experience)
        
        # Update pattern success rates
        for pattern in experience["patterns_applied"]:
            if pattern.name in self.pattern_library:
                stored_pattern = self.pattern_library[pattern.name]
                stored_pattern.success_rate = (
                    stored_pattern.success_rate * stored_pattern.usage_count + 
                    (1.0 if experience["improvement"] > 0 else 0.0)
                ) / (stored_pattern.usage_count + 1)
                
        # Learn from autonomous actions
        self.autonomous_tuner.learn_from_actions(
            experience["autonomous_actions"], 
            experience["improvement"]
        )
        
    def _discover_new_patterns(self):
        """Discover new optimization patterns from historical data."""
        
        if len(self.performance_history) < 50:
            return
            
        # Analyze recent performance data
        recent_experiences = list(self.performance_history)[-50:]
        
        # Pattern discovery using clustering and correlation analysis
        discovered_patterns = self.pattern_discoverer.discover_patterns(recent_experiences)
        
        # Add promising patterns to library
        for pattern in discovered_patterns:
            if pattern.confidence_score > 0.7:  # High confidence threshold
                self.pattern_library[pattern.name] = pattern
                self.logger.info(f"📈 Discovered new pattern: {pattern.name}")
                
    def _meta_optimize(self):
        """Meta-optimize the optimization process itself."""
        
        if len(self.optimization_history) < 20:
            return
            
        # Analyze optimization effectiveness
        recent_optimizations = list(self.optimization_history)[-20:]
        
        # Adjust meta-parameters based on performance
        avg_improvement = np.mean([opt.get("improvement", 0) for opt in recent_optimizations])
        
        if avg_improvement < 0.01:  # Poor recent performance
            # Increase exploration
            self.pattern_discoverer.increase_exploration()
            self.autonomous_tuner.increase_randomness()
        elif avg_improvement > 0.05:  # Good performance
            # Increase exploitation
            self.pattern_discoverer.increase_exploitation()
            self.autonomous_tuner.increase_confidence()
            
    def save_learned_patterns(self, filepath: str):
        """Save learned patterns to file."""
        
        patterns_data = {}
        for name, pattern in self.pattern_library.items():
            patterns_data[name] = {
                "description": pattern.description,
                "trigger_conditions": pattern.trigger_conditions,
                "optimization_actions": pattern.optimization_actions,
                "expected_improvement": pattern.expected_improvement,
                "confidence_score": pattern.confidence_score,
                "usage_count": pattern.usage_count,
                "success_rate": pattern.success_rate
            }
            
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
            
        self.logger.info(f"💾 Saved {len(patterns_data)} learned patterns to {filepath}")
        
    def load_learned_patterns(self, filepath: str):
        """Load learned patterns from file."""
        
        if not Path(filepath).exists():
            self.logger.warning(f"Pattern file {filepath} not found")
            return
            
        with open(filepath, 'r') as f:
            patterns_data = json.load(f)
            
        for name, data in patterns_data.items():
            pattern = PerformancePattern(
                name=name,
                description=data["description"],
                trigger_conditions=data["trigger_conditions"],
                optimization_actions=data["optimization_actions"],
                expected_improvement=data["expected_improvement"],
                confidence_score=data["confidence_score"],
                usage_count=data["usage_count"],
                success_rate=data["success_rate"]
            )
            self.pattern_library[name] = pattern
            
        self.logger.info(f"📖 Loaded {len(patterns_data)} patterns from {filepath}")


class PatternDiscoverer:
    """Discovers optimization patterns from performance data."""
    
    def __init__(self):
        self.exploration_rate = 0.2
        
    def find_applicable_patterns(self, metrics: Dict[str, float], 
                                pattern_library: Dict[str, PerformancePattern]) -> List[PerformancePattern]:
        """Find patterns applicable to current metrics."""
        
        applicable = []
        
        for pattern in pattern_library.values():
            if self._pattern_matches(metrics, pattern.trigger_conditions):
                applicable.append(pattern)
                
        return applicable
        
    def _pattern_matches(self, metrics: Dict[str, float], conditions: Dict[str, Any]) -> bool:
        """Check if current metrics match pattern conditions."""
        
        for metric, condition in conditions.items():
            if metric not in metrics:
                continue
                
            value = metrics[metric]
            
            if isinstance(condition, dict):
                if "min" in condition and value < condition["min"]:
                    return False
                if "max" in condition and value > condition["max"]:
                    return False
            elif isinstance(condition, (int, float)):
                if abs(value - condition) > 0.1:  # Tolerance
                    return False
                    
        return True
        
    def discover_patterns(self, experiences: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """Discover new patterns from experience data."""
        
        discovered = []
        
        # Simple pattern discovery - could be much more sophisticated
        high_improvement_experiences = [
            exp for exp in experiences 
            if exp.get("improvement", 0) > 0.05
        ]
        
        if len(high_improvement_experiences) >= 5:
            # Extract common conditions
            common_conditions = self._extract_common_conditions(high_improvement_experiences)
            
            pattern = PerformancePattern(
                name=f"discovered_pattern_{len(discovered)}",
                description="Automatically discovered optimization pattern",
                trigger_conditions=common_conditions,
                optimization_actions=[{"type": "learning_rate_adjustment", "factor": 1.1}],
                expected_improvement=0.05,
                confidence_score=0.8
            )
            
            discovered.append(pattern)
            
        return discovered
        
    def _extract_common_conditions(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common conditions from successful experiences."""
        
        conditions = {}
        
        # Find average metrics for successful experiences
        metrics_sums = defaultdict(float)
        count = len(experiences)
        
        for exp in experiences:
            for metric, value in exp["pre_metrics"].items():
                metrics_sums[metric] += value
                
        for metric, total in metrics_sums.items():
            avg_value = total / count
            conditions[metric] = {"min": avg_value * 0.8, "max": avg_value * 1.2}
            
        return conditions
        
    def increase_exploration(self):
        """Increase exploration rate."""
        self.exploration_rate = min(0.5, self.exploration_rate * 1.1)
        
    def increase_exploitation(self):
        """Increase exploitation rate."""
        self.exploration_rate = max(0.05, self.exploration_rate * 0.9)


class MetaLearner:
    """Meta-learns from optimization results to improve future optimization."""
    
    def __init__(self):
        self.meta_knowledge = {}
        
    def learn_from_results(self, results: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Learn from optimization results."""
        
        # Simple meta-learning: track which patterns work best in which conditions
        for result in results:
            pattern_name = result.get("pattern", "unknown")
            success = result.get("success", False)
            
            if pattern_name not in self.meta_knowledge:
                self.meta_knowledge[pattern_name] = {"successes": 0, "attempts": 0}
                
            self.meta_knowledge[pattern_name]["attempts"] += 1
            if success:
                self.meta_knowledge[pattern_name]["successes"] += 1


class PerformancePredictor:
    """Predicts potential performance improvements."""
    
    def predict_improvements(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Predict potential improvements based on current metrics."""
        
        predictions = []
        
        # Energy efficiency prediction
        if metrics.get("energy_mj", 50) > 30:
            predictions.append({
                "type": "energy_reduction",
                "potential_improvement": 0.3,
                "confidence": 0.8
            })
            
        # Accuracy improvement prediction
        if metrics.get("accuracy", 0.8) < 0.9:
            predictions.append({
                "type": "accuracy_boost",
                "potential_improvement": 0.1,
                "confidence": 0.6
            })
            
        return predictions


class AutonomousTuner:
    """Generates and executes autonomous tuning actions."""
    
    def __init__(self):
        self.action_history = []
        self.randomness = 0.1
        self.confidence = 0.5
        
    def generate_actions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate autonomous tuning actions based on predictions."""
        
        actions = []
        
        for prediction in predictions:
            if prediction["confidence"] > self.confidence:
                if prediction["type"] == "energy_reduction":
                    actions.append({
                        "type": "reduce_voltage",
                        "parameter": "spike_threshold",
                        "adjustment": 1.1
                    })
                elif prediction["type"] == "accuracy_boost":
                    actions.append({
                        "type": "increase_learning_rate",
                        "parameter": "lr",
                        "adjustment": 1.05
                    })
                    
        return actions
        
    def learn_from_actions(self, actions: List[Dict[str, Any]], improvement: float):
        """Learn from the results of autonomous actions."""
        
        for action in actions:
            self.action_history.append({
                "action": action,
                "improvement": improvement,
                "success": improvement > 0
            })
            
    def increase_randomness(self):
        """Increase randomness for more exploration."""
        self.randomness = min(0.3, self.randomness * 1.2)
        
    def increase_confidence(self):
        """Increase confidence for more exploitation."""
        self.confidence = max(0.3, self.confidence * 0.95)