"""Multi-chip deployment and scaling for neuromorphic systems."""

import torch
import torch.nn as nn
import torch.distributed as dist
import asyncio
import threading
import time
import queue
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import numpy as np
from enum import Enum

from .models import SpikingTransformer, SpikingViT
from .hardware import NeuromorphicBackend, Loihi2Backend, SpiNNakerBackend
from .monitoring import metrics, PerformanceMetrics
import torch.nn.functional as F


class ScalingStrategy(Enum):
    """Scaling strategies for neuromorphic deployment."""
    HORIZONTAL = "horizontal"  # Add more chips
    VERTICAL = "vertical"     # Use more cores per chip
    HYBRID = "hybrid"         # Mix of both
    AUTO = "auto"            # Automatic scaling


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    LATENCY_AWARE = "latency_aware"
    ENERGY_AWARE = "energy_aware"


@dataclass
class ChipResource:
    """Resource information for a neuromorphic chip."""
    chip_id: str
    chip_type: str  # "loihi2", "spinnaker", "edge"
    total_cores: int
    available_cores: int
    total_memory_mb: int
    available_memory_mb: int
    energy_efficiency: float  # Operations per joule
    latency_ms: float
    utilization: float = 0.0
    status: str = "available"  # "available", "busy", "offline"
    current_load: int = 0
    max_load: int = 100
    
    @property
    def load_percentage(self) -> float:
        """Get current load as percentage."""
        return (self.current_load / self.max_load) * 100 if self.max_load > 0 else 0
    
    @property
    def available_capacity(self) -> float:
        """Get available capacity as percentage."""
        return 100 - self.load_percentage


@dataclass
class ScalingDecision:
    """Decision made by the auto-scaler."""
    action: str  # "scale_up", "scale_down", "rebalance", "no_action"
    target_chips: List[str]
    reason: str
    estimated_improvement: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class DeploymentConfig:
    """Configuration for multi-chip deployment."""
    model_name: str
    target_chips: List[str]
    scaling_strategy: ScalingStrategy = ScalingStrategy.AUTO
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ENERGY_AWARE
    min_chips: int = 1
    max_chips: int = 8
    target_latency_ms: float = 50.0
    target_energy_budget_mj: float = 1.0
    failover_enabled: bool = True
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 80.0  # CPU utilization %
    scale_down_threshold: float = 30.0


class ChipManager:
    """Manage multiple neuromorphic chips."""
    
    def __init__(self):
        self.chips: Dict[str, ChipResource] = {}
        self.backends: Dict[str, NeuromorphicBackend] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def register_chip(self, chip_resource: ChipResource, backend: Optional[NeuromorphicBackend] = None):
        """Register a neuromorphic chip."""
        with self.lock:
            self.chips[chip_resource.chip_id] = chip_resource
            if backend:
                self.backends[chip_resource.chip_id] = backend
            
        self.logger.info(f"Registered chip {chip_resource.chip_id} ({chip_resource.chip_type})")
    
    def unregister_chip(self, chip_id: str):
        """Unregister a chip."""
        with self.lock:
            if chip_id in self.chips:
                del self.chips[chip_id]
            if chip_id in self.backends:
                del self.backends[chip_id]
        
        self.logger.info(f"Unregistered chip {chip_id}")
    
    def get_available_chips(self) -> List[ChipResource]:
        """Get list of available chips."""
        with self.lock:
            return [chip for chip in self.chips.values() if chip.status == "available"]
    
    def get_chip_by_type(self, chip_type: str) -> List[ChipResource]:
        """Get chips of specific type."""
        with self.lock:
            return [chip for chip in self.chips.values() if chip.chip_type == chip_type]
    
    def update_chip_status(self, chip_id: str, status: str, utilization: float = None):
        """Update chip status and utilization."""
        with self.lock:
            if chip_id in self.chips:
                self.chips[chip_id].status = status
                if utilization is not None:
                    self.chips[chip_id].utilization = utilization
    
    def get_least_loaded_chip(self, chip_type: Optional[str] = None) -> Optional[ChipResource]:
        """Get the least loaded available chip."""
        available_chips = self.get_available_chips()
        
        if chip_type:
            available_chips = [c for c in available_chips if c.chip_type == chip_type]
        
        if not available_chips:
            return None
        
        return min(available_chips, key=lambda c: c.load_percentage)
    
    def get_most_efficient_chip(self) -> Optional[ChipResource]:
        """Get the most energy-efficient available chip."""
        available_chips = self.get_available_chips()
        
        if not available_chips:
            return None
        
        return max(available_chips, key=lambda c: c.energy_efficiency)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get overall cluster statistics."""
        with self.lock:
            if not self.chips:
                return {"message": "No chips registered"}
            
            total_chips = len(self.chips)
            available_chips = len([c for c in self.chips.values() if c.status == "available"])
            total_cores = sum(c.total_cores for c in self.chips.values())
            available_cores = sum(c.available_cores for c in self.chips.values())
            avg_utilization = np.mean([c.utilization for c in self.chips.values()])
            
            # Group by chip type
            by_type = defaultdict(int)
            for chip in self.chips.values():
                by_type[chip.chip_type] += 1
            
            return {
                "total_chips": total_chips,
                "available_chips": available_chips,
                "total_cores": total_cores,
                "available_cores": available_cores,
                "core_utilization": ((total_cores - available_cores) / total_cores * 100) if total_cores > 0 else 0,
                "avg_chip_utilization": avg_utilization,
                "chips_by_type": dict(by_type),
                "cluster_health": "healthy" if available_chips > 0 else "degraded"
            }


class LoadBalancer:
    """Load balancer for distributing work across chips."""
    
    def __init__(self, chip_manager: ChipManager, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ENERGY_AWARE):
        self.chip_manager = chip_manager
        self.strategy = strategy
        self.request_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
    
    def select_chip(self, workload_info: Dict[str, Any] = None) -> Optional[ChipResource]:
        """Select the best chip for a workload based on strategy."""
        workload_info = workload_info or {}
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection()
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self.chip_manager.get_least_loaded_chip()
        elif self.strategy == LoadBalancingStrategy.ENERGY_AWARE:
            return self._energy_aware_selection(workload_info)
        elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
            return self._latency_aware_selection(workload_info)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_selection(workload_info)
        else:
            return self.chip_manager.get_least_loaded_chip()
    
    def _round_robin_selection(self) -> Optional[ChipResource]:
        """Simple round-robin selection."""
        available_chips = self.chip_manager.get_available_chips()
        if not available_chips:
            return None
        
        # Use request count to determine next chip
        next_index = len(self.request_history) % len(available_chips)
        return available_chips[next_index]
    
    def _energy_aware_selection(self, workload_info: Dict[str, Any]) -> Optional[ChipResource]:
        """Select chip based on energy efficiency."""
        available_chips = self.chip_manager.get_available_chips()
        if not available_chips:
            return None
        
        # Score chips based on energy efficiency and current load
        def energy_score(chip: ChipResource) -> float:
            efficiency_score = chip.energy_efficiency
            load_penalty = chip.load_percentage / 100.0
            return efficiency_score * (1 - load_penalty)
        
        return max(available_chips, key=energy_score)
    
    def _latency_aware_selection(self, workload_info: Dict[str, Any]) -> Optional[ChipResource]:
        """Select chip based on expected latency."""
        available_chips = self.chip_manager.get_available_chips()
        if not available_chips:
            return None
        
        # Estimate latency based on current load and chip characteristics
        def latency_score(chip: ChipResource) -> float:
            base_latency = chip.latency_ms
            load_factor = 1 + (chip.load_percentage / 100.0)
            estimated_latency = base_latency * load_factor
            return 1.0 / estimated_latency  # Higher score for lower latency
        
        return max(available_chips, key=latency_score)
    
    def _weighted_selection(self, workload_info: Dict[str, Any]) -> Optional[ChipResource]:
        """Weighted selection based on multiple factors."""
        available_chips = self.chip_manager.get_available_chips()
        if not available_chips:
            return None
        
        def weighted_score(chip: ChipResource) -> float:
            # Normalize and weight different factors
            efficiency_weight = 0.4
            latency_weight = 0.3
            availability_weight = 0.3
            
            efficiency_score = chip.energy_efficiency / 1000.0  # Normalize
            latency_score = 1.0 / (chip.latency_ms + 1)  # Inverse latency
            availability_score = chip.available_capacity / 100.0
            
            return (efficiency_weight * efficiency_score + 
                   latency_weight * latency_score +
                   availability_weight * availability_score)
        
        return max(available_chips, key=weighted_score)
    
    def record_request(self, chip_id: str, workload_info: Dict[str, Any], response_time_ms: float):
        """Record a completed request for learning."""
        request_record = {
            'chip_id': chip_id,
            'workload_info': workload_info,
            'response_time_ms': response_time_ms,
            'timestamp': time.time()
        }
        self.request_history.append(request_record)


class AutoScaler:
    """Automatic scaling controller for neuromorphic clusters."""
    
    def __init__(self, chip_manager: ChipManager, config: DeploymentConfig):
        self.chip_manager = chip_manager
        self.config = config
        self.scaling_history = deque(maxlen=100)
        self.metrics_window = deque(maxlen=60)  # 1 minute window
        self.logger = logging.getLogger(__name__)
        
        # Scaling state
        self.last_scaling_action = 0
        self.cooldown_period = 60  # seconds
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start automatic scaling monitoring."""
        self.monitoring_active = True
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic scaling monitoring."""
        self.monitoring_active = False
        self.logger.info("Auto-scaling monitoring stopped")
    
    def evaluate_scaling_decision(self) -> ScalingDecision:
        """Evaluate whether scaling action is needed."""
        if not self.config.auto_scaling_enabled:
            return ScalingDecision("no_action", [], "Auto-scaling disabled", 0.0)
        
        # Check cooldown period
        if time.time() - self.last_scaling_action < self.cooldown_period:
            return ScalingDecision("no_action", [], "In cooldown period", 0.0)
        
        cluster_stats = self.chip_manager.get_cluster_stats()
        available_chips = self.chip_manager.get_available_chips()
        
        if not available_chips:
            return ScalingDecision("no_action", [], "No available chips", 0.0)
        
        avg_utilization = cluster_stats.get("avg_chip_utilization", 0)
        active_chips = len(available_chips)
        
        # Scale up decision
        if (avg_utilization > self.config.scale_up_threshold and 
            active_chips < self.config.max_chips):
            
            # Find chips that can be activated
            all_chips = list(self.chip_manager.chips.values())
            inactive_chips = [c for c in all_chips if c.status != "available"]
            
            if inactive_chips:
                target_chips = [inactive_chips[0].chip_id]  # Scale up by one chip
                improvement = self._estimate_scale_up_improvement(target_chips)
                
                return ScalingDecision(
                    "scale_up", 
                    target_chips,
                    f"High utilization ({avg_utilization:.1f}%)",
                    improvement
                )
        
        # Scale down decision
        elif (avg_utilization < self.config.scale_down_threshold and 
              active_chips > self.config.min_chips):
            
            # Find least utilized chip to remove
            least_utilized = min(available_chips, key=lambda c: c.utilization)
            target_chips = [least_utilized.chip_id]
            improvement = self._estimate_scale_down_improvement(target_chips)
            
            return ScalingDecision(
                "scale_down",
                target_chips,
                f"Low utilization ({avg_utilization:.1f}%)",
                improvement
            )
        
        # Rebalancing decision
        elif self._needs_rebalancing(available_chips):
            target_chips = [c.chip_id for c in available_chips]
            
            return ScalingDecision(
                "rebalance",
                target_chips,
                "Load imbalance detected",
                0.1  # Small improvement expected
            )
        
        return ScalingDecision("no_action", [], "No scaling needed", 0.0)
    
    def execute_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision."""
        try:
            if decision.action == "scale_up":
                return self._scale_up(decision.target_chips)
            elif decision.action == "scale_down":
                return self._scale_down(decision.target_chips)
            elif decision.action == "rebalance":
                return self._rebalance_load(decision.target_chips)
            else:
                return True  # No action needed
                
        except Exception as e:
            self.logger.error(f"Failed to execute scaling decision: {e}")
            return False
        finally:
            self.last_scaling_action = time.time()
            self.scaling_history.append(decision)
    
    def _scale_up(self, chip_ids: List[str]) -> bool:
        """Scale up by activating chips."""
        for chip_id in chip_ids:
            if chip_id in self.chip_manager.chips:
                self.chip_manager.update_chip_status(chip_id, "available")
                self.logger.info(f"Scaled up: activated chip {chip_id}")
        return True
    
    def _scale_down(self, chip_ids: List[str]) -> bool:
        """Scale down by deactivating chips."""
        for chip_id in chip_ids:
            if chip_id in self.chip_manager.chips:
                # In a real implementation, would gracefully drain workloads first
                self.chip_manager.update_chip_status(chip_id, "offline")
                self.logger.info(f"Scaled down: deactivated chip {chip_id}")
        return True
    
    def _rebalance_load(self, chip_ids: List[str]) -> bool:
        """Rebalance load across chips."""
        # In a real implementation, would redistribute workloads
        self.logger.info(f"Rebalanced load across {len(chip_ids)} chips")
        return True
    
    def _estimate_scale_up_improvement(self, chip_ids: List[str]) -> float:
        """Estimate improvement from scaling up."""
        # Simplified estimation: assume linear improvement
        current_chips = len(self.chip_manager.get_available_chips())
        new_chips = len(chip_ids)
        return (new_chips / (current_chips + new_chips)) * 100
    
    def _estimate_scale_down_improvement(self, chip_ids: List[str]) -> float:
        """Estimate improvement from scaling down (energy savings)."""
        # Simplified estimation: energy savings from unused chips
        return len(chip_ids) * 10  # 10% improvement per chip removed
    
    def _needs_rebalancing(self, chips: List[ChipResource]) -> bool:
        """Check if load rebalancing is needed."""
        if len(chips) < 2:
            return False
        
        utilizations = [c.utilization for c in chips]
        std_dev = np.std(utilizations)
        return std_dev > 20  # Rebalance if standard deviation > 20%


class DistributedInferenceEngine:
    """Distributed inference engine for multi-chip deployment."""
    
    def __init__(self, chip_manager: ChipManager, load_balancer: LoadBalancer):
        self.chip_manager = chip_manager
        self.load_balancer = load_balancer
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.request_queue = queue.Queue()
        self.result_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        
    def deploy_model(self, model: nn.Module, model_name: str, config: DeploymentConfig) -> bool:
        """Deploy a model across multiple chips."""
        target_chips = config.target_chips or [c.chip_id for c in self.chip_manager.get_available_chips()]
        
        if not target_chips:
            self.logger.error("No chips available for deployment")
            return False
        
        deployed_count = 0
        for chip_id in target_chips:
            if chip_id in self.chip_manager.backends:
                try:
                    backend = self.chip_manager.backends[chip_id]
                    success = backend.deploy_model(model, f"{model_name}_{chip_id}")
                    if success:
                        deployed_count += 1
                        self.logger.info(f"Deployed {model_name} to chip {chip_id}")
                    else:
                        self.logger.warning(f"Failed to deploy {model_name} to chip {chip_id}")
                except Exception as e:
                    self.logger.error(f"Error deploying to chip {chip_id}: {e}")
        
        success = deployed_count > 0
        if success:
            self.logger.info(f"Successfully deployed {model_name} to {deployed_count}/{len(target_chips)} chips")
        
        return success
    
    async def distributed_inference(self, inputs: torch.Tensor, model_name: str, 
                                  config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
        """Perform distributed inference across multiple chips."""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Select chip based on load balancing strategy
            selected_chip = self.load_balancer.select_chip({
                'model_name': model_name,
                'input_shape': list(inputs.shape),
                'timestamp': start_time
            })
            
            if not selected_chip:
                raise RuntimeError("No available chips for inference")
            
            # Execute inference on selected chip
            result = await self._execute_inference_on_chip(
                inputs, model_name, selected_chip.chip_id
            )
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Record metrics
            self.load_balancer.record_request(
                selected_chip.chip_id,
                {'model_name': model_name},
                response_time
            )
            
            # Update performance tracking
            self.successful_requests += 1
            self.avg_response_time = (
                (self.avg_response_time * (self.successful_requests - 1) + response_time) / 
                self.successful_requests
            )
            
            # Record performance metrics
            perf_metrics = PerformanceMetrics(
                inference_time_ms=response_time,
                model_name=model_name,
                hardware_type=selected_chip.chip_type,
                timestamp=time.time()
            )
            
            if 'accuracy' in result:
                perf_metrics.accuracy = result['accuracy']
            if 'energy_mj' in result:
                perf_metrics.energy_consumption_mj = result['energy_mj']
            
            metrics.record_performance_metrics(perf_metrics)
            
            return {
                'result': result,
                'chip_id': selected_chip.chip_id,
                'response_time_ms': response_time,
                'success': True
            }
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Distributed inference failed: {e}")
            
            return {
                'error': str(e),
                'success': False,
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    async def _execute_inference_on_chip(self, inputs: torch.Tensor, model_name: str, 
                                       chip_id: str) -> Dict[str, Any]:
        """Execute inference on a specific chip."""
        if chip_id not in self.chip_manager.backends:
            raise RuntimeError(f"No backend available for chip {chip_id}")
        
        backend = self.chip_manager.backends[chip_id]
        
        # Update chip status
        self.chip_manager.update_chip_status(chip_id, "busy")
        
        try:
            # Execute inference
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                backend.run_inference,
                f"{model_name}_{chip_id}",
                inputs
            )
            
            return result
            
        finally:
            # Reset chip status
            self.chip_manager.update_chip_status(chip_id, "available")
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_percent': success_rate,
            'avg_response_time_ms': self.avg_response_time,
            'active_chips': len(self.chip_manager.get_available_chips()),
            'cluster_stats': self.chip_manager.get_cluster_stats()
        }


class NeuromorphicCluster:
    """High-level interface for managing neuromorphic clusters."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig("default_model", [])
        self.chip_manager = ChipManager()
        self.load_balancer = LoadBalancer(self.chip_manager, self.config.load_balancing)
        self.auto_scaler = AutoScaler(self.chip_manager, self.config)
        self.inference_engine = DistributedInferenceEngine(self.chip_manager, self.load_balancer)
        self.logger = logging.getLogger(__name__)
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def add_chip(self, chip_type: str, chip_id: str, cores: int = 128, 
                memory_mb: int = 1024, backend: Optional[NeuromorphicBackend] = None):
        """Add a chip to the cluster."""
        # Create chip resource with hardware-specific defaults
        energy_efficiency = {
            "loihi2": 1000000,  # Very high efficiency
            "spinnaker": 500000,
            "edge": 100000,
            "gpu": 50000,
            "cpu": 10000
        }.get(chip_type, 50000)
        
        latency_ms = {
            "loihi2": 1.0,
            "spinnaker": 5.0,
            "edge": 10.0,
            "gpu": 15.0,
            "cpu": 50.0
        }.get(chip_type, 25.0)
        
        chip_resource = ChipResource(
            chip_id=chip_id,
            chip_type=chip_type,
            total_cores=cores,
            available_cores=cores,
            total_memory_mb=memory_mb,
            available_memory_mb=memory_mb,
            energy_efficiency=energy_efficiency,
            latency_ms=latency_ms
        )
        
        # Create backend if not provided
        if backend is None:
            if chip_type == "loihi2":
                backend = Loihi2Backend()
            elif chip_type == "spinnaker":
                backend = SpiNNakerBackend()
            # Add more backend types as needed
        
        self.chip_manager.register_chip(chip_resource, backend)
        self.logger.info(f"Added {chip_type} chip {chip_id} to cluster")
    
    def remove_chip(self, chip_id: str):
        """Remove a chip from the cluster."""
        self.chip_manager.unregister_chip(chip_id)
        self.logger.info(f"Removed chip {chip_id} from cluster")
    
    def deploy_model(self, model: nn.Module, model_name: str) -> bool:
        """Deploy a model to the cluster."""
        return self.inference_engine.deploy_model(model, model_name, self.config)
    
    async def inference(self, inputs: torch.Tensor, model_name: str) -> Dict[str, Any]:
        """Perform distributed inference."""
        return await self.inference_engine.distributed_inference(inputs, model_name, self.config)
    
    def start_auto_scaling(self):
        """Start automatic scaling."""
        self.auto_scaler.start_monitoring()
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.auto_scaler.stop_monitoring()
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Auto-scaling stopped")
    
    def _monitoring_loop(self):
        """Background monitoring and scaling loop."""
        while not self.stop_monitoring.wait(30):  # Check every 30 seconds
            try:
                decision = self.auto_scaler.evaluate_scaling_decision()
                if decision.action != "no_action":
                    self.logger.info(f"Scaling decision: {decision.action} - {decision.reason}")
                    success = self.auto_scaler.execute_scaling_decision(decision)
                    if success:
                        self.logger.info(f"Successfully executed {decision.action}")
                    else:
                        self.logger.warning(f"Failed to execute {decision.action}")
                        
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        return {
            'cluster_stats': self.chip_manager.get_cluster_stats(),
            'inference_stats': self.inference_engine.get_inference_stats(),
            'auto_scaling_active': self.auto_scaler.monitoring_active,
            'scaling_history': [
                {
                    'action': d.action,
                    'reason': d.reason,
                    'timestamp': d.timestamp,
                    'improvement': d.estimated_improvement
                }
                for d in list(self.auto_scaler.scaling_history)[-10:]  # Last 10 decisions
            ],
            'config': {
                'model_name': self.config.model_name,
                'scaling_strategy': self.config.scaling_strategy.value,
                'load_balancing': self.config.load_balancing.value,
                'min_chips': self.config.min_chips,
                'max_chips': self.config.max_chips,
                'auto_scaling_enabled': self.config.auto_scaling_enabled
            }
        }


# Convenience functions for easy cluster management
def create_neuromorphic_cluster(model_name: str, scaling_strategy: str = "auto",
                              load_balancing: str = "energy_aware") -> NeuromorphicCluster:
    """Create a neuromorphic cluster with default configuration."""
    config = DeploymentConfig(
        model_name=model_name,
        target_chips=[],
        scaling_strategy=ScalingStrategy(scaling_strategy),
        load_balancing=LoadBalancingStrategy(load_balancing)
    )
    
    return NeuromorphicCluster(config)


def setup_demo_cluster() -> NeuromorphicCluster:
    """Set up a demo cluster for testing."""
    cluster = create_neuromorphic_cluster("demo_model")
    
    # Add various chip types
    cluster.add_chip("loihi2", "loihi2_0", cores=128, memory_mb=2048)
    cluster.add_chip("spinnaker", "spinnaker_0", cores=256, memory_mb=1024)
    cluster.add_chip("edge", "edge_0", cores=64, memory_mb=512)
    cluster.add_chip("gpu", "gpu_0", cores=2048, memory_mb=8192)
    
    return cluster