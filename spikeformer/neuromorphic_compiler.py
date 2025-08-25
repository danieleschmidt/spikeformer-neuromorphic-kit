"""Neuromorphic Intermediate Representation (NIR) Compiler - Revolutionary hardware-software co-optimization."""

import torch
import torch.nn as nn
import numpy as np
import json
import yaml
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from enum import Enum
import logging
import time

from .neurons import create_neuron
from .models import SpikingConfig
from .hardware import NeuromorphicDeployer


class HardwareTarget(Enum):
    """Supported neuromorphic hardware targets."""
    LOIHI2 = "loihi2"
    SPINNAKER2 = "spinnaker2"
    BRAINSCALES2 = "brainscales2"
    AKIDA = "akida"
    DYNAP_SE = "dynap_se"
    GENERIC_ASYNC = "generic_async"
    FPGA_NEUROMORPHIC = "fpga_neuromorphic"


class OptimizationLevel(Enum):
    """Compiler optimization levels."""
    O0 = "none"          # No optimization
    O1 = "basic"         # Basic optimizations
    O2 = "aggressive"    # Aggressive optimizations
    O3 = "experimental"  # Experimental optimizations


class NIRDataType(Enum):
    """NIR data types for neuromorphic computation."""
    SPIKE = "spike"
    MEMBRANE_POTENTIAL = "membrane_potential"
    SYNAPTIC_CURRENT = "synaptic_current"
    THRESHOLD = "threshold"
    WEIGHT = "weight"
    DELAY = "delay"
    PLASTICITY_TRACE = "plasticity_trace"


@dataclass
class HardwareConstraints:
    """Hardware-specific constraints and capabilities."""
    max_neurons_per_core: int = 1024
    max_synapses_per_neuron: int = 4096
    max_axonal_delay: int = 63
    weight_precision_bits: int = 8
    threshold_precision_bits: int = 16
    max_firing_rate_hz: float = 1000.0
    power_budget_watts: float = 1.0
    memory_bandwidth_gbps: float = 100.0
    latency_budget_ms: float = 10.0
    
    # Hardware-specific features
    supports_plasticity: bool = True
    supports_stdp: bool = True
    supports_homeostasis: bool = False
    supports_multicompartment: bool = False
    supports_dendritic_computation: bool = False
    supports_axonal_delays: bool = True
    
    # Resource limits
    total_cores: int = 128
    shared_memory_mb: int = 512
    local_memory_per_core_kb: int = 256


@dataclass
class NIRNode:
    """Node in Neuromorphic Intermediate Representation."""
    node_id: str
    node_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hardware_mapping: Optional[Dict[str, Any]] = None


@dataclass  
class NIRGraph:
    """Complete NIR graph representation."""
    nodes: Dict[str, NIRNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


class NIRTransform(ABC):
    """Abstract base class for NIR transformations."""
    
    @abstractmethod
    def apply(self, graph: NIRGraph) -> NIRGraph:
        """Apply transformation to NIR graph."""
        pass
    
    @abstractmethod
    def is_applicable(self, graph: NIRGraph, constraints: HardwareConstraints) -> bool:
        """Check if transformation is applicable given constraints."""
        pass


class SparsityOptimization(NIRTransform):
    """Optimize for sparse computation patterns."""
    
    def __init__(self, target_sparsity: float = 0.9):
        self.target_sparsity = target_sparsity
        
    def apply(self, graph: NIRGraph) -> NIRGraph:
        """Apply sparsity optimizations."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        for node_id, node in optimized_graph.nodes.items():
            if node.node_type == "synaptic_layer":
                # Prune weak connections
                weights = node.parameters.get('weights', torch.zeros(100, 100))
                if isinstance(weights, torch.Tensor):
                    # Calculate sparsity threshold
                    weight_magnitudes = torch.abs(weights.flatten())
                    threshold_idx = int((1.0 - self.target_sparsity) * len(weight_magnitudes))
                    if threshold_idx > 0:
                        sparsity_threshold = torch.kthvalue(weight_magnitudes, threshold_idx)[0]
                        
                        # Apply sparsity mask
                        sparse_mask = torch.abs(weights) >= sparsity_threshold
                        sparse_weights = weights * sparse_mask
                        
                        # Update node parameters
                        node.parameters['weights'] = sparse_weights
                        node.parameters['sparsity_mask'] = sparse_mask
                        node.parameters['actual_sparsity'] = 1.0 - (sparse_mask.sum().float() / sparse_mask.numel())
                        
                        # Add sparse computation hints
                        node.metadata['sparse_computation'] = True
                        node.metadata['compression_ratio'] = node.parameters['actual_sparsity']
        
        return optimized_graph
    
    def is_applicable(self, graph: NIRGraph, constraints: HardwareConstraints) -> bool:
        """Check if sparsity optimization is beneficial."""
        # Always beneficial for large networks
        total_synapses = sum(
            node.parameters.get('num_synapses', 0) 
            for node in graph.nodes.values() 
            if node.node_type == "synaptic_layer"
        )
        return total_synapses > 1000


class TemporalOptimization(NIRTransform):
    """Optimize temporal computation patterns."""
    
    def apply(self, graph: NIRGraph) -> NIRGraph:
        """Apply temporal optimizations."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # Analyze temporal dependencies
        temporal_chains = self._find_temporal_chains(graph)
        
        # Optimize each temporal chain
        for chain in temporal_chains:
            self._optimize_temporal_chain(optimized_graph, chain)
        
        # Add temporal batching
        self._add_temporal_batching(optimized_graph)
        
        return optimized_graph
    
    def _find_temporal_chains(self, graph: NIRGraph) -> List[List[str]]:
        """Find chains of temporally dependent computations."""
        chains = []
        visited = set()
        
        for node_id in graph.nodes:
            if node_id not in visited:
                chain = self._trace_temporal_chain(graph, node_id, visited)
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _trace_temporal_chain(self, graph: NIRGraph, start_node: str, visited: set) -> List[str]:
        """Trace a temporal dependency chain from a starting node."""
        chain = [start_node]
        visited.add(start_node)
        
        # Find temporal successors
        for edge_start, edge_end in graph.edges:
            if edge_start == start_node and edge_end not in visited:
                node = graph.nodes[edge_end]
                if node.metadata.get('temporal_dependency', False):
                    chain.extend(self._trace_temporal_chain(graph, edge_end, visited))
                    break
        
        return chain
    
    def _optimize_temporal_chain(self, graph: NIRGraph, chain: List[str]):
        """Optimize a single temporal chain."""
        for i, node_id in enumerate(chain):
            node = graph.nodes[node_id]
            
            # Add pipeline stage information
            node.metadata['pipeline_stage'] = i
            node.metadata['chain_id'] = hash(tuple(chain))
            
            # Optimize buffering
            if i > 0:  # Not first node
                node.metadata['input_buffering'] = True
                node.parameters['buffer_size'] = 32
    
    def _add_temporal_batching(self, graph: NIRGraph):
        """Add temporal batching for efficient processing."""
        for node in graph.nodes.values():
            if node.node_type in ['neuron_population', 'synaptic_layer']:
                # Add batching parameters
                node.parameters['temporal_batch_size'] = 16
                node.metadata['supports_batching'] = True
    
    def is_applicable(self, graph: NIRGraph, constraints: HardwareConstraints) -> bool:
        """Check if temporal optimization is applicable."""
        # Beneficial for deep temporal networks
        max_depth = self._calculate_graph_depth(graph)
        return max_depth > 3
    
    def _calculate_graph_depth(self, graph: NIRGraph) -> int:
        """Calculate maximum depth of the computation graph."""
        depths = {}
        
        def dfs_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            
            max_pred_depth = 0
            for edge_start, edge_end in graph.edges:
                if edge_end == node_id:
                    pred_depth = dfs_depth(edge_start)
                    max_pred_depth = max(max_pred_depth, pred_depth)
            
            depths[node_id] = max_pred_depth + 1
            return depths[node_id]
        
        return max(dfs_depth(node_id) for node_id in graph.nodes)


class MemoryOptimization(NIRTransform):
    """Optimize memory usage patterns."""
    
    def apply(self, graph: NIRGraph) -> NIRGraph:
        """Apply memory optimizations."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # Analyze memory usage
        memory_analysis = self._analyze_memory_usage(graph)
        
        # Apply memory optimizations
        self._optimize_weight_storage(optimized_graph, memory_analysis)
        self._optimize_activation_memory(optimized_graph, memory_analysis)
        self._add_memory_compression(optimized_graph)
        
        return optimized_graph
    
    def _analyze_memory_usage(self, graph: NIRGraph) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        analysis = {
            'total_parameters': 0,
            'activation_memory': 0,
            'weight_memory': 0,
            'buffer_memory': 0,
            'hotspots': []
        }
        
        for node_id, node in graph.nodes.items():
            if 'weights' in node.parameters:
                weights = node.parameters['weights']
                if isinstance(weights, torch.Tensor):
                    weight_memory = weights.numel() * 4  # Assume 4 bytes per weight
                    analysis['weight_memory'] += weight_memory
                    analysis['total_parameters'] += weights.numel()
                    
                    # Identify memory hotspots
                    if weight_memory > 1e6:  # > 1MB
                        analysis['hotspots'].append({
                            'node_id': node_id,
                            'memory_mb': weight_memory / 1e6,
                            'type': 'weights'
                        })
        
        return analysis
    
    def _optimize_weight_storage(self, graph: NIRGraph, analysis: Dict[str, Any]):
        """Optimize weight storage format."""
        for hotspot in analysis['hotspots']:
            if hotspot['type'] == 'weights':
                node = graph.nodes[hotspot['node_id']]
                weights = node.parameters['weights']
                
                # Apply weight quantization
                if isinstance(weights, torch.Tensor):
                    # 8-bit quantization
                    weight_scale = torch.max(torch.abs(weights)) / 127.0
                    quantized_weights = torch.round(weights / weight_scale).clamp(-128, 127).to(torch.int8)
                    
                    node.parameters['weights'] = quantized_weights
                    node.parameters['weight_scale'] = weight_scale
                    node.metadata['weight_quantized'] = True
                    node.metadata['compression_ratio'] = 4.0  # 32-bit to 8-bit
    
    def _optimize_activation_memory(self, graph: NIRGraph, analysis: Dict[str, Any]):
        """Optimize activation memory usage."""
        for node in graph.nodes.values():
            if node.node_type == 'neuron_population':
                # Add activation compression
                node.parameters['activation_compression'] = 'delta'
                node.metadata['compressed_activations'] = True
    
    def _add_memory_compression(self, graph: NIRGraph):
        """Add memory compression optimizations."""
        graph.optimization_hints['memory_compression'] = {
            'enable_weight_sharing': True,
            'enable_activation_checkpointing': True,
            'compression_algorithm': 'lz4'
        }
    
    def is_applicable(self, graph: NIRGraph, constraints: HardwareConstraints) -> bool:
        """Check if memory optimization is needed."""
        total_memory = sum(
            node.parameters.get('memory_usage', 0)
            for node in graph.nodes.values()
        )
        return total_memory > constraints.shared_memory_mb * 1e6 * 0.8  # 80% of memory


class HardwareSpecificOptimization(NIRTransform):
    """Hardware-specific optimizations."""
    
    def __init__(self, target_hardware: HardwareTarget, constraints: HardwareConstraints):
        self.target_hardware = target_hardware
        self.constraints = constraints
        
    def apply(self, graph: NIRGraph) -> NIRGraph:
        """Apply hardware-specific optimizations."""
        if self.target_hardware == HardwareTarget.LOIHI2:
            return self._optimize_for_loihi2(graph)
        elif self.target_hardware == HardwareTarget.SPINNAKER2:
            return self._optimize_for_spinnaker2(graph)
        elif self.target_hardware == HardwareTarget.AKIDA:
            return self._optimize_for_akida(graph)
        else:
            return self._optimize_generic(graph)
    
    def _optimize_for_loihi2(self, graph: NIRGraph) -> NIRGraph:
        """Optimize for Intel Loihi 2."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # Loihi 2 specific optimizations
        for node in optimized_graph.nodes.values():
            if node.node_type == 'neuron_population':
                # Use Loihi 2 neuron models
                node.parameters['neuron_model'] = 'loihi_lif'
                node.parameters['use_graded_spikes'] = True
                node.parameters['mantissa_bits'] = 23
                
                # Core allocation hints
                neurons_per_core = min(node.parameters.get('size', 1024), 
                                     self.constraints.max_neurons_per_core)
                node.hardware_mapping = {
                    'target_cores': max(1, node.parameters.get('size', 1024) // neurons_per_core),
                    'neurons_per_core': neurons_per_core
                }
                
            elif node.node_type == 'synaptic_layer':
                # Loihi 2 synaptic optimizations
                node.parameters['use_dendrite_compartments'] = True
                node.parameters['synaptic_delay_bits'] = 6  # Max 63 timesteps
                
        # Add Loihi 2 specific global parameters
        optimized_graph.global_parameters.update({
            'time_step_us': 1.0,  # 1 microsecond timestep
            'spike_input_format': 'graded',
            'core_interconnect': 'mesh',
            'plasticity_enabled': self.constraints.supports_plasticity
        })
        
        return optimized_graph
    
    def _optimize_for_spinnaker2(self, graph: NIRGraph) -> NIRGraph:
        """Optimize for SpiNNaker 2."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # SpiNNaker 2 specific optimizations
        for node in optimized_graph.nodes.values():
            if node.node_type == 'neuron_population':
                # ARM-optimized neuron models
                node.parameters['neuron_model'] = 'spinnaker_lif'
                node.parameters['use_fixed_point'] = True
                node.parameters['time_step_ms'] = 1.0
                
                # Core mapping for ARM cores
                node.hardware_mapping = {
                    'arm_core_assignment': 'round_robin',
                    'packet_routing': 'multicast'
                }
                
        # Add SpiNNaker specific parameters
        optimized_graph.global_parameters.update({
            'packet_format': 'spinnaker',
            'routing_algorithm': 'ner',  # Nearest-neighbor Emergency Routing
            'real_time_factor': 1.0
        })
        
        return optimized_graph
    
    def _optimize_for_akida(self, graph: NIRGraph) -> NIRGraph:
        """Optimize for Akida neuromorphic processor."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # Akida specific optimizations
        for node in optimized_graph.nodes.values():
            if node.node_type == 'neuron_population':
                # Akida neuron constraints
                node.parameters['neuron_model'] = 'akida_if'
                node.parameters['activation_bits'] = 4
                node.parameters['weight_bits'] = 4
                
            elif node.node_type == 'synaptic_layer':
                # Akida convolution-focused optimizations
                if 'conv' in node.metadata.get('layer_type', ''):
                    node.parameters['use_akida_conv'] = True
                    node.parameters['pooling_optimization'] = True
        
        optimized_graph.global_parameters.update({
            'quantization': '4bit',
            'inference_mode': 'edge',
            'power_optimization': 'aggressive'
        })
        
        return optimized_graph
    
    def _optimize_generic(self, graph: NIRGraph) -> NIRGraph:
        """Generic optimizations for unknown hardware."""
        optimized_graph = NIRGraph(
            nodes=graph.nodes.copy(),
            edges=graph.edges.copy(),
            global_parameters=graph.global_parameters.copy()
        )
        
        # Conservative generic optimizations
        for node in optimized_graph.nodes.values():
            node.metadata['generic_optimization'] = True
            
        return optimized_graph
    
    def is_applicable(self, graph: NIRGraph, constraints: HardwareConstraints) -> bool:
        """Always applicable."""
        return True


class NeuromorphicCompiler:
    """Main compiler for neuromorphic hardware deployment."""
    
    def __init__(self, target_hardware: HardwareTarget = HardwareTarget.LOIHI2,
                 optimization_level: OptimizationLevel = OptimizationLevel.O2):
        self.target_hardware = target_hardware
        self.optimization_level = optimization_level
        self.constraints = self._get_hardware_constraints(target_hardware)
        
        # Available transformations
        self.transformations = self._create_transformation_pipeline()
        
        # Compilation statistics
        self.compilation_stats = {
            'total_time_seconds': 0.0,
            'transformations_applied': [],
            'memory_savings_mb': 0.0,
            'energy_savings_percent': 0.0,
            'performance_improvement_percent': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _get_hardware_constraints(self, target: HardwareTarget) -> HardwareConstraints:
        """Get hardware constraints for target platform."""
        if target == HardwareTarget.LOIHI2:
            return HardwareConstraints(
                max_neurons_per_core=1024,
                max_synapses_per_neuron=4096,
                max_axonal_delay=63,
                weight_precision_bits=8,
                supports_plasticity=True,
                supports_stdp=True,
                total_cores=128,
                power_budget_watts=1.0
            )
        elif target == HardwareTarget.SPINNAKER2:
            return HardwareConstraints(
                max_neurons_per_core=256,
                max_synapses_per_neuron=1024,
                weight_precision_bits=16,
                supports_plasticity=True,
                total_cores=1024,  # 18 cores per chip, many chips
                power_budget_watts=5.0,
                memory_bandwidth_gbps=50.0
            )
        elif target == HardwareTarget.AKIDA:
            return HardwareConstraints(
                max_neurons_per_core=512,
                weight_precision_bits=4,
                threshold_precision_bits=4,
                supports_plasticity=False,  # Inference-focused
                power_budget_watts=0.1,  # Ultra-low power
                supports_multicompartment=False
            )
        else:
            return HardwareConstraints()  # Default constraints
    
    def _create_transformation_pipeline(self) -> List[NIRTransform]:
        """Create compilation transformation pipeline based on optimization level."""
        transformations = []
        
        if self.optimization_level in [OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3]:
            transformations.append(SparsityOptimization(target_sparsity=0.9))
            
        if self.optimization_level in [OptimizationLevel.O2, OptimizationLevel.O3]:
            transformations.append(TemporalOptimization())
            transformations.append(MemoryOptimization())
            
        if self.optimization_level == OptimizationLevel.O3:
            # Experimental optimizations
            transformations.append(SparsityOptimization(target_sparsity=0.95))  # More aggressive
            
        # Always apply hardware-specific optimizations
        transformations.append(HardwareSpecificOptimization(self.target_hardware, self.constraints))
        
        return transformations
    
    def compile_model(self, model: nn.Module, input_shape: Tuple[int, ...],
                     output_path: str = None) -> Dict[str, Any]:
        """Compile a PyTorch model for neuromorphic deployment."""
        self.logger.info(f"Starting compilation for {self.target_hardware.value}")
        start_time = time.time()
        
        # Convert model to NIR
        nir_graph = self._convert_to_nir(model, input_shape)
        
        # Apply optimization transformations
        optimized_graph = self._apply_optimizations(nir_graph)
        
        # Generate hardware-specific code
        deployment_package = self._generate_deployment_package(optimized_graph)
        
        # Save compilation results
        if output_path:
            self._save_compilation_results(deployment_package, output_path)
        
        # Update compilation statistics
        self.compilation_stats['total_time_seconds'] = time.time() - start_time
        
        self.logger.info(f"Compilation completed in {self.compilation_stats['total_time_seconds']:.2f}s")
        
        return deployment_package
    
    def _convert_to_nir(self, model: nn.Module, input_shape: Tuple[int, ...]) -> NIRGraph:
        """Convert PyTorch model to NIR representation."""
        nir_graph = NIRGraph()
        
        # Extract model architecture
        node_id = 0
        layer_mapping = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                # Create NIR node for each layer
                nir_node = self._create_nir_node_from_module(f"node_{node_id}", module, name)
                nir_graph.nodes[nir_node.node_id] = nir_node
                layer_mapping[name] = nir_node.node_id
                node_id += 1
        
        # Create edges based on model structure
        self._create_nir_edges(model, nir_graph, layer_mapping)
        
        # Add global parameters
        nir_graph.global_parameters = {
            'input_shape': input_shape,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_name': model.__class__.__name__
        }
        
        return nir_graph
    
    def _create_nir_node_from_module(self, node_id: str, module: nn.Module, name: str) -> NIRNode:
        """Create NIR node from PyTorch module."""
        
        if isinstance(module, nn.Linear):
            return NIRNode(
                node_id=node_id,
                node_type='synaptic_layer',
                parameters={
                    'weights': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None,
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'num_synapses': module.in_features * module.out_features
                },
                metadata={'original_layer': name, 'layer_type': 'linear'}
            )
            
        elif isinstance(module, nn.Conv2d):
            return NIRNode(
                node_id=node_id,
                node_type='synaptic_layer',
                parameters={
                    'weights': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'num_synapses': module.weight.numel()
                },
                metadata={'original_layer': name, 'layer_type': 'conv'}
            )
            
        elif 'Spiking' in module.__class__.__name__ or 'LIF' in module.__class__.__name__:
            # Spiking neuron layer
            return NIRNode(
                node_id=node_id,
                node_type='neuron_population',
                parameters={
                    'neuron_model': 'LIF',
                    'threshold': getattr(module, 'threshold', 1.0),
                    'tau_mem': getattr(module, 'tau_mem', 20.0),
                    'tau_syn': getattr(module, 'tau_syn', 5.0),
                    'size': getattr(module, 'size', 100)
                },
                metadata={'original_layer': name, 'layer_type': 'spiking_neuron'}
            )
            
        else:
            # Generic layer
            return NIRNode(
                node_id=node_id,
                node_type='generic_layer',
                parameters={'original_module': str(module)},
                metadata={'original_layer': name, 'layer_type': 'generic'}
            )
    
    def _create_nir_edges(self, model: nn.Module, nir_graph: NIRGraph, layer_mapping: Dict[str, str]):
        """Create edges in NIR graph based on model connectivity."""
        # Simplified edge creation - assumes sequential connectivity
        node_ids = list(nir_graph.nodes.keys())
        
        for i in range(len(node_ids) - 1):
            source_id = node_ids[i]
            target_id = node_ids[i + 1]
            nir_graph.edges.append((source_id, target_id))
            
            # Update input/output connections
            nir_graph.nodes[source_id].outputs.append(target_id)
            nir_graph.nodes[target_id].inputs.append(source_id)
    
    def _apply_optimizations(self, nir_graph: NIRGraph) -> NIRGraph:
        """Apply optimization transformations to NIR graph."""
        current_graph = nir_graph
        
        for transformation in self.transformations:
            if transformation.is_applicable(current_graph, self.constraints):
                self.logger.info(f"Applying {transformation.__class__.__name__}")
                current_graph = transformation.apply(current_graph)
                self.compilation_stats['transformations_applied'].append(transformation.__class__.__name__)
        
        return current_graph
    
    def _generate_deployment_package(self, nir_graph: NIRGraph) -> Dict[str, Any]:
        """Generate hardware-specific deployment package."""
        
        # Calculate resource requirements
        resource_analysis = self._analyze_resource_requirements(nir_graph)
        
        # Generate hardware configuration
        hardware_config = self._generate_hardware_config(nir_graph, resource_analysis)
        
        # Generate runtime code
        runtime_code = self._generate_runtime_code(nir_graph)
        
        # Create deployment package
        deployment_package = {
            'nir_graph': asdict(nir_graph),
            'hardware_config': hardware_config,
            'runtime_code': runtime_code,
            'resource_analysis': resource_analysis,
            'target_hardware': self.target_hardware.value,
            'optimization_level': self.optimization_level.value,
            'compilation_stats': self.compilation_stats,
            'deployment_instructions': self._generate_deployment_instructions()
        }
        
        return deployment_package
    
    def _analyze_resource_requirements(self, nir_graph: NIRGraph) -> Dict[str, Any]:
        """Analyze resource requirements for the compiled model."""
        analysis = {
            'total_neurons': 0,
            'total_synapses': 0,
            'memory_requirements_mb': 0.0,
            'estimated_power_watts': 0.0,
            'cores_required': 0,
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        for node in nir_graph.nodes.values():
            if node.node_type == 'neuron_population':
                neurons = node.parameters.get('size', 0)
                analysis['total_neurons'] += neurons
                analysis['estimated_power_watts'] += neurons * 1e-9  # 1 nW per neuron
                
            elif node.node_type == 'synaptic_layer':
                synapses = node.parameters.get('num_synapses', 0)
                analysis['total_synapses'] += synapses
                
                # Memory for weights
                weights = node.parameters.get('weights', torch.tensor([]))
                if isinstance(weights, torch.Tensor):
                    weight_memory = weights.numel() * 4  # 4 bytes per weight
                    if node.metadata.get('weight_quantized', False):
                        weight_memory //= 4  # Quantized to 8-bit
                    analysis['memory_requirements_mb'] += weight_memory / 1e6
        
        # Calculate required cores
        analysis['cores_required'] = max(1, analysis['total_neurons'] // self.constraints.max_neurons_per_core)
        
        # Identify bottlenecks
        if analysis['memory_requirements_mb'] > self.constraints.shared_memory_mb:
            analysis['bottlenecks'].append('memory')
            
        if analysis['cores_required'] > self.constraints.total_cores:
            analysis['bottlenecks'].append('compute_cores')
            
        if analysis['estimated_power_watts'] > self.constraints.power_budget_watts:
            analysis['bottlenecks'].append('power')
        
        return analysis
    
    def _generate_hardware_config(self, nir_graph: NIRGraph, resource_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hardware-specific configuration."""
        
        config = {
            'target_platform': self.target_hardware.value,
            'core_allocation': {},
            'memory_layout': {},
            'routing_configuration': {},
            'timing_configuration': {}
        }
        
        # Core allocation
        core_id = 0
        for node_id, node in nir_graph.nodes.items():
            if node.node_type == 'neuron_population':
                neurons_per_core = min(node.parameters.get('size', 0), 
                                     self.constraints.max_neurons_per_core)
                required_cores = max(1, node.parameters.get('size', 0) // neurons_per_core)
                
                config['core_allocation'][node_id] = {
                    'cores': list(range(core_id, core_id + required_cores)),
                    'neurons_per_core': neurons_per_core
                }
                core_id += required_cores
        
        # Memory layout
        memory_offset = 0
        for node_id, node in nir_graph.nodes.items():
            if 'weights' in node.parameters:
                weights = node.parameters['weights']
                if isinstance(weights, torch.Tensor):
                    memory_size = weights.numel() * 4  # 4 bytes per weight
                    if node.metadata.get('weight_quantized', False):
                        memory_size //= 4
                    
                    config['memory_layout'][node_id] = {
                        'offset': memory_offset,
                        'size_bytes': memory_size,
                        'type': 'weights'
                    }
                    memory_offset += memory_size
        
        # Timing configuration
        config['timing_configuration'] = {
            'timestep_us': nir_graph.global_parameters.get('timestep_us', 1.0),
            'simulation_time_ms': 100.0,  # Default simulation time
            'real_time_factor': 1.0
        }
        
        return config
    
    def _generate_runtime_code(self, nir_graph: NIRGraph) -> Dict[str, str]:
        """Generate runtime code for deployment."""
        
        code = {
            'initialization': self._generate_initialization_code(nir_graph),
            'main_loop': self._generate_main_loop_code(nir_graph),
            'cleanup': self._generate_cleanup_code(),
            'utilities': self._generate_utility_functions()
        }
        
        return code
    
    def _generate_initialization_code(self, nir_graph: NIRGraph) -> str:
        """Generate initialization code."""
        init_code = f"""
// Neuromorphic hardware initialization for {self.target_hardware.value}
// Generated by NIR Compiler

#include <stdio.h>
#include <stdlib.h>
#include "neuromorphic_runtime.h"

int initialize_network(void) {{
    printf("Initializing network for {self.target_hardware.value}\\n");
    
    // Initialize {len(nir_graph.nodes)} nodes
"""
        
        for node_id, node in nir_graph.nodes.items():
            if node.node_type == 'neuron_population':
                size = node.parameters.get('size', 100)
                threshold = node.parameters.get('threshold', 1.0)
                init_code += f"""
    // Initialize neuron population {node_id}
    create_neuron_population("{node_id}", {size}, {threshold});
"""
            elif node.node_type == 'synaptic_layer':
                init_code += f"""
    // Initialize synaptic layer {node_id}
    create_synaptic_layer("{node_id}");
    load_weights("{node_id}", weights_{node_id});
"""
        
        init_code += """
    return 0;
}
"""
        return init_code
    
    def _generate_main_loop_code(self, nir_graph: NIRGraph) -> str:
        """Generate main simulation loop code."""
        loop_code = f"""
// Main simulation loop
int run_simulation(int timesteps) {{
    for (int t = 0; t < timesteps; t++) {{
        // Process all nodes in topological order
"""
        
        # Generate code for each node in topological order
        for node_id, node in nir_graph.nodes.items():
            if node.node_type == 'neuron_population':
                loop_code += f"""
        // Update neuron population {node_id}
        update_neurons("{node_id}", t);
"""
            elif node.node_type == 'synaptic_layer':
                loop_code += f"""
        // Process synaptic layer {node_id}
        process_synapses("{node_id}", t);
"""
        
        loop_code += """
        // Synchronize timestep
        synchronize_timestep(t);
    }
    return 0;
}
"""
        return loop_code
    
    def _generate_cleanup_code(self) -> str:
        """Generate cleanup code."""
        return """
// Cleanup resources
void cleanup_network(void) {
    printf("Cleaning up network resources\\n");
    free_all_neurons();
    free_all_synapses();
    cleanup_hardware();
}
"""
    
    def _generate_utility_functions(self) -> str:
        """Generate utility functions."""
        return f"""
// Utility functions for {self.target_hardware.value}

void print_network_stats(void) {{
    printf("Network Statistics:\\n");
    printf("Target Hardware: {self.target_hardware.value}\\n");
    printf("Optimization Level: {self.optimization_level.value}\\n");
}}

double get_power_consumption(void) {{
    // Return estimated power consumption in watts
    return estimate_power();
}}

void save_spike_data(const char* filename) {{
    // Save spike data to file
    FILE* file = fopen(filename, "w");
    if (file) {{
        export_spikes(file);
        fclose(file);
    }}
}}
"""
    
    def _generate_deployment_instructions(self) -> List[str]:
        """Generate deployment instructions."""
        instructions = [
            f"1. Target platform: {self.target_hardware.value}",
            f"2. Required cores: {self.compilation_stats.get('cores_required', 'Unknown')}",
            f"3. Memory requirements: {self.compilation_stats.get('memory_mb', 'Unknown')} MB",
            "4. Compile runtime code with appropriate SDK",
            "5. Load weight data to hardware memory",
            "6. Initialize network connections",
            "7. Start simulation with desired timesteps",
            "8. Monitor performance and power consumption"
        ]
        
        return instructions
    
    def _save_compilation_results(self, deployment_package: Dict[str, Any], output_path: str):
        """Save compilation results to files."""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Save NIR graph as JSON
        with open(os.path.join(output_path, 'nir_graph.json'), 'w') as f:
            json.dump(deployment_package['nir_graph'], f, indent=2, default=str)
        
        # Save hardware config as YAML
        with open(os.path.join(output_path, 'hardware_config.yaml'), 'w') as f:
            yaml.dump(deployment_package['hardware_config'], f, default_flow_style=False)
        
        # Save runtime code
        runtime_code = deployment_package['runtime_code']
        for code_type, code_content in runtime_code.items():
            filename = f"{code_type}.c"
            with open(os.path.join(output_path, filename), 'w') as f:
                f.write(code_content)
        
        # Save compilation report
        with open(os.path.join(output_path, 'compilation_report.json'), 'w') as f:
            json.dump(deployment_package['compilation_stats'], f, indent=2)
        
        self.logger.info(f"Compilation results saved to {output_path}")


class CompilerBenchmark:
    """Benchmark the neuromorphic compiler performance."""
    
    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive compiler benchmarks."""
        self.logger.info("🔧 Starting Neuromorphic Compiler Benchmark")
        
        results = {}
        
        # Test different optimization levels
        optimization_results = self._benchmark_optimization_levels()
        results['optimization_levels'] = optimization_results
        
        # Test different hardware targets
        hardware_results = self._benchmark_hardware_targets()
        results['hardware_targets'] = hardware_results
        
        # Test scalability
        scalability_results = self._benchmark_scalability()
        results['scalability'] = scalability_results
        
        # Test transformation effectiveness
        transformation_results = self._benchmark_transformations()
        results['transformations'] = transformation_results
        
        self.logger.info("🎉 Compiler Benchmark Completed!")
        
        return results
    
    def _benchmark_optimization_levels(self) -> Dict[str, Any]:
        """Benchmark different optimization levels."""
        results = {}
        
        # Create test model
        test_model = self._create_test_model()
        
        for opt_level in OptimizationLevel:
            compiler = NeuromorphicCompiler(
                target_hardware=HardwareTarget.LOIHI2,
                optimization_level=opt_level
            )
            
            start_time = time.time()
            deployment_package = compiler.compile_model(test_model, (1, 784))
            compilation_time = time.time() - start_time
            
            results[opt_level.value] = {
                'compilation_time_seconds': compilation_time,
                'transformations_applied': len(deployment_package['compilation_stats']['transformations_applied']),
                'memory_requirements_mb': deployment_package['resource_analysis']['memory_requirements_mb'],
                'estimated_power_watts': deployment_package['resource_analysis']['estimated_power_watts']
            }
        
        return results
    
    def _benchmark_hardware_targets(self) -> Dict[str, Any]:
        """Benchmark different hardware targets."""
        results = {}
        
        test_model = self._create_test_model()
        
        for hardware in [HardwareTarget.LOIHI2, HardwareTarget.SPINNAKER2, HardwareTarget.AKIDA]:
            compiler = NeuromorphicCompiler(
                target_hardware=hardware,
                optimization_level=OptimizationLevel.O2
            )
            
            start_time = time.time()
            deployment_package = compiler.compile_model(test_model, (1, 784))
            compilation_time = time.time() - start_time
            
            results[hardware.value] = {
                'compilation_time_seconds': compilation_time,
                'cores_required': deployment_package['resource_analysis']['cores_required'],
                'memory_mb': deployment_package['resource_analysis']['memory_requirements_mb'],
                'power_watts': deployment_package['resource_analysis']['estimated_power_watts']
            }
        
        return results
    
    def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark compiler scalability with network size."""
        results = {}
        
        network_sizes = [100, 500, 1000, 5000, 10000]  # Number of neurons
        
        for size in network_sizes:
            test_model = self._create_scalable_test_model(size)
            
            compiler = NeuromorphicCompiler(
                target_hardware=HardwareTarget.LOIHI2,
                optimization_level=OptimizationLevel.O2
            )
            
            start_time = time.time()
            deployment_package = compiler.compile_model(test_model, (1, 784))
            compilation_time = time.time() - start_time
            
            results[f"{size}_neurons"] = {
                'compilation_time_seconds': compilation_time,
                'memory_mb': deployment_package['resource_analysis']['memory_requirements_mb'],
                'cores_required': deployment_package['resource_analysis']['cores_required']
            }
        
        return results
    
    def _benchmark_transformations(self) -> Dict[str, Any]:
        """Benchmark individual transformations."""
        results = {}
        
        # Create test NIR graph
        test_model = self._create_test_model()
        compiler = NeuromorphicCompiler()
        nir_graph = compiler._convert_to_nir(test_model, (1, 784))
        
        # Test individual transformations
        transformations = [
            ('sparsity', SparsityOptimization(0.9)),
            ('temporal', TemporalOptimization()),
            ('memory', MemoryOptimization())
        ]
        
        for trans_name, transformation in transformations:
            if transformation.is_applicable(nir_graph, compiler.constraints):
                start_time = time.time()
                optimized_graph = transformation.apply(nir_graph)
                transform_time = time.time() - start_time
                
                # Calculate improvement metrics
                original_nodes = len(nir_graph.nodes)
                optimized_nodes = len(optimized_graph.nodes)
                
                results[trans_name] = {
                    'transformation_time_seconds': transform_time,
                    'nodes_before': original_nodes,
                    'nodes_after': optimized_nodes,
                    'applicable': True
                }
            else:
                results[trans_name] = {'applicable': False}
        
        return results
    
    def _create_test_model(self) -> nn.Module:
        """Create a test model for benchmarking."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        return TestModel()
    
    def _create_scalable_test_model(self, hidden_size: int) -> nn.Module:
        """Create a scalable test model."""
        class ScalableModel(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(784, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 10)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        return ScalableModel(hidden_size)


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark = CompilerBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print("\n🔧 NEUROMORPHIC COMPILER BENCHMARK RESULTS:")
    print("=" * 60)
    
    print(f"\n🎛️  Optimization Levels:")
    for opt_level, metrics in results['optimization_levels'].items():
        print(f"  {opt_level}: {metrics['compilation_time_seconds']:.3f}s, "
              f"{metrics['transformations_applied']} transforms, "
              f"{metrics['memory_requirements_mb']:.2f} MB")
    
    print(f"\n💻 Hardware Targets:")
    for hardware, metrics in results['hardware_targets'].items():
        print(f"  {hardware}: {metrics['compilation_time_seconds']:.3f}s, "
              f"{metrics['cores_required']} cores, "
              f"{metrics['power_watts']:.6f} W")
    
    print(f"\n📈 Scalability:")
    for size, metrics in results['scalability'].items():
        print(f"  {size}: {metrics['compilation_time_seconds']:.3f}s, "
              f"{metrics['cores_required']} cores")
    
    print(f"\n🔄 Transformations:")
    for trans_name, metrics in results['transformations'].items():
        if metrics.get('applicable', False):
            print(f"  {trans_name}: {metrics['transformation_time_seconds']:.6f}s")
        else:
            print(f"  {trans_name}: Not applicable")