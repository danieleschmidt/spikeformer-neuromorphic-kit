"""Emergent Intelligence: Self-Organizing Neuromorphic Systems.

This module implements emergent intelligence algorithms that enable neuromorphic
systems to develop complex behaviors and representations without explicit programming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import math
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class EmergentConfig:
    """Configuration for emergent intelligence systems."""
    network_size: int = 1000
    connection_density: float = 0.1
    emergence_threshold: float = 0.7
    self_organization_rate: float = 0.01
    criticality_parameter: float = 1.0
    avalanche_size_threshold: int = 10
    complexity_measure: str = "lempel_ziv"
    adaptive_topology: bool = True
    homeostatic_regulation: bool = True


class CriticalBrainDynamics(nn.Module):
    """Self-organized criticality in spiking neural networks."""
    
    def __init__(self, network_size: int, connection_density: float = 0.1):
        super().__init__()
        self.network_size = network_size
        self.connection_density = connection_density
        
        # Initialize network with scale-free topology
        self.adjacency = self._create_scale_free_network()
        self.register_buffer('connectivity', self.adjacency)
        
        # Neuron parameters
        self.membrane_potentials = nn.Parameter(torch.zeros(network_size))
        self.thresholds = nn.Parameter(torch.ones(network_size))
        self.refractory_periods = nn.Parameter(torch.zeros(network_size))
        
        # Critical dynamics parameters
        self.branching_ratio = nn.Parameter(torch.tensor(1.0))  # Critical point
        self.avalanche_history = deque(maxlen=1000)
        
    def _create_scale_free_network(self) -> torch.Tensor:
        """Create a scale-free network topology using preferential attachment."""
        adjacency = torch.zeros(self.network_size, self.network_size)
        
        # Start with a small connected graph
        for i in range(3):
            for j in range(i + 1, 3):
                adjacency[i, j] = adjacency[j, i] = 1
        
        # Add nodes with preferential attachment
        for new_node in range(3, self.network_size):
            # Calculate attachment probabilities based on degree
            degrees = adjacency.sum(dim=1)[:new_node]
            if degrees.sum() == 0:
                probs = torch.ones(new_node) / new_node
            else:
                probs = degrees / degrees.sum()
            
            # Connect to existing nodes based on probability
            num_connections = min(3, new_node)
            connected_nodes = torch.multinomial(probs, num_connections, replacement=False)
            
            for target in connected_nodes:
                adjacency[new_node, target] = adjacency[target, new_node] = 1
                
        return adjacency
    
    def forward(self, external_input: torch.Tensor) -> torch.Tensor:
        """Simulate critical brain dynamics."""
        batch_size = external_input.size(0)
        
        # Initialize states
        spikes = torch.zeros(batch_size, self.network_size)
        membrane_v = self.membrane_potentials.unsqueeze(0).expand(batch_size, -1)
        
        # Apply external input
        membrane_v += external_input
        
        # Network dynamics
        for step in range(10):  # Multiple time steps
            # Propagate spikes through network
            network_input = torch.matmul(spikes, self.connectivity)
            membrane_v += network_input * 0.1
            
            # Generate spikes
            spike_mask = (membrane_v > self.thresholds) & (self.refractory_periods == 0)
            new_spikes = spike_mask.float()
            
            # Reset spiked neurons
            membrane_v[spike_mask] = 0
            
            # Update refractory periods
            self.refractory_periods[spike_mask.any(dim=0)] = 3
            self.refractory_periods = torch.clamp(self.refractory_periods - 1, 0, 10)
            
            # Decay membrane potential
            membrane_v *= 0.95
            
            spikes = new_spikes
            
            # Measure avalanche size
            avalanche_size = spikes.sum().item()
            self.avalanche_history.append(avalanche_size)
            
            # Self-organize toward criticality
            if len(self.avalanche_history) >= 100:
                self._adjust_criticality()
        
        return spikes
    
    def _adjust_criticality(self):
        """Adjust network parameters to maintain criticality."""
        recent_avalanches = list(self.avalanche_history)[-100:]
        avg_size = np.mean(recent_avalanches)
        
        # Adjust branching ratio toward critical point
        if avg_size < 5:  # Subcritical
            self.branching_ratio.data += 0.01
        elif avg_size > 20:  # Supercritical
            self.branching_ratio.data -= 0.01
        
        # Adjust thresholds
        self.thresholds.data += 0.001 * (10 - avg_size)
        self.thresholds.data = torch.clamp(self.thresholds.data, 0.5, 2.0)
    
    def get_criticality_metrics(self) -> Dict[str, float]:
        """Compute metrics of critical dynamics."""
        if len(self.avalanche_history) < 50:
            return {"branching_ratio": 0.0, "avalanche_exponent": 0.0}
        
        sizes = np.array(list(self.avalanche_history))
        non_zero_sizes = sizes[sizes > 0]
        
        if len(non_zero_sizes) < 10:
            return {"branching_ratio": 0.0, "avalanche_exponent": 0.0}
        
        # Power law exponent
        try:
            log_sizes = np.log(non_zero_sizes)
            counts, bins = np.histogram(log_sizes, bins=10)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            valid = counts > 0
            
            if valid.sum() > 3:
                slope, _, r_value, _, _ = stats.linregress(bin_centers[valid], np.log(counts[valid]))
                exponent = -slope
            else:
                exponent = 0.0
        except:
            exponent = 0.0
        
        return {
            "branching_ratio": self.branching_ratio.item(),
            "avalanche_exponent": exponent,
            "mean_avalanche_size": np.mean(non_zero_sizes)
        }


class SelfOrganizingMap(nn.Module):
    """Neuromorphic self-organizing map with spike-based learning."""
    
    def __init__(self, input_dim: int, map_size: Tuple[int, int], 
                 learning_rate: float = 0.1, neighborhood_radius: float = 2.0):
        super().__init__()
        self.input_dim = input_dim
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius
        
        # Initialize weight map
        self.weights = nn.Parameter(torch.randn(*map_size, input_dim) * 0.1)
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(map_size[0], dtype=torch.float),
            torch.arange(map_size[1], dtype=torch.float),
            indexing='ij'
        )
        self.register_buffer('coordinates', torch.stack([y_coords, x_coords], dim=2))
        
        # Spike-based activation
        self.spike_threshold = nn.Parameter(torch.ones(*map_size))
        self.membrane_potential = torch.zeros(*map_size)
        
    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """Process spike input through self-organizing map."""
        batch_size, input_dim = spike_input.shape
        
        # Compute distances to all neurons
        expanded_input = spike_input.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, input_dim)
        expanded_weights = self.weights.unsqueeze(0)  # (1, map_y, map_x, input_dim)
        
        distances = torch.norm(expanded_input - expanded_weights, dim=3)  # (batch, map_y, map_x)
        
        # Find best matching units (BMUs)
        flat_distances = distances.view(batch_size, -1)
        bmu_indices = torch.argmin(flat_distances, dim=1)
        bmu_coords = torch.stack([
            bmu_indices // self.map_size[1],
            bmu_indices % self.map_size[1]
        ], dim=1).float()
        
        # Update membrane potentials
        for i in range(batch_size):
            bmu = bmu_coords[i]
            
            # Calculate neighborhood function
            coord_distances = torch.norm(self.coordinates - bmu, dim=2)
            neighborhood = torch.exp(-coord_distances**2 / (2 * self.neighborhood_radius**2))
            
            # Update weights (Hebbian-like learning)
            weight_delta = self.learning_rate * neighborhood.unsqueeze(2) * (
                spike_input[i] - self.weights
            )
            self.weights.data += weight_delta
            
            # Update membrane potentials
            self.membrane_potential += neighborhood * 0.5
        
        # Generate spikes
        spike_output = (self.membrane_potential > self.spike_threshold).float()
        
        # Reset spiked neurons
        self.membrane_potential[spike_output.bool()] = 0
        
        # Decay membrane potential
        self.membrane_potential *= 0.9
        
        return spike_output
    
    def get_feature_map(self) -> torch.Tensor:
        """Get the learned feature map."""
        return self.weights.detach()


class EmergentPatternDetector(nn.Module):
    """Detector for emergent patterns in spike trains."""
    
    def __init__(self, input_dim: int, pattern_memory_size: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.pattern_memory_size = pattern_memory_size
        
        # Pattern memory bank
        self.pattern_memory = []
        self.pattern_frequencies = defaultdict(int)
        self.novelty_threshold = 0.8
        
        # Spike pattern encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Emergence detector
        self.emergence_detector = nn.LSTM(64, 32, batch_first=True)
        self.emergence_classifier = nn.Linear(32, 1)
        
    def forward(self, spike_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Detect emergent patterns in spike sequences."""
        batch_size, seq_len, input_dim = spike_sequence.shape
        
        # Encode spike patterns
        encoded_patterns = []
        for t in range(seq_len):
            pattern_encoding = self.pattern_encoder(spike_sequence[:, t, :])
            encoded_patterns.append(pattern_encoding)
        
        encoded_sequence = torch.stack(encoded_patterns, dim=1)
        
        # Detect temporal patterns
        lstm_output, _ = self.emergence_detector(encoded_sequence)
        emergence_scores = torch.sigmoid(self.emergence_classifier(lstm_output))
        
        # Analyze pattern novelty
        novelty_scores = self._compute_novelty(encoded_sequence)
        
        # Detect emergent behaviors
        emergent_events = self._detect_emergence(emergence_scores, novelty_scores)
        
        return {
            "emergence_scores": emergence_scores,
            "novelty_scores": novelty_scores,
            "emergent_events": emergent_events,
            "pattern_encodings": encoded_sequence
        }
    
    def _compute_novelty(self, encoded_sequence: torch.Tensor) -> torch.Tensor:
        """Compute novelty scores for encoded patterns."""
        batch_size, seq_len, encoding_dim = encoded_sequence.shape
        novelty_scores = torch.zeros(batch_size, seq_len)
        
        for b in range(batch_size):
            for t in range(seq_len):
                current_pattern = encoded_sequence[b, t].detach().cpu().numpy()
                
                if not self.pattern_memory:
                    novelty_scores[b, t] = 1.0
                else:
                    # Find most similar pattern in memory
                    similarities = []
                    for stored_pattern in self.pattern_memory:
                        similarity = np.dot(current_pattern, stored_pattern) / (
                            np.linalg.norm(current_pattern) * np.linalg.norm(stored_pattern) + 1e-8
                        )
                        similarities.append(similarity)
                    
                    max_similarity = max(similarities)
                    novelty_scores[b, t] = 1.0 - max_similarity
                
                # Store novel patterns
                if novelty_scores[b, t] > self.novelty_threshold:
                    if len(self.pattern_memory) >= self.pattern_memory_size:
                        self.pattern_memory.pop(0)
                    self.pattern_memory.append(current_pattern)
        
        return novelty_scores
    
    def _detect_emergence(self, emergence_scores: torch.Tensor, 
                         novelty_scores: torch.Tensor) -> torch.Tensor:
        """Detect emergent events based on emergence and novelty scores."""
        # Combine emergence and novelty
        combined_scores = 0.7 * emergence_scores.squeeze(-1) + 0.3 * novelty_scores
        
        # Threshold for emergence detection
        emergence_threshold = 0.8
        emergent_events = (combined_scores > emergence_threshold).float()
        
        return emergent_events


class AdaptiveTopologyNetwork(nn.Module):
    """Network with adaptive topology that evolves based on activity."""
    
    def __init__(self, initial_size: int, max_size: int = 2000, 
                 growth_rate: float = 0.01, pruning_threshold: float = 0.01):
        super().__init__()
        self.current_size = initial_size
        self.max_size = max_size
        self.growth_rate = growth_rate
        self.pruning_threshold = pruning_threshold
        
        # Dynamic adjacency matrix
        self.adjacency = nn.Parameter(torch.zeros(max_size, max_size))
        
        # Initialize with random connections
        for i in range(initial_size):
            for j in range(i + 1, initial_size):
                if random.random() < 0.1:
                    self.adjacency.data[i, j] = self.adjacency.data[j, i] = random.random()
        
        # Neuron states
        self.neuron_activities = torch.zeros(max_size)
        self.connection_strengths = torch.zeros(max_size, max_size)
        
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive network."""
        batch_size = input_spikes.size(0)
        
        # Current active network
        active_network = self.adjacency[:self.current_size, :self.current_size]
        
        # Network dynamics
        network_state = torch.zeros(batch_size, self.current_size)
        network_state[:, :input_spikes.size(1)] = input_spikes
        
        # Propagate through network
        for step in range(5):
            network_input = torch.matmul(network_state, active_network)
            network_state = torch.sigmoid(network_state + network_input * 0.1)
            
            # Update activity tracking
            self.neuron_activities[:self.current_size] += network_state.mean(dim=0)
        
        # Adapt topology
        self._adapt_topology()
        
        return network_state
    
    def _adapt_topology(self):
        """Adapt network topology based on activity patterns."""
        # Growth: Add new neurons if activity is high
        if self.current_size < self.max_size:
            avg_activity = self.neuron_activities[:self.current_size].mean()
            
            if avg_activity > 0.8 and random.random() < self.growth_rate:
                new_neuron_idx = self.current_size
                
                # Connect new neuron to highly active neurons
                for i in range(self.current_size):
                    if self.neuron_activities[i] > avg_activity:
                        strength = random.random() * 0.5
                        self.adjacency.data[new_neuron_idx, i] = strength
                        self.adjacency.data[i, new_neuron_idx] = strength
                
                self.current_size += 1
        
        # Pruning: Remove weak connections
        active_adj = self.adjacency[:self.current_size, :self.current_size]
        weak_connections = torch.abs(active_adj) < self.pruning_threshold
        self.adjacency.data[:self.current_size, :self.current_size][weak_connections] = 0
        
        # Decay activity
        self.neuron_activities *= 0.99
    
    def get_topology_metrics(self) -> Dict[str, float]:
        """Compute network topology metrics."""
        active_adj = self.adjacency[:self.current_size, :self.current_size].detach()
        
        # Connection density
        total_possible = self.current_size * (self.current_size - 1) / 2
        actual_connections = (torch.abs(active_adj) > 0).sum().item() / 2
        density = actual_connections / total_possible if total_possible > 0 else 0
        
        # Average path length (approximate)
        try:
            binary_adj = (torch.abs(active_adj) > 0).float()
            path_lengths = []
            for i in range(min(10, self.current_size)):  # Sample for efficiency
                distances = torch.full((self.current_size,), float('inf'))
                distances[i] = 0
                
                for _ in range(self.current_size):
                    new_distances = distances.clone()
                    for j in range(self.current_size):
                        if distances[j] != float('inf'):
                            neighbors = binary_adj[j].nonzero().squeeze()
                            if neighbors.numel() > 0:
                                new_distances[neighbors] = torch.min(
                                    new_distances[neighbors], distances[j] + 1
                                )
                    
                    if torch.equal(distances, new_distances):
                        break
                    distances = new_distances
                
                finite_distances = distances[distances != float('inf')]
                if len(finite_distances) > 1:
                    path_lengths.extend(finite_distances[1:].tolist())
            
            avg_path_length = np.mean(path_lengths) if path_lengths else 0
        except:
            avg_path_length = 0
        
        return {
            "network_size": self.current_size,
            "connection_density": density,
            "average_path_length": avg_path_length,
            "total_activity": self.neuron_activities[:self.current_size].sum().item()
        }


class EmergentIntelligenceFramework:
    """Complete framework for emergent intelligence in neuromorphic systems."""
    
    def __init__(self, config: EmergentConfig):
        self.config = config
        
        # Core components
        self.critical_dynamics = CriticalBrainDynamics(
            config.network_size, config.connection_density
        )
        self.self_organizing_map = SelfOrganizingMap(
            input_dim=64, map_size=(20, 20)
        )
        self.pattern_detector = EmergentPatternDetector(input_dim=64)
        
        if config.adaptive_topology:
            self.adaptive_network = AdaptiveTopologyNetwork(
                initial_size=config.network_size // 2
            )
        
        # Metrics tracking
        self.emergence_history = []
        self.complexity_history = []
        
    def process_input(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Process input through emergent intelligence pipeline."""
        results = {}
        
        # Critical brain dynamics
        critical_output = self.critical_dynamics(input_data)
        results["critical_spikes"] = critical_output
        results["criticality_metrics"] = self.critical_dynamics.get_criticality_metrics()
        
        # Self-organizing feature extraction
        som_output = self.self_organizing_map(input_data)
        results["feature_map"] = som_output
        
        # Pattern detection (requires sequence)
        if input_data.dim() == 3:  # (batch, seq, features)
            pattern_results = self.pattern_detector(input_data)
            results.update(pattern_results)
        
        # Adaptive topology (if enabled)
        if hasattr(self, 'adaptive_network'):
            adaptive_output = self.adaptive_network(input_data)
            results["adaptive_output"] = adaptive_output
            results["topology_metrics"] = self.adaptive_network.get_topology_metrics()
        
        # Compute complexity measures
        complexity = self._compute_complexity(critical_output)
        results["complexity"] = complexity
        
        # Track emergence
        emergence_level = self._assess_emergence(results)
        results["emergence_level"] = emergence_level
        
        self.emergence_history.append(emergence_level)
        self.complexity_history.append(complexity)
        
        return results
    
    def _compute_complexity(self, spike_data: torch.Tensor) -> float:
        """Compute complexity measures of spike patterns."""
        if self.config.complexity_measure == "lempel_ziv":
            return self._lempel_ziv_complexity(spike_data)
        elif self.config.complexity_measure == "entropy":
            return self._shannon_entropy(spike_data)
        else:
            return 0.0
    
    def _lempel_ziv_complexity(self, data: torch.Tensor) -> float:
        """Compute Lempel-Ziv complexity."""
        # Convert to binary string
        binary_data = (data > 0).int().flatten()
        binary_string = ''.join(map(str, binary_data.tolist()))
        
        if len(binary_string) == 0:
            return 0.0
        
        # Lempel-Ziv algorithm
        i, c = 0, 1
        n = len(binary_string)
        
        while i + c <= n:
            substring = binary_string[i:i+c]
            prefix = binary_string[:i+c-1]
            
            if substring in prefix:
                c += 1
            else:
                i += c
                c = 1
        
        # Normalize by theoretical maximum
        max_complexity = n / math.log2(n) if n > 1 else 1
        return (i + c - 1) / max_complexity
    
    def _shannon_entropy(self, data: torch.Tensor) -> float:
        """Compute Shannon entropy."""
        probs = torch.histc(data, bins=10, min=0, max=1)
        probs = probs / probs.sum()
        probs = probs[probs > 0]
        
        if len(probs) <= 1:
            return 0.0
        
        entropy = -(probs * torch.log2(probs)).sum().item()
        return entropy / math.log2(len(probs))  # Normalize
    
    def _assess_emergence(self, results: Dict[str, Any]) -> float:
        """Assess overall emergence level."""
        emergence_indicators = []
        
        # Criticality indicator
        if "criticality_metrics" in results:
            branching_ratio = results["criticality_metrics"]["branching_ratio"]
            # Closer to 1.0 indicates criticality
            criticality_score = 1.0 - abs(branching_ratio - 1.0)
            emergence_indicators.append(criticality_score)
        
        # Pattern novelty
        if "novelty_scores" in results:
            avg_novelty = results["novelty_scores"].mean().item()
            emergence_indicators.append(avg_novelty)
        
        # Complexity
        if "complexity" in results:
            emergence_indicators.append(results["complexity"])
        
        # Network adaptation
        if "topology_metrics" in results:
            density = results["topology_metrics"]["connection_density"]
            # Moderate density indicates good organization
            density_score = 1.0 - abs(density - 0.15) / 0.15
            emergence_indicators.append(max(0, density_score))
        
        if not emergence_indicators:
            return 0.0
        
        return np.mean(emergence_indicators)
    
    def get_emergence_trajectory(self) -> Dict[str, List[float]]:
        """Get trajectory of emergence over time."""
        return {
            "emergence_levels": self.emergence_history.copy(),
            "complexity_levels": self.complexity_history.copy()
        }
    
    def is_emergent_behavior_detected(self) -> bool:
        """Check if emergent behavior is currently detected."""
        if len(self.emergence_history) < 10:
            return False
        
        recent_emergence = np.mean(self.emergence_history[-10:])
        return recent_emergence > self.config.emergence_threshold


# Factory function
def create_emergent_intelligence_system(config: Optional[EmergentConfig] = None) -> EmergentIntelligenceFramework:
    """Create an emergent intelligence framework."""
    if config is None:
        config = EmergentConfig()
    
    logger.info(f"Creating emergent intelligence system with config: {config}")
    
    framework = EmergentIntelligenceFramework(config)
    
    logger.info("Emergent intelligence system created successfully")
    logger.info(f"Network size: {config.network_size}")
    logger.info(f"Emergence threshold: {config.emergence_threshold}")
    
    return framework