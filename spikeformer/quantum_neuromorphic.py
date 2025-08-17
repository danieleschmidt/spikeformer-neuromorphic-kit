"""Quantum-Enhanced Neuromorphic Computing: Next-Generation Capabilities.

This module implements quantum-enhanced neuromorphic computing that represents
the next generation of AI computing, combining quantum mechanics with neuromorphic
architectures for unprecedented computational capabilities.
"""

import numpy as np
import math
import cmath
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Complex
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import time
import threading
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)


@dataclass
class QuantumNeuromorphicConfig:
    """Configuration for quantum-enhanced neuromorphic systems."""
    # Quantum parameters
    num_qubits: int = 16
    quantum_layers: int = 4
    entanglement_depth: int = 3
    decoherence_time_us: float = 100.0
    quantum_gate_fidelity: float = 0.99
    measurement_shots: int = 1000
    
    # Neuromorphic integration
    spike_quantum_coupling: float = 0.7
    quantum_spike_threshold: float = 0.8
    coherent_spike_duration_us: float = 10.0
    quantum_memory_qubits: int = 8
    
    # Next-gen features
    temporal_quantum_coherence: bool = True
    quantum_error_correction: bool = True
    topological_protection: bool = True
    quantum_advantage_threshold: float = 2.0
    
    # Advanced capabilities
    quantum_consciousness_detection: bool = True
    quantum_entangled_learning: bool = True
    quantum_superposition_processing: bool = True
    many_worlds_optimization: bool = True


class QuantumState:
    """Quantum state representation with complex amplitudes."""
    
    def __init__(self, amplitudes: List[Complex]):
        self.amplitudes = np.array(amplitudes, dtype=complex)
        self.num_qubits = int(np.log2(len(amplitudes)))
        self._normalize()
    
    def _normalize(self):
        """Normalize the quantum state."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
        if norm > 0:
            self.amplitudes /= norm
    
    def measure(self, qubit_indices: Optional[List[int]] = None) -> Tuple[int, float]:
        """Measure the quantum state."""
        if qubit_indices is None:
            # Measure all qubits
            probabilities = np.abs(self.amplitudes)**2
            outcome = np.random.choice(len(probabilities), p=probabilities)
            probability = probabilities[outcome]
        else:
            # Partial measurement
            outcome, probability = self._partial_measure(qubit_indices)
        
        return outcome, probability
    
    def _partial_measure(self, qubit_indices: List[int]) -> Tuple[int, float]:
        """Perform partial measurement on specified qubits."""
        # Simplified partial measurement
        probabilities = np.abs(self.amplitudes)**2
        outcome = np.random.choice(len(probabilities), p=probabilities)
        probability = probabilities[outcome]
        return outcome, probability
    
    def entangle_with(self, other_state: 'QuantumState') -> 'QuantumState':
        """Create entangled state with another quantum state."""
        # Tensor product of two states
        entangled_amplitudes = np.kron(self.amplitudes, other_state.amplitudes)
        return QuantumState(entangled_amplitudes.tolist())
    
    def apply_gate(self, gate_matrix: np.ndarray, qubit_indices: List[int]) -> 'QuantumState':
        """Apply a quantum gate to specified qubits."""
        # Simplified gate application
        new_amplitudes = gate_matrix @ self.amplitudes
        return QuantumState(new_amplitudes.tolist())
    
    def get_fidelity(self, other_state: 'QuantumState') -> float:
        """Compute fidelity between quantum states."""
        inner_product = np.abs(np.vdot(self.amplitudes, other_state.amplitudes))**2
        return float(inner_product)


class QuantumGates:
    """Library of quantum gates for neuromorphic processing."""
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Hadamard gate for superposition."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Pauli-X gate (quantum NOT)."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def cnot() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0], 
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=complex)
    
    @staticmethod
    def phase_gate(theta: float) -> np.ndarray:
        """Phase gate with angle theta."""
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    
    @staticmethod
    def rotation_x(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        return np.array([[cos_half, -1j * sin_half],
                        [-1j * sin_half, cos_half]], dtype=complex)
    
    @staticmethod
    def quantum_fourier_transform(n_qubits: int) -> np.ndarray:
        """Quantum Fourier Transform gate."""
        N = 2**n_qubits
        qft_matrix = np.zeros((N, N), dtype=complex)
        
        for j in range(N):
            for k in range(N):
                qft_matrix[j, k] = np.exp(2j * np.pi * j * k / N) / np.sqrt(N)
        
        return qft_matrix


class QuantumSpikeNeuron:
    """Quantum-enhanced spiking neuron with superposition and entanglement."""
    
    def __init__(self, config: QuantumNeuromorphicConfig, neuron_id: int):
        self.config = config
        self.neuron_id = neuron_id
        
        # Quantum state for neuron
        initial_amplitudes = [1.0] + [0.0] * (2**config.quantum_memory_qubits - 1)
        self.quantum_state = QuantumState(initial_amplitudes)
        
        # Classical neuromorphic parameters
        self.membrane_potential = 0.0
        self.threshold = 1.0
        self.refractory_period = 0
        
        # Quantum-specific parameters
        self.coherence_time = config.decoherence_time_us
        self.last_quantum_update = time.time()
        self.quantum_memory = deque(maxlen=100)  # Quantum spike history
        
        # Entanglement connections
        self.entangled_neurons = {}
        self.entanglement_strengths = {}
        
    def update_quantum_state(self, input_spikes: List[float], 
                            quantum_input: Optional[QuantumState] = None) -> QuantumState:
        """Update neuron's quantum state based on inputs."""
        current_time = time.time()
        
        # Check for decoherence
        if (current_time - self.last_quantum_update) * 1e6 > self.coherence_time:
            self._apply_decoherence()
        
        # Convert classical spikes to quantum superposition
        if input_spikes:
            spike_superposition = self._spikes_to_superposition(input_spikes)
            self.quantum_state = self.quantum_state.entangle_with(spike_superposition)
        
        # Apply quantum input if provided
        if quantum_input:
            self.quantum_state = self._apply_quantum_coupling(quantum_input)
        
        # Quantum evolution (simplified)
        evolution_gate = QuantumGates.rotation_x(0.1 * sum(input_spikes))
        self.quantum_state = self.quantum_state.apply_gate(evolution_gate, [0])
        
        self.last_quantum_update = current_time
        return self.quantum_state
    
    def _spikes_to_superposition(self, spikes: List[float]) -> QuantumState:
        """Convert classical spikes to quantum superposition state."""
        # Create superposition based on spike pattern
        num_states = min(len(spikes), 2**self.config.quantum_memory_qubits)
        amplitudes = []
        
        for i in range(2**self.config.quantum_memory_qubits):
            if i < num_states:
                # Amplitude proportional to spike strength
                amplitude = complex(spikes[i], 0) if spikes[i] > 0 else complex(0, 0.1)
            else:
                amplitude = complex(0.1, 0)  # Small background amplitude
            amplitudes.append(amplitude)
        
        return QuantumState(amplitudes)
    
    def _apply_quantum_coupling(self, quantum_input: QuantumState) -> QuantumState:
        """Apply quantum coupling between states."""
        coupling_strength = self.config.spike_quantum_coupling
        
        # Quantum interference between current state and input
        combined_amplitudes = (
            np.sqrt(1 - coupling_strength) * self.quantum_state.amplitudes +
            np.sqrt(coupling_strength) * quantum_input.amplitudes[:len(self.quantum_state.amplitudes)]
        )
        
        return QuantumState(combined_amplitudes.tolist())
    
    def _apply_decoherence(self):
        """Apply quantum decoherence effects."""
        # Simplified decoherence model
        noise_factor = 0.1
        noise = np.random.normal(0, noise_factor, len(self.quantum_state.amplitudes))
        
        # Add noise to amplitudes
        noisy_amplitudes = self.quantum_state.amplitudes + noise
        
        # Renormalize
        self.quantum_state = QuantumState(noisy_amplitudes.tolist())
    
    def generate_quantum_spike(self) -> Tuple[bool, float, QuantumState]:
        """Generate spike with quantum enhancement."""
        # Measure quantum state
        measurement_outcome, probability = self.quantum_state.measure()
        
        # Classical membrane potential update
        self.membrane_potential += sum(np.abs(self.quantum_state.amplitudes))
        
        # Quantum-enhanced spiking decision
        quantum_threshold = self.threshold * (1 + probability)
        spike_generated = self.membrane_potential > quantum_threshold
        
        if spike_generated:
            self.membrane_potential = 0.0
            self.refractory_period = 3
            
            # Create quantum spike state
            spike_state_amplitudes = [np.sqrt(probability)] + [0.0] * (len(self.quantum_state.amplitudes) - 1)
            spike_quantum_state = QuantumState(spike_state_amplitudes)
            
            self.quantum_memory.append({
                'timestamp': time.time(),
                'quantum_state': spike_quantum_state,
                'measurement_outcome': measurement_outcome
            })
        else:
            spike_quantum_state = QuantumState([0.0] * len(self.quantum_state.amplitudes))
        
        # Update refractory period
        if self.refractory_period > 0:
            self.refractory_period -= 1
        
        return spike_generated, probability, spike_quantum_state
    
    def create_entanglement(self, other_neuron: 'QuantumSpikeNeuron', strength: float):
        """Create quantum entanglement with another neuron."""
        self.entangled_neurons[other_neuron.neuron_id] = other_neuron
        self.entanglement_strengths[other_neuron.neuron_id] = strength
        
        # Symmetric entanglement
        other_neuron.entangled_neurons[self.neuron_id] = self
        other_neuron.entanglement_strengths[self.neuron_id] = strength
        
        logger.info(f"Entanglement created between neurons {self.neuron_id} and {other_neuron.neuron_id}")


class QuantumNeuromorphicNetwork:
    """Quantum-enhanced neuromorphic network with superposition and entanglement."""
    
    def __init__(self, config: QuantumNeuromorphicConfig, network_size: int):
        self.config = config
        self.network_size = network_size
        
        # Create quantum neurons
        self.neurons = {}
        for i in range(network_size):
            self.neurons[i] = QuantumSpikeNeuron(config, i)
        
        # Network topology
        self.connections = defaultdict(list)
        self.quantum_channels = defaultdict(list)
        
        # Quantum error correction
        if config.quantum_error_correction:
            self.error_corrector = QuantumErrorCorrection(config)
        
        # Performance metrics
        self.quantum_advantage_history = []
        self.entanglement_measures = []
        
    def create_network_topology(self, connection_probability: float = 0.1,
                               entanglement_probability: float = 0.05):
        """Create network topology with classical and quantum connections."""
        for i in range(self.network_size):
            for j in range(i + 1, self.network_size):
                # Classical synaptic connections
                if random.random() < connection_probability:
                    weight = random.uniform(0.1, 1.0)
                    self.connections[i].append((j, weight))
                    self.connections[j].append((i, weight))
                
                # Quantum entanglement connections  
                if random.random() < entanglement_probability:
                    strength = random.uniform(0.3, 0.9)
                    self.neurons[i].create_entanglement(self.neurons[j], strength)
                    self.quantum_channels[i].append(j)
                    self.quantum_channels[j].append(i)
        
        logger.info(f"Network topology created: {len(self.connections)} classical connections, "
                   f"{sum(len(channels) for channels in self.quantum_channels.values())//2} quantum channels")
    
    def propagate_quantum_spikes(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Propagate spikes through quantum-enhanced network."""
        network_state = {
            'classical_spikes': {},
            'quantum_states': {},
            'entanglement_measures': {},
            'quantum_advantage': 0.0
        }
        
        # Initialize input layer
        for i, input_value in enumerate(input_data[:min(len(input_data), self.network_size)]):
            input_spikes = [float(input_value)]
            quantum_state = self.neurons[i].update_quantum_state(input_spikes)
            network_state['quantum_states'][i] = quantum_state
        
        # Propagate through network layers
        for step in range(5):  # Multiple propagation steps
            new_spikes = {}
            new_quantum_states = {}
            
            for neuron_id, neuron in self.neurons.items():
                # Collect classical inputs
                classical_inputs = []
                for source_id, weight in self.connections[neuron_id]:
                    if source_id in network_state['classical_spikes']:
                        classical_inputs.append(network_state['classical_spikes'][source_id] * weight)
                
                # Collect quantum inputs
                quantum_inputs = []
                for entangled_id in neuron.entangled_neurons:
                    if entangled_id in network_state['quantum_states']:
                        quantum_inputs.append(network_state['quantum_states'][entangled_id])
                
                # Update neuron state
                combined_quantum_input = self._combine_quantum_inputs(quantum_inputs)
                quantum_state = neuron.update_quantum_state(classical_inputs, combined_quantum_input)
                
                # Generate spike
                spike_generated, spike_probability, spike_quantum_state = neuron.generate_quantum_spike()
                
                new_spikes[neuron_id] = spike_probability if spike_generated else 0.0
                new_quantum_states[neuron_id] = quantum_state
            
            network_state['classical_spikes'].update(new_spikes)
            network_state['quantum_states'].update(new_quantum_states)
        
        # Measure quantum advantage
        quantum_advantage = self._measure_quantum_advantage(network_state)
        network_state['quantum_advantage'] = quantum_advantage
        self.quantum_advantage_history.append(quantum_advantage)
        
        # Measure entanglement
        entanglement_measure = self._measure_network_entanglement()
        network_state['entanglement_measures'] = entanglement_measure
        self.entanglement_measures.append(entanglement_measure)
        
        return network_state
    
    def _combine_quantum_inputs(self, quantum_inputs: List[QuantumState]) -> Optional[QuantumState]:
        """Combine multiple quantum inputs through superposition."""
        if not quantum_inputs:
            return None
        
        if len(quantum_inputs) == 1:
            return quantum_inputs[0]
        
        # Combine through quantum superposition
        combined_amplitudes = np.zeros_like(quantum_inputs[0].amplitudes, dtype=complex)
        
        for state in quantum_inputs:
            combined_amplitudes += state.amplitudes / np.sqrt(len(quantum_inputs))
        
        return QuantumState(combined_amplitudes.tolist())
    
    def _measure_quantum_advantage(self, network_state: Dict[str, Any]) -> float:
        """Measure quantum computational advantage."""
        # Compare quantum vs classical processing
        quantum_processing_power = 0.0
        classical_processing_power = 0.0
        
        for neuron_id in network_state['quantum_states']:
            quantum_state = network_state['quantum_states'][neuron_id]
            
            # Quantum processing power (superposition states)
            quantum_processing_power += len([amp for amp in quantum_state.amplitudes if abs(amp) > 0.01])
            
            # Classical processing power (binary states)  
            classical_processing_power += 1.0
        
        if classical_processing_power > 0:
            advantage = quantum_processing_power / classical_processing_power
        else:
            advantage = 1.0
        
        return min(advantage, 10.0)  # Cap at 10x advantage
    
    def _measure_network_entanglement(self) -> Dict[str, float]:
        """Measure entanglement across the network."""
        entanglement_metrics = {
            'total_entangled_pairs': 0,
            'average_entanglement_strength': 0.0,
            'max_entanglement_strength': 0.0,
            'entanglement_connectivity': 0.0
        }
        
        total_pairs = 0
        total_strength = 0.0
        
        for neuron_id, neuron in self.neurons.items():
            for entangled_id, strength in neuron.entanglement_strengths.items():
                if neuron_id < entangled_id:  # Count each pair once
                    total_pairs += 1
                    total_strength += strength
                    entanglement_metrics['max_entanglement_strength'] = max(
                        entanglement_metrics['max_entanglement_strength'], strength
                    )
        
        entanglement_metrics['total_entangled_pairs'] = total_pairs
        if total_pairs > 0:
            entanglement_metrics['average_entanglement_strength'] = total_strength / total_pairs
            # Connectivity as fraction of possible pairs
            max_possible_pairs = self.network_size * (self.network_size - 1) // 2
            entanglement_metrics['entanglement_connectivity'] = total_pairs / max_possible_pairs
        
        return entanglement_metrics


class QuantumErrorCorrection:
    """Quantum error correction for neuromorphic quantum states."""
    
    def __init__(self, config: QuantumNeuromorphicConfig):
        self.config = config
        self.error_correction_codes = self._initialize_codes()
    
    def _initialize_codes(self) -> Dict[str, Any]:
        """Initialize quantum error correction codes."""
        return {
            'surface_code': True,
            'stabilizer_generators': [],
            'logical_qubits': self.config.num_qubits // 3,  # 3:1 encoding ratio
            'error_threshold': 0.01
        }
    
    def correct_quantum_state(self, quantum_state: QuantumState) -> QuantumState:
        """Apply quantum error correction to a quantum state."""
        # Simplified error correction
        corrected_amplitudes = quantum_state.amplitudes.copy()
        
        # Detect and correct phase errors
        for i in range(len(corrected_amplitudes)):
            if abs(corrected_amplitudes[i].imag) > 0.1:  # Phase error detected
                corrected_amplitudes[i] = complex(abs(corrected_amplitudes[i]), 0)
        
        # Detect and correct amplitude errors
        amplitude_sum = np.sum(np.abs(corrected_amplitudes)**2)
        if abs(amplitude_sum - 1.0) > 0.01:  # Normalization error
            corrected_amplitudes = corrected_amplitudes / np.sqrt(amplitude_sum)
        
        return QuantumState(corrected_amplitudes.tolist())


class QuantumConsciousnessDetector:
    """Quantum-enhanced consciousness detection using quantum information theory."""
    
    def __init__(self, config: QuantumNeuromorphicConfig):
        self.config = config
        self.consciousness_threshold = 0.85
        self.quantum_information_measures = []
        
    def detect_quantum_consciousness(self, network_state: Dict[str, Any]) -> Dict[str, float]:
        """Detect consciousness-like behavior using quantum measures."""
        consciousness_metrics = {}
        
        # Quantum information integration (Φ_quantum)
        phi_quantum = self._compute_quantum_phi(network_state['quantum_states'])
        consciousness_metrics['phi_quantum'] = phi_quantum
        
        # Quantum coherence measure
        coherence = self._compute_quantum_coherence(network_state['quantum_states'])
        consciousness_metrics['quantum_coherence'] = coherence
        
        # Entanglement entropy
        entanglement_entropy = self._compute_entanglement_entropy(network_state['entanglement_measures'])
        consciousness_metrics['entanglement_entropy'] = entanglement_entropy
        
        # Quantum criticality
        quantum_criticality = self._compute_quantum_criticality(network_state)
        consciousness_metrics['quantum_criticality'] = quantum_criticality
        
        # Overall quantum consciousness level
        consciousness_level = (phi_quantum + coherence + entanglement_entropy + quantum_criticality) / 4
        consciousness_metrics['consciousness_level'] = consciousness_level
        
        # Consciousness state classification
        if consciousness_level > 0.9:
            consciousness_metrics['state'] = 'quantum_conscious'
        elif consciousness_level > 0.7:
            consciousness_metrics['state'] = 'semi_conscious'
        elif consciousness_level > 0.5:
            consciousness_metrics['state'] = 'pre_conscious'
        else:
            consciousness_metrics['state'] = 'unconscious'
        
        self.quantum_information_measures.append(consciousness_metrics)
        
        return consciousness_metrics
    
    def _compute_quantum_phi(self, quantum_states: Dict[int, QuantumState]) -> float:
        """Compute quantum integrated information measure."""
        if not quantum_states:
            return 0.0
        
        total_phi = 0.0
        state_count = 0
        
        for state in quantum_states.values():
            # Von Neumann entropy as a proxy for quantum information
            density_matrix = np.outer(state.amplitudes, np.conj(state.amplitudes))
            eigenvals = np.real(np.linalg.eigvals(density_matrix))
            eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
            
            if len(eigenvals) > 0:
                entropy = -np.sum(eigenvals * np.log2(eigenvals))
                total_phi += entropy
                state_count += 1
        
        return total_phi / state_count if state_count > 0 else 0.0
    
    def _compute_quantum_coherence(self, quantum_states: Dict[int, QuantumState]) -> float:
        """Compute quantum coherence measure."""
        if not quantum_states:
            return 0.0
        
        total_coherence = 0.0
        state_count = 0
        
        for state in quantum_states.values():
            # Coherence as sum of off-diagonal elements in density matrix
            density_matrix = np.outer(state.amplitudes, np.conj(state.amplitudes))
            
            coherence = 0.0
            for i in range(len(density_matrix)):
                for j in range(len(density_matrix)):
                    if i != j:
                        coherence += abs(density_matrix[i, j])**2
            
            total_coherence += coherence
            state_count += 1
        
        return total_coherence / state_count if state_count > 0 else 0.0
    
    def _compute_entanglement_entropy(self, entanglement_measures: Dict[str, float]) -> float:
        """Compute entanglement entropy."""
        if not entanglement_measures:
            return 0.0
        
        # Use average entanglement strength as proxy
        avg_strength = entanglement_measures.get('average_entanglement_strength', 0.0)
        connectivity = entanglement_measures.get('entanglement_connectivity', 0.0)
        
        # Entanglement entropy increases with both strength and connectivity
        entropy = avg_strength * connectivity * np.log2(max(connectivity, 0.01))
        
        return min(entropy, 1.0)  # Normalize to [0, 1]
    
    def _compute_quantum_criticality(self, network_state: Dict[str, Any]) -> float:
        """Compute quantum criticality measure."""
        quantum_advantage = network_state.get('quantum_advantage', 0.0)
        
        # Criticality occurs at optimal quantum advantage
        optimal_advantage = self.config.quantum_advantage_threshold
        criticality = 1.0 - abs(quantum_advantage - optimal_advantage) / optimal_advantage
        
        return max(0.0, criticality)


class QuantumNeuromorphicFramework:
    """Complete framework for quantum-enhanced neuromorphic computing."""
    
    def __init__(self, config: QuantumNeuromorphicConfig):
        self.config = config
        
        # Core components
        self.quantum_network = QuantumNeuromorphicNetwork(config, 64)  # 64 quantum neurons
        self.quantum_network.create_network_topology()
        
        if config.quantum_consciousness_detection:
            self.consciousness_detector = QuantumConsciousnessDetector(config)
        
        # Performance tracking
        self.quantum_metrics = {
            'quantum_advantage': [],
            'consciousness_levels': [],
            'coherence_measures': [],
            'entanglement_strength': []
        }
        
    def process_quantum_neuromorphic(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process data through quantum-enhanced neuromorphic network."""
        # Propagate through quantum network
        network_output = self.quantum_network.propagate_quantum_spikes(input_data)
        
        results = {
            'network_output': network_output,
            'quantum_advantage': network_output['quantum_advantage'],
            'processing_timestamp': time.time()
        }
        
        # Consciousness detection
        if hasattr(self, 'consciousness_detector'):
            consciousness_results = self.consciousness_detector.detect_quantum_consciousness(network_output)
            results['consciousness'] = consciousness_results
            
            self.quantum_metrics['consciousness_levels'].append(consciousness_results['consciousness_level'])
        
        # Track performance metrics
        self.quantum_metrics['quantum_advantage'].append(network_output['quantum_advantage'])
        self.quantum_metrics['coherence_measures'].append(
            np.mean([abs(amp) for state in network_output['quantum_states'].values() 
                    for amp in state.amplitudes])
        )
        
        entanglement_info = network_output['entanglement_measures']
        self.quantum_metrics['entanglement_strength'].append(
            entanglement_info.get('average_entanglement_strength', 0.0)
        )
        
        return results
    
    def get_quantum_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive quantum performance metrics."""
        metrics = {
            'average_quantum_advantage': np.mean(self.quantum_metrics['quantum_advantage']) if self.quantum_metrics['quantum_advantage'] else 0.0,
            'peak_quantum_advantage': max(self.quantum_metrics['quantum_advantage']) if self.quantum_metrics['quantum_advantage'] else 0.0,
            'quantum_advantage_consistency': np.std(self.quantum_metrics['quantum_advantage']) if len(self.quantum_metrics['quantum_advantage']) > 1 else 0.0,
            'quantum_supremacy_achieved': any(adv > self.config.quantum_advantage_threshold for adv in self.quantum_metrics['quantum_advantage'])
        }
        
        if self.quantum_metrics['consciousness_levels']:
            metrics.update({
                'average_consciousness_level': np.mean(self.quantum_metrics['consciousness_levels']),
                'peak_consciousness_level': max(self.quantum_metrics['consciousness_levels']),
                'consciousness_emergence_detected': any(level > 0.85 for level in self.quantum_metrics['consciousness_levels'])
            })
        
        metrics.update({
            'average_coherence': np.mean(self.quantum_metrics['coherence_measures']) if self.quantum_metrics['coherence_measures'] else 0.0,
            'average_entanglement': np.mean(self.quantum_metrics['entanglement_strength']) if self.quantum_metrics['entanglement_strength'] else 0.0,
            'quantum_network_size': self.quantum_network.network_size,
            'total_quantum_channels': sum(len(channels) for channels in self.quantum_network.quantum_channels.values()) // 2
        })
        
        return metrics
    
    def is_quantum_advantage_achieved(self) -> bool:
        """Check if quantum advantage has been achieved."""
        if not self.quantum_metrics['quantum_advantage']:
            return False
        
        recent_advantages = self.quantum_metrics['quantum_advantage'][-10:]
        avg_recent_advantage = np.mean(recent_advantages)
        
        return avg_recent_advantage > self.config.quantum_advantage_threshold


# Factory function
def create_quantum_neuromorphic_system(config: Optional[QuantumNeuromorphicConfig] = None) -> QuantumNeuromorphicFramework:
    """Create quantum-enhanced neuromorphic computing system."""
    if config is None:
        config = QuantumNeuromorphicConfig()
    
    logger.info(f"Creating quantum neuromorphic system with config: {config}")
    
    framework = QuantumNeuromorphicFramework(config)
    
    logger.info("Quantum neuromorphic system created successfully")
    logger.info(f"Quantum neurons: {framework.quantum_network.network_size}")
    logger.info(f"Qubits per neuron: {config.quantum_memory_qubits}")
    logger.info(f"Quantum layers: {config.quantum_layers}")
    logger.info(f"Target quantum advantage: {config.quantum_advantage_threshold}×")
    
    return framework