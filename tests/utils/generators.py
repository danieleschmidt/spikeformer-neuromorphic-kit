"""Test data generators for neuromorphic testing."""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union
import random


class SpikeTrainGenerator:
    """Generate realistic spike trains for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def poisson_spikes(
        self,
        shape: Tuple[int, ...],
        rate: float = 0.1,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Generate Poisson spike trains.
        
        Args:
            shape: Output tensor shape (batch, time, ...)
            rate: Spike rate (probability per timestep)
            device: Target device
            
        Returns:
            Binary spike tensor
        """
        random_vals = torch.rand(shape, device=device)
        return (random_vals < rate).float()
    
    def regular_spikes(
        self,
        shape: Tuple[int, ...],
        interval: int = 10,
        phase: int = 0,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Generate regular spike trains.
        
        Args:
            shape: Output tensor shape (batch, time, ...)
            interval: Spike interval in timesteps
            phase: Phase offset
            device: Target device
            
        Returns:
            Binary spike tensor
        """
        spikes = torch.zeros(shape, device=device)
        timesteps = shape[1]
        
        for t in range(phase, timesteps, interval):
            spikes[:, t, ...] = 1.0
            
        return spikes
    
    def burst_spikes(
        self,
        shape: Tuple[int, ...],
        burst_length: int = 5,
        burst_interval: int = 20,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Generate burst spike patterns.
        
        Args:
            shape: Output tensor shape (batch, time, ...)
            burst_length: Length of each burst
            burst_interval: Interval between bursts
            device: Target device
            
        Returns:
            Binary spike tensor
        """
        spikes = torch.zeros(shape, device=device)
        timesteps = shape[1]
        
        t = 0
        while t < timesteps:
            # Generate burst
            burst_end = min(t + burst_length, timesteps)
            spikes[:, t:burst_end, ...] = 1.0
            t += burst_interval
            
        return spikes
    
    def gaussian_rate_modulated(
        self,
        shape: Tuple[int, ...],
        base_rate: float = 0.05,
        modulation_strength: float = 0.1,
        modulation_period: int = 50,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Generate rate-modulated spike trains.
        
        Args:
            shape: Output tensor shape (batch, time, ...)
            base_rate: Base firing rate
            modulation_strength: Strength of rate modulation
            modulation_period: Period of modulation
            device: Target device
            
        Returns:
            Binary spike tensor
        """
        batch_size, timesteps = shape[0], shape[1]
        
        # Create rate modulation
        time_indices = torch.arange(timesteps, device=device, dtype=torch.float32)
        rate_modulation = modulation_strength * torch.sin(2 * np.pi * time_indices / modulation_period)
        instantaneous_rate = base_rate + rate_modulation
        instantaneous_rate = torch.clamp(instantaneous_rate, 0.0, 1.0)
        
        # Generate spikes based on modulated rate
        random_vals = torch.rand(shape, device=device)
        rate_expanded = instantaneous_rate.view(1, timesteps, *([1] * (len(shape) - 2)))
        rate_expanded = rate_expanded.expand(shape)
        
        return (random_vals < rate_expanded).float()


class ModelGenerator:
    """Generate test models for neuromorphic testing."""
    
    @staticmethod
    def simple_mlp(
        input_size: int = 784,
        hidden_sizes: List[int] = [128, 64],
        output_size: int = 10,
        activation: str = 'relu'
    ) -> torch.nn.Module:
        """Generate simple MLP for testing.
        
        Args:
            input_size: Input dimension
            hidden_sizes: Hidden layer sizes
            output_size: Output dimension
            activation: Activation function name
            
        Returns:
            MLP model
        """
        layers = []
        prev_size = input_size
        
        activation_fn = {
            'relu': torch.nn.ReLU,
            'tanh': torch.nn.Tanh,
            'sigmoid': torch.nn.Sigmoid
        }[activation]
        
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                activation_fn()
            ])
            prev_size = hidden_size
        
        layers.append(torch.nn.Linear(prev_size, output_size))
        
        return torch.nn.Sequential(*layers)
    
    @staticmethod
    def mini_transformer(
        vocab_size: int = 1000,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_length: int = 32
    ) -> torch.nn.Module:
        """Generate mini transformer for testing.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_seq_length: Maximum sequence length
            
        Returns:
            Transformer model
        """
        class MiniTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, d_model)
                self.pos_encoding = torch.nn.Parameter(torch.randn(max_seq_length, d_model))
                
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True
                )
                self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
                self.classifier = torch.nn.Linear(d_model, vocab_size)
            
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
                x = self.transformer(x)
                return self.classifier(x)
        
        return MiniTransformer()
    
    @staticmethod
    def conv_net(
        input_channels: int = 3,
        num_classes: int = 10,
        architecture: str = 'simple'
    ) -> torch.nn.Module:
        """Generate convolutional network for testing.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            architecture: Architecture type ('simple', 'resnet-like')
            
        Returns:
            CNN model
        """
        if architecture == 'simple':
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 4 * 4, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, num_classes)
            )
        elif architecture == 'resnet-like':
            class ResBlock(torch.nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
                    self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
                    self.relu = torch.nn.ReLU()
                
                def forward(self, x):
                    residual = x
                    x = self.relu(self.conv1(x))
                    x = self.conv2(x)
                    return self.relu(x + residual)
            
            return torch.nn.Sequential(
                torch.nn.Conv2d(input_channels, 32, 7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                ResBlock(32),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                ResBlock(64),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(64, num_classes)
            )


class DatasetGenerator:
    """Generate test datasets for neuromorphic testing."""
    
    @staticmethod
    def synthetic_classification(
        num_samples: int = 1000,
        num_features: int = 20,
        num_classes: int = 5,
        noise_level: float = 0.1,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic classification dataset.
        
        Args:
            num_samples: Number of samples
            num_features: Number of features
            num_classes: Number of classes
            noise_level: Noise level
            device: Target device
            
        Returns:
            (features, labels) tensors
        """
        # Generate class centers
        centers = torch.randn(num_classes, num_features, device=device) * 2
        
        # Generate samples
        samples_per_class = num_samples // num_classes
        features = []
        labels = []
        
        for class_idx in range(num_classes):
            # Generate samples around class center
            class_samples = centers[class_idx].unsqueeze(0) + \
                           torch.randn(samples_per_class, num_features, device=device) * noise_level
            class_labels = torch.full((samples_per_class,), class_idx, device=device)
            
            features.append(class_samples)
            labels.append(class_labels)
        
        # Handle remaining samples
        remaining = num_samples - samples_per_class * num_classes
        if remaining > 0:
            extra_class = torch.randint(0, num_classes, (1,)).item()
            extra_samples = centers[extra_class].unsqueeze(0) + \
                           torch.randn(remaining, num_features, device=device) * noise_level
            extra_labels = torch.full((remaining,), extra_class, device=device)
            features.append(extra_samples)
            labels.append(extra_labels)
        
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        # Shuffle
        indices = torch.randperm(len(features))
        return features[indices], labels[indices]
    
    @staticmethod
    def temporal_pattern_dataset(
        num_samples: int = 500,
        sequence_length: int = 100,
        num_features: int = 10,
        num_patterns: int = 3,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate temporal pattern recognition dataset.
        
        Args:
            num_samples: Number of sequences
            sequence_length: Length of each sequence
            num_features: Number of features per timestep
            num_patterns: Number of different patterns
            device: Target device
            
        Returns:
            (sequences, labels) tensors
        """
        sequences = []
        labels = []
        
        # Define pattern templates
        pattern_templates = []
        for pattern_idx in range(num_patterns):
            # Create distinctive temporal patterns
            template = torch.zeros(sequence_length, num_features, device=device)
            
            # Pattern-specific characteristics
            if pattern_idx == 0:  # Rising pattern
                for t in range(sequence_length):
                    template[t, :] = torch.sigmoid(torch.tensor(t / sequence_length * 6 - 3))
            elif pattern_idx == 1:  # Oscillatory pattern
                for t in range(sequence_length):
                    template[t, :] = torch.sin(torch.tensor(t * 2 * np.pi / sequence_length * 3))
            else:  # Decaying pattern
                for t in range(sequence_length):
                    template[t, :] = torch.exp(torch.tensor(-t / sequence_length * 3))
            
            pattern_templates.append(template)
        
        # Generate samples
        samples_per_pattern = num_samples // num_patterns
        
        for pattern_idx in range(num_patterns):
            for _ in range(samples_per_pattern):
                # Add noise to template
                noise = torch.randn_like(pattern_templates[pattern_idx]) * 0.1
                sequence = pattern_templates[pattern_idx] + noise
                
                sequences.append(sequence)
                labels.append(pattern_idx)
        
        # Handle remaining samples
        remaining = num_samples - samples_per_pattern * num_patterns
        for _ in range(remaining):
            pattern_idx = torch.randint(0, num_patterns, (1,)).item()
            noise = torch.randn_like(pattern_templates[pattern_idx]) * 0.1
            sequence = pattern_templates[pattern_idx] + noise
            
            sequences.append(sequence)
            labels.append(pattern_idx)
        
        sequences = torch.stack(sequences)
        labels = torch.tensor(labels, device=device)
        
        # Shuffle
        indices = torch.randperm(len(sequences))
        return sequences[indices], labels[indices]
    
    @staticmethod
    def spike_pattern_dataset(
        num_samples: int = 200,
        timesteps: int = 50,
        num_neurons: int = 20,
        spike_rate: float = 0.2,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate spike pattern classification dataset.
        
        Args:
            num_samples: Number of spike pattern samples
            timesteps: Number of timesteps
            num_neurons: Number of neurons
            spike_rate: Base spike rate
            device: Target device
            
        Returns:
            (spike_patterns, labels) tensors
        """
        spike_gen = SpikeTrainGenerator()
        
        patterns = []
        labels = []
        
        # Generate different spike pattern types
        pattern_types = ['poisson', 'regular', 'burst']
        samples_per_type = num_samples // len(pattern_types)
        
        for type_idx, pattern_type in enumerate(pattern_types):
            for _ in range(samples_per_type):
                shape = (1, timesteps, num_neurons)
                
                if pattern_type == 'poisson':
                    pattern = spike_gen.poisson_spikes(shape, spike_rate, device)
                elif pattern_type == 'regular':
                    pattern = spike_gen.regular_spikes(shape, interval=int(1/spike_rate), device=device)
                else:  # burst
                    pattern = spike_gen.burst_spikes(shape, burst_length=5, burst_interval=15, device=device)
                
                patterns.append(pattern.squeeze(0))
                labels.append(type_idx)
        
        # Handle remaining samples
        remaining = num_samples - samples_per_type * len(pattern_types)
        for _ in range(remaining):
            type_idx = torch.randint(0, len(pattern_types), (1,)).item()
            pattern_type = pattern_types[type_idx]
            shape = (1, timesteps, num_neurons)
            
            if pattern_type == 'poisson':
                pattern = spike_gen.poisson_spikes(shape, spike_rate, device)
            elif pattern_type == 'regular':
                pattern = spike_gen.regular_spikes(shape, interval=int(1/spike_rate), device=device)
            else:  # burst
                pattern = spike_gen.burst_spikes(shape, burst_length=5, burst_interval=15, device=device)
            
            patterns.append(pattern.squeeze(0))
            labels.append(type_idx)
        
        patterns = torch.stack(patterns)
        labels = torch.tensor(labels, device=device)
        
        # Shuffle
        indices = torch.randperm(len(patterns))
        return patterns[indices], labels[indices]


class ConfigurationGenerator:
    """Generate test configurations for neuromorphic testing."""
    
    @staticmethod
    def conversion_configs() -> List[Dict[str, Any]]:
        """Generate various conversion configurations for testing."""
        configs = []
        
        # Basic configurations
        for timesteps in [10, 32, 100]:
            for threshold in [0.5, 1.0, 2.0]:
                for neuron_model in ['LIF', 'PLIF']:
                    configs.append({
                        'timesteps': timesteps,
                        'threshold': threshold,
                        'neuron_model': neuron_model,
                        'spike_encoding': 'rate',
                        'surrogate_gradient': 'fast_sigmoid'
                    })
        
        # Advanced configurations
        configs.extend([
            {
                'timesteps': 50,
                'threshold': 1.0,
                'neuron_model': 'AdLIF',
                'spike_encoding': 'temporal',
                'surrogate_gradient': 'super_spike',
                'adaptive_threshold': True,
                'adaptation_rate': 0.1
            },
            {
                'timesteps': 200,
                'threshold': 0.8,
                'neuron_model': 'LIF',
                'spike_encoding': 'poisson',
                'surrogate_gradient': 'straight_through',
                'noise_level': 0.01,
                'dropout_rate': 0.1
            }
        ])
        
        return configs
    
    @staticmethod
    def hardware_configs() -> List[Dict[str, Any]]:
        """Generate hardware deployment configurations for testing."""
        return [
            {
                'hardware': 'loihi2',
                'num_chips': 1,
                'core_allocation': 'automatic',
                'precision': 8,
                'max_fanin': 64,
                'max_fanout': 128,
                'routing_algorithm': 'shortest_path'
            },
            {
                'hardware': 'loihi2',
                'num_chips': 4,
                'core_allocation': 'manual',
                'precision': 16,
                'max_fanin': 32,
                'max_fanout': 64,
                'routing_algorithm': 'minimum_congestion'
            },
            {
                'hardware': 'spinnaker',
                'num_boards': 1,
                'routing_algorithm': 'neighbor_aware',
                'time_scale_factor': 1000,
                'packet_ordering': True,
                'live_output': False
            },
            {
                'hardware': 'cpu_simulation',
                'precision': 'float32',
                'optimization_level': 2,
                'parallel_processing': True,
                'memory_efficient': True
            }
        ]
    
    @staticmethod
    def benchmark_configs() -> List[Dict[str, Any]]:
        """Generate benchmark configurations for testing."""
        return [
            {
                'model_size': 'tiny',
                'batch_size': 1,
                'num_runs': 10,
                'warmup_runs': 2,
                'measure_energy': True,
                'measure_latency': True,
                'measure_accuracy': True
            },
            {
                'model_size': 'small',
                'batch_size': 8,
                'num_runs': 5,
                'warmup_runs': 1,
                'measure_energy': True,
                'measure_latency': True,
                'measure_accuracy': True
            },
            {
                'model_size': 'base',
                'batch_size': 32,
                'num_runs': 3,
                'warmup_runs': 1,
                'measure_energy': False,
                'measure_latency': True,
                'measure_accuracy': True
            }
        ]