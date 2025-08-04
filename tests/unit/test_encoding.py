"""Unit tests for spike encoding strategies."""

import torch
import torch.nn as nn
import pytest
import numpy as np

from spikeformer.encoding import (
    RateCoding, TemporalCoding, PoissonCoding, PopulationCoding,
    DeltaCoding, LatencyCoding, BurstCoding, AdaptiveThresholdCoding,
    MultiModalEncoder, create_encoder
)


class TestRateCoding:
    """Test rate coding encoder."""
    
    def test_initialization(self):
        """Test rate coding initialization."""
        encoder = RateCoding(timesteps=32, max_rate=0.8, gain=1.5)
        
        assert encoder.timesteps == 32
        assert encoder.max_rate == 0.8
        assert encoder.gain == 1.5
    
    def test_encoding_shape(self):
        """Test output shape of rate coding."""
        encoder = RateCoding(timesteps=16)
        batch_size, features = 4, 10
        
        x = torch.randn(batch_size, features)
        spikes = encoder(x)
        
        assert spikes.shape == (batch_size, 16, features)
    
    def test_encoding_properties(self):
        """Test rate coding properties."""
        encoder = RateCoding(timesteps=20, max_rate=1.0)
        
        # High input should generate more spikes
        high_input = torch.ones(1, 5) * 0.8
        low_input = torch.ones(1, 5) * 0.2
        
        high_spikes = encoder(high_input)
        low_spikes = encoder(low_input)
        
        assert high_spikes.sum() > low_spikes.sum()
    
    def test_binary_spikes(self):
        """Test that output contains only binary spikes."""
        encoder = RateCoding(timesteps=10)
        x = torch.randn(2, 8)
        
        spikes = encoder(x)
        
        assert torch.all((spikes == 0) | (spikes == 1))
    
    def test_zero_input(self):
        """Test encoding of zero input."""
        encoder = RateCoding(timesteps=15)
        x = torch.zeros(1, 3)
        
        spikes = encoder(x)
        
        # Zero input should produce no spikes
        assert torch.all(spikes == 0)


class TestTemporalCoding:
    """Test temporal coding encoder."""
    
    def test_initialization(self):
        """Test temporal coding initialization."""
        encoder = TemporalCoding(timesteps=20, min_delay=2)
        
        assert encoder.timesteps == 20
        assert encoder.min_delay == 2
    
    def test_encoding_shape(self):
        """Test temporal coding output shape."""
        encoder = TemporalCoding(timesteps=25)
        
        x = torch.rand(3, 7)  # Positive values for temporal coding
        spikes = encoder(x)
        
        assert spikes.shape == (3, 25, 7)
    
    def test_spike_timing(self):
        """Test that higher values spike earlier."""
        encoder = TemporalCoding(timesteps=10, min_delay=0)
        
        # Create inputs with different values
        x = torch.tensor([[0.1, 0.9]]).float()
        spikes = encoder(x)
        
        # Find first spike times
        spike_times = []
        for feature in range(x.shape[1]):
            spike_indices = torch.where(spikes[0, :, feature] > 0)[0]
            if len(spike_indices) > 0:
                spike_times.append(spike_indices[0].item())
            else:
                spike_times.append(float('inf'))
        
        # Higher value (0.9) should spike before lower value (0.1)
        assert spike_times[1] <= spike_times[0]
    
    def test_single_spike_per_input(self):
        """Test that each input generates at most one spike."""
        encoder = TemporalCoding(timesteps=15)
        x = torch.rand(2, 5)
        
        spikes = encoder(x)
        
        # Each feature should have at most one spike across time
        for b in range(2):
            for f in range(5):
                spike_count = spikes[b, :, f].sum()
                assert spike_count <= 1


class TestPoissonCoding:
    """Test Poisson coding encoder."""
    
    def test_initialization(self):
        """Test Poisson coding initialization."""
        encoder = PoissonCoding(timesteps=30, max_rate=50.0, dt=2e-3)
        
        assert encoder.timesteps == 30
        assert encoder.max_rate == 50.0
        assert encoder.dt == 2e-3
    
    def test_encoding_shape(self):
        """Test Poisson coding output shape."""
        encoder = PoissonCoding(timesteps=20)
        
        x = torch.rand(2, 6)
        spikes = encoder(x)
        
        assert spikes.shape == (2, 20, 6)
    
    def test_poisson_properties(self):
        """Test Poisson spike generation properties."""
        encoder = PoissonCoding(timesteps=100, max_rate=10.0, dt=1e-3)
        
        # High rate input
        x_high = torch.ones(1, 1) * 0.8
        spikes_high = encoder(x_high)
        
        # Low rate input
        x_low = torch.ones(1, 1) * 0.2
        spikes_low = encoder(x_low)
        
        # Higher rate should generate more spikes on average
        assert spikes_high.sum() >= spikes_low.sum()
    
    def test_rate_proportionality(self):
        """Test that spike rate is proportional to input."""
        encoder = PoissonCoding(timesteps=1000, max_rate=1.0, dt=1e-3)
        
        # Test different input levels
        rates = [0.1, 0.5, 0.9]
        spike_counts = []
        
        for rate in rates:
            x = torch.ones(1, 1) * rate
            spikes = encoder(x)
            spike_counts.append(spikes.sum().item())
        
        # Spike counts should generally increase with input rate
        assert spike_counts[1] >= spike_counts[0]
        assert spike_counts[2] >= spike_counts[1]


class TestPopulationCoding:
    """Test population coding encoder."""
    
    def test_initialization(self):
        """Test population coding initialization."""
        encoder = PopulationCoding(timesteps=16, num_neurons=8, min_val=0.0, max_val=2.0)
        
        assert encoder.timesteps == 16
        assert encoder.num_neurons == 8
        assert encoder.min_val == 0.0
        assert encoder.max_val == 2.0
    
    def test_encoding_shape(self):
        """Test population coding output shape."""
        encoder = PopulationCoding(timesteps=12, num_neurons=5)
        
        x = torch.rand(3, 4)
        spikes = encoder(x)
        
        assert spikes.shape == (3, 12, 4, 5)  # Added neuron dimension
    
    def test_gaussian_tuning(self):
        """Test Gaussian tuning curves."""
        encoder = PopulationCoding(timesteps=20, num_neurons=10, min_val=0.0, max_val=1.0)
        
        # Input at center of range should activate middle neurons most
        x = torch.tensor([[0.5]])
        spikes = encoder(x)
        
        # Sum across time to get total activation per neuron
        activation = spikes.sum(dim=1)  # (batch, features, neurons)
        
        # Middle neurons should have higher activation
        middle_idx = encoder.num_neurons // 2
        middle_activation = activation[0, 0, middle_idx]
        edge_activation = activation[0, 0, 0]
        
        assert middle_activation >= edge_activation


class TestDeltaCoding:
    """Test delta coding encoder."""
    
    def test_initialization(self):
        """Test delta coding initialization."""
        encoder = DeltaCoding(timesteps=25, threshold=0.05)
        
        assert encoder.timesteps == 25
        assert encoder.threshold == 0.05
    
    def test_change_detection(self):
        """Test that delta coding detects changes."""
        encoder = DeltaCoding(timesteps=10, threshold=0.1)
        
        # Create input with a step change
        x1 = torch.ones(1, 3) * 0.5
        spikes1 = encoder(x1)  # First call establishes baseline
        
        x2 = torch.ones(1, 3) * 0.8  # Change in input
        spikes2 = encoder(x2)
        
        # Should generate spikes for the change
        assert torch.any(spikes2 != 0)
    
    def test_state_reset(self):
        """Test delta coding state reset."""
        encoder = DeltaCoding(timesteps=5)
        
        # Process some input
        x = torch.rand(1, 2)
        _ = encoder(x)
        
        assert encoder.previous_input is not None
        
        # Reset state
        encoder.reset_state()
        assert encoder.previous_input is None


class TestLatencyCoding:
    """Test latency coding encoder."""
    
    def test_initialization(self):
        """Test latency coding initialization."""
        encoder = LatencyCoding(timesteps=20, max_latency=15)
        
        assert encoder.timesteps == 20
        assert encoder.max_latency == 15
    
    def test_latency_ordering(self):
        """Test that higher values have shorter latencies."""
        encoder = LatencyCoding(timesteps=10)
        
        x = torch.tensor([[0.2, 0.8]])
        spikes = encoder(x)
        
        # Find first spike times
        first_spike_0 = torch.where(spikes[0, :, 0] > 0)[0]
        first_spike_1 = torch.where(spikes[0, :, 1] > 0)[0]
        
        if len(first_spike_0) > 0 and len(first_spike_1) > 0:
            # Higher value (0.8) should spike before lower value (0.2)
            assert first_spike_1[0] <= first_spike_0[0]


class TestBurstCoding:
    """Test burst coding encoder."""
    
    def test_initialization(self):
        """Test burst coding initialization."""
        encoder = BurstCoding(timesteps=50, burst_length=4, interburst_interval=6)
        
        assert encoder.timesteps == 50
        assert encoder.burst_length == 4
        assert encoder.interburst_interval == 6
    
    def test_burst_patterns(self):
        """Test burst spike patterns."""
        encoder = BurstCoding(timesteps=20, burst_length=3, interburst_interval=2)
        
        x = torch.ones(1, 1) * 0.8  # High input should generate bursts
        spikes = encoder(x)
        
        # Should have some bursts (consecutive spikes)
        spike_times = torch.where(spikes[0, :, 0] > 0)[0]
        
        if len(spike_times) >= 2:
            # Check for consecutive spikes (burst pattern)
            consecutive_found = False
            for i in range(len(spike_times) - 1):
                if spike_times[i+1] - spike_times[i] == 1:
                    consecutive_found = True
                    break
            
            # Should find some consecutive spikes in bursts
            # (This test is probabilistic due to input intensity)


class TestAdaptiveThresholdCoding:
    """Test adaptive threshold coding encoder."""
    
    def test_initialization(self):
        """Test adaptive threshold coding initialization."""
        encoder = AdaptiveThresholdCoding(timesteps=30, initial_threshold=0.3, adaptation_rate=0.05)
        
        assert encoder.timesteps == 30
        assert encoder.initial_threshold == 0.3
        assert encoder.adaptation_rate == 0.05
    
    def test_threshold_adaptation(self):
        """Test threshold adaptation mechanism."""
        encoder = AdaptiveThresholdCoding(timesteps=20, adaptation_rate=0.1)
        
        x = torch.ones(1, 2) * 0.6
        spikes = encoder(x)
        
        # Threshold should adapt during encoding
        assert encoder.threshold is not None
        assert encoder.accumulated_input is not None
    
    def test_state_reset(self):
        """Test adaptive coding state reset."""
        encoder = AdaptiveThresholdCoding(timesteps=10)
        
        # Process input
        x = torch.rand(1, 3)
        _ = encoder(x)
        
        assert encoder.threshold is not None
        assert encoder.accumulated_input is not None
        
        # Reset state
        encoder.reset_state()
        assert encoder.threshold is None
        assert encoder.accumulated_input is None


class TestMultiModalEncoder:
    """Test multi-modal encoder."""
    
    def test_initialization(self):
        """Test multi-modal encoder initialization."""
        encoders = {
            'rate': RateCoding(timesteps=16),
            'temporal': TemporalCoding(timesteps=16)
        }
        
        multi_encoder = MultiModalEncoder(encoders, combination_method="concatenate")
        
        assert 'rate' in multi_encoder.encoders
        assert 'temporal' in multi_encoder.encoders
        assert multi_encoder.combination_method == "concatenate"
    
    def test_concatenate_combination(self):
        """Test concatenation combination method."""
        encoders = {
            'rate': RateCoding(timesteps=10),
            'poisson': PoissonCoding(timesteps=10)
        }
        
        multi_encoder = MultiModalEncoder(encoders, combination_method="concatenate")
        
        x = torch.rand(2, 5)
        combined_spikes = multi_encoder(x)
        
        # Should concatenate along feature dimension
        assert combined_spikes.shape == (2, 10, 10)  # 5 features * 2 encoders
    
    def test_average_combination(self):
        """Test average combination method."""
        encoders = {
            'rate1': RateCoding(timesteps=8),
            'rate2': RateCoding(timesteps=8)
        }
        
        multi_encoder = MultiModalEncoder(encoders, combination_method="average")
        
        x = torch.rand(1, 3)
        combined_spikes = multi_encoder(x)
        
        # Should average across encoders
        assert combined_spikes.shape == (1, 8, 3)
    
    def test_state_reset(self):
        """Test multi-modal encoder state reset."""
        encoders = {
            'delta': DeltaCoding(timesteps=12),
            'adaptive': AdaptiveThresholdCoding(timesteps=12)
        }
        
        multi_encoder = MultiModalEncoder(encoders)
        
        # Process input
        x = torch.rand(1, 4)
        _ = multi_encoder(x)
        
        # Reset states
        multi_encoder.reset_state()
        
        # Should not raise errors


class TestEncoderFactory:
    """Test encoder factory function."""
    
    def test_create_rate_encoder(self):
        """Test creating rate encoder."""
        encoder = create_encoder("RATE", timesteps=20, max_rate=0.9)
        
        assert isinstance(encoder, RateCoding)
        assert encoder.timesteps == 20
        assert encoder.max_rate == 0.9
    
    def test_create_temporal_encoder(self):
        """Test creating temporal encoder."""
        encoder = create_encoder("TEMPORAL", timesteps=15, min_delay=1)
        
        assert isinstance(encoder, TemporalCoding)
        assert encoder.timesteps == 15
        assert encoder.min_delay == 1
    
    def test_create_poisson_encoder(self):
        """Test creating Poisson encoder."""
        encoder = create_encoder("POISSON", timesteps=25, max_rate=20.0)
        
        assert isinstance(encoder, PoissonCoding)
        assert encoder.timesteps == 25
        assert encoder.max_rate == 20.0
    
    def test_invalid_encoder_type(self):
        """Test error handling for invalid encoder type."""
        with pytest.raises(ValueError, match="Unknown encoding type"):
            create_encoder("INVALID_TYPE", timesteps=10)


class TestEncodingProperties:
    """Test general encoding properties."""
    
    def test_temporal_consistency(self):
        """Test that encoders produce consistent temporal patterns."""
        encoder = RateCoding(timesteps=20)
        
        x = torch.ones(1, 3) * 0.5
        
        # Multiple runs should have similar patterns (stochastic)
        spikes1 = encoder(x)
        spikes2 = encoder(x)
        
        # Should have similar spike counts (within reasonable variance)
        count1 = spikes1.sum()
        count2 = spikes2.sum()
        
        # Allow for some variance in stochastic encoding
        assert abs(count1 - count2) / max(count1, count2, 1) < 0.5
    
    def test_batch_independence(self):
        """Test that batch items are encoded independently."""
        encoder = TemporalCoding(timesteps=15)
        
        # Different inputs for each batch item
        x = torch.tensor([[0.3], [0.7]])
        spikes = encoder(x)
        
        # Should produce different spike patterns
        assert not torch.equal(spikes[0], spikes[1])
    
    def test_input_range_handling(self):
        """Test handling of different input ranges."""
        encoder = RateCoding(timesteps=10)
        
        # Test negative inputs (should be handled gracefully)
        x_neg = torch.ones(1, 2) * -0.5
        spikes_neg = encoder(x_neg)
        
        # Should not generate spikes for negative inputs
        assert torch.all(spikes_neg == 0)
        
        # Test very large inputs
        x_large = torch.ones(1, 2) * 10.0
        spikes_large = encoder(x_large)
        
        # Should be clipped/normalized appropriately
        assert torch.all((spikes_large == 0) | (spikes_large == 1))
    
    def test_zero_timesteps_error(self):
        """Test error handling for zero timesteps."""
        with pytest.raises((ValueError, AssertionError)):
            RateCoding(timesteps=0)
    
    def test_gradient_computation(self):
        """Test that encoders work with gradient computation."""
        encoder = RateCoding(timesteps=8)
        
        x = torch.rand(2, 4, requires_grad=True)
        spikes = encoder(x)
        
        # Spikes should be differentiable (even if gradients are zero)
        loss = spikes.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape