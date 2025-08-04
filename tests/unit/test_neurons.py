"""Unit tests for spiking neuron models."""

import torch
import torch.nn as nn
import pytest
import numpy as np

from spikeformer.neurons import (
    LifNeuron, AdLifNeuron, IzhikevichNeuron,
    SpikingLayer, TemporalBatchNorm, 
    SurrogateGradient, SpikingFunction,
    create_neuron
)


class TestSurrogateGradient:
    """Test surrogate gradient functions."""
    
    def test_fast_sigmoid(self):
        """Test fast sigmoid surrogate gradient."""
        x = torch.randn(10, 5)
        grad = SurrogateGradient.fast_sigmoid(x)
        
        assert grad.shape == x.shape
        assert torch.all(grad >= 0)
        assert torch.all(grad <= 1)
    
    def test_straight_through_estimator(self):
        """Test straight-through estimator."""
        x = torch.randn(10, 5)
        grad = SurrogateGradient.straight_through_estimator(x)
        
        assert grad.shape == x.shape
        assert torch.all(grad == 1)
    
    def test_triangular(self):
        """Test triangular surrogate gradient."""
        x = torch.randn(10, 5)
        grad = SurrogateGradient.triangular(x)
        
        assert grad.shape == x.shape
        assert torch.all(grad >= 0)
        assert torch.all(grad <= 1)


class TestSpikingFunction:
    """Test differentiable spiking function."""
    
    def test_forward(self):
        """Test forward pass generates spikes."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        threshold = 1.0
        
        spikes = SpikingFunction.apply(x, threshold, "fast_sigmoid")
        
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(spikes, expected)
    
    def test_backward_gradient(self):
        """Test backward pass with surrogate gradients."""
        x = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
        threshold = 1.0
        
        spikes = SpikingFunction.apply(x, threshold, "fast_sigmoid")
        loss = spikes.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLifNeuron:
    """Test Leaky Integrate-and-Fire neuron."""
    
    def test_initialization(self):
        """Test LIF neuron initialization."""
        neuron = LifNeuron(threshold=1.5, tau_mem=30.0)
        
        assert neuron.threshold == 1.5
        assert neuron.tau_mem == 30.0
        assert neuron.membrane_potential is None
    
    def test_forward_pass(self):
        """Test forward pass through LIF neuron."""
        neuron = LifNeuron(threshold=1.0)
        batch_size, timesteps, features = 2, 10, 5
        
        x = torch.randn(batch_size, timesteps, features)
        spikes = neuron(x)
        
        assert spikes.shape == (batch_size, timesteps, features)
        assert torch.all((spikes == 0) | (spikes == 1))
    
    def test_membrane_dynamics(self):
        """Test membrane potential dynamics."""
        neuron = LifNeuron(threshold=1.0, tau_mem=20.0)
        
        # Constant input should cause integration
        constant_input = torch.ones(1, 5, 1) * 0.5
        spikes = neuron(constant_input)
        
        # Should eventually spike with constant positive input
        assert torch.any(spikes > 0)
    
    def test_reset_mechanism(self):
        """Test neuron reset mechanisms."""
        # Test subtract reset
        neuron_subtract = LifNeuron(threshold=1.0, reset="subtract")
        x = torch.ones(1, 3, 1) * 2.0  # Strong input
        spikes_subtract = neuron_subtract(x)
        
        # Test zero reset
        neuron_zero = LifNeuron(threshold=1.0, reset="zero")
        neuron_zero.reset_state()
        spikes_zero = neuron_zero(x)
        
        assert spikes_subtract.shape == spikes_zero.shape
    
    def test_state_reset(self):
        """Test neuron state reset."""
        neuron = LifNeuron()
        
        # Run forward pass to initialize state
        x = torch.randn(2, 5, 3)
        _ = neuron(x)
        
        assert neuron.membrane_potential is not None
        
        # Reset state
        neuron.reset_state()
        assert neuron.membrane_potential is None


class TestAdLifNeuron:
    """Test Adaptive LIF neuron."""
    
    def test_initialization(self):
        """Test AdLIF neuron initialization."""
        neuron = AdLifNeuron(threshold=1.0, tau_adp=100.0, beta=0.2)
        
        assert neuron.threshold == 1.0
        assert neuron.tau_adp == 100.0
        assert neuron.beta == 0.2
        assert neuron.adaptive_threshold is None
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold mechanism."""
        neuron = AdLifNeuron(threshold=1.0, beta=0.5)
        
        # High frequency input should increase threshold
        high_freq_input = torch.ones(1, 10, 1) * 1.5
        spikes = neuron(high_freq_input)
        
        assert torch.any(spikes > 0)
        assert neuron.adaptive_threshold is not None
    
    def test_state_reset(self):
        """Test AdLIF state reset."""
        neuron = AdLifNeuron()
        
        x = torch.randn(1, 5, 2)
        _ = neuron(x)
        
        assert neuron.membrane_potential is not None
        assert neuron.adaptive_threshold is not None
        
        neuron.reset_state()
        assert neuron.membrane_potential is None
        assert neuron.adaptive_threshold is None


class TestIzhikevichNeuron:
    """Test Izhikevich neuron model."""
    
    def test_initialization(self):
        """Test Izhikevich neuron initialization."""
        neuron = IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0)
        
        assert neuron.a == 0.02
        assert neuron.b == 0.2
        assert neuron.c == -65.0
        assert neuron.d == 8.0
    
    def test_complex_dynamics(self):
        """Test complex neuron dynamics."""
        neuron = IzhikevichNeuron()
        
        # Test with varied input
        x = torch.sin(torch.linspace(0, 4*np.pi, 50)).unsqueeze(0).unsqueeze(2) * 10
        spikes = neuron(x)
        
        assert spikes.shape == x.shape
        assert torch.any(spikes > 0)  # Should generate some spikes
    
    def test_state_reset(self):
        """Test Izhikevich state reset."""
        neuron = IzhikevichNeuron()
        
        x = torch.randn(1, 5, 1)
        _ = neuron(x)
        
        assert neuron.membrane_potential is not None
        assert neuron.recovery_variable is not None
        
        neuron.reset_state()
        assert neuron.membrane_potential is None
        assert neuron.recovery_variable is None


class TestSpikingLayer:
    """Test spiking layer combining linear transformation and neurons."""
    
    def test_initialization(self):
        """Test spiking layer initialization."""
        layer = SpikingLayer(10, 5, neuron_type="LIF", threshold=1.0)
        
        assert layer.linear.in_features == 10
        assert layer.linear.out_features == 5
        assert isinstance(layer.neuron, LifNeuron)
    
    def test_neuron_types(self):
        """Test different neuron types."""
        # LIF neuron
        lif_layer = SpikingLayer(5, 3, neuron_type="LIF")
        assert isinstance(lif_layer.neuron, LifNeuron)
        
        # AdLIF neuron
        adlif_layer = SpikingLayer(5, 3, neuron_type="ADLIF")
        assert isinstance(adlif_layer.neuron, AdLifNeuron)
        
        # Izhikevich neuron
        izh_layer = SpikingLayer(5, 3, neuron_type="IZHIKEVICH")
        assert isinstance(izh_layer.neuron, IzhikevichNeuron)
    
    def test_forward_3d_input(self):
        """Test forward pass with 3D input."""
        layer = SpikingLayer(10, 5)
        batch_size, timesteps, features = 2, 8, 10
        
        x = torch.randn(batch_size, timesteps, features)
        output = layer(x)
        
        assert output.shape == (batch_size, timesteps, 5)
    
    def test_forward_4d_input(self):
        """Test forward pass with 4D input."""
        layer = SpikingLayer(10, 5)
        batch_size, timesteps, seq_len, features = 2, 8, 12, 10
        
        x = torch.randn(batch_size, timesteps, seq_len, features)
        output = layer(x)
        
        assert output.shape == (batch_size, timesteps, seq_len, 5)
    
    def test_state_reset(self):
        """Test layer state reset."""
        layer = SpikingLayer(5, 3)
        
        x = torch.randn(1, 5, 5)
        _ = layer(x)
        
        # Reset should not raise error
        layer.reset_state()


class TestTemporalBatchNorm:
    """Test temporal batch normalization."""
    
    def test_initialization(self):
        """Test temporal batch norm initialization."""
        norm = TemporalBatchNorm(10)
        
        assert norm.num_features == 10
        assert norm.weight.shape == (10,)
        assert norm.bias.shape == (10,)
        assert norm.running_mean.shape == (10,)
        assert norm.running_var.shape == (10,)
    
    def test_forward_training(self):
        """Test forward pass in training mode."""
        norm = TemporalBatchNorm(5)
        batch_size, timesteps, features = 4, 10, 5
        
        x = torch.randn(batch_size, timesteps, features)
        output = norm(x)
        
        assert output.shape == x.shape
        # Check normalization (approximately zero mean)
        assert torch.abs(output.mean()) < 0.1
    
    def test_forward_eval(self):
        """Test forward pass in evaluation mode."""
        norm = TemporalBatchNorm(3)
        norm.eval()
        
        x = torch.randn(2, 8, 3)
        output = norm(x)
        
        assert output.shape == x.shape
    
    def test_running_statistics(self):
        """Test running statistics update."""
        norm = TemporalBatchNorm(4)
        
        # Initial values
        initial_mean = norm.running_mean.clone()
        initial_var = norm.running_var.clone()
        
        # Forward pass should update statistics
        x = torch.randn(3, 5, 4)
        _ = norm(x)
        
        assert not torch.allclose(norm.running_mean, initial_mean)
        assert not torch.allclose(norm.running_var, initial_var)


class TestNeuronFactory:
    """Test neuron factory function."""
    
    def test_create_lif_neuron(self):
        """Test creating LIF neuron."""
        neuron = create_neuron("LIF", threshold=1.5)
        
        assert isinstance(neuron, LifNeuron)
        assert neuron.threshold == 1.5
    
    def test_create_adlif_neuron(self):
        """Test creating AdLIF neuron."""
        neuron = create_neuron("ADLIF", threshold=1.0, beta=0.3)
        
        assert isinstance(neuron, AdLifNeuron)
        assert neuron.threshold == 1.0
        assert neuron.beta == 0.3
    
    def test_create_izhikevich_neuron(self):
        """Test creating Izhikevich neuron."""
        neuron = create_neuron("IZHIKEVICH", a=0.05)
        
        assert isinstance(neuron, IzhikevichNeuron)
        assert neuron.a == 0.05
    
    def test_invalid_neuron_type(self):
        """Test error handling for invalid neuron type."""
        with pytest.raises(ValueError, match="Unknown neuron type"):
            create_neuron("INVALID_TYPE")


class TestGradientFlow:
    """Test gradient flow through spiking neurons."""
    
    def test_lif_gradient_flow(self):
        """Test gradient flow through LIF neuron."""
        neuron = LifNeuron()
        x = torch.randn(1, 5, 3, requires_grad=True)
        
        spikes = neuron(x)
        loss = spikes.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_spiking_layer_gradient_flow(self):
        """Test gradient flow through spiking layer."""
        layer = SpikingLayer(5, 3)
        x = torch.randn(2, 4, 5, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check layer parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestNeuronProperties:
    """Test neuron properties and edge cases."""
    
    def test_zero_input(self):
        """Test neuron behavior with zero input."""
        neuron = LifNeuron()
        x = torch.zeros(1, 10, 3)
        
        spikes = neuron(x)
        
        # Zero input should not generate spikes
        assert torch.all(spikes == 0)
    
    def test_negative_input(self):
        """Test neuron behavior with negative input."""
        neuron = LifNeuron()
        x = torch.ones(1, 5, 2) * -1.0
        
        spikes = neuron(x)
        
        # Negative input should not generate spikes with positive threshold
        assert torch.all(spikes == 0)
    
    def test_batch_consistency(self):
        """Test that different batch items are processed independently."""
        neuron = LifNeuron(threshold=1.0)
        
        # Different inputs for each batch item
        x = torch.stack([
            torch.ones(5, 3) * 0.5,  # Low input
            torch.ones(5, 3) * 2.0   # High input
        ])
        
        spikes = neuron(x)
        
        # High input batch should have more spikes
        assert spikes[1].sum() > spikes[0].sum()
    
    def test_temporal_consistency(self):
        """Test temporal dynamics consistency."""
        neuron = LifNeuron(threshold=1.0, tau_mem=10.0)
        
        # Step input
        x = torch.zeros(1, 20, 1)
        x[:, 5:15, :] = 1.0  # Input from timestep 5 to 15
        
        spikes = neuron(x)
        
        # Should not spike before input starts
        assert torch.all(spikes[:, :5, :] == 0)
        
        # Should spike during or shortly after input
        assert torch.any(spikes[:, 5:18, :] > 0)