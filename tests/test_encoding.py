"""Tests for spike encoding modules."""
import pytest
import torch

from spikeformer.encoding import RateEncoder, LatencyEncoder, DirectEncoder


class TestRateEncoder:
    def test_output_shape(self):
        enc = RateEncoder(timesteps=16)
        x = torch.rand(4, 32)
        spikes = enc(x)
        assert spikes.shape == (4, 16, 32)

    def test_binary_output(self):
        enc = RateEncoder(timesteps=8)
        x = torch.rand(4, 32)
        spikes = enc(x)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_rate_proportional_to_value(self):
        """Higher input → higher spike rate on average."""
        enc = RateEncoder(timesteps=1000, normalize=False)
        # Two groups: low (0.1) and high (0.9)
        x = torch.tensor([[0.1, 0.9]])
        spikes = enc(x)  # (1, 1000, 2)
        low_rate = spikes[0, :, 0].mean().item()
        high_rate = spikes[0, :, 1].mean().item()
        assert high_rate > low_rate


class TestLatencyEncoder:
    def test_output_shape(self):
        enc = LatencyEncoder(timesteps=16)
        x = torch.rand(4, 32)
        spikes = enc(x)
        assert spikes.shape == (4, 16, 32)

    def test_one_spike_per_neuron(self):
        enc = LatencyEncoder(timesteps=16)
        x = torch.rand(4, 32)
        spikes = enc(x)
        # Each neuron should fire exactly once
        spike_counts = spikes.sum(dim=1)  # (4, 32)
        assert (spike_counts == 1).all()

    def test_higher_value_fires_earlier(self):
        enc = LatencyEncoder(timesteps=100)
        x = torch.tensor([[0.1, 0.9]])
        spikes = enc(x)  # (1, 100, 2)
        # Find first spike time for each neuron
        low_t = spikes[0, :, 0].argmax().item()   # first 1 in time
        high_t = spikes[0, :, 1].argmax().item()
        assert high_t < low_t, f"High-value neuron should fire first: {high_t} vs {low_t}"


class TestDirectEncoder:
    def test_output_shape(self):
        enc = DirectEncoder(timesteps=8)
        x = torch.randn(4, 32)
        out = enc(x)
        assert out.shape == (4, 8, 32)

    def test_values_repeated(self):
        enc = DirectEncoder(timesteps=4)
        x = torch.randn(2, 8)
        out = enc(x)
        for t in range(4):
            assert torch.allclose(out[:, t, :], x)
