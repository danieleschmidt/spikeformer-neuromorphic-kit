"""Tests for spiking neuron models."""
import pytest
import torch
import torch.nn as nn

from spikeformer.neurons import LIFNeuron, SpikeRateTracker, spike_fn


class TestSpikeFn:
    def test_fires_above_threshold(self):
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        s = spike_fn(x, threshold=1.0)
        assert s[0].item() == 0.0  # below
        assert s[1].item() == 1.0  # at threshold → fire
        assert s[2].item() == 1.0
        assert s[3].item() == 1.0

    def test_binary_output(self):
        x = torch.randn(10, 20)
        s = spike_fn(x, threshold=0.5)
        assert set(s.unique().tolist()).issubset({0.0, 1.0})

    def test_gradient_flows(self):
        x = torch.randn(4, 8, requires_grad=True)
        s = spike_fn(x, threshold=1.0, beta=4.0)
        loss = s.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # Surrogate gradient should be non-negative
        assert (x.grad >= 0).all()


class TestLIFNeuron:
    def test_output_shape(self):
        lif = LIFNeuron()
        x = torch.randn(4, 8, 32)  # (batch, time, features)
        spikes, rate = lif(x)
        assert spikes.shape == x.shape

    def test_binary_spikes(self):
        lif = LIFNeuron(threshold=0.3)
        x = torch.randn(4, 8, 16)
        spikes, _ = lif(x)
        assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

    def test_spike_rate_range(self):
        lif = LIFNeuron(threshold=1.0)
        x = torch.randn(8, 16, 32)
        _, rate = lif(x)
        assert 0.0 <= rate.item() <= 1.0

    def test_backprop(self):
        lif = LIFNeuron(threshold=1.0)
        x = torch.randn(4, 8, 16, requires_grad=True)
        spikes, _ = lif(x)
        loss = spikes.sum()
        loss.backward()
        assert x.grad is not None

    def test_learnable_decay(self):
        lif = LIFNeuron(tau_mem=20.0)
        params = list(lif.parameters())
        assert len(params) == 1, "LIFNeuron should have one learnable parameter (log_alpha)"
        assert params[0].requires_grad

    def test_alpha_in_valid_range(self):
        lif = LIFNeuron(tau_mem=20.0)
        alpha = lif.alpha.item()
        assert 0.0 < alpha < 1.0, f"alpha={alpha} out of range"

    def test_stats_tracking(self):
        lif = LIFNeuron(track_rates=True)
        x = torch.ones(4, 8, 16) * 2.0  # should fire a lot
        lif.reset_stats()
        lif(x)
        rate = lif.spike_rate
        assert 0.0 < rate <= 1.0

    def test_reset_stats(self):
        lif = LIFNeuron(track_rates=True)
        x = torch.ones(4, 8, 16) * 2.0
        lif(x)
        lif.reset_stats()
        assert lif.spike_rate == 0.0

    def test_subtract_reset(self):
        """After firing, membrane should be reduced by threshold (not zeroed)."""
        lif = LIFNeuron(threshold=1.0, tau_mem=1000.0)  # almost no decay
        # Single time step with large input — should fire and subtract
        x = torch.tensor([[[10.0]]])  # (1, 1, 1) — way above threshold
        spikes, _ = lif(x)
        assert spikes[0, 0, 0].item() == 1.0


class TestSpikeRateTracker:
    def test_tracker_collects_rates(self):
        from spikeformer import SpikeFormer
        model = SpikeFormer(
            input_dim=8, dim=16, num_heads=2, num_layers=1,
            num_classes=3, timesteps=4
        )
        x = torch.randn(2, 8)
        with SpikeRateTracker(model) as tracker:
            model(x)
        assert len(tracker.rates) > 0
        for name, rate in tracker.rates.items():
            assert 0.0 <= rate <= 1.0, f"{name}: rate={rate} out of range"
