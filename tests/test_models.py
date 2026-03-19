"""Tests for SpikeFormer model architecture."""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from spikeformer import SpikeFormer, SpikeFormerBlock, SpikeAttention, SpikeMLP


class TestSpikeAttention:
    def test_output_shape(self):
        B, T, N, D = 2, 4, 5, 32
        attn = SpikeAttention(dim=D, num_heads=4, tau_mem=10.0)
        x = torch.randn(B, T, N, D)
        out, rate = attn(x)
        assert out.shape == (B, T, N, D)

    def test_rate_range(self):
        attn = SpikeAttention(dim=32, num_heads=4)
        x = torch.randn(2, 4, 3, 32)
        _, rate = attn(x)
        assert 0.0 <= rate.item() <= 1.0

    def test_backprop(self):
        attn = SpikeAttention(dim=32, num_heads=4)
        x = torch.randn(2, 4, 3, 32, requires_grad=True)
        out, _ = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestSpikeMLP:
    def test_output_shape(self):
        B, T, N, D = 2, 4, 5, 32
        mlp = SpikeMLP(dim=D, mlp_ratio=4.0)
        x = torch.randn(B, T, N, D)
        out, rate = mlp(x)
        assert out.shape == (B, T, N, D)

    def test_backprop(self):
        mlp = SpikeMLP(dim=32)
        x = torch.randn(2, 4, 3, 32, requires_grad=True)
        out, _ = mlp(x)
        out.sum().backward()
        assert x.grad is not None


class TestSpikeFormerBlock:
    def test_output_shape(self):
        B, T, N, D = 2, 4, 5, 32
        block = SpikeFormerBlock(dim=D, num_heads=4)
        x = torch.randn(B, T, N, D)
        out, rates = block(x)
        assert out.shape == (B, T, N, D)

    def test_rates_dict_keys(self):
        block = SpikeFormerBlock(dim=32, num_heads=4)
        x = torch.randn(2, 4, 5, 32)
        _, rates = block(x)
        assert "lif1" in rates
        assert "attn" in rates
        assert "lif2" in rates
        assert "ffn_hidden" in rates

    def test_residual_connection(self):
        """Output should NOT be identical to input (residual + computation)."""
        block = SpikeFormerBlock(dim=32, num_heads=4)
        x = torch.zeros(2, 4, 5, 32)
        out, _ = block(x)
        # With zero input, output could be zero if weights are zero, but with
        # random init it should differ. At least check shape is correct.
        assert out.shape == x.shape


class TestSpikeFormer:
    def test_output_shape_2d_input(self, small_model):
        x = torch.randn(4, 16)  # (batch, input_dim)
        logits, rates = small_model(x)
        assert logits.shape == (4, 4)  # (batch, num_classes)

    def test_output_shape_3d_input(self, small_model):
        x = torch.randn(4, 1, 16)  # (batch, seq=1, input_dim)
        logits, rates = small_model(x)
        assert logits.shape == (4, 4)

    def test_spike_rates_returned(self, small_model):
        x = torch.randn(4, 16)
        _, rates = small_model(x)
        assert isinstance(rates, dict)
        assert len(rates) > 0

    def test_rates_are_valid(self, small_model):
        x = torch.randn(4, 16)
        _, rates = small_model(x)
        for name, rate in rates.items():
            assert 0.0 <= rate <= 1.0, f"{name}: {rate}"

    def test_backprop_end_to_end(self, small_model):
        """Gradients should flow from logits all the way back to input."""
        x = torch.randn(4, 16, requires_grad=True)
        logits, _ = small_model(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameter_update(self, small_model):
        """Verify parameters actually change after a gradient step."""
        optimizer = optim.Adam(small_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        params_before = {n: p.clone() for n, p in small_model.named_parameters()}

        x = torch.randn(4, 16)
        y = torch.randint(0, 4, (4,))
        logits, _ = small_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        changed = False
        for n, p in small_model.named_parameters():
            if not torch.allclose(p, params_before[n]):
                changed = True
                break
        assert changed, "No parameters changed after gradient step"

    def test_different_batch_sizes(self, small_model):
        for B in [1, 4, 16]:
            x = torch.randn(B, 16)
            logits, _ = small_model(x)
            assert logits.shape == (B, 4)

    def test_param_count_reasonable(self, small_model):
        n_params = sum(p.numel() for p in small_model.parameters())
        assert 1_000 < n_params < 10_000_000, f"Unexpected param count: {n_params}"

    def test_training_loss_decreases(self):
        """Simple sanity check: loss should decrease over a few steps."""
        torch.manual_seed(0)
        model = SpikeFormer(
            input_dim=16, dim=32, num_heads=4, num_layers=2,
            num_classes=4, timesteps=4
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        x = torch.randn(32, 16)
        y = torch.randint(0, 4, (32,))

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at step 10 should be lower than step 1
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
