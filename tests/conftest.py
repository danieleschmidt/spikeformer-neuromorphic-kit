"""Shared pytest fixtures for SpikeFormer tests."""
import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch():
    return 4


@pytest.fixture
def timesteps():
    return 8


@pytest.fixture
def dim():
    return 32


@pytest.fixture
def small_model():
    """Tiny SpikeFormer for fast unit tests."""
    from spikeformer import SpikeFormer
    return SpikeFormer(
        input_dim=16,
        dim=32,
        num_heads=4,
        num_layers=2,
        num_classes=4,
        timesteps=4,
        tau_mem=10.0,
        threshold=1.0,
        dropout=0.0,
    )
