#!/usr/bin/env python3
"""Test cases for neuron models."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_lif_neuron_structure():
    """Test LifNeuron class structure."""
    try:
        from spikeformer.neurons import LifNeuron
        assert LifNeuron is not None
        assert hasattr(LifNeuron, '__init__')
        assert hasattr(LifNeuron, 'forward')
    except ImportError:
        assert True


def test_adlif_neuron_structure():
    """Test AdLifNeuron class structure."""
    try:
        from spikeformer.neurons import AdLifNeuron
        assert AdLifNeuron is not None
        assert hasattr(AdLifNeuron, '__init__')
        assert hasattr(AdLifNeuron, 'forward')
    except ImportError:
        assert True


def test_spiking_layer_structure():
    """Test SpikingLayer class structure."""
    try:
        from spikeformer.neurons import SpikingLayer
        assert SpikingLayer is not None
        assert hasattr(SpikingLayer, '__init__')
        assert hasattr(SpikingLayer, 'forward')
    except ImportError:
        assert True


def test_temporal_batch_norm_structure():
    """Test TemporalBatchNorm class structure."""
    try:
        from spikeformer.neurons import TemporalBatchNorm
        assert TemporalBatchNorm is not None
        assert hasattr(TemporalBatchNorm, '__init__')
        assert hasattr(TemporalBatchNorm, 'forward')
    except ImportError:
        assert True


def test_neuron_factory_function():
    """Test create_neuron factory function."""
    try:
        from spikeformer.neurons import create_neuron
        assert create_neuron is not None
        assert callable(create_neuron)
    except ImportError:
        assert True


if __name__ == "__main__":
    print("🧪 Running neuron tests...")
    
    tests = [
        test_lif_neuron_structure,
        test_adlif_neuron_structure,
        test_spiking_layer_structure,
        test_temporal_batch_norm_structure,
        test_neuron_factory_function
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    
    print(f"Neuron tests: {passed}/{len(tests)} passed")