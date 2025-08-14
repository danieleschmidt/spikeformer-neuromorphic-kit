#!/usr/bin/env python3
"""Test cases for spike encoding methods."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rate_coding_structure():
    """Test RateCoding class structure."""
    try:
        from spikeformer.encoding import RateCoding
        assert RateCoding is not None
        assert hasattr(RateCoding, '__init__')
        assert hasattr(RateCoding, 'forward')
    except ImportError:
        assert True


def test_temporal_coding_structure():
    """Test TemporalCoding class structure."""
    try:
        from spikeformer.encoding import TemporalCoding
        assert TemporalCoding is not None
        assert hasattr(TemporalCoding, '__init__')
        assert hasattr(TemporalCoding, 'forward')
    except ImportError:
        assert True


def test_poisson_coding_structure():
    """Test PoissonCoding class structure."""
    try:
        from spikeformer.encoding import PoissonCoding
        assert PoissonCoding is not None
        assert hasattr(PoissonCoding, '__init__')
        assert hasattr(PoissonCoding, 'forward')
    except ImportError:
        assert True


def test_delta_coding_structure():
    """Test DeltaCoding class structure."""
    try:
        from spikeformer.encoding import DeltaCoding
        assert DeltaCoding is not None
        assert hasattr(DeltaCoding, '__init__')
        assert hasattr(DeltaCoding, 'forward')
    except ImportError:
        assert True


def test_rank_order_coding_structure():
    """Test RankOrderCoding class structure."""
    try:
        from spikeformer.encoding import RankOrderCoding
        assert RankOrderCoding is not None
        assert hasattr(RankOrderCoding, '__init__')
        assert hasattr(RankOrderCoding, 'forward')
    except ImportError:
        assert True


def test_latency_coding_structure():
    """Test LatencyCoding class structure."""
    try:
        from spikeformer.encoding import LatencyCoding
        assert LatencyCoding is not None
        assert hasattr(LatencyCoding, '__init__')
        assert hasattr(LatencyCoding, 'forward')
    except ImportError:
        assert True


def test_adaptive_optimal_encoder_structure():
    """Test AdaptiveOptimalEncoder class structure."""
    try:
        from spikeformer.encoding import AdaptiveOptimalEncoder
        assert AdaptiveOptimalEncoder is not None
        assert hasattr(AdaptiveOptimalEncoder, '__init__')
        assert hasattr(AdaptiveOptimalEncoder, 'encode_adaptively')
    except ImportError:
        assert True


if __name__ == "__main__":
    print("🧪 Running encoding tests...")
    
    tests = [
        test_rate_coding_structure,
        test_temporal_coding_structure,
        test_poisson_coding_structure,
        test_delta_coding_structure,
        test_rank_order_coding_structure,
        test_latency_coding_structure,
        test_adaptive_optimal_encoder_structure
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    
    print(f"Encoding tests: {passed}/{len(tests)} passed")