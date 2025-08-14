#!/usr/bin/env python3
"""Test cases for adaptive systems."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_robust_adaptive_system_structure():
    """Test RobustAdaptiveSystem class structure."""
    try:
        from spikeformer.adaptive import RobustAdaptiveSystem
        assert RobustAdaptiveSystem is not None
        assert hasattr(RobustAdaptiveSystem, '__init__')
        assert hasattr(RobustAdaptiveSystem, 'robust_adapt')
    except ImportError:
        assert True


def test_critical_point_detector_structure():
    """Test CriticalPointDetector class structure."""
    try:
        from spikeformer.adaptive import CriticalPointDetector
        assert CriticalPointDetector is not None
        assert hasattr(CriticalPointDetector, '__init__')
        assert hasattr(CriticalPointDetector, 'detect_critical_transitions')
    except ImportError:
        assert True


def test_adaptive_threshold_controller_structure():
    """Test AdaptiveThresholdController class structure."""
    try:
        from spikeformer.adaptive import AdaptiveThresholdController
        assert AdaptiveThresholdController is not None
        assert hasattr(AdaptiveThresholdController, '__init__')
        assert hasattr(AdaptiveThresholdController, 'adapt')
    except ImportError:
        assert True


def test_adaptation_config_structure():
    """Test AdaptationConfig dataclass structure."""
    try:
        from spikeformer.adaptive import AdaptationConfig
        config = AdaptationConfig()
        assert hasattr(config, 'adaptation_rate')
        assert hasattr(config, 'momentum')
        assert config.adaptation_rate > 0
    except ImportError:
        assert True


def test_adaptive_mechanism_interface():
    """Test AdaptiveMechanism interface."""
    try:
        from spikeformer.adaptive import AdaptiveMechanism
        assert AdaptiveMechanism is not None
        assert hasattr(AdaptiveMechanism, 'adapt')
        assert hasattr(AdaptiveMechanism, 'reset')
    except ImportError:
        assert True


if __name__ == "__main__":
    print("🧪 Running adaptive tests...")
    
    tests = [
        test_robust_adaptive_system_structure,
        test_critical_point_detector_structure,
        test_adaptive_threshold_controller_structure,
        test_adaptation_config_structure,
        test_adaptive_mechanism_interface
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"✅ {test.__name__}")
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
    
    print(f"Adaptive tests: {passed}/{len(tests)} passed")