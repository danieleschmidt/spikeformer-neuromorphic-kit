#!/usr/bin/env python3
"""Validate hardware configuration files."""

import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any


def validate_loihi2_config(config: Dict[str, Any]) -> bool:
    """Validate Loihi 2 configuration."""
    required_fields = ['num_chips', 'partition_strategy', 'memory_allocation']
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate constraints
    if config.get('num_chips', 0) > 128:
        print("❌ num_chips exceeds maximum (128)")
        return False
        
    valid_strategies = ['layer_wise', 'neuron_wise', 'hybrid']
    if config.get('partition_strategy') not in valid_strategies:
        print(f"❌ Invalid partition_strategy. Must be one of: {valid_strategies}")
        return False
    
    return True


def validate_spinnaker_config(config: Dict[str, Any]) -> bool:
    """Validate SpiNNaker configuration."""
    required_fields = ['machine_width', 'machine_height', 'time_scale_factor']
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate board dimensions
    width = config.get('machine_width', 0)
    height = config.get('machine_height', 0)
    
    if width <= 0 or height <= 0:
        print("❌ Invalid machine dimensions")
        return False
        
    if width * height > 1024:  # Reasonable limit
        print("❌ Machine size too large")
        return False
    
    return True


def validate_edge_config(config: Dict[str, Any]) -> bool:
    """Validate edge device configuration."""
    required_fields = ['target_device', 'power_budget_mw', 'memory_limit_mb']
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate constraints
    if config.get('power_budget_mw', 0) <= 0:
        print("❌ Invalid power budget")
        return False
        
    if config.get('memory_limit_mb', 0) <= 0:
        print("❌ Invalid memory limit")
        return False
    
    return True


def validate_config_file(file_path: Path) -> bool:
    """Validate a single configuration file."""
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                config = json.load(f)
            else:
                print(f"⚠️  Skipping unsupported file type: {file_path}")
                return True
    except Exception as e:
        print(f"❌ Failed to parse {file_path}: {e}")
        return False
    
    if not isinstance(config, dict):
        print(f"❌ Configuration must be a dictionary: {file_path}")
        return False
    
    # Determine config type and validate accordingly
    if 'loihi' in str(file_path).lower() or config.get('hardware_type') == 'loihi2':
        return validate_loihi2_config(config)
    elif 'spinnaker' in str(file_path).lower() or config.get('hardware_type') == 'spinnaker':
        return validate_spinnaker_config(config)
    elif 'edge' in str(file_path).lower() or config.get('hardware_type') == 'edge':
        return validate_edge_config(config)
    else:
        # Generic validation for unknown config types
        print(f"✅ {file_path}: Generic validation passed")
        return True


def main() -> bool:
    """Main validation function."""
    repo_root = Path.cwd()
    
    # Find configuration files
    config_patterns = [
        'hardware_configs/**/*.json',
        'hardware_configs/**/*.yaml',
        'hardware_configs/**/*.yml',
        'configs/**/*.json',
        'configs/**/*.yaml',
        'configs/**/*.yml'
    ]
    
    config_files = []
    for pattern in config_patterns:
        config_files.extend(repo_root.glob(pattern))
    
    if not config_files:
        print("✅ No hardware configuration files found")
        return True
    
    all_valid = True
    for config_file in config_files:
        print(f"Validating {config_file}...")
        if not validate_config_file(config_file):
            all_valid = False
    
    if all_valid:
        print("✅ All hardware configurations are valid")
    else:
        print("❌ Some hardware configurations are invalid")
    
    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)