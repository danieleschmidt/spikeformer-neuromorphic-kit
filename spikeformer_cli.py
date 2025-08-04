#!/usr/bin/env python3
"""
SpikeFormer CLI - Simplified Command Line Interface
Provides core neuromorphic functionality without heavy dependencies.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import our basic demo components
from demo_basic import (
    SimpleModelConfig, SimpleANN, SpikingConfig, BasicConverter,
    SpikingMLP, RateEncoder, TemporalEncoder, BasicHardwareSimulator,
    BasicEnergyProfiler, run_quick_demo, run_complete_demo
)


def create_model_from_config(config_path: str) -> SimpleANN:
    """Create model from JSON configuration file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    model_config = SimpleModelConfig(
        input_size=config_data.get('input_size', 784),
        hidden_sizes=config_data.get('hidden_sizes', [256, 128]),
        output_size=config_data.get('output_size', 10)
    )
    
    return SimpleANN(model_config)


def save_model_config(model: SimpleANN, output_path: str):
    """Save model configuration to JSON file."""
    config_data = {
        'input_size': model.config.input_size,
        'hidden_sizes': model.config.hidden_sizes,
        'output_size': model.config.output_size,
        'total_parameters': model.total_parameters,
        'architecture_type': 'feedforward'
    }
    
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"📁 Model configuration saved to {output_path}")


def cmd_convert(args):
    """Convert ANN model to SNN."""
    print(f"🔄 Converting model: {args.input}")
    
    # Load or create model
    if args.input.endswith('.json'):
        ann_model = create_model_from_config(args.input)
    else:
        # Create default model if not JSON config
        model_config = SimpleModelConfig(
            input_size=args.input_size,
            hidden_sizes=args.hidden_layers,
            output_size=args.output_size
        )
        ann_model = SimpleANN(model_config)
    
    # Configure spiking parameters
    spiking_config = SpikingConfig(
        timesteps=args.timesteps,
        threshold=args.threshold,
        tau_mem=args.tau_mem
    )
    
    # Convert to SNN
    converter = BasicConverter(spiking_config)
    snn_model = converter.convert(ann_model)
    
    # Save configuration
    if args.output:
        output_config = {
            'spiking_config': {
                'timesteps': spiking_config.timesteps,
                'threshold': spiking_config.threshold,
                'tau_mem': spiking_config.tau_mem
            },
            'original_model': {
                'input_size': ann_model.config.input_size,
                'hidden_sizes': ann_model.config.hidden_sizes,
                'output_size': ann_model.config.output_size,
                'total_parameters': ann_model.total_parameters
            },
            'conversion_timestamp': time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_config, f, indent=2)
        
        print(f"✅ Conversion results saved to {args.output}")
    
    return snn_model


def cmd_deploy(args):
    """Deploy model to neuromorphic hardware."""
    print(f"🚀 Deploying to {args.platform.upper()} hardware")
    
    # Load model configuration
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            config_data = json.load(f)
        
        # Reconstruct model
        if 'spiking_config' in config_data:
            # It's a converted SNN
            spiking_config = SpikingConfig(
                timesteps=config_data['spiking_config']['timesteps'],
                threshold=config_data['spiking_config']['threshold'],
                tau_mem=config_data['spiking_config']['tau_mem']
            )
            
            original_config = config_data['original_model']
            layer_sizes = ([original_config['input_size']] + 
                          original_config['hidden_sizes'] + 
                          [original_config['output_size']])
            
            model = SpikingMLP(layer_sizes, spiking_config)
        else:
            # It's an ANN config - convert first
            ann_model = create_model_from_config(args.input)
            converter = BasicConverter(SpikingConfig())
            model = converter.convert(ann_model)
    else:
        print("❌ Error: Please provide a valid model configuration file (.json)")
        return
    
    # Deploy to hardware
    simulator = BasicHardwareSimulator(args.platform)
    deployment_info = simulator.deploy(model)
    
    # Run benchmark
    if not args.no_benchmark:
        print(f"📊 Running benchmark...")
        metrics = simulator.benchmark(model, num_samples=args.benchmark_samples)
        deployment_info['benchmark_metrics'] = {
            'energy_per_inference_mj': metrics.energy_per_inference_mj,
            'latency_ms': metrics.latency_ms,
            'throughput_inferences_per_sec': metrics.throughput_inferences_per_sec,
            'power_consumption_mw': metrics.power_consumption_mw,
            'spike_rate': metrics.spike_rate
        }
    
    # Save deployment results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"✅ Deployment results saved to {args.output}")
    
    return deployment_info


def cmd_profile(args):
    """Profile model energy consumption."""
    print(f"⚡ Profiling energy consumption")
    
    # Load model
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            config_data = json.load(f)
        
        # Create ANN for comparison
        if 'original_model' in config_data:
            ann_config = SimpleModelConfig(
                input_size=config_data['original_model']['input_size'],
                hidden_sizes=config_data['original_model']['hidden_sizes'],
                output_size=config_data['original_model']['output_size']
            )
        else:
            ann_config = SimpleModelConfig(
                input_size=config_data['input_size'],
                hidden_sizes=config_data['hidden_sizes'],
                output_size=config_data['output_size']
            )
        
        ann_model = SimpleANN(ann_config)
        
        # Create SNN
        if 'spiking_config' in config_data:
            spiking_config = SpikingConfig(
                timesteps=config_data['spiking_config']['timesteps'],
                threshold=config_data['spiking_config']['threshold'],
                tau_mem=config_data['spiking_config']['tau_mem']
            )
        else:
            spiking_config = SpikingConfig()
        
        converter = BasicConverter(spiking_config)
        snn_model = converter.convert(ann_model)
    else:
        print("❌ Error: Please provide a valid model configuration file (.json)")
        return
    
    # Profile energy consumption
    profiler = BasicEnergyProfiler()
    
    # Generate test inputs
    test_inputs = [[0.5] * ann_model.config.input_size for _ in range(args.num_samples)]
    
    results = profiler.profile_ann_vs_snn(ann_model, snn_model, test_inputs)
    
    # Add platform-specific results
    platform_results = {}
    for platform in args.platforms:
        simulator = BasicHardwareSimulator(platform)
        metrics = simulator.benchmark(snn_model, num_samples=5)
        platform_results[platform] = {
            'energy_per_inference_mj': metrics.energy_per_inference_mj,
            'latency_ms': metrics.latency_ms,
            'throughput_inferences_per_sec': metrics.throughput_inferences_per_sec,
            'spike_rate': metrics.spike_rate
        }
    
    results['platform_comparison'] = platform_results
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Profiling results saved to {args.output}")
    
    return results


def cmd_encode(args):
    """Test spike encoding strategies."""
    print(f"🔥 Testing spike encoding: {args.encoding}")
    
    # Generate or load test data
    if args.input:
        with open(args.input, 'r') as f:
            if args.input.endswith('.json'):
                data = json.load(f)
                test_values = data.get('values', [0.1, 0.3, 0.5, 0.7, 0.9])
            else:
                test_values = [float(line.strip()) for line in f.readlines()]
    else:
        # Generate random test values
        import random
        test_values = [random.uniform(0, 1) for _ in range(args.num_values)]
    
    print(f"📊 Input values: {len(test_values)} values")
    print(f"   Range: [{min(test_values):.3f}, {max(test_values):.3f}]")
    
    # Apply encoding
    if args.encoding.lower() == 'rate':
        encoder = RateEncoder(timesteps=args.timesteps, max_rate=args.max_rate)
    elif args.encoding.lower() == 'temporal':
        encoder = TemporalEncoder(timesteps=args.timesteps)
    else:
        print(f"❌ Error: Unknown encoding type: {args.encoding}")
        return
    
    # Encode values
    spike_trains = encoder.encode(test_values)
    
    # Analyze results
    total_spikes = sum(sum(train) for train in spike_trains)
    spike_rate = total_spikes / (len(spike_trains) * args.timesteps)
    
    results = {
        'encoding_type': args.encoding,
        'timesteps': args.timesteps,
        'num_values': len(test_values),
        'total_spikes': total_spikes,
        'spike_rate': spike_rate,
        'spikes_per_value': total_spikes / len(test_values),
        'encoding_efficiency': spike_rate
    }
    
    print(f"✅ Encoding complete:")
    print(f"   Total spikes: {total_spikes}")
    print(f"   Spike rate: {spike_rate:.3f}")
    print(f"   Spikes per value: {results['spikes_per_value']:.2f}")
    
    # Save results
    if args.output:
        results['spike_trains'] = spike_trains  # Include raw data
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Encoding results saved to {args.output}")
    
    return results


def cmd_info(args):
    """Show system and toolkit information."""
    print("🧠 SpikeFormer Neuromorphic Toolkit")
    print("=" * 50)
    
    print(f"Version: 0.1.0 (Basic Demo)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    print("\n📊 Available Features:")
    print("  ✅ ANN to SNN Conversion")
    print("  ✅ Spike Encoding (Rate, Temporal)")
    print("  ✅ Hardware Simulation (Loihi2, SpiNNaker, CPU)")
    print("  ✅ Energy Profiling")
    print("  ✅ Performance Benchmarking")
    
    print("\n🔌 Supported Hardware Platforms:")
    print("  • Loihi 2 (Intel)")
    print("  • SpiNNaker (Manchester)")
    print("  • CPU (Baseline)")
    
    print("\n🧪 Neuron Models:")
    print("  • LIF (Leaky Integrate-and-Fire)")
    print("  • More models available in full version")
    
    print(f"\n💡 Quick start: python3 {sys.argv[0]} demo --quick")


def cmd_demo(args):
    """Run demonstration."""
    if args.quick:
        run_quick_demo()
    else:
        results = run_complete_demo()
        
        if args.save_results:
            output_file = args.save_results
            json_results = {
                "energy_reduction_factor": results["energy_results"]["energy_reduction_factor"],
                "best_platform": results["best_platform"],
                "benchmark_summary": {
                    platform: {
                        "energy_mj": metrics.energy_per_inference_mj,
                        "latency_ms": metrics.latency_ms,
                        "spike_rate": metrics.spike_rate
                    }
                    for platform, metrics in results["benchmark_results"].items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"📁 Demo results saved to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpikeFormer Neuromorphic Toolkit - Basic CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick demo
    python3 spikeformer_cli.py demo --quick
    
    # Convert model
    python3 spikeformer_cli.py convert model.json --timesteps 64 --output snn_model.json
    
    # Deploy to hardware
    python3 spikeformer_cli.py deploy snn_model.json --platform loihi2 --output deployment.json
    
    # Profile energy
    python3 spikeformer_cli.py profile snn_model.json --platforms loihi2 spinnaker --output profile.json
    
    # Test encoding
    python3 spikeformer_cli.py encode --encoding rate --timesteps 32 --num-values 100
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert ANN to SNN')
    convert_parser.add_argument('input', help='Input model config file or path')
    convert_parser.add_argument('--output', '-o', help='Output file for converted model')
    convert_parser.add_argument('--timesteps', type=int, default=32, help='Number of timesteps')
    convert_parser.add_argument('--threshold', type=float, default=1.0, help='Spike threshold')
    convert_parser.add_argument('--tau-mem', type=float, default=20.0, help='Membrane time constant')
    convert_parser.add_argument('--input-size', type=int, default=784, help='Input size (if not JSON)')
    convert_parser.add_argument('--hidden-layers', nargs='+', type=int, default=[256, 128], help='Hidden layer sizes')
    convert_parser.add_argument('--output-size', type=int, default=10, help='Output size')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model to hardware')
    deploy_parser.add_argument('input', help='Model configuration file')
    deploy_parser.add_argument('--platform', choices=['loihi2', 'spinnaker', 'cpu'], default='loihi2', help='Target platform')
    deploy_parser.add_argument('--output', '-o', help='Output file for deployment results')
    deploy_parser.add_argument('--no-benchmark', action='store_true', help='Skip benchmarking')
    deploy_parser.add_argument('--benchmark-samples', type=int, default=10, help='Number of benchmark samples')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile energy consumption')
    profile_parser.add_argument('input', help='Model configuration file')
    profile_parser.add_argument('--platforms', nargs='+', choices=['loihi2', 'spinnaker', 'cpu'], 
                               default=['loihi2', 'spinnaker'], help='Platforms to profile')
    profile_parser.add_argument('--num-samples', type=int, default=10, help='Number of test samples')
    profile_parser.add_argument('--output', '-o', help='Output file for profiling results')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Test spike encoding')
    encode_parser.add_argument('--encoding', choices=['rate', 'temporal'], default='rate', help='Encoding type')
    encode_parser.add_argument('--timesteps', type=int, default=32, help='Number of timesteps')
    encode_parser.add_argument('--max-rate', type=float, default=0.8, help='Maximum spike rate')
    encode_parser.add_argument('--num-values', type=int, default=10, help='Number of test values')
    encode_parser.add_argument('--input', help='Input file with values')
    encode_parser.add_argument('--output', '-o', help='Output file for encoding results')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--quick', action='store_true', help='Run quick demo')
    demo_parser.add_argument('--save-results', help='Save demo results to file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == 'convert':
            cmd_convert(args)
        elif args.command == 'deploy':
            cmd_deploy(args)
        elif args.command == 'profile':
            cmd_profile(args)
        elif args.command == 'encode':
            cmd_encode(args)
        elif args.command == 'demo':
            cmd_demo(args)
        elif args.command == 'info':
            cmd_info(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()