"""CLI commands for energy profiling and performance analysis."""

import click
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
import matplotlib.pyplot as plt
import numpy as np

from ..profiling import EnergyProfiler, PowerMonitor, EnergyComparison
from ..models import SpikingTransformer, SpikingViT


def profile_main(model_path: str, data_path: str, platform: str = "cpu",
                num_runs: int = 10, output_path: Optional[str] = None,
                verbose: bool = False):
    """Main profiling function."""
    logger = logging.getLogger(__name__)
    
    click.echo(f"Profiling model: {model_path}")
    click.echo(f"Platform: {platform}")
    click.echo(f"Number of runs: {num_runs}")
    
    # Load model
    try:
        model = load_profiling_model(model_path)
        click.echo(f"✓ Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        click.echo(f"✗ Failed to load model: {e}", err=True)
        return
    
    # Load test data
    try:
        test_data = load_profiling_data(data_path)
        click.echo(f"✓ Loaded test data: {test_data.shape}")
    except Exception as e:
        click.echo(f"✗ Failed to load data: {e}", err=True)
        return
    
    # Setup profiling
    profiler = create_profiler(platform)
    
    # Run profiling
    click.echo(f"Starting profiling on {platform}...")
    
    try:
        results = profiler.profile_inference(model, test_data, num_runs)
        
        # Display results
        display_profiling_results(results)
        
        # Save results
        if output_path:
            save_profiling_results(results, Path(output_path), model_path, platform)
            click.echo(f"✓ Saved results to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Profiling failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


def load_profiling_model(model_path: str) -> torch.nn.Module:
    """Load model for profiling."""
    model_data = torch.load(model_path, map_location='cpu')
    
    if isinstance(model_data, dict) and 'model_state_dict' in model_data:
        # Structured model data
        model_class = model_data.get('model_class')
        config = model_data.get('conversion_config', {})
        
        if model_class == 'SpikingTransformer':
            model = SpikingTransformer(
                vocab_size=config.get('vocab_size', 30522),
                hidden_size=config.get('hidden_size', 768),
                num_layers=config.get('num_layers', 12),
                num_heads=config.get('num_heads', 12),
                intermediate_size=config.get('intermediate_size', 3072),
                timesteps=config.get('timesteps', 32)
            )
        elif model_class == 'SpikingViT':
            model = SpikingViT(
                timesteps=config.get('timesteps', 32),
                num_classes=config.get('num_classes', 1000)
            )
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        model.load_state_dict(model_data['model_state_dict'])
    else:
        # Direct model
        model = model_data
    
    model.eval()
    return model


def load_profiling_data(data_path: str) -> torch.Tensor:
    """Load test data for profiling."""
    data_path_obj = Path(data_path)
    
    if data_path_obj.suffix in ['.pt', '.pth']:
        # PyTorch tensor
        data = torch.load(data_path, map_location='cpu')
        
        if isinstance(data, dict):
            # Use test/validation data if available
            data = data.get('test', data.get('val', data.get('validation')))
            
        if isinstance(data, (list, tuple)):
            data = data[0]  # Take inputs only
            
        return data[:10]  # Limit to 10 samples for profiling
    
    else:
        # Generate mock data
        click.echo("⚠ Using mock data for profiling")
        return torch.randn(10, 128, 768)  # Mock transformer input


def create_profiler(platform: str) -> EnergyProfiler:
    """Create appropriate profiler for platform."""
    from ..profiling import CPUEnergyMonitor, GPUEnergyMonitor, NeuromorphicEnergyMonitor
    
    if platform == "cpu":
        return EnergyProfiler([CPUEnergyMonitor()])
    elif platform == "gpu":
        return EnergyProfiler([GPUEnergyMonitor()])
    elif platform in ["loihi2", "spinnaker"]:
        return EnergyProfiler([NeuromorphicEnergyMonitor(platform)])
    else:
        # Default to CPU
        return EnergyProfiler([CPUEnergyMonitor()])


def display_profiling_results(results: Dict[str, Any]):
    """Display profiling results in a formatted way."""
    click.echo("\nProfiling Results:")
    click.echo("=" * 50)
    
    click.echo(f"Average Energy per Inference: {results['avg_energy_per_inference_mj']:.2f} mJ")
    click.echo(f"Total Energy Consumed: {results['total_energy_joules']:.4f} J")
    click.echo(f"Average Latency: {results['avg_latency_ms']:.2f} ms")
    click.echo(f"Number of Runs: {results['num_runs']}")
    
    # Energy breakdown by component
    if 'energy_per_component' in results:
        click.echo("\nEnergy Breakdown by Component:")
        for component, energy in results['energy_per_component'].items():
            click.echo(f"  {component}: {energy * 1000:.2f} mJ ({energy / results['total_energy_joules'] * 100:.1f}%)")
    
    # Performance metrics
    throughput = results['num_runs'] / (results['avg_latency_ms'] / 1000 * results['num_runs'])
    click.echo(f"\nThroughput: {throughput:.1f} inferences/sec")
    
    efficiency_score = 1000 / results['avg_energy_per_inference_mj']  # inferences per Joule
    click.echo(f"Energy Efficiency: {efficiency_score:.1f} inferences/J")


def save_profiling_results(results: Dict[str, Any], output_path: Path,
                          model_path: str, platform: str):
    """Save profiling results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive report
    report = {
        'profiling_metadata': {
            'model_path': model_path,
            'platform': platform,
            'timestamp': time.time(),
            'spikeformer_version': '0.1.0'
        },
        'results': results,
        'summary': {
            'avg_energy_mj': results['avg_energy_per_inference_mj'],
            'avg_latency_ms': results['avg_latency_ms'],
            'throughput_per_sec': results['num_runs'] / (results['avg_latency_ms'] / 1000 * results['num_runs']),
            'efficiency_score': 1000 / results['avg_energy_per_inference_mj']
        }
    }
    
    # Save as JSON
    if output_path.suffix == '.json' or not output_path.suffix:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Save as text report
    text_path = output_path.with_suffix('.txt')
    with open(text_path, 'w') as f:
        f.write(f"SpikeFormer Profiling Report\n")
        f.write(f"Generated: {time.ctime()}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Platform: {platform}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Average Energy per Inference: {results['avg_energy_per_inference_mj']:.2f} mJ\n")
        f.write(f"  Total Energy Consumed: {results['total_energy_joules']:.4f} J\n")
        f.write(f"  Average Latency: {results['avg_latency_ms']:.2f} ms\n")
        f.write(f"  Throughput: {report['summary']['throughput_per_sec']:.1f} inferences/sec\n")
        f.write(f"  Energy Efficiency: {report['summary']['efficiency_score']:.1f} inferences/J\n")


@click.command()
@click.argument('ann_model', type=click.Path(exists=True))
@click.argument('snn_model', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for comparison')
def compare_ann_snn(ann_model: str, snn_model: str, test_data: str, output: Optional[str]):
    """Compare energy consumption between ANN and SNN models."""
    click.echo("Comparing ANN vs SNN energy consumption")
    
    # Load models
    ann = torch.load(ann_model, map_location='cpu')
    snn = load_profiling_model(snn_model)
    
    # Load test data
    test_tensor = load_profiling_data(test_data)
    
    # Compare energy consumption
    comparison = EnergyComparison()
    results = comparison.compare_ann_vs_snn(ann, snn, test_tensor)
    
    # Display results
    click.echo("\nComparison Results:")
    click.echo(f"ANN Energy: {results['ann_results']['avg_energy_per_inference_mj']:.2f} mJ")
    click.echo(f"SNN Energy: {results['snn_results']['avg_energy_per_inference_mj']:.2f} mJ")
    click.echo(f"Energy Reduction: {results['energy_reduction_ratio']:.1%}")
    click.echo(f"Energy Improvement: {results['energy_improvement_factor']:.1f}x")
    
    # Latency comparison
    latency_comp = results['latency_comparison']
    click.echo(f"\nLatency Comparison:")
    click.echo(f"ANN Latency: {latency_comp['ann_latency_ms']:.2f} ms")
    click.echo(f"SNN Latency: {latency_comp['snn_latency_ms']:.2f} ms")
    click.echo(f"Latency Ratio: {latency_comp['latency_ratio']:.2f}x")
    
    if output:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"✓ Saved comparison to {output}")


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path(exists=True))
@click.option('--platforms', default='cpu,gpu,loihi2,spinnaker',
              help='Comma-separated list of platforms')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
def benchmark_platforms(model_path: str, test_data: str, platforms: str, output: Optional[str]):
    """Benchmark model across multiple platforms."""
    platform_list = [p.strip() for p in platforms.split(',')]
    
    click.echo(f"Benchmarking across platforms: {', '.join(platform_list)}")
    
    model = load_profiling_model(model_path)
    test_tensor = load_profiling_data(test_data)
    
    results = {}
    
    for platform in platform_list:
        click.echo(f"\nProfiling on {platform}...")
        
        try:
            profiler = create_profiler(platform)
            platform_results = profiler.profile_inference(model, test_tensor, num_runs=5)
            results[platform] = platform_results
            
            click.echo(f"  ✓ {platform}: {platform_results['avg_energy_per_inference_mj']:.2f} mJ, "
                      f"{platform_results['avg_latency_ms']:.2f} ms")
            
        except Exception as e:
            click.echo(f"  ✗ {platform}: Failed - {e}")
            results[platform] = {"error": str(e)}
    
    # Generate comparison report
    successful_platforms = [p for p, r in results.items() if "error" not in r]
    
    if successful_platforms:
        click.echo("\nPlatform Comparison:")
        
        # Find best energy efficiency
        best_energy = min(results[p]['avg_energy_per_inference_mj'] for p in successful_platforms)
        best_energy_platform = [p for p in successful_platforms 
                               if results[p]['avg_energy_per_inference_mj'] == best_energy][0]
        
        click.echo(f"Most energy efficient: {best_energy_platform} ({best_energy:.2f} mJ)")
        
        # Find lowest latency
        best_latency = min(results[p]['avg_latency_ms'] for p in successful_platforms)
        best_latency_platform = [p for p in successful_platforms 
                                if results[p]['avg_latency_ms'] == best_latency][0]
        
        click.echo(f"Lowest latency: {best_latency_platform} ({best_latency:.2f} ms)")
    
    if output:
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'platform_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plots if matplotlib is available
        try:
            create_comparison_plots(results, output_path)
            click.echo(f"✓ Saved results and plots to {output_path}")
        except ImportError:
            click.echo(f"✓ Saved results to {output_path} (matplotlib not available for plots)")


def create_comparison_plots(results: Dict[str, Any], output_path: Path):
    """Create comparison plots."""
    successful_platforms = [p for p, r in results.items() if "error" not in r]
    
    if len(successful_platforms) < 2:
        return
    
    # Energy comparison
    platforms = successful_platforms
    energies = [results[p]['avg_energy_per_inference_mj'] for p in platforms]
    latencies = [results[p]['avg_latency_ms'] for p in platforms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Energy plot
    bars1 = ax1.bar(platforms, energies)
    ax1.set_ylabel('Energy (mJ)')
    ax1.set_title('Energy Consumption by Platform')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, energy in zip(bars1, energies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.01,
                f'{energy:.2f}', ha='center', va='bottom')
    
    # Latency plot
    bars2 = ax2.bar(platforms, latencies)
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency by Platform')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, latency in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latencies)*0.01,
                f'{latency:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'platform_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Energy efficiency plot
    plt.figure(figsize=(10, 6))
    efficiencies = [1000 / results[p]['avg_energy_per_inference_mj'] for p in platforms]
    
    bars = plt.bar(platforms, efficiencies)
    plt.ylabel('Efficiency (inferences/J)')
    plt.title('Energy Efficiency by Platform')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, eff in zip(bars, efficiencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiencies)*0.01,
                f'{eff:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--duration', default=60, help='Monitoring duration in seconds')
@click.option('--sampling-rate', default=1.0, help='Sampling rate in Hz')
@click.option('--output', '-o', type=click.Path(), help='Output file for monitoring data')
def monitor_power(model_path: str, duration: int, sampling_rate: float, output: Optional[str]):
    """Monitor real-time power consumption."""
    click.echo(f"Monitoring power consumption for {duration} seconds")
    
    model = load_profiling_model(model_path)
    
    # Setup power monitor
    monitor = PowerMonitor(sampling_rate_hz=sampling_rate)
    monitor.start_monitoring()
    
    click.echo("Power monitoring started (press Ctrl+C to stop early)")
    
    try:
        # Run inference continuously while monitoring
        test_input = torch.randn(1, 128, 768)
        
        start_time = time.time()
        inference_count = 0
        
        while time.time() - start_time < duration:
            with torch.no_grad():
                _ = model(test_input)
            inference_count += 1
            
            # Display current power every 5 seconds
            if inference_count % 100 == 0:
                current_power = monitor.get_current_power()
                elapsed = time.time() - start_time
                click.echo(f"\rElapsed: {elapsed:.1f}s, Power: {current_power:.2f}W, "
                          f"Inferences: {inference_count}", nl=False)
            
            time.sleep(0.01)  # Small delay
    
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped by user")
    
    monitor.stop_monitoring()
    
    # Get statistics
    stats = monitor.get_power_statistics()
    
    click.echo(f"\nPower Statistics:")
    click.echo(f"  Average Power: {stats.get('avg_power_watts', 0):.2f} W")
    click.echo(f"  Peak Power: {stats.get('max_power_watts', 0):.2f} W")
    click.echo(f"  Min Power: {stats.get('min_power_watts', 0):.2f} W")
    click.echo(f"  Total Inferences: {inference_count}")
    
    if output:
        monitor.export_power_profile(output)
        click.echo(f"✓ Saved power profile to {output}")


@click.command()
@click.argument('profiling_data', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['json', 'csv', 'html']),
              default='html', help='Report format')
@click.option('--output', '-o', type=click.Path(), help='Output file')
def generate_report(profiling_data: str, format: str, output: Optional[str]):
    """Generate comprehensive profiling report."""
    click.echo(f"Generating {format.upper()} report from {profiling_data}")
    
    # Load profiling data
    with open(profiling_data, 'r') as f:
        data = json.load(f)
    
    if format == 'html':
        generate_html_report(data, output or 'profiling_report.html')
    elif format == 'csv':
        generate_csv_report(data, output or 'profiling_report.csv')
    elif format == 'json':
        # Already in JSON format
        if output:
            with open(output, 'w') as f:
                json.dump(data, f, indent=2)
    
    click.echo(f"✓ Report generated: {output}")


def generate_html_report(data: Dict[str, Any], output_path: str):
    """Generate HTML profiling report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SpikeFormer Profiling Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .value {{ font-weight: bold; color: #2e8b57; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SpikeFormer Profiling Report</h1>
            <p>Generated: {time.ctime()}</p>
            <p>Model: {data.get('profiling_metadata', {}).get('model_path', 'N/A')}</p>
            <p>Platform: {data.get('profiling_metadata', {}).get('platform', 'N/A')}</p>
        </div>
        
        <h2>Summary</h2>
        <div class="metric">Average Energy per Inference: <span class="value">{data.get('summary', {}).get('avg_energy_mj', 0):.2f} mJ</span></div>
        <div class="metric">Average Latency: <span class="value">{data.get('summary', {}).get('avg_latency_ms', 0):.2f} ms</span></div>
        <div class="metric">Throughput: <span class="value">{data.get('summary', {}).get('throughput_per_sec', 0):.1f} inferences/sec</span></div>
        <div class="metric">Energy Efficiency: <span class="value">{data.get('summary', {}).get('efficiency_score', 0):.1f} inferences/J</span></div>
        
        <h2>Detailed Results</h2>
        <pre>{json.dumps(data.get('results', {}), indent=2)}</pre>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def generate_csv_report(data: Dict[str, Any], output_path: str):
    """Generate CSV profiling report."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Metric', 'Value', 'Unit'])
        
        # Summary metrics
        summary = data.get('summary', {})
        writer.writerow(['Average Energy per Inference', f"{summary.get('avg_energy_mj', 0):.2f}", 'mJ'])
        writer.writerow(['Average Latency', f"{summary.get('avg_latency_ms', 0):.2f}", 'ms'])
        writer.writerow(['Throughput', f"{summary.get('throughput_per_sec', 0):.1f}", 'inferences/sec'])
        writer.writerow(['Energy Efficiency', f"{summary.get('efficiency_score', 0):.1f}", 'inferences/J'])


if __name__ == '__main__':
    profile_main()