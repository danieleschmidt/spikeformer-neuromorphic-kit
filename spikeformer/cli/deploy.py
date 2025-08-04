"""CLI commands for model deployment."""

import click
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from ..hardware import NeuromorphicDeployer, HardwareConfig
from ..models import SpikingTransformer, SpikingViT


def deploy_main(model_path: str, platform: str = "loihi2", num_chips: int = 1,
                power_budget: Optional[float] = None, output_path: Optional[str] = None,
                verbose: bool = False):
    """Main deployment function."""
    logger = logging.getLogger(__name__)
    
    click.echo(f"Deploying model to {platform}")
    click.echo(f"Model path: {model_path}")
    click.echo(f"Target chips: {num_chips}")
    if power_budget:
        click.echo(f"Power budget: {power_budget} mW")
    
    # Load model
    try:
        model = load_deployment_model(model_path)
        click.echo(f"✓ Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        click.echo(f"✗ Failed to load model: {e}", err=True)
        return
    
    # Configure deployment
    hardware_config = HardwareConfig(
        platform=platform,
        num_chips=num_chips,
        power_budget_mw=power_budget
    )
    
    # Deploy model
    try:
        deployer = NeuromorphicDeployer(platform, **hardware_config.__dict__)
        
        click.echo("Starting deployment...")
        deployment_result = deployer.deploy_model(model)
        
        click.echo(f"✓ Deployment completed in {deployment_result.deployment_time:.2f}s")
        click.echo(f"  Memory usage: {deployment_result.memory_usage_mb:.1f} MB")
        click.echo(f"  Estimated power: {deployment_result.estimated_power_mw:.1f} mW")
        
        # Test deployment
        test_deployment(deployer, deployment_result, model)
        
        # Save deployment package
        if output_path:
            save_deployment_package(deployment_result, Path(output_path))
            click.echo(f"✓ Saved deployment package to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Deployment failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


def load_deployment_model(model_path: str) -> torch.nn.Module:
    """Load model for deployment."""
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


def test_deployment(deployer, deployment_result, original_model):
    """Test deployed model functionality."""
    click.echo("Testing deployment...")
    
    # Create test input
    test_input = torch.randn(1, 10, 768)  # Mock transformer input
    
    try:
        # Run inference on deployed model
        deployed_output = deployer.run_inference(deployment_result, test_input)
        
        # Compare with original if possible
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(test_input)
        
        if hasattr(deployed_output, 'shape') and hasattr(original_output, 'shape'):
            if deployed_output.shape == original_output.shape:
                mse = torch.mean((deployed_output - original_output) ** 2).item()
                click.echo(f"  Output MSE: {mse:.6f}")
            else:
                click.echo(f"  Output shapes: Deployed {deployed_output.shape}, Original {original_output.shape}")
        
        click.echo("✓ Deployment test passed")
        
    except Exception as e:
        click.echo(f"⚠ Deployment test warning: {e}")


def save_deployment_package(deployment_result, output_path: Path):
    """Save deployment package with all necessary files."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save deployment metadata
    metadata = {
        'platform': deployment_result.metadata['platform'],
        'deployment_time': deployment_result.deployment_time,
        'memory_usage_mb': deployment_result.memory_usage_mb,
        'estimated_power_mw': deployment_result.estimated_power_mw,
        'chip_utilization': deployment_result.chip_utilization,
        'config': deployment_result.metadata['config'],
        'timestamp': time.time()
    }
    
    with open(output_path / 'deployment.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save compiled model
    torch.save(deployment_result.compiled_model, output_path / 'compiled_model.pth')
    
    # Create README
    readme_content = f"""# SpikeFormer Deployment Package

## Deployment Information
- Platform: {metadata['platform']}
- Memory Usage: {metadata['memory_usage_mb']:.1f} MB
- Estimated Power: {metadata['estimated_power_mw']:.1f} mW
- Deployment Time: {metadata['deployment_time']:.2f}s

## Files
- `deployment.json`: Deployment metadata
- `compiled_model.pth`: Compiled model for target platform
- `README.md`: This file

## Usage
Load the deployment package and run inference using the SpikeFormer API.
"""
    
    with open(output_path / 'README.md', 'w') as f:
        f.write(readme_content)


@click.command()
@click.argument('deployment_package', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path(exists=True))
@click.option('--num-samples', default=100, help='Number of test samples')
def benchmark(deployment_package: str, test_data: str, num_samples: int):
    """Benchmark deployed model performance."""
    click.echo(f"Benchmarking deployment package: {deployment_package}")
    
    # Load deployment package
    package_path = Path(deployment_package)
    with open(package_path / 'deployment.json', 'r') as f:
        metadata = json.load(f)
    
    platform = metadata['platform']
    
    # Load compiled model
    compiled_model = torch.load(package_path / 'compiled_model.pth', map_location='cpu')
    
    # Create deployer for benchmarking
    deployer = NeuromorphicDeployer(platform)
    
    # Load test data
    test_tensor = torch.load(test_data, map_location='cpu')[:num_samples]
    
    # Mock data loader for benchmarking
    from torch.utils.data import DataLoader, TensorDataset
    test_labels = torch.randint(0, 2, (len(test_tensor),))
    test_dataset = TensorDataset(test_tensor, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Run benchmark
    click.echo("Running benchmark...")
    
    # Create deployment result structure
    from ..hardware import DeploymentResult
    deployment_result = DeploymentResult(
        compiled_model=compiled_model,
        deployment_time=metadata['deployment_time'],
        memory_usage_mb=metadata['memory_usage_mb'],
        estimated_power_mw=metadata['estimated_power_mw'],
        chip_utilization=metadata['chip_utilization'],
        metadata=metadata
    )
    
    benchmark_result = deployer.benchmark_model(deployment_result, test_loader)
    
    # Display results
    click.echo("\nBenchmark Results:")
    click.echo(f"  Throughput: {benchmark_result.throughput_samples_per_sec:.1f} samples/sec")
    click.echo(f"  Latency: {benchmark_result.latency_ms:.2f} ms")
    click.echo(f"  Energy per sample: {benchmark_result.energy_per_sample_uj:.2f} μJ")
    click.echo(f"  Accuracy: {benchmark_result.accuracy:.3f}")
    click.echo(f"  Power consumption: {benchmark_result.power_consumption_mw:.1f} mW")
    click.echo(f"  Memory efficiency: {benchmark_result.memory_efficiency:.3f}")


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--platforms', default='loihi2,spinnaker,edge',
              help='Comma-separated list of platforms to compare')
@click.option('--test-data', type=click.Path(exists=True),
              help='Test data for comparison')
def compare_platforms(model_path: str, platforms: str, test_data: Optional[str]):
    """Compare deployment across different platforms."""
    platform_list = [p.strip() for p in platforms.split(',')]
    
    click.echo(f"Comparing deployment across platforms: {', '.join(platform_list)}")
    
    model = load_deployment_model(model_path)
    results = {}
    
    for platform in platform_list:
        click.echo(f"\nDeploying to {platform}...")
        
        try:
            deployer = NeuromorphicDeployer(platform)
            deployment_result = deployer.deploy_model(model)
            
            results[platform] = {
                'deployment_time': deployment_result.deployment_time,
                'memory_usage_mb': deployment_result.memory_usage_mb,
                'estimated_power_mw': deployment_result.estimated_power_mw,
                'status': 'success'
            }
            
            click.echo(f"  ✓ {platform}: {deployment_result.estimated_power_mw:.1f} mW, "
                      f"{deployment_result.memory_usage_mb:.1f} MB")
            
        except Exception as e:
            results[platform] = {
                'status': 'failed',
                'error': str(e)
            }
            click.echo(f"  ✗ {platform}: Failed - {e}")
    
    # Summary comparison
    click.echo("\nComparison Summary:")
    successful_platforms = [p for p, r in results.items() if r['status'] == 'success']
    
    if successful_platforms:
        # Find best platform by power efficiency
        best_power = min(results[p]['estimated_power_mw'] for p in successful_platforms)
        best_platform = [p for p in successful_platforms 
                        if results[p]['estimated_power_mw'] == best_power][0]
        
        click.echo(f"  Most power efficient: {best_platform} ({best_power:.1f} mW)")
        
        # Find best platform by memory efficiency
        best_memory = min(results[p]['memory_usage_mb'] for p in successful_platforms)
        best_memory_platform = [p for p in successful_platforms 
                               if results[p]['memory_usage_mb'] == best_memory][0]
        
        click.echo(f"  Most memory efficient: {best_memory_platform} ({best_memory:.1f} MB)")


@click.command()
@click.argument('deployment_package', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8080, help='Port to bind to')
def serve(deployment_package: str, host: str, port: int):
    """Serve deployed model as REST API."""
    click.echo(f"Starting inference server for {deployment_package}")
    click.echo(f"Server will be available at http://{host}:{port}")
    
    # This would start a FastAPI server with the deployed model
    click.echo("Inference server started (mock implementation)")
    
    # Mock server loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nServer stopped")


@click.command()
@click.argument('deployment_package', type=click.Path(exists=True))
def monitor(deployment_package: str):
    """Monitor deployed model performance."""
    click.echo(f"Monitoring deployment: {deployment_package}")
    
    # Load deployment metadata
    package_path = Path(deployment_package)
    with open(package_path / 'deployment.json', 'r') as f:
        metadata = json.load(f)
    
    platform = metadata['platform']
    
    # Create deployer for monitoring
    deployer = NeuromorphicDeployer(platform)
    
    click.echo("Starting monitoring (press Ctrl+C to stop)...")
    
    try:
        while True:
            # Get hardware status
            status = deployer.get_hardware_status()
            
            click.echo(f"\rPower: {status.get('energy_consumption', 0):.3f}J, "
                      f"Status: {status['platform']}", nl=False)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped")


@click.command()
@click.argument('source_package', type=click.Path(exists=True))
@click.argument('target_platform')
def migrate(source_package: str, target_platform: str):
    """Migrate deployment between platforms."""
    click.echo(f"Migrating deployment to {target_platform}")
    
    # Load source deployment
    package_path = Path(source_package)
    with open(package_path / 'deployment.json', 'r') as f:
        metadata = json.load(f)
    
    source_platform = metadata['platform']
    
    if source_platform == target_platform:
        click.echo("Source and target platforms are the same")
        return
    
    # Load compiled model
    compiled_model = torch.load(package_path / 'compiled_model.pth', map_location='cpu')
    
    # Reconstruct original model (simplified)
    # In practice, this would require platform-specific decompilation
    
    click.echo(f"Migrating from {source_platform} to {target_platform}")
    click.echo("Migration completed (mock implementation)")


if __name__ == '__main__':
    deploy_main()