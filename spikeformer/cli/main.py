"""Main CLI entry point for SpikeFormer."""

import click
import sys
import logging
from pathlib import Path
from typing import Optional

from .. import __version__
from .convert import convert_main
from .train import train_main  
from .deploy import deploy_main
from .profile import profile_main


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-file', type=click.Path(), help='Log file path')
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx, verbose: bool, log_file: Optional[str]):
    """SpikeFormer: Complete toolkit for spiking neural networks.
    
    Convert pre-trained transformers to energy-efficient spiking networks
    and deploy them on neuromorphic hardware platforms.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['log_file'] = log_file
    
    setup_logging(verbose, log_file)


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), 
              help='Output path for converted model')
@click.option('--timesteps', '-t', default=32, 
              help='Number of timesteps for spike encoding')
@click.option('--threshold', default=1.0, 
              help='Spike threshold for neurons')
@click.option('--neuron-model', default='LIF', 
              type=click.Choice(['LIF', 'ADLIF', 'IZHIKEVICH']),
              help='Neuron model type')
@click.option('--calibration-samples', default=100,
              help='Number of samples for threshold calibration')
@click.pass_context
def convert(ctx, model_path: str, output: Optional[str], timesteps: int,
           threshold: float, neuron_model: str, calibration_samples: int):
    """Convert ANN model to SNN."""
    convert_main(
        model_path=model_path,
        output_path=output,
        timesteps=timesteps,
        threshold=threshold,
        neuron_model=neuron_model,
        calibration_samples=calibration_samples,
        verbose=ctx.obj['verbose']
    )


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for trained model')
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Training batch size')
@click.option('--learning-rate', default=1e-4, help='Learning rate')
@click.option('--hardware-aware', is_flag=True,
              help='Enable hardware-aware training')
@click.pass_context
def train(ctx, model_path: str, data_path: str, output: Optional[str],
         epochs: int, batch_size: int, learning_rate: float, hardware_aware: bool):
    """Train spiking neural network."""
    train_main(
        model_path=model_path,
        data_path=data_path,
        output_path=output,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hardware_aware=hardware_aware,
        verbose=ctx.obj['verbose']
    )


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--platform', default='loihi2',
              type=click.Choice(['loihi2', 'spinnaker', 'edge']),
              help='Target hardware platform')
@click.option('--num-chips', default=1, help='Number of chips to use')
@click.option('--power-budget', type=float, help='Power budget in mW')
@click.option('--output', '-o', type=click.Path(),
              help='Output directory for deployment package')
@click.pass_context
def deploy(ctx, model_path: str, platform: str, num_chips: int,
          power_budget: Optional[float], output: Optional[str]):
    """Deploy model to neuromorphic hardware."""
    deploy_main(
        model_path=model_path,
        platform=platform,
        num_chips=num_chips,
        power_budget=power_budget,
        output_path=output,
        verbose=ctx.obj['verbose']
    )


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--platform', default='cpu',
              type=click.Choice(['cpu', 'gpu', 'loihi2', 'spinnaker']),
              help='Platform to profile on')
@click.option('--num-runs', default=10, help='Number of profiling runs')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for profiling results')
@click.pass_context
def profile(ctx, model_path: str, data_path: str, platform: str,
           num_runs: int, output: Optional[str]):
    """Profile model energy consumption."""
    profile_main(
        model_path=model_path,
        data_path=data_path,
        platform=platform,
        num_runs=num_runs,
        output_path=output,
        verbose=ctx.obj['verbose']
    )


@cli.command()
def info():
    """Show system information and available hardware."""
    import torch
    import psutil
    from ..health import get_health_summary
    
    click.echo(f"SpikeFormer v{__version__}")
    click.echo("=" * 50)
    
    # System info
    click.echo("System Information:")
    click.echo(f"  Python: {sys.version.split()[0]}")
    click.echo(f"  PyTorch: {torch.__version__}")
    click.echo(f"  CPU Cores: {psutil.cpu_count()}")
    click.echo(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        click.echo(f"  CUDA: {torch.version.cuda}")
        click.echo(f"  GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            click.echo(f"    GPU {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")
    else:
        click.echo("  CUDA: Not available")
    
    # Neuromorphic hardware
    click.echo("\nNeuromorphic Hardware:")
    
    # Check Loihi 2
    try:
        import nxsdk
        click.echo(f"  Loihi 2: Available (NxSDK {getattr(nxsdk, '__version__', 'unknown')})")
    except ImportError:
        click.echo("  Loihi 2: Not available (NxSDK not installed)")
    
    # Check SpiNNaker
    try:
        import spynnaker
        click.echo(f"  SpiNNaker: Available (sPyNNaker {getattr(spynnaker, '__version__', 'unknown')})")
    except ImportError:
        click.echo("  SpiNNaker: Not available (sPyNNaker not installed)")
    
    # Health check
    click.echo("\nHealth Status:")
    health = get_health_summary()
    click.echo(f"  Overall Status: {health['status'].upper()}")
    
    for check_name, check_result in health['checks'].items():
        status_color = 'green' if check_result['status'] == 'healthy' else 'yellow' if check_result['status'] == 'degraded' else 'red'
        click.echo(f"  {check_name}: ", nl=False)
        click.secho(check_result['status'].upper(), fg=status_color)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start SpikeFormer API server."""
    try:
        import uvicorn
        from ..api import app
        
        click.echo(f"Starting SpikeFormer API server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=reload)
    except ImportError:
        click.echo("API server dependencies not installed. Install with: pip install spikeformer[api]")
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def batch(config_file: str):
    """Run batch operations from config file."""
    import yaml
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    click.echo(f"Running batch operations from {config_file}")
    
    for operation in config.get('operations', []):
        op_type = operation.get('type')
        
        if op_type == 'convert':
            click.echo(f"Converting {operation['model_path']}")
            convert_main(**operation.get('params', {}))
        elif op_type == 'train':
            click.echo(f"Training {operation['model_path']}")
            train_main(**operation.get('params', {}))
        elif op_type == 'deploy':
            click.echo(f"Deploying {operation['model_path']}")
            deploy_main(**operation.get('params', {}))
        elif op_type == 'profile':
            click.echo(f"Profiling {operation['model_path']}")
            profile_main(**operation.get('params', {}))
        else:
            click.echo(f"Unknown operation type: {op_type}")


# Aliases for common commands
@cli.command(name='c')
@click.pass_context
def convert_alias(ctx):
    """Alias for convert command."""
    ctx.invoke(convert)


@cli.command(name='t')
@click.pass_context  
def train_alias(ctx):
    """Alias for train command."""
    ctx.invoke(train)


@cli.command(name='d')
@click.pass_context
def deploy_alias(ctx):
    """Alias for deploy command."""
    ctx.invoke(deploy)


@cli.command(name='p')
@click.pass_context
def profile_alias(ctx):
    """Alias for profile command."""
    ctx.invoke(profile)


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()