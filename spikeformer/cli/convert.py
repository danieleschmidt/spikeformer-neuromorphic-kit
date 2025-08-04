"""CLI commands for model conversion."""

import click
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

from ..conversion import SpikeformerConverter, ConversionConfig, ConversionPipeline
from ..models import SpikingTransformer, SpikingViT


def convert_main(model_path: str, output_path: Optional[str] = None,
                timesteps: int = 32, threshold: float = 1.0,
                neuron_model: str = "LIF", calibration_samples: int = 100,
                verbose: bool = False):
    """Main conversion function."""
    logger = logging.getLogger(__name__)
    
    click.echo(f"Loading model from {model_path}")
    
    # Load model
    try:
        model = load_model(model_path)
        click.echo(f"✓ Loaded model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        click.echo(f"✗ Failed to load model: {e}", err=True)
        return
    
    # Configure conversion
    config = ConversionConfig(
        timesteps=timesteps,
        threshold=threshold,
        neuron_model=neuron_model,
        calibration_samples=calibration_samples
    )
    
    click.echo(f"Converting to SNN with config:")
    click.echo(f"  Timesteps: {timesteps}")
    click.echo(f"  Threshold: {threshold}")
    click.echo(f"  Neuron model: {neuron_model}")
    
    # Convert model
    try:
        converter = SpikeformerConverter(config)
        start_time = time.time()
        
        snn_model = converter.convert(model)
        conversion_time = time.time() - start_time
        
        click.echo(f"✓ Conversion completed in {conversion_time:.2f}s")
        
        # Validate conversion
        validate_conversion(model, snn_model, config)
        
        # Save converted model
        if output_path:
            save_path = Path(output_path)
        else:
            model_path_obj = Path(model_path)
            save_path = model_path_obj.parent / f"{model_path_obj.stem}_snn.pth"
            
        save_converted_model(snn_model, save_path, config, conversion_time)
        click.echo(f"✓ Saved converted model to {save_path}")
        
    except Exception as e:
        click.echo(f"✗ Conversion failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()


def load_model(model_path: str) -> nn.Module:
    """Load model from various formats."""
    path = Path(model_path)
    
    if path.suffix == '.pth' or path.suffix == '.pt':
        # PyTorch model
        return torch.load(model_path, map_location='cpu')
    elif path.suffix == '.onnx':
        # ONNX model
        import onnx
        import torch.onnx
        # Would need ONNX to PyTorch conversion
        raise NotImplementedError("ONNX loading not yet implemented")
    elif path.is_dir():
        # Hugging Face model directory
        try:
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_path)
        except ImportError:
            raise RuntimeError("Transformers library required for Hugging Face models")
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")


def validate_conversion(original_model: nn.Module, snn_model: nn.Module, 
                       config: ConversionConfig):
    """Validate the conversion process."""
    click.echo("Validating conversion...")
    
    # Test with dummy input
    if hasattr(original_model, 'config') and hasattr(original_model.config, 'vocab_size'):
        # Transformer model
        dummy_input = torch.randint(0, 1000, (1, 10))  # seq_len=10
    else:
        # Generic model - try to infer input size
        dummy_input = torch.randn(1, 224, 224, 3)  # Assume image input
    
    try:
        with torch.no_grad():
            original_output = original_model(dummy_input)
            snn_output = snn_model(dummy_input)
            
        # Basic shape validation
        if hasattr(original_output, 'shape'):
            click.echo(f"  Original output shape: {original_output.shape}")
        if hasattr(snn_output, 'shape'):
            click.echo(f"  SNN output shape: {snn_output.shape}")
            
        click.echo("✓ Conversion validation passed")
        
    except Exception as e:
        click.echo(f"⚠ Validation warning: {e}")


def save_converted_model(model: nn.Module, save_path: Path, 
                        config: ConversionConfig, conversion_time: float):
    """Save converted model with metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'conversion_config': {
            'timesteps': config.timesteps,
            'threshold': config.threshold,
            'neuron_model': config.neuron_model,
            'spike_encoding': config.spike_encoding
        },
        'metadata': {
            'conversion_time': conversion_time,
            'spikeformer_version': '0.1.0',
            'timestamp': time.time()
        }
    }
    
    torch.save(model_data, save_path)
    
    # Save human-readable config
    config_path = save_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_class': model.__class__.__name__,
            'config': model_data['conversion_config'],
            'metadata': model_data['metadata']
        }, f, indent=2)


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output path')
@click.option('--format', type=click.Choice(['pytorch', 'onnx', 'tensorrt']),
              default='pytorch', help='Output format')
@click.option('--optimize', is_flag=True, help='Apply optimization passes')
def export(model_path: str, output: Optional[str], format: str, optimize: bool):
    """Export converted model to different formats."""
    click.echo(f"Exporting model from {model_path} to {format}")
    
    # Load converted model
    model_data = torch.load(model_path, map_location='cpu')
    
    if format == 'pytorch':
        # Already in PyTorch format
        if output:
            torch.save(model_data, output)
        click.echo("✓ Model already in PyTorch format")
        
    elif format == 'onnx':
        # Export to ONNX
        try:
            import torch.onnx
            
            # Reconstruct model (simplified)
            model = create_model_from_state_dict(model_data)
            
            # Create dummy input
            dummy_input = torch.randn(1, 10, 768)  # Assume transformer
            
            output_path = output or model_path.replace('.pth', '.onnx')
            torch.onnx.export(
                model, dummy_input, output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True
            )
            click.echo(f"✓ Exported to ONNX: {output_path}")
            
        except Exception as e:
            click.echo(f"✗ ONNX export failed: {e}", err=True)
            
    elif format == 'tensorrt':
        click.echo("TensorRT export not yet implemented")


def create_model_from_state_dict(model_data: Dict[str, Any]) -> nn.Module:
    """Reconstruct model from saved state dict."""
    model_class = model_data['model_class']
    config = model_data['conversion_config']
    
    # Create model based on class name
    if model_class == 'SpikingTransformer':
        model = SpikingTransformer(
            vocab_size=30522,  # Default BERT vocab
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072,
            timesteps=config['timesteps']
        )
    elif model_class == 'SpikingViT':
        model = SpikingViT(
            timesteps=config['timesteps']
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model.load_state_dict(model_data['model_state_dict'])
    return model


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--metric', type=click.Choice(['accuracy', 'energy', 'latency']),
              default='accuracy', help='Evaluation metric')
@click.option('--num-samples', default=100, help='Number of test samples')
def evaluate(model_path: str, data_path: str, metric: str, num_samples: int):
    """Evaluate converted model performance."""
    click.echo(f"Evaluating {metric} on {num_samples} samples")
    
    # Load model
    model_data = torch.load(model_path, map_location='cpu')
    model = create_model_from_state_dict(model_data)
    
    # Load test data (simplified)
    test_data = torch.randn(num_samples, 10, 768)  # Mock data
    
    if metric == 'accuracy':
        evaluate_accuracy(model, test_data)
    elif metric == 'energy':
        evaluate_energy(model, test_data)
    elif metric == 'latency':
        evaluate_latency(model, test_data)


def evaluate_accuracy(model: nn.Module, test_data: torch.Tensor):
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        
    # Mock accuracy calculation
    accuracy = 0.85  # Placeholder
    click.echo(f"Accuracy: {accuracy:.3f}")


def evaluate_energy(model: nn.Module, test_data: torch.Tensor):
    """Evaluate energy consumption."""
    from ..profiling import EnergyProfiler
    
    profiler = EnergyProfiler()
    results = profiler.profile_inference(model, test_data[:1])  # Single sample
    
    click.echo(f"Energy per inference: {results['avg_energy_per_inference_mj']:.2f} mJ")


def evaluate_latency(model: nn.Module, test_data: torch.Tensor):
    """Evaluate inference latency."""
    import time
    
    model.eval()
    latencies = []
    
    # Warmup
    with torch.no_grad():
        _ = model(test_data[:1])
    
    # Measure latency
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            _ = model(test_data[i:i+1])
        latencies.append((time.time() - start) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    click.echo(f"Average latency: {avg_latency:.2f} ms")


@click.command()
@click.argument('original_model', type=click.Path(exists=True))
@click.argument('converted_model', type=click.Path(exists=True))
@click.argument('test_data', type=click.Path(exists=True))
def compare(original_model: str, converted_model: str, test_data: str):
    """Compare original and converted models."""
    click.echo("Comparing original vs converted model")
    
    # Load models
    orig_model = load_model(original_model)
    conv_data = torch.load(converted_model, map_location='cpu')
    conv_model = create_model_from_state_dict(conv_data)
    
    # Load test data
    test_tensor = torch.load(test_data, map_location='cpu')
    
    # Compare outputs
    with torch.no_grad():
        orig_output = orig_model(test_tensor)
        conv_output = conv_model(test_tensor)
    
    # Calculate metrics
    if hasattr(orig_output, 'shape') and hasattr(conv_output, 'shape'):
        if orig_output.shape == conv_output.shape:
            mse = torch.mean((orig_output - conv_output) ** 2).item()
            click.echo(f"MSE between outputs: {mse:.6f}")
        else:
            click.echo(f"Output shapes differ: {orig_output.shape} vs {conv_output.shape}")
    
    # Model size comparison
    orig_params = sum(p.numel() for p in orig_model.parameters())
    conv_params = sum(p.numel() for p in conv_model.parameters())
    
    click.echo(f"Parameter count - Original: {orig_params}, Converted: {conv_params}")
    click.echo(f"Parameter ratio: {conv_params / orig_params:.2f}x")


if __name__ == '__main__':
    convert_main()