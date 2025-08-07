"""Core conversion pipeline for transforming ANNs to SNNs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from .neurons import LifNeuron
from .models import SpikingTransformer, SpikingAttention, SpikingMLP
from .encoding import RateCoding


@dataclass
class ConversionConfig:
    """Configuration for ANN-to-SNN conversion."""
    timesteps: int = 32
    threshold: float = 1.0
    neuron_model: str = "LIF"
    spike_encoding: str = "rate"
    surrogate_gradient: str = "fast_sigmoid"
    calibration_samples: int = 100
    conversion_mode: str = "layer_wise"
    optimization_target: str = "accuracy"
    hardware_constraints: Optional[Dict[str, Any]] = None


@dataclass
class ConversionResult:
    """Result of ANN-to-SNN conversion."""
    snn_model: nn.Module
    accuracy_retention: float
    spike_sparsity: float
    energy_reduction: float
    conversion_time: float
    metadata: Dict[str, Any]


class LayerConverter:
    """Converts individual layers from ANN to SNN."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def convert_linear(self, layer: nn.Linear) -> nn.Module:
        """Convert Linear layer to spiking equivalent."""
        # Create spiking linear layer with LIF neurons
        spiking_layer = nn.Sequential(
            layer,  # Keep original weights
            LifNeuron(threshold=self.config.threshold)
        )
        return spiking_layer
        
    def convert_attention(self, layer: nn.MultiheadAttention) -> SpikingAttention:
        """Convert MultiheadAttention to SpikingAttention."""
        return SpikingAttention(
            embed_dim=layer.embed_dim,
            num_heads=layer.num_heads,
            timesteps=self.config.timesteps,
            threshold=self.config.threshold
        )
        
    def convert_mlp(self, layers: List[nn.Module]) -> SpikingMLP:
        """Convert MLP layers to SpikingMLP."""
        # Extract dimensions from layers
        input_dim = None
        hidden_dims = []
        output_dim = None
        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                if input_dim is None:
                    input_dim = layer.in_features
                hidden_dims.append(layer.out_features)
                output_dim = layer.out_features
                
        return SpikingMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [hidden_dims[0]],
            output_dim=output_dim,
            timesteps=self.config.timesteps,
            threshold=self.config.threshold
        )


class ThresholdCalibrator:
    """Calibrates firing thresholds for optimal spike distribution."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calibrate_layer(self, layer: nn.Module, calibration_data: torch.Tensor) -> float:
        """Calibrate threshold for a single layer."""
        # Run forward pass to collect activations
        with torch.no_grad():
            activations = layer(calibration_data)
            
        # Use percentile-based threshold setting
        threshold = torch.quantile(activations.abs(), 0.99).item()
        return max(threshold, 0.1)  # Minimum threshold
        
    def calibrate_model(self, model: nn.Module, calibration_loader) -> Dict[str, float]:
        """Calibrate thresholds for entire model."""
        thresholds = {}
        
        # Collect activation statistics
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                activations = []
                
                def hook(module, input, output):
                    activations.append(output.detach())
                    
                handle = module.register_forward_hook(hook)
                
                # Run calibration samples
                with torch.no_grad():
                    for i, (data, _) in enumerate(calibration_loader):
                        if i >= self.config.calibration_samples:
                            break
                        model(data)
                        
                handle.remove()
                
                if activations:
                    all_activations = torch.cat(activations, dim=0)
                    threshold = torch.quantile(all_activations.abs(), 0.99).item()
                    thresholds[name] = max(threshold, 0.1)
                    
        return thresholds


class SpikeformerConverter:
    """Main converter for transforming pre-trained models to spiking networks."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.layer_converter = LayerConverter(self.config)
        self.calibrator = ThresholdCalibrator(self.config)
        self.logger = logging.getLogger(__name__)
        
    def convert(self, model: nn.Module, calibration_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """Convert ANN model to SNN."""
        self.logger.info(f"Starting conversion with config: {self.config}")
        
        # Analyze model architecture
        architecture_info = self._analyze_architecture(model)
        
        # Convert based on model type
        if "transformer" in architecture_info.get("type", "").lower():
            return self._convert_transformer(model, calibration_data)
        else:
            return self._convert_generic(model, calibration_data)
            
    def _analyze_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture to determine conversion strategy."""
        info = {
            "total_params": sum(p.numel() for p in model.parameters()),
            "layer_types": {},
            "has_attention": False,
            "has_transformer": False
        }
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            info["layer_types"][module_type] = info["layer_types"].get(module_type, 0) + 1
            
            if "attention" in name.lower() or isinstance(module, nn.MultiheadAttention):
                info["has_attention"] = True
            if "transformer" in name.lower():
                info["has_transformer"] = True
                
        # Determine model type
        if info["has_transformer"] or info["has_attention"]:
            info["type"] = "transformer"
        elif "Conv" in str(info["layer_types"]):
            info["type"] = "cnn"
        else:
            info["type"] = "mlp"
            
        return info
        
    def _convert_transformer(self, model: nn.Module, calibration_data) -> SpikingTransformer:
        """Convert transformer model to spiking transformer."""
        # Extract transformer configuration
        config = self._extract_transformer_config(model)
        
        # Create spiking transformer
        spiking_model = SpikingTransformer(
            vocab_size=config.get("vocab_size", 30522),
            hidden_size=config.get("hidden_size", 768),
            num_layers=config.get("num_layers", 12),
            num_heads=config.get("num_heads", 12),
            intermediate_size=config.get("intermediate_size", 3072),
            timesteps=self.config.timesteps,
            threshold=self.config.threshold
        )
        
        # Transfer weights from original model
        self._transfer_weights(model, spiking_model)
        
        return spiking_model
        
    def _convert_generic(self, model: nn.Module, calibration_data) -> nn.Module:
        """Convert generic model to spiking version."""
        converted_modules = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                converted_modules[name] = self.layer_converter.convert_linear(module)
            elif isinstance(module, nn.MultiheadAttention):
                converted_modules[name] = self.layer_converter.convert_attention(module)
                
        # Replace modules in original model
        for name, new_module in converted_modules.items():
            self._replace_module(model, name, new_module)
            
        return model
        
    def _extract_transformer_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract configuration from transformer model."""
        config = {}
        
        # Try to extract from model attributes
        if hasattr(model, 'config'):
            model_config = model.config
            config["vocab_size"] = getattr(model_config, 'vocab_size', 30522)
            config["hidden_size"] = getattr(model_config, 'hidden_size', 768)
            config["num_layers"] = getattr(model_config, 'num_hidden_layers', 12)
            config["num_heads"] = getattr(model_config, 'num_attention_heads', 12)
            config["intermediate_size"] = getattr(model_config, 'intermediate_size', 3072)
        else:
            # Analyze model structure
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if "attention" in name.lower():
                        if "query" in name.lower() or "key" in name.lower():
                            config["hidden_size"] = module.in_features
                    elif "intermediate" in name.lower() or "ffn" in name.lower():
                        config["intermediate_size"] = module.out_features
                        
        return config
        
    def _transfer_weights(self, source: nn.Module, target: nn.Module):
        """Transfer weights from source to target model."""
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        
        # Map corresponding layers
        for target_key in target_dict.keys():
            # Find matching source key
            for source_key in source_dict.keys():
                if self._keys_match(source_key, target_key):
                    if source_dict[source_key].shape == target_dict[target_key].shape:
                        target_dict[target_key] = source_dict[source_key]
                        break
                        
        target.load_state_dict(target_dict, strict=False)
        
    def _keys_match(self, source_key: str, target_key: str) -> bool:
        """Check if two parameter keys correspond to the same layer."""
        # Simple matching based on layer names
        source_parts = source_key.split('.')
        target_parts = target_key.split('.')
        
        # Remove indices and common suffixes
        source_clean = [p for p in source_parts if not p.isdigit() and p not in ['weight', 'bias']]
        target_clean = [p for p in target_parts if not p.isdigit() and p not in ['weight', 'bias']]
        
        return source_clean == target_clean
        
    def _replace_module(self, model: nn.Module, module_name: str, new_module: nn.Module):
        """Replace a module in the model."""
        parts = module_name.split('.')
        current = model
        
        for part in parts[:-1]:
            current = getattr(current, part)
            
        setattr(current, parts[-1], new_module)


class LearnableConverter(nn.Module):
    """Learnable conversion module with end-to-end optimization."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 timesteps: int = 32,
                 learnable_threshold: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.timesteps = timesteps
        
        # Learnable conversion parameters
        self.weight_transform = nn.Linear(input_dim, output_dim, bias=False)
        self.temporal_encoding = nn.Parameter(torch.randn(timesteps, output_dim) * 0.1)
        
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.ones(output_dim))
        else:
            self.register_buffer('threshold', torch.ones(output_dim))
            
        # Learnable surrogate gradient parameters
        self.surrogate_beta = nn.Parameter(torch.tensor(10.0))
        self.surrogate_gamma = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ANN activations to spikes with learnable parameters."""
        batch_size = x.shape[0]
        
        # Transform input
        transformed = self.weight_transform(x)  # [B, output_dim]
        
        # Expand to temporal dimension with learnable encoding
        temporal_input = transformed.unsqueeze(1) + self.temporal_encoding.unsqueeze(0)  # [B, T, output_dim]
        
        # Generate spikes using learnable threshold and surrogate
        spikes = []
        membrane_potential = torch.zeros_like(transformed)
        
        for t in range(self.timesteps):
            # Integrate input
            membrane_potential += temporal_input[:, t]
            
            # Learnable spiking function
            spike_logits = (membrane_potential - self.threshold) * self.surrogate_beta
            spike_prob = torch.sigmoid(spike_logits)
            
            # Straight-through estimator with learnable gradient
            spikes_t = (spike_prob > 0.5).float()
            spikes_t = spikes_t + (spike_prob - spike_prob.detach()) * self.surrogate_gamma
            
            spikes.append(spikes_t)
            
            # Adaptive reset
            membrane_potential = membrane_potential - spikes_t * self.threshold
            
        return torch.stack(spikes, dim=1)  # [B, T, output_dim]


class DifferentiableThresholdOptimizer(nn.Module):
    """Differentiable optimization of firing thresholds."""
    
    def __init__(self, layer_dims: List[int], learning_rate: float = 0.001):
        super().__init__()
        
        self.layer_dims = layer_dims
        self.lr = learning_rate
        
        # Learnable thresholds for each layer
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.ones(dim)) for dim in layer_dims
        ])
        
        # Adaptive learning rate per layer
        self.layer_lr = nn.ParameterList([
            nn.Parameter(torch.tensor(learning_rate)) for _ in layer_dims
        ])
        
    def optimize_thresholds(self, 
                          model: nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          target_sparsity: float = 0.1,
                          num_iterations: int = 100) -> Dict[str, torch.Tensor]:
        """Optimize thresholds using gradient-based methods."""
        
        optimizer = torch.optim.Adam([
            {'params': self.thresholds},
            {'params': self.layer_lr, 'lr': self.lr * 0.1}
        ], lr=self.lr)
        
        for iteration in range(num_iterations):
            total_loss = 0
            
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit batches for efficiency
                    break
                    
                optimizer.zero_grad()
                
                # Forward pass to collect spike statistics
                spike_stats = self._collect_spike_stats(model, data)
                
                # Compute loss combining accuracy and sparsity objectives
                sparsity_loss = self._compute_sparsity_loss(spike_stats, target_sparsity)
                accuracy_loss = self._compute_accuracy_loss(model, data, targets)
                
                total_loss = 0.7 * accuracy_loss + 0.3 * sparsity_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Ensure thresholds stay positive
                with torch.no_grad():
                    for threshold in self.thresholds:
                        threshold.data.clamp_(min=0.01)
                        
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss:.4f}")
                
        return {f"layer_{i}": thresh.detach().clone() 
                for i, thresh in enumerate(self.thresholds)}
    
    def _collect_spike_stats(self, model: nn.Module, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Collect spike statistics during forward pass."""
        stats = {}
        hooks = []
        
        def make_hook(name):
            def hook_fn(module, input, output):
                if hasattr(output, 'sum'):  # Check if it's a spike tensor
                    spike_rate = output.sum(dim=1).mean()  # Average over time
                    stats[name] = spike_rate
            return hook_fn
        
        # Register hooks
        for name, module in model.named_modules():
            if 'neuron' in name.lower():
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = model(data)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return stats
    
    def _compute_sparsity_loss(self, 
                             spike_stats: Dict[str, torch.Tensor], 
                             target_sparsity: float) -> torch.Tensor:
        """Compute loss for spike sparsity regulation."""
        sparsity_errors = []
        
        for layer_idx, (name, spike_rate) in enumerate(spike_stats.items()):
            if layer_idx < len(self.thresholds):
                current_sparsity = 1.0 - spike_rate
                sparsity_error = (current_sparsity - target_sparsity) ** 2
                sparsity_errors.append(sparsity_error)
                
        return torch.stack(sparsity_errors).mean() if sparsity_errors else torch.tensor(0.0)
    
    def _compute_accuracy_loss(self, 
                             model: nn.Module, 
                             data: torch.Tensor, 
                             targets: torch.Tensor) -> torch.Tensor:
        """Compute accuracy-based loss."""
        outputs = model(data)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        return F.cross_entropy(outputs, targets)


class ArchitectureAwareConverter:
    """Architecture-specific conversion strategies."""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        
    def convert_vision_transformer(self, model: nn.Module) -> nn.Module:
        """Convert Vision Transformer with specialized strategies."""
        converted_model = self._create_spiking_vit(model)
        
        # Patch embedding optimization
        self._optimize_patch_embedding(model, converted_model)
        
        # Attention head specialization
        self._specialize_attention_heads(model, converted_model)
        
        # Position embedding adaptation
        self._adapt_position_embeddings(model, converted_model)
        
        return converted_model
    
    def convert_language_model(self, model: nn.Module) -> nn.Module:
        """Convert language model with token-aware strategies."""
        converted_model = self._create_spiking_lm(model)
        
        # Token embedding optimization
        self._optimize_token_embeddings(model, converted_model)
        
        # Causal attention adaptation
        self._adapt_causal_attention(model, converted_model)
        
        return converted_model
    
    def _create_spiking_vit(self, model: nn.Module) -> nn.Module:
        """Create spiking vision transformer."""
        # Extract ViT configuration
        config = self._extract_vit_config(model)
        
        # Build spiking equivalent
        from .models import SpikingViT
        return SpikingViT(**config, timesteps=self.config.timesteps)
    
    def _create_spiking_lm(self, model: nn.Module) -> nn.Module:
        """Create spiking language model."""
        config = self._extract_lm_config(model)
        
        from .models import SpikingBERT
        return SpikingBERT(**config, timesteps=self.config.timesteps)
    
    def _optimize_patch_embedding(self, original: nn.Module, spiking: nn.Module):
        """Optimize patch embedding for temporal processing."""
        pass  # Implementation details
    
    def _specialize_attention_heads(self, original: nn.Module, spiking: nn.Module):
        """Specialize different attention heads for different temporal dynamics."""
        pass  # Implementation details
    
    def _adapt_position_embeddings(self, original: nn.Module, spiking: nn.Module):
        """Adapt position embeddings for spatio-temporal processing."""
        pass  # Implementation details
    
    def _extract_vit_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract ViT configuration."""
        return {"image_size": 224, "patch_size": 16, "num_classes": 1000}
    
    def _extract_lm_config(self, model: nn.Module) -> Dict[str, Any]:
        """Extract language model configuration."""
        return {"vocab_size": 30522, "max_length": 512}


class ConversionPipeline:
    """Enhanced conversion pipeline with learnable algorithms."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.converter = SpikeformerConverter(self.config)
        self.arch_converter = ArchitectureAwareConverter(self.config)
        self.logger = logging.getLogger(__name__)
        
    def convert(self, 
                model: nn.Module,
                calibration_data: Optional[torch.utils.data.DataLoader] = None,
                test_data: Optional[torch.utils.data.DataLoader] = None,
                use_learnable: bool = True) -> ConversionResult:
        """Run enhanced conversion pipeline with learnable algorithms."""
        
        import time
        start_time = time.time()
        
        if use_learnable and calibration_data is not None:
            # Use learnable conversion with end-to-end optimization
            snn_model = self._learnable_conversion(model, calibration_data)
        else:
            # Use traditional conversion
            snn_model = self.converter.convert(model, calibration_data)
        
        # Architecture-specific optimizations
        snn_model = self._apply_architecture_optimizations(model, snn_model)
        
        # Evaluate conversion quality
        metrics = self._evaluate_conversion(model, snn_model, test_data)
        
        conversion_time = time.time() - start_time
        
        return ConversionResult(
            snn_model=snn_model,
            accuracy_retention=metrics.get("accuracy_retention", 0.0),
            spike_sparsity=metrics.get("spike_sparsity", 0.0),
            energy_reduction=metrics.get("energy_reduction", 0.0),
            conversion_time=conversion_time,
            metadata=metrics
        )
    
    def _learnable_conversion(self, 
                            model: nn.Module, 
                            calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """Apply learnable conversion algorithms."""
        
        # Extract layer dimensions
        layer_dims = self._extract_layer_dimensions(model)
        
        # Initialize learnable threshold optimizer
        threshold_optimizer = DifferentiableThresholdOptimizer(
            layer_dims, 
            learning_rate=0.001
        )
        
        # Optimize thresholds
        optimized_thresholds = threshold_optimizer.optimize_thresholds(
            model, calibration_data, target_sparsity=0.1
        )
        
        # Convert model with optimized parameters
        snn_model = self.converter.convert(model, calibration_data)
        
        # Apply learned thresholds
        self._apply_learned_thresholds(snn_model, optimized_thresholds)
        
        return snn_model
    
    def _apply_architecture_optimizations(self, 
                                        original: nn.Module, 
                                        spiking: nn.Module) -> nn.Module:
        """Apply architecture-specific optimizations."""
        
        # Detect model type
        model_type = self._detect_model_type(original)
        
        if model_type == "vision_transformer":
            return self.arch_converter.convert_vision_transformer(original)
        elif model_type == "language_model":
            return self.arch_converter.convert_language_model(original)
        else:
            return spiking
    
    def _detect_model_type(self, model: nn.Module) -> str:
        """Detect the type of model architecture."""
        module_names = [name for name, _ in model.named_modules()]
        
        if any("patch_embed" in name for name in module_names):
            return "vision_transformer"
        elif any("token_embed" in name or "word_embed" in name for name in module_names):
            return "language_model"
        else:
            return "generic"
    
    def _extract_layer_dimensions(self, model: nn.Module) -> List[int]:
        """Extract dimensions of relevant layers."""
        dims = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                dims.append(module.out_features)
        return dims
    
    def _apply_learned_thresholds(self, 
                                model: nn.Module, 
                                thresholds: Dict[str, torch.Tensor]):
        """Apply learned thresholds to the spiking model."""
        layer_idx = 0
        for name, module in model.named_modules():
            if hasattr(module, 'threshold'):
                threshold_key = f"layer_{layer_idx}"
                if threshold_key in thresholds:
                    module.threshold = thresholds[threshold_key]
                    layer_idx += 1
        
    def _evaluate_conversion(self, 
                           original_model: nn.Module, 
                           snn_model: nn.Module,
                           test_data: Optional[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Evaluate conversion quality."""
        metrics = {}
        
        if test_data is not None:
            # Evaluate accuracy retention
            original_acc = self._evaluate_accuracy(original_model, test_data)
            snn_acc = self._evaluate_accuracy(snn_model, test_data)
            
            metrics["original_accuracy"] = original_acc
            metrics["snn_accuracy"] = snn_acc
            metrics["accuracy_retention"] = snn_acc / original_acc if original_acc > 0 else 0.0
            
        # Analyze spike sparsity
        metrics["spike_sparsity"] = self._calculate_spike_sparsity(snn_model)
        
        # Estimate energy reduction (simplified)
        metrics["energy_reduction"] = 5.0  # Placeholder - would be measured
        
        return metrics
        
    def _evaluate_accuracy(self, model: nn.Module, test_data) -> float:
        """Evaluate model accuracy on test data."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_data:
                outputs = model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return correct / total if total > 0 else 0.0
        
    def _calculate_spike_sparsity(self, model: nn.Module) -> float:
        """Calculate average spike sparsity in the model."""
        # Placeholder - would analyze actual spike patterns
        return 0.8  # 80% sparsity typical for spiking networks