"""Advanced multi-modal fusion mechanisms for neuromorphic computing."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import math
from collections import defaultdict

from .models import SpikingAttention, SpikingMLP
from .neurons import LifNeuron, SpikingLayer
from .encoding import RateCoding


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    fusion_type: str = "cross_attention"  # cross_attention, tensor_fusion, adaptive_gate
    shared_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    temperature: float = 1.0
    gate_activation: str = "sigmoid"
    timesteps: int = 32


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion for spiking modalities."""
    
    def __init__(self, config: FusionConfig, modality_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        
        # Projection layers for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, config.shared_dim)
            for modality, dim in modality_dims.items()
        })
        
        # Cross-modal attention layers
        self.cross_attentions = nn.ModuleDict()
        for i, mod_a in enumerate(self.modalities):
            for mod_b in self.modalities[i+1:]:
                key = f"{mod_a}_{mod_b}"
                self.cross_attentions[key] = SpikingAttention(
                    embed_dim=config.shared_dim,
                    num_heads=config.num_heads,
                    timesteps=config.timesteps,
                    dropout=config.dropout
                )
        
        # Fusion network
        self.fusion_mlp = SpikingMLP(
            in_features=config.shared_dim * len(modality_dims),
            hidden_features=config.shared_dim * 2,
            out_features=config.shared_dim,
            timesteps=config.timesteps
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.shared_dim)
        self.output_neuron = LifNeuron(threshold=1.0)
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform cross-modal attention fusion."""
        batch_size = next(iter(modality_inputs.values())).size(0)
        timesteps = next(iter(modality_inputs.values())).size(1)
        
        # Project all modalities to shared dimension
        projected_modalities = {}
        for modality, input_tensor in modality_inputs.items():
            # input_tensor: (B, T, seq_len, dim)
            B, T, S, D = input_tensor.shape
            projected = self.modality_projections[modality](
                input_tensor.view(B * T * S, D)
            ).view(B, T, S, self.config.shared_dim)
            projected_modalities[modality] = projected
        
        # Compute cross-modal attentions
        fused_representations = []
        
        for i, mod_a in enumerate(self.modalities):
            for j, mod_b in enumerate(self.modalities):
                if i != j:
                    key = f"{mod_a}_{mod_b}" if i < j else f"{mod_b}_{mod_a}"
                    
                    if key in self.cross_attentions:
                        # Apply cross-modal attention
                        query = projected_modalities[mod_a]
                        key_val = projected_modalities[mod_b]
                        
                        # Reshape for attention: (B*T, seq_len, dim)
                        B, T, S_q, D = query.shape
                        _, _, S_kv, _ = key_val.shape
                        
                        query_flat = query.view(B * T, S_q, D)
                        key_flat = key_val.view(B * T, S_kv, D)
                        
                        cross_attended = self.cross_attentions[key](
                            query_flat, key_flat, key_flat
                        )
                        
                        # Reshape back
                        cross_attended = cross_attended.view(B, T, S_q, D)
                        fused_representations.append(cross_attended)
        
        # Combine all cross-modal representations
        if fused_representations:
            # Average pool across sequence dimension and concatenate
            pooled_reps = []
            for rep in fused_representations:
                pooled = rep.mean(dim=2)  # (B, T, D)
                pooled_reps.append(pooled)
            
            # Concatenate along feature dimension
            concatenated = torch.cat(pooled_reps, dim=-1)  # (B, T, total_D)
            
            # Apply fusion MLP
            fused_output = self.fusion_mlp(concatenated)  # (B, T, shared_dim)
            
            # Normalize and apply spiking dynamics
            B, T, D = fused_output.shape
            normalized = self.output_norm(fused_output.view(B * T, D))
            normalized = normalized.view(B, T, D)
            
            return self.output_neuron(normalized)
        
        else:
            # Fallback: simple concatenation and fusion
            all_modalities = torch.cat([
                proj.mean(dim=2) for proj in projected_modalities.values()
            ], dim=-1)
            
            return self.fusion_mlp(all_modalities)


class TensorFusion(nn.Module):
    """Tensor fusion network for multi-modal spike integration."""
    
    def __init__(self, config: FusionConfig, modality_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        
        # Unimodal networks
        self.unimodal_networks = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.unimodal_networks[modality] = nn.Sequential(
                nn.Linear(dim, config.shared_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.shared_dim, config.shared_dim)
            )
        
        # Tensor fusion dimensions
        num_modalities = len(modality_dims)
        tensor_dim = (config.shared_dim + 1) ** num_modalities
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(tensor_dim, config.shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_dim * 2, config.shared_dim)
        )
        
        # Spiking output
        self.output_neuron = LifNeuron(threshold=1.0)
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform tensor fusion."""
        batch_size = next(iter(modality_inputs.values())).size(0)
        timesteps = next(iter(modality_inputs.values())).size(1)
        
        # Process unimodal representations
        unimodal_reps = []
        for modality, input_tensor in modality_inputs.items():
            # Average pool across sequence dimension
            pooled = input_tensor.mean(dim=2)  # (B, T, dim)
            
            # Apply unimodal network
            B, T, D = pooled.shape
            processed = self.unimodal_networks[modality](
                pooled.view(B * T, D)
            ).view(B, T, self.config.shared_dim)
            
            # Add constant term for tensor fusion
            ones = torch.ones(B, T, 1, device=processed.device)
            augmented = torch.cat([processed, ones], dim=-1)
            unimodal_reps.append(augmented)
        
        # Compute outer product for tensor fusion
        tensor_product = unimodal_reps[0]
        for rep in unimodal_reps[1:]:
            # Outer product: (B, T, D1) x (B, T, D2) -> (B, T, D1*D2)
            tensor_product = torch.einsum('btd,bte->btde', tensor_product, rep)
            tensor_product = tensor_product.flatten(start_dim=2)
        
        # Apply fusion network
        B, T, tensor_dim = tensor_product.shape
        fused = self.fusion_network(tensor_product.view(B * T, tensor_dim))
        fused = fused.view(B, T, self.config.shared_dim)
        
        return self.output_neuron(fused)


class AdaptiveGatedFusion(nn.Module):
    """Adaptive gated fusion with learned modality weights."""
    
    def __init__(self, config: FusionConfig, modality_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.encoders[modality] = nn.Sequential(
                nn.Linear(dim, config.shared_dim),
                nn.LayerNorm(config.shared_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
        
        # Gating network
        total_input_dim = sum(modality_dims.values())
        self.gate_network = nn.Sequential(
            nn.Linear(total_input_dim, config.shared_dim),
            nn.ReLU(),
            nn.Linear(config.shared_dim, len(modality_dims)),
            nn.Sigmoid() if config.gate_activation == "sigmoid" else nn.Softmax(dim=-1)
        )
        
        # Context-aware fusion
        self.context_attention = nn.MultiheadAttention(
            config.shared_dim, 
            config.num_heads, 
            dropout=config.dropout,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion_layer = SpikingMLP(
            in_features=config.shared_dim,
            hidden_features=config.shared_dim * 2,
            out_features=config.shared_dim,
            timesteps=config.timesteps
        )
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform adaptive gated fusion."""
        batch_size = next(iter(modality_inputs.values())).size(0)
        timesteps = next(iter(modality_inputs.values())).size(1)
        
        # Encode each modality
        encoded_modalities = {}
        pooled_for_gating = []
        
        for modality, input_tensor in modality_inputs.items():
            # Average pool across sequence dimension
            pooled = input_tensor.mean(dim=2)  # (B, T, dim)
            pooled_for_gating.append(pooled)
            
            # Encode
            B, T, D = pooled.shape
            encoded = self.encoders[modality](pooled.view(B * T, D))
            encoded = encoded.view(B, T, self.config.shared_dim)
            encoded_modalities[modality] = encoded
        
        # Compute gating weights
        gating_input = torch.cat(pooled_for_gating, dim=-1)  # (B, T, total_dim)
        B, T, total_D = gating_input.shape
        
        gate_weights = self.gate_network(gating_input.view(B * T, total_D))
        gate_weights = gate_weights.view(B, T, len(self.modalities))
        
        # Apply gating to modalities
        gated_modalities = []
        for i, modality in enumerate(self.modalities):
            weight = gate_weights[:, :, i:i+1]  # (B, T, 1)
            gated = encoded_modalities[modality] * weight
            gated_modalities.append(gated)
        
        # Combine gated modalities
        combined = torch.stack(gated_modalities, dim=2)  # (B, T, num_mod, D)
        B, T, M, D = combined.shape
        
        # Apply context attention across modalities
        combined_reshaped = combined.view(B * T, M, D)
        attended, _ = self.context_attention(
            combined_reshaped, combined_reshaped, combined_reshaped
        )
        
        # Average across modalities
        fused = attended.mean(dim=1)  # (B*T, D)
        fused = fused.view(B, T, D)
        
        # Final fusion with spiking dynamics
        return self.fusion_layer(fused)


class MultiModalSpikeTransformer(nn.Module):
    """Complete multi-modal spiking transformer with advanced fusion."""
    
    def __init__(self, fusion_config: FusionConfig, modality_dims: Dict[str, int],
                 num_classes: int, num_layers: int = 6):
        super().__init__()
        self.fusion_config = fusion_config
        self.modality_dims = modality_dims
        self.num_classes = num_classes
        
        # Fusion layer
        if fusion_config.fusion_type == "cross_attention":
            self.fusion = CrossModalAttentionFusion(fusion_config, modality_dims)
        elif fusion_config.fusion_type == "tensor_fusion":
            self.fusion = TensorFusion(fusion_config, modality_dims)
        elif fusion_config.fusion_type == "adaptive_gate":
            self.fusion = AdaptiveGatedFusion(fusion_config, modality_dims)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_config.fusion_type}")
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            SpikingTransformerLayer(
                d_model=fusion_config.shared_dim,
                nhead=fusion_config.num_heads,
                timesteps=fusion_config.timesteps,
                dropout=fusion_config.dropout
            )
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_config.shared_dim, fusion_config.shared_dim),
            nn.ReLU(),
            nn.Dropout(fusion_config.dropout),
            nn.Linear(fusion_config.shared_dim, num_classes)
        )
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-modal transformer."""
        # Fusion
        fused_representation = self.fusion(modality_inputs)  # (B, T, shared_dim)
        
        # Transformer processing
        x = fused_representation
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling across time dimension
        pooled = x.mean(dim=1)  # (B, shared_dim)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class SpikingTransformerLayer(nn.Module):
    """Single spiking transformer layer."""
    
    def __init__(self, d_model: int, nhead: int, timesteps: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = SpikingAttention(
            embed_dim=d_model,
            num_heads=nhead,
            timesteps=timesteps,
            dropout=dropout
        )
        
        self.feed_forward = SpikingMLP(
            in_features=d_model,
            hidden_features=d_model * 4,
            out_features=d_model,
            timesteps=timesteps,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layer."""
        B, T, D = x.shape
        
        # Self-attention with residual connection
        x_reshaped = x.view(B * T, 1, D)  # Treat each timestep as separate sequence
        attn_out = self.self_attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.view(B, T, D)
        
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x