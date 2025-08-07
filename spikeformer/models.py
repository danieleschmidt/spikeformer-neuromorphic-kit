"""Spiking neural network models and architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass

from .neurons import LifNeuron, SpikingLayer, TemporalBatchNorm, create_neuron
from .encoding import RateCoding, create_encoder


@dataclass
class SpikingConfig:
    """Configuration for spiking models."""
    timesteps: int = 32
    threshold: float = 1.0
    neuron_model: str = "LIF"
    spike_encoding: str = "rate"
    surrogate_gradient: str = "fast_sigmoid"
    tau_mem: float = 20.0
    tau_adp: float = 200.0
    beta: float = 0.1
    dropout: float = 0.1
    layer_norm: bool = True


class SpikingEmbedding(nn.Module):
    """Spiking embedding layer for discrete inputs."""
    
    def __init__(self, vocab_size: int, embed_dim: int, timesteps: int, 
                 padding_idx: Optional[int] = None, threshold: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.encoder = RateCoding(timesteps=timesteps)
        self.neuron = LifNeuron(threshold=threshold)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking embedding."""
        # Get embeddings
        embeddings = self.embedding(input_ids)  # (batch, seq, embed_dim)
        
        # Encode to spikes
        spike_embeddings = self.encoder(embeddings)  # (batch, time, seq, embed_dim)
        
        # Apply neuron dynamics
        return self.neuron(spike_embeddings)


class SpikingPositionalEncoding(nn.Module):
    """Spiking positional encoding with temporal dynamics."""
    
    def __init__(self, d_model: int, max_len: int = 5000, timesteps: int = 32):
        super().__init__()
        self.d_model = d_model
        self.timesteps = timesteps
        
        # Create standard positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Spiking components
        self.encoder = RateCoding(timesteps=timesteps)
        self.neuron = LifNeuron(threshold=1.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add spiking positional encoding."""
        seq_len = x.size(2)  # (batch, time, seq, features)
        
        # Get positional encoding
        pos_enc = self.pe[:, :seq_len]  # (1, seq, d_model)
        
        # Convert to spikes
        pos_spikes = self.encoder(pos_enc)  # (1, time, seq, d_model)
        pos_spikes = pos_spikes.expand(x.size(0), -1, -1, -1)
        
        # Add to input
        return x + pos_spikes


class SparseSpikingAttention(nn.Module):
    """Sparse spike-based attention with event-driven computation."""
    
    def __init__(self, embed_dim: int, num_heads: int, timesteps: int = 32,
                 threshold: float = 1.0, dropout: float = 0.1, 
                 sparsity_ratio: float = 0.1, use_event_driven: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.head_dim = embed_dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        self.use_event_driven = use_event_driven
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections with specialized neurons
        self.q_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ATTENTION", threshold=threshold)
        self.k_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ATTENTION", threshold=threshold)
        self.v_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="STOCHASTIC", threshold=threshold)
        
        # Sparse attention components
        self.attention_sparsity = nn.Parameter(torch.tensor(sparsity_ratio))
        self.spike_threshold_adaptation = nn.Parameter(torch.ones(num_heads))
        
        # Temporal attention integration
        self.temporal_weights = nn.Parameter(torch.randn(timesteps, num_heads) * 0.1)
        
        # Event-driven spike accumulator
        if use_event_driven:
            self.spike_memory = nn.Parameter(torch.zeros(1, num_heads, 1, embed_dim))
            self.memory_decay = nn.Parameter(torch.tensor(0.9))
        
        # Output projection with adaptive sparsity
        self.out_proj = SpikingLayer(embed_dim, embed_dim, 
                                   neuron_type="ADAPTIVE", threshold=threshold)
        
        # Attention computation
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through sparse spiking attention."""
        batch_size, time_steps, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V with spike-based projections
        Q = self.q_proj(x)  # (batch, time, seq, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, time_steps, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, time_steps, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, time_steps, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(2, 3)  # (batch, time, heads, seq, head_dim)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        
        # Event-driven sparse attention computation
        attended_values = []
        spike_events = []
        
        for t in range(time_steps):
            # Compute attention scores
            scores = torch.matmul(Q[:, t], K[:, t].transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Temporal weighting for spiking dynamics
            temporal_weight = self.temporal_weights[t].view(1, self.num_heads, 1, 1)
            scores = scores * temporal_weight
            
            # Sparse attention: only compute attention for top-k highest scores
            if self.training and self.use_event_driven:
                # Dynamic sparsity based on spike activity
                k = max(1, int(seq_len * self.attention_sparsity))
                top_k_values, top_k_indices = torch.topk(scores, k, dim=-1)
                
                # Create sparse attention mask
                sparse_mask = torch.zeros_like(scores)
                sparse_mask.scatter_(-1, top_k_indices, 1.0)
                
                # Apply sparse mask
                scores = scores * sparse_mask
                
            # Spike-based softmax with adaptive thresholds
            attn_weights = self._spiking_softmax(scores, t)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attended = torch.matmul(attn_weights, V[:, t])  # (batch, heads, seq, head_dim)
            
            # Event-driven memory update
            if self.use_event_driven:
                spike_event = (attended > self.spike_threshold_adaptation.view(1, -1, 1, 1)).float()
                spike_events.append(spike_event.sum())
                
                # Update spike memory
                self.spike_memory.data = (self.memory_decay * self.spike_memory.data + 
                                        (1 - self.memory_decay) * attended.mean(dim=(0, 2), keepdim=True))
                
                # Memory-modulated attention
                attended = attended + self.spike_memory * 0.1
            
            attended_values.append(attended)
            
        # Stack timesteps and reshape
        attended_output = torch.stack(attended_values, dim=1)  # (batch, time, heads, seq, head_dim)
        attended_output = attended_output.transpose(2, 3)  # (batch, time, seq, heads, head_dim)
        attended_output = attended_output.contiguous().view(
            batch_size, time_steps, seq_len, embed_dim
        )
        
        # Adaptive sparsity output projection
        output = self.out_proj(attended_output)
        
        # Return output with spike statistics for analysis
        if self.training and spike_events:
            self._update_sparsity_stats(spike_events)
            
        return output
    
    def _spiking_softmax(self, scores: torch.Tensor, timestep: int) -> torch.Tensor:
        """Spike-based softmax approximation with temporal dynamics."""
        # Temperature scheduling based on timestep
        temperature = 1.0 + 0.5 * torch.cos(torch.tensor(timestep / self.timesteps * math.pi))
        
        # Spike-based softmax using winner-take-all dynamics
        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        normalized_scores = (scores - max_scores) / temperature
        
        # Probabilistic spiking
        spike_probs = torch.sigmoid(normalized_scores * 5.0)  # Sharp sigmoid
        
        if self.training:
            # Stochastic spiking during training
            spikes = torch.bernoulli(spike_probs)
            # Straight-through estimator
            attn_weights = spikes + (spike_probs - spike_probs.detach())
        else:
            # Deterministic during inference
            attn_weights = (spike_probs > 0.5).float()
            
        # Normalize to maintain attention properties
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return attn_weights
    
    def _update_sparsity_stats(self, spike_events: List[torch.Tensor]):
        """Update sparsity statistics for adaptive control."""
        avg_spikes = torch.stack(spike_events).mean()
        target_spikes = self.sparsity_ratio * self.timesteps * self.num_heads
        
        # Adaptive sparsity adjustment
        if avg_spikes > target_spikes * 1.1:
            self.attention_sparsity.data *= 0.95  # Increase sparsity
        elif avg_spikes < target_spikes * 0.9:
            self.attention_sparsity.data *= 1.05  # Decrease sparsity
            
        # Clamp sparsity ratio
        self.attention_sparsity.data.clamp_(0.01, 0.5)
        
    def reset_state(self):
        """Reset neuron states and spike memory."""
        self.q_proj.reset_state()
        self.k_proj.reset_state()
        self.v_proj.reset_state()
        self.out_proj.reset_state()
        
        if self.use_event_driven:
            self.spike_memory.data.zero_()


class SpikingAttention(SparseSpikingAttention):
    """Standard spiking attention (alias for backward compatibility)."""
    
    def __init__(self, embed_dim: int, num_heads: int, timesteps: int = 32,
                 threshold: float = 1.0, dropout: float = 0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            timesteps=timesteps,
            threshold=threshold,
            dropout=dropout,
            sparsity_ratio=0.1,
            use_event_driven=False  # Standard mode
        )


class TemporalSpikingAttention(nn.Module):
    """Multi-scale temporal attention with spike timing."""
    
    def __init__(self, embed_dim: int, num_heads: int, timesteps: int = 32,
                 threshold: float = 1.0, dropout: float = 0.1,
                 temporal_scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.temporal_scales = temporal_scales
        self.head_dim = embed_dim // num_heads
        
        # Multi-scale attention heads
        self.scale_attentions = nn.ModuleList([
            SparseSpikingAttention(
                embed_dim=embed_dim // len(temporal_scales),
                num_heads=num_heads // len(temporal_scales),
                timesteps=timesteps,
                threshold=threshold,
                dropout=dropout,
                sparsity_ratio=0.1,
                use_event_driven=True
            )
            for scale in temporal_scales
        ])
        
        # Temporal pooling for different scales
        self.temporal_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=scale, stride=1, padding=scale//2)
            for scale in temporal_scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-scale temporal attention."""
        batch_size, time_steps, seq_len, embed_dim = x.shape
        
        # Split input across scales
        scale_dim = embed_dim // len(self.temporal_scales)
        scale_outputs = []
        
        for i, (attention, pool) in enumerate(zip(self.scale_attentions, self.temporal_pools)):
            # Extract scale-specific features
            start_idx = i * scale_dim
            end_idx = (i + 1) * scale_dim
            scale_input = x[:, :, :, start_idx:end_idx]
            
            # Temporal pooling for this scale
            if self.temporal_scales[i] > 1:
                # Reshape for pooling: (batch*seq, features, time)
                reshaped = scale_input.permute(0, 2, 3, 1).contiguous()
                reshaped = reshaped.view(batch_size * seq_len, scale_dim, time_steps)
                
                pooled = pool(reshaped)
                pooled = pooled.view(batch_size, seq_len, scale_dim, -1)
                scale_input = pooled.permute(0, 3, 1, 2).contiguous()
            
            # Apply scale-specific attention
            scale_output = attention(scale_input, mask)
            
            # Interpolate back to original temporal resolution if needed
            if scale_output.shape[1] != time_steps:
                scale_output = F.interpolate(
                    scale_output.permute(0, 3, 1, 2).contiguous(),
                    size=(time_steps, seq_len),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1).contiguous()
                
            scale_outputs.append(scale_output)
        
        # Concatenate scale outputs
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        
        # Fuse scales
        fused_output = self.scale_fusion(multi_scale_output)
        
        return fused_output


class SpikingMLP(nn.Module):
    """Spiking Multi-Layer Perceptron."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 timesteps: int = 32, threshold: float = 1.0, 
                 neuron_type: str = "LIF", dropout: float = 0.1):
        super().__init__()
        self.timesteps = timesteps
        
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(SpikingLayer(
                current_dim, hidden_dim, 
                neuron_type=neuron_type, 
                threshold=threshold
            ))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            
        # Output layer
        layers.append(SpikingLayer(
            current_dim, output_dim,
            neuron_type=neuron_type,
            threshold=threshold
        ))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking MLP."""
        for layer in self.layers:
            if isinstance(layer, SpikingLayer):
                x = layer(x)
            elif isinstance(layer, nn.Dropout):
                # Apply dropout across time dimension
                if x.dim() == 4:  # (batch, time, seq, features)
                    batch_size, time_steps, seq_len, features = x.shape
                    x = x.view(-1, features)
                    x = layer(x)
                    x = x.view(batch_size, time_steps, seq_len, features)
                else:
                    x = layer(x)
                    
        return x
        
    def reset_state(self):
        """Reset all layer states."""
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()


class SpikingTransformerBlock(nn.Module):
    """Single transformer block with spiking components."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 timesteps: int = 32, threshold: float = 1.0, 
                 dropout: float = 0.1, layer_norm: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Self-attention
        self.self_attn = SpikingAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            timesteps=timesteps,
            threshold=threshold,
            dropout=dropout
        )
        
        # MLP
        self.mlp = SpikingMLP(
            input_dim=embed_dim,
            hidden_dims=[mlp_dim],
            output_dim=embed_dim,
            timesteps=timesteps,
            threshold=threshold,
            dropout=dropout
        )
        
        # Layer normalization (adapted for spikes)
        if layer_norm:
            self.norm1 = TemporalBatchNorm(embed_dim)
            self.norm2 = TemporalBatchNorm(embed_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        
        return x
        
    def reset_state(self):
        """Reset all component states."""
        self.self_attn.reset_state()
        self.mlp.reset_state()


class SpikingTransformer(nn.Module):
    """Complete spiking transformer model."""
    
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int,
                 num_heads: int, intermediate_size: int, max_position_embeddings: int = 512,
                 timesteps: int = 32, threshold: float = 1.0, 
                 dropout: float = 0.1, num_classes: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.timesteps = timesteps
        
        # Embedding layers
        self.embeddings = SpikingEmbedding(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            timesteps=timesteps,
            threshold=threshold
        )
        
        self.position_embeddings = SpikingPositionalEncoding(
            d_model=hidden_size,
            max_len=max_position_embeddings,
            timesteps=timesteps
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            SpikingTransformerBlock(
                embed_dim=hidden_size,
                num_heads=num_heads,
                mlp_dim=intermediate_size,
                timesteps=timesteps,
                threshold=threshold,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        if num_classes:
            self.classifier = SpikingMLP(
                input_dim=hidden_size,
                hidden_dims=[],
                output_dim=num_classes,
                timesteps=timesteps,
                threshold=threshold
            )
        else:
            self.classifier = None
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through spiking transformer."""
        # Embeddings
        x = self.embeddings(input_ids)  # (batch, time, seq, hidden)
        x = self.position_embeddings(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        # Classification head if present
        if self.classifier:
            # Pool over sequence dimension (mean over time and sequence)
            pooled = x.mean(dim=(1, 2))  # (batch, hidden)
            logits = self.classifier(pooled.unsqueeze(1).unsqueeze(1))  # Add time/seq dims
            return logits.squeeze(1).squeeze(1)  # Remove time/seq dims
        
        return x
        
    def reset_state(self):
        """Reset all model states."""
        for layer in self.layers:
            layer.reset_state()
        if self.classifier:
            self.classifier.reset_state()


class SpikingViT(nn.Module):
    """Spiking Vision Transformer."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, num_classes: int = 1000,
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 12,
                 mlp_dim: int = 3072, timesteps: int = 32, threshold: float = 1.0,
                 dropout: float = 0.1, in_channels: int = 3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Spike encoding for patches
        self.spike_encoder = RateCoding(timesteps=timesteps)
        self.patch_neuron = LifNeuron(threshold=threshold)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                timesteps=timesteps,
                threshold=threshold,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.head = SpikingMLP(
            input_dim=embed_dim,
            hidden_dims=[],
            output_dim=num_classes,
            timesteps=timesteps,
            threshold=threshold
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking ViT."""
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Convert to spikes
        x = self.spike_encoder(x)  # (batch, time, seq, embed_dim)
        x = self.patch_neuron(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Classification head (use CLS token)
        cls_output = x[:, :, 0]  # (batch, time, embed_dim)
        cls_pooled = cls_output.mean(dim=1)  # Pool over time
        
        logits = self.head(cls_pooled.unsqueeze(1).unsqueeze(1))  # Add dims for MLP
        return logits.squeeze(1).squeeze(1)
        
    def reset_state(self):
        """Reset all model states."""
        self.patch_neuron.reset_state()
        for block in self.blocks:
            block.reset_state()
        self.head.reset_state()


class SpikingBERT(SpikingTransformer):
    """BERT-style spiking transformer."""
    
    def __init__(self, vocab_size: int = 30522, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12, 
                 intermediate_size: int = 3072, max_position_embeddings: int = 512,
                 timesteps: int = 32, threshold: float = 1.0, dropout: float = 0.1):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            timesteps=timesteps,
            threshold=threshold,
            dropout=dropout
        )
        
        # BERT-specific heads
        self.pooler = SpikingMLP(
            input_dim=hidden_size,
            hidden_dims=[hidden_size],
            output_dim=hidden_size,
            timesteps=timesteps,
            threshold=threshold
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """BERT forward pass."""
        # Get transformer output
        sequence_output = super().forward(input_ids, attention_mask)
        
        # Pooled output (CLS token)
        cls_output = sequence_output[:, :, 0]  # (batch, time, hidden)
        pooled_output = self.pooler(cls_output.mean(dim=1, keepdim=True).unsqueeze(1))  # Pool time
        pooled_output = pooled_output.squeeze(1).squeeze(1)
        
        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output
        }