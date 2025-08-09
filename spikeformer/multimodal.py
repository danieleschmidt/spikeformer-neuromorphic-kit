"""Multi-modal spike processing for neuromorphic computing with cross-modal learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
from collections import defaultdict

from .models import SpikingTransformer, SpikingAttention, SpikingMLP
from .neurons import LifNeuron, SpikingLayer
from .encoding import RateCoding, TemporalCoding, PoissonCoding


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal spiking models."""
    modalities: List[str]
    fusion_strategy: str = "cross_attention"
    temporal_alignment: str = "adaptive"
    cross_modal_learning: bool = True
    modality_weights: Optional[Dict[str, float]] = None
    fusion_layers: int = 2
    shared_representation_dim: int = 512
    timesteps: int = 32
    
    
class ModalityEncoder(ABC):
    """Abstract base class for modality-specific encoders."""
    
    @abstractmethod
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode input data to spike representation."""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output dimension of encoded spikes."""
        pass


class VisionSpikeEncoder(ModalityEncoder):
    """Vision encoder for spike-based processing."""
    
    def __init__(self, input_channels: int = 3, patch_size: int = 16, 
                 embed_dim: int = 768, timesteps: int = 32):
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Convolutional patch extraction
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Spike encoding
        self.spike_encoder = RateCoding(timesteps=timesteps)
        self.patch_neuron = LifNeuron(threshold=1.0)
        
        # Temporal dynamics for vision
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode vision input to spikes."""
        batch_size = input_data.shape[0]
        
        # Extract patches
        patches = self.patch_embed(input_data)  # (B, embed_dim, H', W')
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Convert to spikes
        spike_patches = self.spike_encoder(patches)  # (B, T, num_patches, embed_dim)
        spike_patches = self.patch_neuron(spike_patches)
        
        # Apply temporal dynamics
        B, T, P, E = spike_patches.shape
        spike_patches_flat = spike_patches.view(B * P, E, T)
        temporal_spikes = self.temporal_conv(spike_patches_flat)
        spike_patches = temporal_spikes.view(B, P, E, T).transpose(2, 3)
        
        return spike_patches  # (B, T, num_patches, embed_dim)
    
    def get_output_dim(self) -> int:
        return self.embed_dim


class AudioSpikeEncoder(ModalityEncoder):
    """Audio encoder for spike-based processing."""
    
    def __init__(self, input_features: int = 80, embed_dim: int = 768, timesteps: int = 32):
        self.input_features = input_features
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Audio feature projection
        self.feature_proj = nn.Linear(input_features, embed_dim)
        
        # Cochlear-inspired spike encoding
        self.cochlear_encoder = self._create_cochlear_filterbank()
        self.spike_encoder = PoissonCoding(timesteps=timesteps)
        
        # Temporal processing for audio
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
    def _create_cochlear_filterbank(self) -> nn.Module:
        """Create cochlear-inspired filterbank for audio encoding."""
        return nn.ModuleList([
            nn.Conv1d(1, 32, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11, 15]  # Multiple temporal scales
        ])
    
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode audio input to spikes."""
        batch_size, seq_len, features = input_data.shape
        
        # Apply cochlear filterbank
        audio_filtered = []
        for conv_layer in self.cochlear_encoder:
            # Reshape for 1D convolution: (B*features, 1, seq_len)
            reshaped = input_data.transpose(1, 2).contiguous().view(-1, 1, seq_len)
            filtered = conv_layer(reshaped)
            # Reshape back: (B, features, filtered_seq_len)
            filtered = filtered.view(batch_size, features, -1)
            audio_filtered.append(filtered)
        
        # Concatenate filtered signals
        audio_features = torch.cat(audio_filtered, dim=1)  # (B, total_features, seq_len)
        audio_features = audio_features.transpose(1, 2)  # (B, seq_len, total_features)
        
        # Project to embedding dimension
        embedded = self.feature_proj(audio_features)  # (B, seq_len, embed_dim)
        
        # Convert to spikes
        spike_audio = self.spike_encoder(embedded)  # (B, T, seq_len, embed_dim)
        
        # Apply temporal attention across time steps
        B, T, S, E = spike_audio.shape
        spike_audio_reshaped = spike_audio.view(B * S, T, E)
        attended_spikes, _ = self.temporal_attention(
            spike_audio_reshaped, spike_audio_reshaped, spike_audio_reshaped
        )
        spike_audio = attended_spikes.view(B, S, T, E).transpose(1, 2)
        
        return spike_audio  # (B, T, seq_len, embed_dim)
    
    def get_output_dim(self) -> int:
        return self.embed_dim


class TextSpikeEncoder(ModalityEncoder):
    """Text encoder for spike-based processing."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, timesteps: int = 32):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.timesteps = timesteps
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Spike encoding with semantic structure
        self.semantic_encoder = TemporalCoding(timesteps=timesteps)
        self.token_neuron = LifNeuron(threshold=0.8)  # Lower threshold for discrete tokens
        
        # Positional encoding for text
        self.pos_encoding = nn.Parameter(torch.randn(1024, embed_dim) * 0.1)
        
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode text input to spikes."""
        batch_size, seq_len = input_data.shape
        
        # Token embedding
        token_embeds = self.token_embedding(input_data)  # (B, seq_len, embed_dim)
        
        # Add positional encoding
        pos_embeds = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        combined_embeds = token_embeds + pos_embeds
        
        # Convert to spikes with semantic structure
        spike_tokens = self.semantic_encoder(combined_embeds)  # (B, T, seq_len, embed_dim)
        spike_tokens = self.token_neuron(spike_tokens)
        
        return spike_tokens  # (B, T, seq_len, embed_dim)
    
    def get_output_dim(self) -> int:
        return self.embed_dim


class CrossModalAttention(nn.Module):
    """Cross-modal attention for spike-based multi-modal fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, timesteps: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.head_dim = embed_dim // num_heads
        
        # Cross-modal projections
        self.q_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ATTENTION")
        self.k_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ATTENTION") 
        self.v_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ATTENTION")
        
        # Modality-specific attention weights
        self.modality_weights = nn.Parameter(torch.ones(2, num_heads))
        
        # Output projection
        self.out_proj = SpikingLayer(embed_dim, embed_dim, neuron_type="ADAPTIVE")
        
        # Temporal synchronization
        self.temporal_sync = nn.Parameter(torch.randn(timesteps, num_heads) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query_spikes: torch.Tensor, key_spikes: torch.Tensor, 
                value_spikes: torch.Tensor, 
                query_modality: int = 0, key_modality: int = 1) -> torch.Tensor:
        """Cross-modal attention between different spike modalities."""
        
        batch_size, timesteps, seq_len_q, _ = query_spikes.shape
        seq_len_k = key_spikes.shape[2]
        seq_len_v = value_spikes.shape[2]
        
        # Project Q, K, V
        Q = self.q_proj(query_spikes)  # (B, T, seq_q, embed_dim)
        K = self.k_proj(key_spikes)    # (B, T, seq_k, embed_dim)
        V = self.v_proj(value_spikes)  # (B, T, seq_v, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, timesteps, seq_len_q, self.num_heads, self.head_dim)
        K = K.view(batch_size, timesteps, seq_len_k, self.num_heads, self.head_dim)
        V = V.view(batch_size, timesteps, seq_len_v, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: (B, T, heads, seq, head_dim)
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3) 
        V = V.transpose(2, 3)
        
        # Cross-modal attention computation
        attended_values = []
        
        for t in range(timesteps):
            # Compute attention scores
            scores = torch.matmul(Q[:, t], K[:, t].transpose(-2, -1)) / self.scale
            
            # Apply modality-specific weighting
            modality_weight = (self.modality_weights[query_modality] * 
                             self.modality_weights[key_modality]).view(1, self.num_heads, 1, 1)
            scores = scores * modality_weight
            
            # Temporal synchronization
            temporal_weight = self.temporal_sync[t].view(1, self.num_heads, 1, 1)
            scores = scores + temporal_weight
            
            # Spike-based softmax (competitive dynamics)
            attention_weights = self._spike_softmax(scores)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attended = torch.matmul(attention_weights, V[:, t])
            attended_values.append(attended)
        
        # Stack timesteps: (B, T, heads, seq_q, head_dim)
        attended_output = torch.stack(attended_values, dim=1)
        
        # Reshape back: (B, T, seq_q, embed_dim)
        attended_output = attended_output.transpose(2, 3).contiguous()
        attended_output = attended_output.view(batch_size, timesteps, seq_len_q, self.embed_dim)
        
        # Output projection
        output = self.out_proj(attended_output)
        
        return output
    
    def _spike_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Spike-based competitive softmax."""
        # Winner-take-all dynamics with spikes
        max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
        normalized = scores - max_scores
        
        # Convert to spike probabilities
        spike_probs = torch.sigmoid(normalized * 5.0)  # Sharp activation
        
        # Competitive normalization
        spike_probs = spike_probs / (spike_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return spike_probs


class MultiModalSpikeFusion(nn.Module):
    """Multi-modal fusion layer for spiking neural networks."""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.num_modalities = len(config.modalities)
        
        # Cross-modal attention layers
        self.cross_attentions = nn.ModuleList([
            CrossModalAttention(
                config.shared_representation_dim,
                num_heads=8,
                timesteps=config.timesteps
            )
            for _ in range(config.fusion_layers)
        ])
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict({
            modality: SpikingLayer(
                config.shared_representation_dim,
                config.shared_representation_dim,
                neuron_type="ADAPTIVE"
            )
            for modality in config.modalities
        })
        
        # Fusion strategies
        if config.fusion_strategy == "cross_attention":
            self.fusion_layer = self._cross_attention_fusion
        elif config.fusion_strategy == "concatenation":
            self.fusion_layer = self._concatenation_fusion
        elif config.fusion_strategy == "gated":
            self.fusion_layer = self._gated_fusion
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")
        
        # Adaptive modality weighting
        if config.modality_weights is None:
            self.modality_weights = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)
        else:
            weights = torch.tensor([config.modality_weights.get(mod, 1.0) for mod in config.modalities])
            self.register_buffer("modality_weights", weights / weights.sum())
        
        # Temporal alignment
        if config.temporal_alignment == "adaptive":
            self.temporal_aligner = AdaptiveTemporalAlignment(config.timesteps, self.num_modalities)
        else:
            self.temporal_aligner = None
            
    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multi-modal spike representations."""
        
        # Ensure all modalities are present
        available_modalities = list(modality_inputs.keys())
        for modality in self.config.modalities:
            if modality not in available_modalities:
                raise ValueError(f"Missing modality: {modality}")
        
        # Project modalities to shared representation space
        projected_modalities = {}
        for modality, spike_input in modality_inputs.items():
            if modality in self.modality_projections:
                projected = self.modality_projections[modality](spike_input)
                projected_modalities[modality] = projected
        
        # Temporal alignment if enabled
        if self.temporal_aligner is not None:
            projected_modalities = self.temporal_aligner(projected_modalities)
        
        # Apply cross-modal attention
        enhanced_modalities = {}
        for i, (modality, spikes) in enumerate(projected_modalities.items()):
            enhanced = spikes
            
            # Cross-attention with other modalities
            for j, (other_modality, other_spikes) in enumerate(projected_modalities.items()):
                if i != j:  # Don't attend to self
                    for cross_attn in self.cross_attentions:
                        attended = cross_attn(
                            enhanced, other_spikes, other_spikes,
                            query_modality=i, key_modality=j
                        )
                        # Residual connection
                        enhanced = enhanced + attended
            
            enhanced_modalities[modality] = enhanced
        
        # Fusion
        fused_representation = self.fusion_layer(enhanced_modalities)
        
        return fused_representation
    
    def _cross_attention_fusion(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Cross-attention based fusion."""
        # Use the first modality as query, others as key/value
        modality_names = list(modalities.keys())
        query_spikes = modalities[modality_names[0]]
        
        # Attend to all other modalities
        attended_results = [query_spikes]  # Include original query
        
        for i, other_modality in enumerate(modality_names[1:], 1):
            other_spikes = modalities[other_modality]
            attended = self.cross_attentions[0](
                query_spikes, other_spikes, other_spikes,
                query_modality=0, key_modality=i
            )
            attended_results.append(attended)
        
        # Weighted fusion
        weighted_results = []
        for i, attended in enumerate(attended_results):
            weight = self.modality_weights[i] if i < len(self.modality_weights) else 1.0
            weighted_results.append(weight * attended)
        
        return sum(weighted_results)
    
    def _concatenation_fusion(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenation-based fusion."""
        # Concatenate along feature dimension
        modality_spikes = list(modalities.values())
        concatenated = torch.cat(modality_spikes, dim=-1)
        
        # Project back to shared dimension
        fusion_proj = nn.Linear(
            concatenated.shape[-1], 
            self.config.shared_representation_dim
        ).to(concatenated.device)
        
        return fusion_proj(concatenated)
    
    def _gated_fusion(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Gated fusion with learned attention."""
        modality_list = list(modalities.values())
        stacked_modalities = torch.stack(modality_list, dim=-2)  # (B, T, seq, num_mod, embed)
        
        # Compute attention weights over modalities
        attention_scores = torch.sum(stacked_modalities, dim=-1)  # (B, T, seq, num_mod)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T, seq, num_mod, 1)
        fused = torch.sum(stacked_modalities * attention_weights, dim=-2)
        
        return fused


class AdaptiveTemporalAlignment(nn.Module):
    """Adaptive temporal alignment for multi-modal spikes."""
    
    def __init__(self, timesteps: int, num_modalities: int):
        super().__init__()
        self.timesteps = timesteps
        self.num_modalities = num_modalities
        
        # Learnable temporal offsets for each modality
        self.temporal_offsets = nn.Parameter(torch.zeros(num_modalities, timesteps))
        
        # Adaptive alignment network
        self.alignment_net = nn.Sequential(
            nn.Linear(timesteps * num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, num_modalities * timesteps),
            nn.Tanh()  # Bounded offsets
        )
        
    def forward(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adaptively align temporal dynamics across modalities."""
        
        modality_list = list(modalities.items())
        batch_size = modality_list[0][1].shape[0]
        
        # Compute temporal statistics for alignment
        temporal_stats = []
        for modality_name, spikes in modality_list:
            # Compute temporal activity pattern
            temporal_activity = spikes.sum(dim=(2, 3))  # (B, T)
            temporal_stats.append(temporal_activity)
        
        stacked_stats = torch.stack(temporal_stats, dim=1)  # (B, num_mod, T)
        stacked_stats_flat = stacked_stats.view(batch_size, -1)  # (B, num_mod * T)
        
        # Compute adaptive alignment offsets
        adaptive_offsets = self.alignment_net(stacked_stats_flat)  # (B, num_mod * T)
        adaptive_offsets = adaptive_offsets.view(batch_size, self.num_modalities, self.timesteps)
        
        # Combine with learned offsets
        total_offsets = self.temporal_offsets.unsqueeze(0) + 0.1 * adaptive_offsets
        
        # Apply temporal shifting
        aligned_modalities = {}
        for i, (modality_name, spikes) in enumerate(modality_list):
            # Apply temporal offset (simplified - in practice would use more sophisticated resampling)
            offset = total_offsets[:, i]  # (B, T)
            
            # Simple temporal weighting (placeholder for more complex alignment)
            temporal_weights = F.softmax(offset, dim=-1).unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1)
            aligned_spikes = spikes * temporal_weights
            
            aligned_modalities[modality_name] = aligned_spikes
        
        return aligned_modalities


class MultiModalSpikingTransformer(nn.Module):
    """Complete multi-modal spiking transformer."""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        
        # Modality encoders
        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            if modality == "vision":
                self.encoders[modality] = VisionSpikeEncoder(
                    embed_dim=config.shared_representation_dim,
                    timesteps=config.timesteps
                )
            elif modality == "audio":
                self.encoders[modality] = AudioSpikeEncoder(
                    embed_dim=config.shared_representation_dim,
                    timesteps=config.timesteps
                )
            elif modality == "text":
                self.encoders[modality] = TextSpikeEncoder(
                    vocab_size=30000,  # Default vocab size
                    embed_dim=config.shared_representation_dim,
                    timesteps=config.timesteps
                )
        
        # Multi-modal fusion
        self.fusion_layer = MultiModalSpikeFusion(config)
        
        # Shared spiking transformer
        self.shared_transformer = SpikingTransformer(
            vocab_size=1,  # Not used directly
            hidden_size=config.shared_representation_dim,
            num_layers=6,
            num_heads=8,
            intermediate_size=config.shared_representation_dim * 4,
            timesteps=config.timesteps
        )
        
        # Task-specific heads
        self.classification_head = SpikingLayer(
            config.shared_representation_dim, 1000,  # 1000 classes default
            neuron_type="ADAPTIVE"
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, inputs: Dict[str, torch.Tensor], task: str = "classification") -> torch.Tensor:
        """Forward pass through multi-modal spiking transformer."""
        
        # Encode each modality
        encoded_modalities = {}
        for modality, input_data in inputs.items():
            if modality in self.encoders:
                encoded = self.encoders[modality].encode(input_data)
                encoded_modalities[modality] = encoded
                self.logger.debug(f"Encoded {modality}: {encoded.shape}")
        
        # Fuse modalities
        fused_representation = self.fusion_layer(encoded_modalities)
        self.logger.debug(f"Fused representation: {fused_representation.shape}")
        
        # Process through shared transformer (adapt input format)
        # Create dummy input_ids for transformer interface
        batch_size, timesteps, seq_len, embed_dim = fused_representation.shape
        dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=fused_representation.device)
        
        # Override embedding with our fused representation
        original_embedding = self.shared_transformer.embeddings
        self.shared_transformer.embeddings = lambda x: fused_representation.transpose(1, 2)  # (B, seq, T, embed)
        
        # Process
        transformed = self.shared_transformer(dummy_ids)  # This will use our override
        
        # Restore original embedding
        self.shared_transformer.embeddings = original_embedding
        
        # Task-specific processing
        if task == "classification":
            # Global pooling over sequence and time
            pooled = transformed.mean(dim=1)  # Pool over sequence
            if len(pooled.shape) > 2:
                pooled = pooled.mean(dim=1)  # Pool over time if present
            
            logits = self.classification_head(pooled.unsqueeze(1).unsqueeze(1)).squeeze()
            return logits
        else:
            return transformed
    
    def get_cross_modal_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Extract cross-modal attention weights for analysis."""
        attention_weights = {}
        
        for i, cross_attn in enumerate(self.fusion_layer.cross_attentions):
            # This would require storing attention weights during forward pass
            # Implementation depends on specific needs
            pass
        
        return attention_weights


def create_multimodal_model(modalities: List[str], 
                          fusion_strategy: str = "cross_attention",
                          shared_dim: int = 512) -> MultiModalSpikingTransformer:
    """Factory function to create multi-modal spiking model."""
    
    config = MultiModalConfig(
        modalities=modalities,
        fusion_strategy=fusion_strategy,
        temporal_alignment="adaptive",
        cross_modal_learning=True,
        fusion_layers=2,
        shared_representation_dim=shared_dim,
        timesteps=32
    )
    
    return MultiModalSpikingTransformer(config)


# Example usage and testing
if __name__ == "__main__":
    # Create multi-modal model
    model = create_multimodal_model(
        modalities=["vision", "audio", "text"],
        fusion_strategy="cross_attention",
        shared_dim=768
    )
    
    print(f"Created multi-modal model with modalities: {model.config.modalities}")
    print(f"Fusion strategy: {model.config.fusion_strategy}")
    print(f"Shared representation dim: {model.config.shared_representation_dim}")
    
    # Test with dummy data
    dummy_inputs = {
        "vision": torch.randn(2, 3, 224, 224),  # Batch of images
        "audio": torch.randn(2, 100, 80),       # Batch of audio features  
        "text": torch.randint(0, 1000, (2, 50)) # Batch of token sequences
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_inputs, task="classification")
        print(f"Output shape: {output.shape}")
        
    print("Multi-modal spiking transformer test completed successfully!")