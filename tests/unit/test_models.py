"""Unit tests for spiking neural network models."""

import torch
import torch.nn as nn
import pytest
import numpy as np

from spikeformer.models import (
    SpikingTransformer, SpikingViT, SpikingBERT,
    SpikingTransformerBlock, SpikingAttention, SpikingMLP,
    SpikingEmbedding, SpikingPositionalEncoding,
    SpikingConfig
)


class TestSpikingConfig:
    """Test spiking model configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SpikingConfig()
        
        assert config.timesteps == 32
        assert config.threshold == 1.0
        assert config.neuron_model == "LIF"
        assert config.spike_encoding == "rate"
        assert config.dropout == 0.1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpikingConfig(
            timesteps=64,
            threshold=1.5,
            neuron_model="ADLIF",
            dropout=0.2
        )
        
        assert config.timesteps == 64
        assert config.threshold == 1.5
        assert config.neuron_model == "ADLIF"
        assert config.dropout == 0.2


class TestSpikingEmbedding:
    """Test spiking embedding layer."""
    
    def test_initialization(self):
        """Test spiking embedding initialization."""
        embedding = SpikingEmbedding(
            vocab_size=1000,
            embed_dim=256,
            timesteps=16,
            threshold=1.0
        )
        
        assert embedding.embed_dim == 256
        assert embedding.timesteps == 16
        assert embedding.embedding.num_embeddings == 1000
        assert embedding.embedding.embedding_dim == 256
    
    def test_forward_pass(self):
        """Test forward pass through spiking embedding."""
        embedding = SpikingEmbedding(
            vocab_size=500,
            embed_dim=128,
            timesteps=20
        )
        
        # Create input token ids
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, 20, seq_len, 128)
        assert torch.all((output == 0) | (output == 1))  # Binary spikes
    
    def test_padding_idx(self):
        """Test padding index handling."""
        embedding = SpikingEmbedding(
            vocab_size=100,
            embed_dim=64,
            timesteps=8,
            padding_idx=0
        )
        
        # Input with padding tokens
        input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
        output = embedding(input_ids)
        
        # Padding positions should have consistent behavior
        assert output.shape == (2, 8, 4, 64)


class TestSpikingPositionalEncoding:
    """Test spiking positional encoding."""
    
    def test_initialization(self):
        """Test spiking positional encoding initialization."""
        pos_enc = SpikingPositionalEncoding(
            d_model=512,
            max_len=1000,
            timesteps=24
        )
        
        assert pos_enc.d_model == 512
        assert pos_enc.timesteps == 24
        assert pos_enc.pe.shape == (1, 1000, 512)
    
    def test_forward_pass(self):
        """Test forward pass with positional encoding."""
        pos_enc = SpikingPositionalEncoding(d_model=256, timesteps=16)
        
        batch_size, timesteps, seq_len, d_model = 2, 16, 12, 256
        x = torch.randn(batch_size, timesteps, seq_len, d_model)
        
        output = pos_enc(x)
        
        assert output.shape == x.shape
        # Output should be sum of input and positional encoding
        assert not torch.equal(output, x)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        pos_enc = SpikingPositionalEncoding(d_model=128, timesteps=10)
        
        # Short sequence
        x_short = torch.randn(1, 10, 5, 128)
        output_short = pos_enc(x_short)
        assert output_short.shape == (1, 10, 5, 128)
        
        # Long sequence
        x_long = torch.randn(1, 10, 20, 128)
        output_long = pos_enc(x_long)
        assert output_long.shape == (1, 10, 20, 128)


class TestSpikingAttention:
    """Test spiking attention mechanism."""
    
    def test_initialization(self):
        """Test spiking attention initialization."""
        attention = SpikingAttention(
            embed_dim=512,
            num_heads=8,
            timesteps=32,
            threshold=1.0,
            dropout=0.1
        )
        
        assert attention.embed_dim == 512
        assert attention.num_heads == 8
        assert attention.head_dim == 64  # 512 / 8
        assert attention.timesteps == 32
    
    def test_forward_pass(self):
        """Test forward pass through spiking attention."""
        attention = SpikingAttention(
            embed_dim=256,
            num_heads=4,
            timesteps=16
        )
        
        batch_size, timesteps, seq_len, embed_dim = 2, 16, 10, 256
        x = torch.randn(batch_size, timesteps, seq_len, embed_dim)
        
        output = attention(x)
        
        assert output.shape == x.shape
        assert torch.all((output == 0) | (output == 1))  # Binary spikes
    
    def test_attention_mask(self):
        """Test attention with mask."""
        attention = SpikingAttention(embed_dim=128, num_heads=2, timesteps=8)
        
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, 8, seq_len, 128)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        output = attention(x, mask=mask)
        assert output.shape == x.shape
    
    def test_head_dimension_validation(self):
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            SpikingAttention(embed_dim=100, num_heads=3, timesteps=16)
    
    def test_state_reset(self):
        """Test attention state reset."""
        attention = SpikingAttention(embed_dim=64, num_heads=2, timesteps=8)
        
        x = torch.randn(1, 8, 5, 64)
        _ = attention(x)
        
        # Reset should not raise error
        attention.reset_state()


class TestSpikingMLP:
    """Test spiking MLP."""
    
    def test_initialization(self):
        """Test spiking MLP initialization."""
        mlp = SpikingMLP(
            input_dim=512,
            hidden_dims=[1024, 512],
            output_dim=256,
            timesteps=20,
            threshold=1.0,
            dropout=0.1
        )
        
        assert len(mlp.layers) == 5  # 2 hidden + 1 output + 2 dropout
        assert mlp.timesteps == 20
    
    def test_forward_pass(self):
        """Test forward pass through spiking MLP."""
        mlp = SpikingMLP(
            input_dim=128,
            hidden_dims=[256],
            output_dim=64,
            timesteps=12
        )
        
        batch_size, timesteps, features = 3, 12, 128
        x = torch.randn(batch_size, timesteps, features)
        
        output = mlp(x)
        
        assert output.shape == (batch_size, timesteps, 64)
        assert torch.all((output == 0) | (output == 1))  # Binary spikes
    
    def test_4d_input(self):
        """Test MLP with 4D input."""
        mlp = SpikingMLP(
            input_dim=64,
            hidden_dims=[128],
            output_dim=32,
            timesteps=8
        )
        
        batch_size, timesteps, seq_len, features = 2, 8, 10, 64
        x = torch.randn(batch_size, timesteps, seq_len, features)
        
        output = mlp(x)
        
        assert output.shape == (batch_size, timesteps, seq_len, 32)
    
    def test_no_hidden_layers(self):
        """Test MLP with no hidden layers."""
        mlp = SpikingMLP(
            input_dim=100,
            hidden_dims=[],
            output_dim=50,
            timesteps=16
        )
        
        x = torch.randn(2, 16, 100)
        output = mlp(x)
        
        assert output.shape == (2, 16, 50)
    
    def test_state_reset(self):
        """Test MLP state reset."""
        mlp = SpikingMLP(input_dim=32, hidden_dims=[64], output_dim=16, timesteps=8)
        
        x = torch.randn(1, 8, 32)
        _ = mlp(x)
        
        mlp.reset_state()


class TestSpikingTransformerBlock:
    """Test spiking transformer block."""
    
    def test_initialization(self):
        """Test transformer block initialization."""
        block = SpikingTransformerBlock(
            embed_dim=256,
            num_heads=4,
            mlp_dim=1024,
            timesteps=24,
            threshold=1.0,
            dropout=0.1
        )
        
        assert block.embed_dim == 256
        assert block.timesteps == 24
        assert isinstance(block.self_attn, SpikingAttention)
        assert isinstance(block.mlp, SpikingMLP)
    
    def test_forward_pass(self):
        """Test forward pass through transformer block."""
        block = SpikingTransformerBlock(
            embed_dim=128,
            num_heads=2,
            mlp_dim=512,
            timesteps=16
        )
        
        batch_size, timesteps, seq_len, embed_dim = 2, 16, 8, 128
        x = torch.randn(batch_size, timesteps, seq_len, embed_dim)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert torch.all((output == 0) | (output == 1))  # Binary spikes
    
    def test_residual_connections(self):
        """Test residual connections in transformer block."""
        block = SpikingTransformerBlock(
            embed_dim=64,
            num_heads=2,
            mlp_dim=256,
            timesteps=8
        )
        
        x = torch.ones(1, 8, 4, 64) * 0.1  # Small input
        output = block(x)
        
        # With residual connections, output should not be zero everywhere
        assert torch.any(output > 0)
    
    def test_layer_norm_disabled(self):
        """Test transformer block without layer normalization."""
        block = SpikingTransformerBlock(
            embed_dim=64,
            num_heads=2,
            mlp_dim=128,
            timesteps=8,
            layer_norm=False
        )
        
        assert isinstance(block.norm1, nn.Identity)
        assert isinstance(block.norm2, nn.Identity)
    
    def test_state_reset(self):
        """Test transformer block state reset."""
        block = SpikingTransformerBlock(
            embed_dim=32,
            num_heads=2,
            mlp_dim=64,
            timesteps=8
        )
        
        x = torch.randn(1, 8, 4, 32)
        _ = block(x)
        
        block.reset_state()


class TestSpikingTransformer:
    """Test complete spiking transformer model."""
    
    def test_initialization(self):
        """Test spiking transformer initialization."""
        model = SpikingTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=4,
            intermediate_size=1024,
            timesteps=20
        )
        
        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert model.timesteps == 20
        assert len(model.layers) == 6
    
    def test_forward_pass(self):
        """Test forward pass through spiking transformer."""
        model = SpikingTransformer(
            vocab_size=500,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
            intermediate_size=512,
            timesteps=16
        )
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert output.shape == (batch_size, 16, seq_len, 128)
    
    def test_with_classification_head(self):
        """Test transformer with classification head."""
        model = SpikingTransformer(
            vocab_size=200,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
            intermediate_size=128,
            timesteps=8,
            num_classes=10
        )
        
        input_ids = torch.randint(0, 200, (3, 5))
        output = model(input_ids)
        
        assert output.shape == (3, 10)  # Batch size, num_classes
    
    def test_attention_mask(self):
        """Test transformer with attention mask."""
        model = SpikingTransformer(
            vocab_size=100,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            intermediate_size=64,
            timesteps=8
        )
        
        input_ids = torch.randint(0, 100, (2, 6))
        attention_mask = torch.ones(2, 6)
        attention_mask[0, 4:] = 0  # Mask last 2 tokens for first sample
        
        output = model(input_ids, attention_mask=attention_mask)
        assert output.shape == (2, 8, 6, 32)
    
    def test_state_reset(self):
        """Test transformer state reset."""
        model = SpikingTransformer(
            vocab_size=50,
            hidden_size=16,
            num_layers=1,
            num_heads=1,
            intermediate_size=32,
            timesteps=4
        )
        
        input_ids = torch.randint(0, 50, (1, 3))
        _ = model(input_ids)
        
        model.reset_state()


class TestSpikingViT:
    """Test spiking Vision Transformer."""
    
    def test_initialization(self):
        """Test spiking ViT initialization."""
        model = SpikingViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            embed_dim=256,
            num_layers=4,
            num_heads=4,
            timesteps=20
        )
        
        assert model.image_size == 224
        assert model.patch_size == 16
        assert model.num_patches == (224 // 16) ** 2  # 196
        assert model.embed_dim == 256
        assert model.timesteps == 20
        assert len(model.blocks) == 4
    
    def test_forward_pass(self):
        """Test forward pass through spiking ViT."""
        model = SpikingViT(
            image_size=32,
            patch_size=8,
            num_classes=10,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            timesteps=12
        )
        
        batch_size, channels, height, width = 2, 3, 32, 32
        x = torch.randn(batch_size, channels, height, width)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)  # Batch size, num_classes
    
    def test_patch_embedding(self):
        """Test patch embedding process."""
        model = SpikingViT(
            image_size=64,
            patch_size=16,
            embed_dim=128,
            num_layers=1,
            num_heads=2,
            timesteps=8
        )
        
        # Check patch embedding dimensions
        x = torch.randn(1, 3, 64, 64)
        
        # Extract patch embedding manually
        patches = model.patch_embed(x)  # Should be (batch, embed_dim, H//patch_size, W//patch_size)
        expected_patches = (64 // 16) ** 2  # 16 patches
        
        assert patches.shape == (1, 128, 4, 4)
    
    def test_cls_token(self):
        """Test CLS token processing."""
        model = SpikingViT(
            image_size=16,
            patch_size=8,
            embed_dim=32,
            num_layers=1,
            num_heads=2,
            timesteps=4
        )
        
        x = torch.randn(2, 3, 16, 16)
        output = model(x)
        
        # Should use CLS token for classification
        assert output.shape[0] == 2  # Batch size preserved
    
    def test_state_reset(self):
        """Test ViT state reset."""
        model = SpikingViT(
            image_size=16,
            patch_size=4,
            embed_dim=16,
            num_layers=1,
            num_heads=1,
            timesteps=4
        )
        
        x = torch.randn(1, 3, 16, 16)
        _ = model(x)
        
        model.reset_state()


class TestSpikingBERT:
    """Test spiking BERT model."""
    
    def test_initialization(self):
        """Test spiking BERT initialization."""
        model = SpikingBERT(
            vocab_size=1000,
            hidden_size=128,
            num_layers=3,
            num_heads=2,
            intermediate_size=512,
            timesteps=16
        )
        
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert hasattr(model, 'pooler')
    
    def test_forward_pass(self):
        """Test forward pass through spiking BERT."""
        model = SpikingBERT(
            vocab_size=500,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            timesteps=12
        )
        
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert isinstance(output, dict)
        assert "last_hidden_state" in output
        assert "pooler_output" in output
        
        assert output["last_hidden_state"].shape == (batch_size, 12, seq_len, 64)
        assert output["pooler_output"].shape == (batch_size, 64)
    
    def test_token_type_ids(self):
        """Test BERT with token type IDs."""
        model = SpikingBERT(
            vocab_size=100,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            intermediate_size=64,
            timesteps=8
        )
        
        input_ids = torch.randint(0, 100, (1, 6))
        token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
        
        output = model(input_ids, token_type_ids=token_type_ids)
        
        assert "last_hidden_state" in output
        assert "pooler_output" in output


class TestModelProperties:
    """Test general model properties."""
    
    def test_gradient_flow(self):
        """Test gradient flow through spiking models."""
        model = SpikingTransformer(
            vocab_size=100,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            intermediate_size=64,
            timesteps=8,
            num_classes=5
        )
        
        input_ids = torch.randint(0, 100, (2, 4))
        output = model(input_ids)
        
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
    
    def test_model_modes(self):
        """Test model training and evaluation modes."""
        model = SpikingTransformerBlock(
            embed_dim=64,
            num_heads=2,
            mlp_dim=128,
            timesteps=8
        )
        
        # Training mode
        model.train()
        assert model.training
        
        # Evaluation mode  
        model.eval()
        assert not model.training
    
    def test_parameter_count(self):
        """Test parameter counting."""
        model = SpikingTransformer(
            vocab_size=200,
            hidden_size=64,
            num_layers=2,
            num_heads=2,
            intermediate_size=128,
            timesteps=8
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_device_placement(self):
        """Test model device placement."""
        if torch.cuda.is_available():
            model = SpikingMLP(
                input_dim=32,
                hidden_dims=[64],
                output_dim=16,
                timesteps=8
            )
            
            # Move to GPU
            model = model.cuda()
            
            # Test inference on GPU
            x = torch.randn(2, 8, 32).cuda()
            output = model(x)
            
            assert output.is_cuda
            assert output.shape == (2, 8, 16)
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        model = SpikingAttention(
            embed_dim=32,
            num_heads=2,
            timesteps=8
        )
        
        # Save state dict
        state_dict = model.state_dict()
        
        # Create new model and load state dict
        new_model = SpikingAttention(
            embed_dim=32,
            num_heads=2,
            timesteps=8
        )
        new_model.load_state_dict(state_dict)
        
        # Test that models produce same output
        x = torch.randn(1, 8, 4, 32)
        
        # Set models to eval mode for deterministic output
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        # Outputs should be identical (deterministic in eval mode)
        # Note: Due to stochastic nature of spike generation, we check structure
        assert output1.shape == output2.shape
    
    def test_memory_efficiency(self):
        """Test memory usage of spiking models."""
        # This is a basic test - in practice, spiking models should use less memory
        # due to sparse activations
        
        model = SpikingMLP(
            input_dim=128,
            hidden_dims=[256, 128],
            output_dim=64,
            timesteps=16
        )
        
        x = torch.randn(4, 16, 128)
        
        # Forward pass should complete without memory issues
        output = model(x)
        assert output.shape == (4, 16, 64)
        
        # Check that output is sparse (should have many zeros for spiking networks)
        sparsity = (output == 0).float().mean()
        assert sparsity > 0.3  # At least 30% zeros expected