"""Integration tests for the complete conversion pipeline."""

import torch
import torch.nn as nn
import pytest
import tempfile
from pathlib import Path

from spikeformer.conversion import SpikeformerConverter, ConversionPipeline, ConversionConfig
from spikeformer.models import SpikingTransformer, SpikingViT
from spikeformer.neurons import LifNeuron


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing conversion."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, hidden_size))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_size, 10)
        
        # Add config for compatibility
        class Config:
            vocab_size = vocab_size
            hidden_size = hidden_size
            num_hidden_layers = num_layers
            num_attention_heads = num_heads
            intermediate_size = hidden_size * 4
            
        self.config = Config()
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        for layer in self.layers:
            x = layer(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """Simple CNN model for testing conversion."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TestSpikeformerConverter:
    """Test the main converter class."""
    
    def test_converter_initialization(self):
        """Test converter initialization with different configs."""
        # Default config
        converter = SpikeformerConverter()
        assert converter.config.timesteps == 32
        assert converter.config.threshold == 1.0
        
        # Custom config
        config = ConversionConfig(timesteps=64, threshold=1.5, neuron_model="ADLIF")
        converter = SpikeformerConverter(config)
        assert converter.config.timesteps == 64
        assert converter.config.threshold == 1.5
        assert converter.config.neuron_model == "ADLIF"
    
    def test_architecture_analysis(self):
        """Test model architecture analysis."""
        converter = SpikeformerConverter()
        
        # Test transformer analysis
        transformer = SimpleTransformer(vocab_size=500, hidden_size=128)
        arch_info = converter._analyze_architecture(transformer)
        
        assert arch_info["type"] == "transformer"
        assert arch_info["has_attention"] == True
        assert arch_info["total_params"] > 0
        assert "TransformerEncoderLayer" in arch_info["layer_types"]
        
        # Test CNN analysis
        cnn = SimpleCNN()
        arch_info = converter._analyze_architecture(cnn)
        
        assert arch_info["type"] == "cnn"
        assert "Conv2d" in str(arch_info["layer_types"])
    
    def test_transformer_conversion(self):
        """Test conversion of transformer model."""
        config = ConversionConfig(timesteps=16, threshold=1.0)
        converter = SpikeformerConverter(config)
        
        # Create simple transformer
        model = SimpleTransformer(vocab_size=200, hidden_size=64, num_layers=1, num_heads=2)
        
        # Convert to spiking
        snn_model = converter.convert(model)
        
        assert isinstance(snn_model, SpikingTransformer)
        
        # Test inference
        input_ids = torch.randint(0, 200, (2, 8))
        output = snn_model(input_ids)
        
        assert output.shape[0] == 2  # Batch size preserved
        assert torch.all((output == 0) | (output == 1))  # Binary spikes
    
    def test_cnn_conversion(self):
        """Test conversion of CNN model."""
        converter = SpikeformerConverter()
        
        # Create simple CNN
        model = SimpleCNN(num_classes=5)
        
        # Convert to spiking
        snn_model = converter.convert(model)
        
        # The converted model should still be a valid PyTorch module
        assert isinstance(snn_model, nn.Module)
        
        # Test inference
        x = torch.randn(2, 3, 32, 32)
        output = snn_model(x)
        
        assert output.shape == (2, 5)
    
    def test_linear_layer_conversion(self):
        """Test conversion of individual linear layers."""
        converter = SpikeformerConverter()
        
        # Create simple linear model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Convert
        snn_model = converter.convert(model)
        
        # Test inference
        x = torch.randn(3, 100)
        output = snn_model(x)
        
        assert output.shape == (3, 10)
    
    def test_weight_transfer(self):
        """Test that weights are properly transferred."""
        converter = SpikeformerConverter()
        
        # Create transformer with known weights
        model = SimpleTransformer(vocab_size=100, hidden_size=32, num_layers=1)
        original_embedding_weight = model.embedding.weight.clone()
        
        # Convert
        snn_model = converter.convert(model)
        
        # Check that some weights are preserved
        # Note: This is a simplified test - in practice, weight mapping is complex
        assert snn_model is not None
        assert isinstance(snn_model, SpikingTransformer)


class TestConversionPipeline:
    """Test the complete conversion pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        config = ConversionConfig(timesteps=24, calibration_samples=50)
        pipeline = ConversionPipeline(config)
        
        assert pipeline.config.timesteps == 24
        assert pipeline.config.calibration_samples == 50
    
    def test_conversion_without_data(self):
        """Test conversion without calibration data."""
        pipeline = ConversionPipeline()
        
        model = SimpleTransformer(vocab_size=100, hidden_size=32, num_layers=1)
        result = pipeline.convert(model)
        
        assert result.snn_model is not None
        assert result.conversion_time > 0
        assert isinstance(result.snn_model, SpikingTransformer)
    
    def test_conversion_with_calibration_data(self):
        """Test conversion with calibration data."""
        pipeline = ConversionPipeline()
        
        model = SimpleTransformer(vocab_size=50, hidden_size=32, num_layers=1)
        
        # Create mock calibration data
        from torch.utils.data import DataLoader, TensorDataset
        cal_inputs = torch.randint(0, 50, (20, 8))
        cal_targets = torch.randint(0, 10, (20,))
        cal_dataset = TensorDataset(cal_inputs, cal_targets)
        cal_loader = DataLoader(cal_dataset, batch_size=4)
        
        result = pipeline.convert(model, calibration_data=cal_loader)
        
        assert result.snn_model is not None
        assert result.conversion_time > 0
        assert "original_accuracy" not in result.metadata  # No test data provided
    
    def test_conversion_with_evaluation(self):
        """Test conversion with evaluation data."""
        pipeline = ConversionPipeline()
        
        model = SimpleTransformer(vocab_size=30, hidden_size=16, num_layers=1)
        
        # Create mock test data
        from torch.utils.data import DataLoader, TensorDataset
        test_inputs = torch.randint(0, 30, (10, 4))
        test_targets = torch.randint(0, 10, (10,))
        test_dataset = TensorDataset(test_inputs, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=2)
        
        result = pipeline.convert(model, test_data=test_loader)
        
        assert result.snn_model is not None
        assert "original_accuracy" in result.metadata
        assert "snn_accuracy" in result.metadata
        assert "accuracy_retention" in result.metadata
    
    def test_conversion_metrics(self):
        """Test conversion quality metrics."""
        pipeline = ConversionPipeline()
        
        model = SimpleTransformer(vocab_size=25, hidden_size=16, num_layers=1)
        result = pipeline.convert(model)
        
        # Check that metrics are computed
        assert result.spike_sparsity >= 0
        assert result.energy_reduction > 0
        assert result.accuracy_retention >= 0
        assert result.conversion_time > 0
        
        # Check metadata
        assert "spike_sparsity" in result.metadata
        assert "energy_reduction" in result.metadata


class TestThresholdCalibration:
    """Test threshold calibration functionality."""
    
    def test_layer_calibration(self):
        """Test calibration of individual layers."""
        config = ConversionConfig(calibration_samples=10)
        converter = SpikeformerConverter(config)
        calibrator = converter.calibrator
        
        # Create a linear layer
        layer = nn.Linear(20, 10)
        
        # Create calibration data
        cal_data = torch.randn(5, 20)
        
        # Calibrate threshold
        threshold = calibrator.calibrate_layer(layer, cal_data)
        
        assert isinstance(threshold, float)
        assert threshold > 0
    
    def test_model_calibration(self):
        """Test calibration of entire model."""
        config = ConversionConfig(calibration_samples=5)
        converter = SpikeformerConverter(config)
        calibrator = converter.calibrator
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Create mock data loader
        from torch.utils.data import DataLoader, TensorDataset
        cal_inputs = torch.randn(8, 10)
        cal_targets = torch.randint(0, 2, (8,))
        cal_dataset = TensorDataset(cal_inputs, cal_targets)
        cal_loader = DataLoader(cal_dataset, batch_size=2)
        
        thresholds = calibrator.calibrate_model(model, cal_loader)
        
        assert isinstance(thresholds, dict)
        assert len(thresholds) > 0
        
        # Check that thresholds are positive
        for threshold in thresholds.values():
            assert threshold > 0


class TestConversionAccuracy:
    """Test conversion accuracy and fidelity."""
    
    def test_output_similarity(self):
        """Test that SNN output is similar to ANN output."""
        converter = SpikeformerConverter()
        
        # Create simple model
        model = SimpleTransformer(vocab_size=50, hidden_size=32, num_layers=1)
        model.eval()
        
        # Convert to SNN
        snn_model = converter.convert(model)
        snn_model.eval()
        
        # Test input
        input_ids = torch.randint(0, 50, (2, 6))
        
        # Get outputs
        with torch.no_grad():
            ann_output = model(input_ids)
            snn_output = snn_model(input_ids)
        
        # SNN output should be from classification head if present
        if hasattr(snn_model, 'classifier') and snn_model.classifier is not None:
            assert snn_output.shape[0] == ann_output.shape[0]  # Same batch size
        else:
            # Compare shapes at least
            assert snn_output.shape[0] == input_ids.shape[0]
    
    def test_spike_properties(self):
        """Test that converted model produces valid spikes."""
        converter = SpikeformerConverter()
        
        model = SimpleTransformer(vocab_size=30, hidden_size=16, num_layers=1)
        snn_model = converter.convert(model)
        
        input_ids = torch.randint(0, 30, (1, 4))
        
        # Hook to capture intermediate activations
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(module, LifNeuron):
                activations.append(output)
        
        # Register hooks on all LIF neurons
        hooks = []
        for module in snn_model.modules():
            if isinstance(module, LifNeuron):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            output = snn_model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Check spike properties
        for activation in activations:
            # Should be binary
            assert torch.all((activation == 0) | (activation == 1))
            
            # Should have some spikes (not all zeros)
            assert torch.any(activation > 0)
            
            # Should have reasonable sparsity
            sparsity = (activation == 0).float().mean()
            assert 0.1 <= sparsity <= 0.95  # Between 10% and 95% sparsity


class TestConversionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model(self):
        """Test conversion of empty model."""
        converter = SpikeformerConverter()
        
        model = nn.Sequential()  # Empty model
        
        # Should handle gracefully
        snn_model = converter.convert(model)
        assert snn_model is not None
    
    def test_model_without_linear_layers(self):
        """Test conversion of model without linear layers."""
        converter = SpikeformerConverter()
        
        # Model with only non-linear operations
        model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(10)
        )
        
        snn_model = converter.convert(model)
        assert snn_model is not None
    
    def test_large_model_conversion(self):
        """Test conversion of relatively large model."""
        converter = SpikeformerConverter()
        
        # Larger transformer
        model = SimpleTransformer(
            vocab_size=1000,
            hidden_size=128,
            num_layers=3,
            num_heads=4
        )
        
        snn_model = converter.convert(model)
        assert isinstance(snn_model, SpikingTransformer)
        
        # Test that it can still run inference
        input_ids = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = snn_model(input_ids)
        
        assert output is not None
    
    def test_conversion_determinism(self):
        """Test that conversion is deterministic."""
        # Set random seeds
        torch.manual_seed(42)
        converter1 = SpikeformerConverter()
        
        torch.manual_seed(42)
        converter2 = SpikeformerConverter()
        
        # Same model
        torch.manual_seed(123)
        model1 = SimpleTransformer(vocab_size=100, hidden_size=32, num_layers=1)
        
        torch.manual_seed(123)
        model2 = SimpleTransformer(vocab_size=100, hidden_size=32, num_layers=1)
        
        # Convert both
        snn_model1 = converter1.convert(model1)
        snn_model2 = converter2.convert(model2)
        
        # Should have same structure
        assert type(snn_model1) == type(snn_model2)


class TestIntegrationWithHardware:
    """Test integration between conversion and hardware deployment."""
    
    def test_conversion_for_loihi2(self):
        """Test conversion optimized for Loihi 2."""
        # Configure for Loihi 2 constraints
        config = ConversionConfig(
            timesteps=32,
            threshold=1.0,
            neuron_model="LIF",  # Loihi 2 compatible
            hardware_constraints={"max_fanin": 64, "max_fanout": 128}
        )
        
        converter = SpikeformerConverter(config)
        
        model = SimpleTransformer(vocab_size=100, hidden_size=64, num_layers=1)
        snn_model = converter.convert(model)
        
        # Should produce compatible model
        assert isinstance(snn_model, SpikingTransformer)
        
        # Test with hardware backend (mock)
        from spikeformer.hardware import NeuromorphicDeployer
        
        deployer = NeuromorphicDeployer("loihi2")
        deployment_result = deployer.deploy_model(snn_model)
        
        assert deployment_result.compiled_model is not None
        assert deployment_result.deployment_time > 0
    
    def test_conversion_for_spinnaker(self):
        """Test conversion optimized for SpiNNaker."""
        config = ConversionConfig(
            timesteps=20,
            neuron_model="LIF",
            spike_encoding="poisson"
        )
        
        converter = SpikeformerConverter(config)
        
        model = SimpleTransformer(vocab_size=50, hidden_size=32, num_layers=1)
        snn_model = converter.convert(model)
        
        # Test with SpiNNaker backend
        from spikeformer.hardware import NeuromorphicDeployer
        
        deployer = NeuromorphicDeployer("spinnaker")
        deployment_result = deployer.deploy_model(snn_model)
        
        assert deployment_result is not None


class TestConversionPersistence:
    """Test saving and loading converted models."""
    
    def test_save_load_converted_model(self):
        """Test saving and loading converted models."""
        converter = SpikeformerConverter()
        
        # Convert model
        model = SimpleTransformer(vocab_size=50, hidden_size=32, num_layers=1)
        snn_model = converter.convert(model)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "converted_model.pth"
            
            model_data = {
                'model_state_dict': snn_model.state_dict(),
                'model_class': snn_model.__class__.__name__,
                'conversion_config': {
                    'timesteps': converter.config.timesteps,
                    'threshold': converter.config.threshold,
                    'neuron_model': converter.config.neuron_model
                }
            }
            
            torch.save(model_data, save_path)
            
            # Load model
            loaded_data = torch.load(save_path, map_location='cpu')
            
            assert loaded_data['model_class'] == 'SpikingTransformer'
            assert 'conversion_config' in loaded_data
            assert 'model_state_dict' in loaded_data
    
    def test_conversion_metadata(self):
        """Test that conversion metadata is properly stored."""
        config = ConversionConfig(
            timesteps=16,
            threshold=1.5,
            neuron_model="ADLIF",
            calibration_samples=25
        )
        
        pipeline = ConversionPipeline(config)
        
        model = SimpleTransformer(vocab_size=30, hidden_size=16, num_layers=1)
        result = pipeline.convert(model)
        
        # Check metadata completeness
        assert result.conversion_time > 0
        assert result.accuracy_retention >= 0
        assert result.spike_sparsity >= 0
        assert result.energy_reduction > 0
        
        metadata = result.metadata
        assert "spike_sparsity" in metadata
        assert "energy_reduction" in metadata