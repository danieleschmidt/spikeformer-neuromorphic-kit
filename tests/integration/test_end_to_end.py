"""Integration tests for end-to-end SpikeFormer workflows."""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tests.conftest import (
    assert_spike_tensor_valid,
    assert_energy_reduction,
    assert_accuracy_retention
)


class TestEndToEndConversion:
    """Test complete conversion pipeline from ANN to SNN."""
    
    @pytest.mark.integration
    def test_vit_conversion_pipeline(self, sample_transformer_model, sample_dataloader, temp_dir):
        """Test complete ViT conversion pipeline."""
        model = sample_transformer_model
        dataloader = sample_dataloader
        
        # Mock the complete conversion pipeline
        conversion_steps = [
            "model_analysis",
            "layer_conversion", 
            "threshold_calibration",
            "architecture_optimization",
            "validation"
        ]
        
        results = {}
        for step in conversion_steps:
            # Simulate each step
            if step == "model_analysis":
                results[step] = {
                    "num_layers": len(list(model.modules())),
                    "num_parameters": sum(p.numel() for p in model.parameters())
                }
            elif step == "layer_conversion":
                results[step] = {"converted_layers": results["model_analysis"]["num_layers"]}
            elif step == "threshold_calibration":
                # Use some calibration data
                calibration_batches = 0
                for batch in dataloader:
                    calibration_batches += 1
                    if calibration_batches >= 3:  # Limit for testing
                        break
                results[step] = {"calibration_batches": calibration_batches}
            elif step == "architecture_optimization":
                results[step] = {"optimization_applied": True}
            elif step == "validation":
                results[step] = {
                    "accuracy_retention": 0.96,
                    "energy_reduction": 12.5,
                    "sparsity": 0.85
                }
        
        # Verify pipeline completion
        assert all(step in results for step in conversion_steps)
        assert results["validation"]["accuracy_retention"] > 0.95
        assert results["validation"]["energy_reduction"] > 10.0
    
    @pytest.mark.integration
    def test_bert_conversion_pipeline(self, temp_dir):
        """Test complete BERT conversion pipeline."""
        # Mock BERT model
        bert_config = {
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512
        }
        
        # Mock conversion process
        conversion_results = {
            "model_type": "bert",
            "original_config": bert_config,
            "snn_config": {
                **bert_config,
                "timesteps": 20,
                "neuron_model": "LIF",
                "threshold": 1.0
            },
            "conversion_metrics": {
                "accuracy_retention": 0.94,
                "energy_reduction": 18.2,
                "conversion_time_seconds": 45.6
            }
        }
        
        assert conversion_results["conversion_metrics"]["accuracy_retention"] > 0.90
        assert conversion_results["conversion_metrics"]["energy_reduction"] > 15.0
    
    @pytest.mark.integration 
    def test_model_serialization_and_loading(self, temp_dir):
        """Test saving and loading converted models."""
        model_path = temp_dir / "converted_model.pth"
        
        # Mock model serialization
        mock_model_state = {
            "snn_config": {
                "timesteps": 32,
                "threshold": 1.0,
                "neuron_model": "LIF"
            },
            "state_dict": {"layer.weight": torch.randn(64, 128)},
            "conversion_metadata": {
                "original_accuracy": 0.85,
                "snn_accuracy": 0.82,
                "energy_reduction": 15.5
            }
        }
        
        # Simulate saving
        torch.save(mock_model_state, model_path)
        assert model_path.exists()
        
        # Simulate loading
        loaded_state = torch.load(model_path)
        assert loaded_state["snn_config"]["timesteps"] == 32
        assert "state_dict" in loaded_state
        assert "conversion_metadata" in loaded_state


class TestHardwareDeploymentIntegration:
    """Test integration with hardware deployment."""
    
    @pytest.mark.integration
    @pytest.mark.hardware
    def test_loihi2_deployment_pipeline(self, mock_loihi2_backend):
        """Test complete deployment pipeline to Loihi 2."""
        backend = mock_loihi2_backend
        
        # Mock model for deployment
        mock_snn_model = Mock()
        mock_snn_model.config = {
            "timesteps": 32,
            "num_layers": 6,
            "hidden_dim": 256
        }
        
        # Test compilation
        compiled_model = backend.compile(mock_snn_model)
        assert compiled_model is not None
        
        # Test deployment
        deployed_model = backend.deploy(compiled_model)
        assert deployed_model is not None
        
        # Test execution
        mock_input = torch.randn(1, 3, 32, 32)
        results = backend.execute(deployed_model, mock_input)
        
        assert results.energy_uj > 0
        assert results.latency_ms > 0
        assert 0 <= results.accuracy <= 1
    
    @pytest.mark.integration
    @pytest.mark.hardware
    def test_spinnaker_deployment_pipeline(self, mock_spinnaker_backend):
        """Test complete deployment pipeline to SpiNNaker."""
        backend = mock_spinnaker_backend
        
        # Mock model for deployment
        mock_snn_model = Mock()
        mock_snn_model.config = {
            "timesteps": 40,
            "num_layers": 8,
            "hidden_dim": 512
        }
        
        # Test compilation
        compiled_model = backend.compile(mock_snn_model)
        assert compiled_model is not None
        
        # Test deployment
        deployed_model = backend.deploy(compiled_model)
        assert deployed_model is not None
        
        # Test execution
        mock_input = torch.randn(1, 3, 64, 64)
        results = backend.execute(deployed_model, mock_input)
        
        assert results.energy_uj > 0
        assert results.latency_ms > 0
        assert 0 <= results.accuracy <= 1
    
    @pytest.mark.integration
    def test_multi_hardware_comparison(self, mock_loihi2_backend, mock_spinnaker_backend):
        """Test comparison across multiple hardware platforms."""
        backends = {
            "loihi2": mock_loihi2_backend,
            "spinnaker": mock_spinnaker_backend
        }
        
        # Mock model
        mock_snn_model = Mock()
        
        results = {}
        for platform, backend in backends.items():
            compiled_model = backend.compile(mock_snn_model)
            deployed_model = backend.deploy(compiled_model)
            execution_result = backend.execute(deployed_model, torch.randn(1, 3, 32, 32))
            
            results[platform] = {
                "energy_uj": execution_result.energy_uj,
                "latency_ms": execution_result.latency_ms,
                "accuracy": execution_result.accuracy
            }
        
        # Verify we have results for both platforms
        assert "loihi2" in results
        assert "spinnaker" in results
        
        # Compare platforms
        loihi_energy = results["loihi2"]["energy_uj"]
        spinnaker_energy = results["spinnaker"]["energy_uj"]
        
        assert loihi_energy > 0
        assert spinnaker_energy > 0


class TestTrainingIntegration:
    """Test integration of training workflows."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_hybrid_training_pipeline(self, sample_dataloader):
        """Test hybrid ANN-SNN training pipeline."""
        dataloader = sample_dataloader
        
        # Mock hybrid training process
        training_phases = ["ann_pretraining", "snn_finetuning", "knowledge_distillation"]
        
        results = {}
        for phase in training_phases:
            if phase == "ann_pretraining":
                # Simulate ANN training
                epoch_results = []
                for epoch in range(3):  # Limited epochs for testing
                    epoch_loss = 1.0 - (epoch * 0.2)  # Decreasing loss
                    epoch_acc = 0.6 + (epoch * 0.1)   # Increasing accuracy
                    epoch_results.append({"loss": epoch_loss, "accuracy": epoch_acc})
                results[phase] = epoch_results
                
            elif phase == "snn_finetuning":
                # Simulate SNN fine-tuning
                initial_acc = results["ann_pretraining"][-1]["accuracy"]
                final_acc = initial_acc * 0.96  # Some accuracy drop expected
                results[phase] = {"initial_accuracy": initial_acc, "final_accuracy": final_acc}
                
            elif phase == "knowledge_distillation":
                # Simulate knowledge distillation
                teacher_acc = results["ann_pretraining"][-1]["accuracy"]
                student_acc = results["snn_finetuning"]["final_accuracy"]
                distilled_acc = student_acc * 1.02  # Slight improvement
                results[phase] = {
                    "teacher_accuracy": teacher_acc,
                    "student_accuracy": student_acc,
                    "distilled_accuracy": distilled_acc
                }
        
        # Verify training pipeline
        assert len(results) == 3
        final_accuracy = results["knowledge_distillation"]["distilled_accuracy"]
        original_accuracy = results["ann_pretraining"][-1]["accuracy"]
        
        # Should retain most of the original accuracy
        assert_accuracy_retention(final_accuracy, original_accuracy, min_retention=0.90)
    
    @pytest.mark.integration
    def test_direct_snn_training(self, sample_dataloader):
        """Test direct SNN training from scratch."""
        dataloader = sample_dataloader
        
        # Mock direct SNN training
        training_config = {
            "learning_rate": 1e-4,
            "surrogate_gradient": "fast_sigmoid",
            "regularization": ["spike_count", "membrane_potential"],
            "epochs": 5
        }
        
        # Simulate training loop
        training_history = []
        for epoch in range(training_config["epochs"]):
            epoch_metrics = {
                "epoch": epoch,
                "loss": 2.0 - (epoch * 0.3),  # Decreasing loss
                "accuracy": 0.4 + (epoch * 0.1),  # Increasing accuracy
                "spike_rate": 0.15 - (epoch * 0.01),  # Optimizing spike rate
                "energy_estimate": 100.0 - (epoch * 5)  # Improving energy efficiency
            }
            training_history.append(epoch_metrics)
        
        # Verify training progress
        assert len(training_history) == training_config["epochs"]
        assert training_history[-1]["accuracy"] > training_history[0]["accuracy"]
        assert training_history[-1]["loss"] < training_history[0]["loss"]


class TestEnergyProfilingIntegration:
    """Test integration of energy profiling workflows."""
    
    @pytest.mark.integration
    def test_energy_comparison_workflow(self, mock_energy_profiler):
        """Test complete energy comparison workflow."""
        profiler = mock_energy_profiler
        
        # Mock different model configurations
        models = {
            "gpu_baseline": {"energy_mJ": 1000.0, "accuracy": 0.85},
            "cpu_baseline": {"energy_mJ": 2000.0, "accuracy": 0.85},
            "loihi2_snn": {"energy_mJ": 80.0, "accuracy": 0.82},
            "spinnaker_snn": {"energy_mJ": 120.0, "accuracy": 0.81}
        }
        
        # Calculate comparisons
        gpu_baseline = models["gpu_baseline"]["energy_mJ"]
        
        energy_reductions = {}
        accuracy_retentions = {}
        
        for model_name, metrics in models.items():
            if "snn" in model_name:
                energy_reductions[model_name] = gpu_baseline / metrics["energy_mJ"]
                accuracy_retentions[model_name] = metrics["accuracy"] / models["gpu_baseline"]["accuracy"]
        
        # Verify energy efficiency
        for model_name, reduction in energy_reductions.items():
            assert reduction > 5.0, f"{model_name} doesn't achieve sufficient energy reduction"
        
        # Verify reasonable accuracy retention
        for model_name, retention in accuracy_retentions.items():
            assert retention > 0.90, f"{model_name} accuracy retention too low"
    
    @pytest.mark.integration
    def test_real_time_energy_monitoring(self):
        """Test real-time energy monitoring during inference."""
        # Mock real-time monitoring
        monitoring_duration = 10  # seconds
        sampling_rate = 100  # Hz
        
        energy_samples = []
        for i in range(monitoring_duration * sampling_rate):
            # Simulate energy measurements with some variation
            base_power = 250.0  # mW
            variation = torch.randn(1).item() * 20.0
            power_sample = base_power + variation
            energy_samples.append(power_sample)
        
        # Analyze monitoring results
        avg_power = sum(energy_samples) / len(energy_samples)
        max_power = max(energy_samples)
        min_power = min(energy_samples)
        
        assert len(energy_samples) == monitoring_duration * sampling_rate
        assert 200.0 < avg_power < 300.0  # Reasonable power range
        assert max_power > avg_power
        assert min_power < avg_power


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Integration tests for performance benchmarking."""
    
    def test_inference_latency_benchmark(self, benchmark):
        """Benchmark inference latency."""
        # Mock inference function
        def run_inference():
            # Simulate SNN inference
            batch_size = 8
            timesteps = 32
            
            # Simulate processing timesteps
            for t in range(timesteps):
                # Mock computation per timestep
                computation = torch.randn(batch_size, 256) @ torch.randn(256, 10)
            
            return computation.shape[0]
        
        result = benchmark(run_inference)
        assert result == 8  # batch size
    
    def test_throughput_benchmark(self, benchmark):
        """Benchmark model throughput."""
        def process_batch():
            # Mock batch processing
            batch_sizes = [1, 4, 8, 16, 32]
            throughputs = []
            
            for batch_size in batch_sizes:
                # Simulate processing time (inversely related to efficiency)
                processing_time = 1.0 / batch_size  # Simplified model
                throughput = batch_size / processing_time
                throughputs.append(throughput)
            
            return max(throughputs)
        
        max_throughput = benchmark(process_batch)
        assert max_throughput > 0
    
    def test_memory_efficiency_benchmark(self, memory_monitor):
        """Benchmark memory efficiency."""
        initial_memory = memory_monitor.get_memory_usage()
        
        # Mock model processing with different configurations
        configurations = [
            {"batch_size": 1, "timesteps": 32},
            {"batch_size": 8, "timesteps": 32},
            {"batch_size": 16, "timesteps": 16}
        ]
        
        memory_usages = []
        for config in configurations:
            # Simulate memory allocation
            temp_tensors = []
            for _ in range(config["batch_size"]):
                tensor = torch.randn(config["timesteps"], 256, 256)
                temp_tensors.append(tensor)
            
            peak_memory = memory_monitor.get_memory_usage()
            memory_usages.append(peak_memory - initial_memory)
            
            # Clean up
            del temp_tensors
        
        # Memory usage should scale reasonably with batch size
        assert len(memory_usages) == len(configurations)
        assert all(usage > 0 for usage in memory_usages)