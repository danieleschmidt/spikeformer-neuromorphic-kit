#!/usr/bin/env python3
"""
Simple demonstration of SpikeFormer core functionality without heavy dependencies.
Generation 1: MAKE IT WORK (Simple) - Basic functionality demo.
"""

import sys
import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class SpikingModelConfig:
    """Simple configuration for spiking models."""
    timesteps: int = 32
    threshold: float = 1.0
    neuron_model: str = "LIF"
    spike_encoding: str = "rate"
    dropout: float = 0.1

class SimpleSpikingTransformer:
    """Simplified spiking transformer for demonstration."""
    
    def __init__(self, config: SpikingModelConfig):
        self.config = config
        self.initialized = True
        self.training_history = []
        print(f"✅ SimpleSpikingTransformer initialized with {config.timesteps} timesteps")
    
    def encode_input(self, data: List[float]) -> List[List[int]]:
        """Simple rate encoding simulation."""
        spikes = []
        for timestep in range(self.config.timesteps):
            timestep_spikes = []
            for value in data:
                # Simple threshold-based spike generation
                spike = 1 if (value * timestep / self.config.timesteps) > self.config.threshold else 0
                timestep_spikes.append(spike)
            spikes.append(timestep_spikes)
        return spikes
    
    def forward(self, input_data: List[float]) -> Dict[str, Any]:
        """Forward pass simulation."""
        start_time = time.time()
        
        # Encode input to spikes
        spikes = self.encode_input(input_data)
        
        # Simulate processing
        total_spikes = sum(sum(timestep) for timestep in spikes)
        sparsity = 1.0 - (total_spikes / (len(spikes) * len(input_data)))
        
        # Simulate output
        output = sum(input_data) / len(input_data) * 0.8  # Simple transformation
        
        result = {
            "output": output,
            "total_spikes": total_spikes,
            "sparsity": sparsity,
            "inference_time_ms": (time.time() - start_time) * 1000,
            "energy_estimate_mj": total_spikes * 0.1  # Mock energy calculation
        }
        
        return result

class SimpleEnergyProfiler:
    """Simple energy profiling simulation."""
    
    def __init__(self):
        self.measurements = []
        self.baseline_gpu_energy = 100.0  # mJ baseline
    
    def profile_inference(self, model: SimpleSpikingTransformer, data: List[float]) -> Dict[str, float]:
        """Profile energy consumption for inference."""
        start_time = time.time()
        result = model.forward(data)
        duration = time.time() - start_time
        
        # Mock energy calculations
        spiking_energy = result["energy_estimate_mj"]
        gpu_energy_reduction = self.baseline_gpu_energy / max(spiking_energy, 0.1)
        
        profile = {
            "spiking_energy_mj": spiking_energy,
            "gpu_baseline_mj": self.baseline_gpu_energy,
            "energy_reduction_factor": gpu_energy_reduction,
            "inference_time_ms": duration * 1000,
            "sparsity": result["sparsity"]
        }
        
        self.measurements.append(profile)
        return profile

class SimpleNeuromorphicDeployer:
    """Simple deployment simulation."""
    
    def __init__(self, target_hardware: str = "loihi2"):
        self.target_hardware = target_hardware
        self.supported_hardware = ["loihi2", "spinnaker", "cpu_simulation"]
        
        if target_hardware not in self.supported_hardware:
            raise ValueError(f"Unsupported hardware: {target_hardware}")
    
    def deploy(self, model: SimpleSpikingTransformer) -> Dict[str, Any]:
        """Simulate deployment to neuromorphic hardware."""
        deployment_info = {
            "target_hardware": self.target_hardware,
            "model_config": asdict(model.config),
            "deployment_time": time.time(),
            "status": "deployed",
            "estimated_power_mw": 50.0 if self.target_hardware == "loihi2" else 100.0,
            "estimated_latency_ms": 5.0 if self.target_hardware == "loihi2" else 10.0
        }
        
        print(f"✅ Model deployed to {self.target_hardware}")
        return deployment_info

def run_basic_functionality_demo():
    """Run basic functionality demonstration."""
    print("🚀 SpikeFormer Basic Functionality Demo - Generation 1")
    print("=" * 60)
    
    # Initialize configuration
    config = SpikingModelConfig(
        timesteps=16,
        threshold=0.5,
        neuron_model="LIF",
        spike_encoding="rate"
    )
    
    # Create model
    print("\n📊 Creating SimpleSpikingTransformer...")
    model = SimpleSpikingTransformer(config)
    
    # Test data
    test_data = [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6]
    print(f"📥 Input data: {test_data}")
    
    # Run inference
    print("\n⚡ Running inference...")
    result = model.forward(test_data)
    print(f"📤 Output: {result['output']:.3f}")
    print(f"⭐ Total spikes: {result['total_spikes']}")
    print(f"🔥 Sparsity: {result['sparsity']:.2%}")
    print(f"⏱️ Inference time: {result['inference_time_ms']:.2f} ms")
    
    # Energy profiling
    print("\n⚡ Energy Profiling...")
    profiler = SimpleEnergyProfiler()
    energy_profile = profiler.profile_inference(model, test_data)
    print(f"🔋 Spiking energy: {energy_profile['spiking_energy_mj']:.2f} mJ")
    print(f"💻 GPU baseline: {energy_profile['gpu_baseline_mj']:.2f} mJ")
    print(f"🎯 Energy reduction: {energy_profile['energy_reduction_factor']:.1f}x")
    
    # Deployment simulation
    print("\n🚀 Deployment Simulation...")
    deployer = SimpleNeuromorphicDeployer("loihi2")
    deployment = deployer.deploy(model)
    print(f"🏭 Target: {deployment['target_hardware']}")
    print(f"⚡ Estimated power: {deployment['estimated_power_mw']:.1f} mW")
    print(f"⏱️ Estimated latency: {deployment['estimated_latency_ms']:.1f} ms")
    
    # Save results
    results = {
        "demo_type": "basic_functionality",
        "generation": "1_make_it_work",
        "timestamp": time.time(),
        "model_config": asdict(config),
        "inference_result": result,
        "energy_profile": energy_profile,
        "deployment_info": deployment,
        "status": "success"
    }
    
    with open("/root/repo/generation_1_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Results saved to: generation_1_demo_results.json")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    try:
        results = run_basic_functionality_demo()
        print("🎉 Generation 1 (MAKE IT WORK) - COMPLETED")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)