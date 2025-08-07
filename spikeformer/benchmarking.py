"""Comprehensive benchmarking framework for neuromorphic computing research."""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
import warnings

from .models import SpikingTransformer, SpikingViT, SpikingBERT
from .profiling import EnergyProfiler, PowerMonitor
from .hardware import NeuromorphicDeployer
from .conversion import ConversionPipeline


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    name: str
    description: str
    models: List[str]
    datasets: List[str] 
    hardware_targets: List[str]
    metrics: List[str]
    repetitions: int = 3
    timeout_seconds: int = 3600
    save_results: bool = True
    output_dir: str = "benchmark_results"
    parallel_execution: bool = True
    statistical_significance: bool = True
    confidence_level: float = 0.95


@dataclass 
class BenchmarkResult:
    """Results from a single benchmark experiment."""
    config_name: str
    model_name: str
    dataset_name: str
    hardware_target: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str
    execution_time: float
    memory_usage: float
    energy_consumption: Optional[float] = None
    error_message: Optional[str] = None


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    @abstractmethod
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        """Calculate metrics for the given model and data."""
        pass


class AccuracyMetric(MetricCalculator):
    """Calculate classification accuracy."""
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                
                if isinstance(outputs, dict):
                    outputs = outputs.get('logits', outputs['last_hidden_state'])
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        return {"accuracy": correct / total if total > 0 else 0.0}


class SpikeSparsityMetric(MetricCalculator):
    """Calculate spike sparsity across the network."""
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        spike_counts = []
        total_activations = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'sum'):  # Spike tensor
                spikes = output.sum().item()
                total = output.numel()
                spike_counts.append(spikes)
                total_activations.append(total)
        
        hooks = []
        for name, module in model.named_modules():
            if 'neuron' in name.lower():
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= 10:  # Limit samples for efficiency
                    break
                data = data.to(device)
                _ = model(data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if spike_counts and total_activations:
            total_spikes = sum(spike_counts)
            total_acts = sum(total_activations)
            sparsity = 1.0 - (total_spikes / total_acts)
        else:
            sparsity = 0.0
            
        return {"spike_sparsity": sparsity}


class LatencyMetric(MetricCalculator):
    """Measure inference latency."""
    
    def __init__(self, warmup_iterations: int = 10, measure_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.measure_iterations = measure_iterations
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        model.eval()
        
        # Get sample data
        sample_data = next(iter(data_loader))[0][:1].to(device)  # Single sample
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(sample_data)
        
        # Synchronize GPU if available
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(self.measure_iterations):
                start_time = time.perf_counter()
                _ = model(sample_data)
                
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                    
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
        
        return {
            "latency_mean_ms": np.mean(latencies) * 1000,
            "latency_std_ms": np.std(latencies) * 1000,
            "latency_p50_ms": np.percentile(latencies, 50) * 1000,
            "latency_p95_ms": np.percentile(latencies, 95) * 1000,
            "latency_p99_ms": np.percentile(latencies, 99) * 1000,
        }


class ThroughputMetric(MetricCalculator):
    """Measure inference throughput."""
    
    def __init__(self, duration_seconds: float = 10.0):
        self.duration_seconds = duration_seconds
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        model.eval()
        
        start_time = time.time()
        samples_processed = 0
        
        with torch.no_grad():
            while time.time() - start_time < self.duration_seconds:
                for data, _ in data_loader:
                    data = data.to(device)
                    _ = model(data)
                    samples_processed += data.size(0)
                    
                    if time.time() - start_time >= self.duration_seconds:
                        break
        
        duration = time.time() - start_time
        throughput = samples_processed / duration
        
        return {"throughput_samples_per_sec": throughput}


class MemoryUsageMetric(MetricCalculator):
    """Measure memory usage during inference."""
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure peak memory during inference
            model.eval()
            sample_data = next(iter(data_loader))[0][:1].to(device)
            
            with torch.no_grad():
                _ = model(sample_data)
                
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            return {"gpu_memory_peak_gb": peak_memory}
        else:
            # CPU memory measurement
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**3  # GB
            
            model.eval()
            sample_data = next(iter(data_loader))[0][:1].to(device)
            
            with torch.no_grad():
                _ = model(sample_data)
                
            peak_memory = process.memory_info().rss / 1024**3  # GB
            return {"cpu_memory_peak_gb": peak_memory - initial_memory}


class EnergyEfficiencyMetric(MetricCalculator):
    """Measure energy efficiency (performance per watt)."""
    
    def calculate(self, model: nn.Module, data_loader, device: str) -> Dict[str, float]:
        # This would integrate with actual power measurement hardware
        # For now, return estimated values based on computation
        
        profiler = EnergyProfiler(backend=device)
        
        model.eval()
        sample_data = next(iter(data_loader))[0][:10].to(device)  # Small batch
        
        with profiler.measure():
            with torch.no_grad():
                for _ in range(10):  # Multiple inferences
                    _ = model(sample_data)
        
        metrics = profiler.get_metrics()
        
        return {
            "energy_per_sample_mj": metrics.get("energy_per_sample", 0.0) * 1000,
            "power_consumption_w": metrics.get("average_power", 0.0),
            "energy_efficiency_samples_per_joule": 1.0 / max(metrics.get("energy_per_sample", 1e-6), 1e-6)
        }


class NeuromorphicBenchmark:
    """Main benchmarking framework for neuromorphic computing."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric calculators
        self.metrics = {
            "accuracy": AccuracyMetric(),
            "spike_sparsity": SpikeSparsityMetric(),
            "latency": LatencyMetric(),
            "throughput": ThroughputMetric(),
            "memory": MemoryUsageMetric(),
            "energy": EnergyEfficiencyMetric(),
        }
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_benchmark(self, 
                     models: Dict[str, nn.Module],
                     datasets: Dict[str, torch.utils.data.DataLoader],
                     hardware_targets: List[str] = ["cpu"]) -> List[BenchmarkResult]:
        """Run comprehensive benchmark across models, datasets, and hardware."""
        
        self.logger.info(f"Starting benchmark: {self.config.name}")
        self.logger.info(f"Models: {list(models.keys())}")
        self.logger.info(f"Datasets: {list(datasets.keys())}")
        self.logger.info(f"Hardware: {hardware_targets}")
        
        # Generate all experiment combinations
        experiments = []
        for model_name in models.keys():
            for dataset_name in datasets.keys():
                for hardware in hardware_targets:
                    for rep in range(self.config.repetitions):
                        experiments.append({
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "hardware": hardware,
                            "repetition": rep
                        })
        
        self.logger.info(f"Total experiments: {len(experiments)}")
        
        # Run experiments
        if self.config.parallel_execution and len(experiments) > 1:
            results = self._run_parallel_experiments(experiments, models, datasets)
        else:
            results = self._run_sequential_experiments(experiments, models, datasets)
        
        # Statistical analysis
        if self.config.statistical_significance:
            results = self._add_statistical_analysis(results)
        
        # Save results
        if self.config.save_results:
            self._save_results(results)
        
        self.results.extend(results)
        return results
    
    def _run_sequential_experiments(self,
                                  experiments: List[Dict[str, Any]],
                                  models: Dict[str, nn.Module],
                                  datasets: Dict[str, torch.utils.data.DataLoader]) -> List[BenchmarkResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, exp in enumerate(experiments):
            self.logger.info(f"Running experiment {i+1}/{len(experiments)}: {exp}")
            
            try:
                result = self._run_single_experiment(
                    exp["model_name"], exp["dataset_name"], exp["hardware"],
                    models[exp["model_name"]], datasets[exp["dataset_name"]]
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Experiment failed: {e}")
                results.append(BenchmarkResult(
                    config_name=self.config.name,
                    model_name=exp["model_name"],
                    dataset_name=exp["dataset_name"],
                    hardware_target=exp["hardware"],
                    metrics={},
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    execution_time=0.0,
                    memory_usage=0.0,
                    error_message=str(e)
                ))
                
        return results
    
    def _run_parallel_experiments(self,
                                experiments: List[Dict[str, Any]],
                                models: Dict[str, nn.Module],
                                datasets: Dict[str, torch.utils.data.DataLoader]) -> List[BenchmarkResult]:
        """Run experiments in parallel."""
        results = []
        
        # Use ThreadPoolExecutor for I/O bound tasks, ProcessPoolExecutor for CPU bound
        executor_class = ThreadPoolExecutor if torch.cuda.is_available() else ProcessPoolExecutor
        max_workers = min(len(experiments), psutil.cpu_count())
        
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            
            for exp in experiments:
                future = executor.submit(
                    self._run_single_experiment,
                    exp["model_name"], exp["dataset_name"], exp["hardware"],
                    models[exp["model_name"]], datasets[exp["dataset_name"]]
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel experiment failed: {e}")
                    
        return results
    
    def _run_single_experiment(self,
                             model_name: str,
                             dataset_name: str, 
                             hardware: str,
                             model: nn.Module,
                             data_loader: torch.utils.data.DataLoader) -> BenchmarkResult:
        """Run a single benchmark experiment."""
        
        start_time = time.time()
        device = torch.device(hardware if torch.cuda.is_available() and hardware.startswith('cuda') else 'cpu')
        
        # Move model to device
        model = model.to(device)
        
        # Reset model state if it's a spiking model
        if hasattr(model, 'reset_state'):
            model.reset_state()
        
        # Memory tracking
        initial_memory = self._get_memory_usage()
        
        # Run metrics
        all_metrics = {}
        for metric_name in self.config.metrics:
            if metric_name in self.metrics:
                try:
                    metric_results = self.metrics[metric_name].calculate(model, data_loader, str(device))
                    all_metrics.update(metric_results)
                except Exception as e:
                    self.logger.warning(f"Metric {metric_name} failed: {e}")
                    all_metrics[f"{metric_name}_error"] = str(e)
        
        # Execution metadata
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - initial_memory
        
        return BenchmarkResult(
            config_name=self.config.name,
            model_name=model_name,
            dataset_name=dataset_name,
            hardware_target=hardware,
            metrics=all_metrics,
            metadata={
                "device": str(device),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            execution_time=execution_time,
            memory_usage=memory_usage
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    def _add_statistical_analysis(self, results: List[BenchmarkResult]) -> List[BenchmarkResult]:
        """Add statistical significance analysis to results."""
        # Group results by experiment configuration
        grouped_results = {}
        
        for result in results:
            key = (result.model_name, result.dataset_name, result.hardware_target)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Add statistical metrics
        for key, group in grouped_results.items():
            if len(group) > 1:
                # Calculate confidence intervals for each metric
                for metric_name in group[0].metrics.keys():
                    if isinstance(group[0].metrics[metric_name], (int, float)):
                        values = [r.metrics[metric_name] for r in group if not np.isnan(r.metrics[metric_name])]
                        if values:
                            mean = np.mean(values)
                            std = np.std(values, ddof=1)
                            n = len(values)
                            
                            # Add statistical metrics
                            for result in group:
                                result.metrics[f"{metric_name}_mean"] = mean
                                result.metrics[f"{metric_name}_std"] = std
                                result.metrics[f"{metric_name}_ci_lower"] = mean - 1.96 * std / np.sqrt(n)
                                result.metrics[f"{metric_name}_ci_upper"] = mean + 1.96 * std / np.sqrt(n)
        
        return results
    
    def _save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        json_results = [asdict(result) for result in results]
        json_path = self.output_dir / f"benchmark_{self.config.name}_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {json_path}")
        
        # Save as CSV for easy analysis
        df = pd.DataFrame([{
            **{k: v for k, v in asdict(result).items() if k != 'metrics'},
            **result.metrics
        } for result in results])
        
        csv_path = self.output_dir / f"benchmark_{self.config.name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to {csv_path}")
    
    def generate_report(self, results: Optional[List[BenchmarkResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No results available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([{
            "model": r.model_name,
            "dataset": r.dataset_name,
            "hardware": r.hardware_target,
            "execution_time": r.execution_time,
            **r.metrics
        } for r in results if r.error_message is None])
        
        if df.empty:
            return {"error": "No successful experiments"}
        
        report = {
            "summary": {
                "total_experiments": len(results),
                "successful_experiments": len(df),
                "failed_experiments": len(results) - len(df),
                "models_tested": df["model"].unique().tolist(),
                "datasets_tested": df["dataset"].unique().tolist(),
                "hardware_tested": df["hardware"].unique().tolist(),
            },
            "performance_analysis": {},
            "statistical_analysis": {},
            "recommendations": []
        }
        
        # Performance analysis by model
        for model in df["model"].unique():
            model_data = df[df["model"] == model]
            
            model_metrics = {}
            for metric in self.config.metrics:
                metric_cols = [col for col in df.columns if col.startswith(metric)]
                if metric_cols:
                    for col in metric_cols:
                        if col in model_data.columns:
                            model_metrics[col] = {
                                "mean": float(model_data[col].mean()),
                                "std": float(model_data[col].std()),
                                "min": float(model_data[col].min()),
                                "max": float(model_data[col].max()),
                            }
            
            report["performance_analysis"][model] = model_metrics
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(df)
        
        return report
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        # Accuracy vs Energy efficiency trade-offs
        if "accuracy" in df.columns and "energy_per_sample_mj" in df.columns:
            efficient_models = df.nlargest(3, "energy_efficiency_samples_per_joule")
            recommendations.append(
                f"Most energy-efficient models: {', '.join(efficient_models['model'].tolist())}"
            )
        
        # Latency recommendations
        if "latency_mean_ms" in df.columns:
            fastest_models = df.nsmallest(3, "latency_mean_ms")
            recommendations.append(
                f"Lowest latency models: {', '.join(fastest_models['model'].tolist())}"
            )
        
        # Memory efficiency
        if "gpu_memory_peak_gb" in df.columns:
            memory_efficient = df.nsmallest(3, "gpu_memory_peak_gb")
            recommendations.append(
                f"Most memory-efficient models: {', '.join(memory_efficient['model'].tolist())}"
            )
        
        # Spike sparsity analysis
        if "spike_sparsity" in df.columns:
            sparse_models = df.nlargest(3, "spike_sparsity")
            recommendations.append(
                f"Highest spike sparsity models: {', '.join(sparse_models['model'].tolist())}"
            )
        
        return recommendations
    
    def plot_results(self, results: Optional[List[BenchmarkResult]] = None, save_plots: bool = True):
        """Generate visualization plots for benchmark results."""
        if results is None:
            results = self.results
            
        if not results:
            self.logger.warning("No results to plot")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "model": r.model_name,
            "dataset": r.dataset_name,
            "hardware": r.hardware_target,
            **r.metrics
        } for r in results if r.error_message is None])
        
        if df.empty:
            self.logger.warning("No successful results to plot")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        n_metrics = len([m for m in self.config.metrics if any(col.startswith(m) for col in df.columns)])
        fig_rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(fig_rows, 2, figsize=(15, 5 * fig_rows))
        
        if n_metrics == 1:
            axes = [axes]
        elif fig_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot each metric
        for metric in self.config.metrics:
            metric_cols = [col for col in df.columns if col.startswith(metric) and not col.endswith(('_mean', '_std', '_ci_lower', '_ci_upper'))]
            
            if not metric_cols:
                continue
                
            ax = axes[plot_idx] if plot_idx < len(axes) else plt.gca()
            
            for col in metric_cols:
                if col in df.columns:
                    # Box plot by model
                    sns.boxplot(data=df, x="model", y=col, ax=ax)
                    ax.set_title(f"{col.replace('_', ' ').title()} by Model")
                    ax.tick_params(axis='x', rotation=45)
                    break
            
            plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"benchmark_plots_{self.config.name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {plot_path}")
        
        plt.show()


def create_standard_benchmarks() -> List[BenchmarkConfig]:
    """Create a set of standard benchmark configurations for neuromorphic research."""
    
    benchmarks = [
        # Performance benchmark
        BenchmarkConfig(
            name="neuromorphic_performance",
            description="Comprehensive performance evaluation of spiking neural networks",
            models=["spiking_transformer", "spiking_vit", "spiking_bert"],
            datasets=["imagenet_subset", "glue_benchmark", "cifar100"],
            hardware_targets=["cpu", "cuda"],
            metrics=["accuracy", "latency", "throughput", "memory"],
            repetitions=5,
            parallel_execution=True,
            statistical_significance=True
        ),
        
        # Energy efficiency benchmark
        BenchmarkConfig(
            name="energy_efficiency",
            description="Energy efficiency comparison between ANN and SNN models",
            models=["ann_baseline", "spiking_equivalent"],
            datasets=["classification_tasks"],
            hardware_targets=["cpu", "cuda", "loihi2"],
            metrics=["accuracy", "energy", "spike_sparsity"],
            repetitions=10,
            parallel_execution=True,
            statistical_significance=True
        ),
        
        # Scalability benchmark
        BenchmarkConfig(
            name="scalability_analysis",
            description="Scalability analysis across different model sizes",
            models=["spiking_small", "spiking_medium", "spiking_large"],
            datasets=["synthetic_data"],
            hardware_targets=["cpu", "cuda"],
            metrics=["latency", "throughput", "memory"],
            repetitions=3,
            parallel_execution=True,
            statistical_significance=True
        ),
        
        # Hardware comparison benchmark
        BenchmarkConfig(
            name="hardware_comparison",
            description="Performance comparison across neuromorphic hardware platforms",
            models=["optimized_spiking_model"],
            datasets=["standard_benchmark_suite"],
            hardware_targets=["cpu", "cuda", "loihi2", "spinnaker", "brainsales"],
            metrics=["accuracy", "latency", "energy", "throughput"],
            repetitions=5,
            parallel_execution=False,  # Hardware-specific
            statistical_significance=True
        )
    ]
    
    return benchmarks


# Example usage and research validation
if __name__ == "__main__":
    # Create a comprehensive benchmark for research validation
    config = BenchmarkConfig(
        name="neuromorphic_research_validation",
        description="Validation benchmark for neuromorphic computing research",
        models=["spiking_transformer_attention", "spiking_transformer_standard", "ann_baseline"],
        datasets=["imagenet_val", "glue_cola", "glue_sst2"],
        hardware_targets=["cpu", "cuda"],
        metrics=["accuracy", "spike_sparsity", "latency", "energy", "memory"],
        repetitions=5,
        save_results=True,
        statistical_significance=True,
        confidence_level=0.95
    )
    
    benchmark = NeuromorphicBenchmark(config)
    
    # This would be used with actual models and datasets:
    # results = benchmark.run_benchmark(models_dict, datasets_dict)
    # report = benchmark.generate_report()
    # benchmark.plot_results()
    
    print("Neuromorphic benchmarking framework ready for research validation")