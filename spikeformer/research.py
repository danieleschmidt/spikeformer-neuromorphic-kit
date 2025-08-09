"""Advanced research framework for neuromorphic computing with automated experimentation."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
from datetime import datetime
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from abc import ABC, abstractmethod
import pickle
import hashlib
import time

from .models import SpikingTransformer, SpikingViT, SpikingBERT
from .benchmarking import NeuromorphicBenchmark, BenchmarkConfig, BenchmarkResult
from .conversion import ConversionPipeline, ConversionConfig
from .profiling import EnergyProfiler


@dataclass
class ResearchHypothesis:
    """Definition of a research hypothesis to be tested."""
    name: str
    description: str
    independent_variables: List[str]
    dependent_variables: List[str]
    expected_outcome: Dict[str, Any]
    significance_level: float = 0.05
    effect_size_threshold: float = 0.1
    metadata: Dict[str, Any] = None


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment."""
    name: str
    hypothesis: ResearchHypothesis
    parameter_space: Dict[str, List[Any]]
    baseline_config: Dict[str, Any]
    sample_size: int
    random_seed: int = 42
    stratification_factors: List[str] = None
    blocking_factors: List[str] = None
    power_analysis: bool = True
    
    
@dataclass
class ExperimentResult:
    """Result of a research experiment."""
    experiment_name: str
    hypothesis_name: str
    configuration: Dict[str, Any]
    measurements: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]
    effect_sizes: Dict[str, float] 
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any]
    timestamp: str
    execution_time: float
    significance: Dict[str, bool]


class StatisticalAnalyzer:
    """Statistical analysis tools for research experiments."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def power_analysis(self, 
                      effect_size: float, 
                      alpha: float = None, 
                      power: float = 0.8) -> int:
        """Calculate required sample size for given power."""
        alpha = alpha or self.alpha
        
        # Cohen's formula for sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    def compare_groups(self, 
                      group1: List[float], 
                      group2: List[float],
                      test_type: str = "auto") -> Dict[str, Any]:
        """Compare two groups using appropriate statistical test."""
        
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        results = {
            "n1": len(group1),
            "n2": len(group2),
            "mean1": np.mean(group1),
            "mean2": np.mean(group2),
            "std1": np.std(group1, ddof=1),
            "std2": np.std(group2, ddof=1),
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) + 
                             (len(group2)-1)*np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        results["cohens_d"] = cohens_d
        
        # Determine test type
        if test_type == "auto":
            # Check normality
            _, p_norm1 = stats.shapiro(group1)
            _, p_norm2 = stats.shapiro(group2)
            
            if p_norm1 > 0.05 and p_norm2 > 0.05:
                # Check equal variance
                _, p_var = stats.levene(group1, group2)
                if p_var > 0.05:
                    test_type = "t_test_equal_var"
                else:
                    test_type = "t_test_unequal_var"
            else:
                test_type = "mann_whitney"
        
        # Perform test
        if test_type == "t_test_equal_var":
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            test_name = "Independent t-test (equal variance)"
        elif test_type == "t_test_unequal_var":
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test (unequal variance)"
        elif test_type == "mann_whitney":
            t_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        results.update({
            "test_name": test_name,
            "test_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "effect_size_category": self._categorize_effect_size(abs(cohens_d))
        })
        
        # Confidence interval for mean difference
        diff = np.mean(group1) - np.mean(group2)
        se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        t_crit = stats.t.ppf(1 - self.alpha/2, len(group1) + len(group2) - 2)
        ci_lower = diff - t_crit * se_diff
        ci_upper = diff + t_crit * se_diff
        
        results["mean_diff_ci"] = (ci_lower, ci_upper)
        
        return results
    
    def anova(self, groups: List[List[float]]) -> Dict[str, Any]:
        """Perform one-way ANOVA."""
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta squared)
        group_means = [np.mean(g) for g in groups]
        group_sizes = [len(g) for g in groups]
        overall_mean = np.mean([val for group in groups for val in group])
        
        ss_between = sum(n * (mean - overall_mean)**2 for n, mean in zip(group_sizes, group_means))
        ss_total = sum((val - overall_mean)**2 for group in groups for val in group)
        eta_squared = ss_between / ss_total
        
        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "eta_squared": eta_squared,
            "effect_size_category": self._categorize_effect_size(np.sqrt(eta_squared))
        }
    
    def _categorize_effect_size(self, effect_size: float) -> str:
        """Categorize effect size according to Cohen's conventions."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"


class ExperimentDesigner:
    """Design optimal experiments using statistical principles."""
    
    def __init__(self, random_seed: int = 42):
        self.random_state = np.random.RandomState(random_seed)
        self.analyzer = StatisticalAnalyzer()
    
    def design_factorial_experiment(self,
                                  factors: Dict[str, List[Any]],
                                  interactions: List[Tuple[str, str]] = None,
                                  replications: int = 3,
                                  randomize: bool = True) -> List[Dict[str, Any]]:
        """Design a full factorial experiment."""
        
        # Generate all factor combinations
        factor_names = list(factors.keys())
        factor_levels = list(factors.values())
        
        combinations = list(itertools.product(*factor_levels))
        
        # Create experimental runs
        runs = []
        for replication in range(replications):
            for combination in combinations:
                run = dict(zip(factor_names, combination))
                run["replication"] = replication + 1
                runs.append(run)
        
        # Randomize run order
        if randomize:
            self.random_state.shuffle(runs)
            
        return runs
    
    def design_response_surface(self,
                              factors: Dict[str, Tuple[float, float]],
                              center_points: int = 5,
                              star_points: bool = True,
                              alpha: float = None) -> List[Dict[str, Any]]:
        """Design a response surface methodology experiment (Central Composite Design)."""
        
        n_factors = len(factors)
        factor_names = list(factors.keys())
        
        if alpha is None:
            alpha = n_factors ** 0.25  # Rotatable design
        
        # Coded levels for factorial points
        factorial_levels = list(itertools.product([-1, 1], repeat=n_factors))
        
        # Star points (axial points)
        star_levels = []
        if star_points:
            for i in range(n_factors):
                point_pos = [0] * n_factors
                point_neg = [0] * n_factors
                point_pos[i] = alpha
                point_neg[i] = -alpha
                star_levels.extend([point_pos, point_neg])
        
        # Center points
        center_levels = [[0] * n_factors] * center_points
        
        # Combine all points
        all_levels = factorial_levels + star_levels + center_levels
        
        # Convert coded levels to actual values
        runs = []
        for level_set in all_levels:
            run = {}
            for i, (factor_name, (low, high)) in enumerate(factors.items()):
                center = (high + low) / 2
                range_val = (high - low) / 2
                actual_value = center + level_set[i] * range_val
                run[factor_name] = actual_value
            runs.append(run)
        
        return runs
    
    def optimize_sample_size(self,
                           effect_size: float,
                           power: float = 0.8,
                           alpha: float = 0.05,
                           pilot_data: Dict[str, List[float]] = None) -> Dict[str, int]:
        """Optimize sample size based on power analysis."""
        
        results = {}
        
        # Basic power analysis
        basic_n = self.analyzer.power_analysis(effect_size, alpha, power)
        results["basic_power_analysis"] = basic_n
        
        # If pilot data available, refine estimate
        if pilot_data:
            for group_name, values in pilot_data.items():
                std_dev = np.std(values, ddof=1)
                # Adjust effect size based on observed variability
                adjusted_effect_size = effect_size / std_dev if std_dev > 0 else effect_size
                adjusted_n = self.analyzer.power_analysis(adjusted_effect_size, alpha, power)
                results[f"adjusted_for_{group_name}"] = adjusted_n
        
        return results


class ResearchFramework:
    """Complete framework for conducting neuromorphic computing research."""
    
    def __init__(self, output_dir: str = "research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.analyzer = StatisticalAnalyzer()
        self.designer = ExperimentDesigner()
        
        # Results storage
        self.experiments: Dict[str, ExperimentResult] = {}
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        
    def register_hypothesis(self, hypothesis: ResearchHypothesis):
        """Register a research hypothesis for testing."""
        self.hypotheses[hypothesis.name] = hypothesis
        self.logger.info(f"Registered hypothesis: {hypothesis.name}")
    
    def design_experiment(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Design an experiment to test a hypothesis."""
        
        # Register hypothesis
        self.register_hypothesis(config.hypothesis)
        
        # Power analysis for sample size
        if config.power_analysis:
            required_n = self.analyzer.power_analysis(
                config.hypothesis.effect_size_threshold,
                config.hypothesis.significance_level
            )
            
            if config.sample_size < required_n:
                self.logger.warning(
                    f"Sample size {config.sample_size} may be insufficient. "
                    f"Recommended: {required_n}"
                )
        
        # Design experimental runs
        if len(config.parameter_space) <= 3:
            # Full factorial for small number of factors
            runs = self.designer.design_factorial_experiment(
                config.parameter_space,
                replications=config.sample_size,
                randomize=True
            )
        else:
            # Response surface for many factors
            factor_ranges = {k: (min(v), max(v)) for k, v in config.parameter_space.items()}
            runs = self.designer.design_response_surface(factor_ranges)
            
        self.logger.info(f"Designed {len(runs)} experimental runs for {config.name}")
        return runs
    
    def run_experiment(self,
                      config: ExperimentConfig,
                      model_factory: Callable[[Dict[str, Any]], nn.Module],
                      evaluation_function: Callable[[nn.Module, Dict[str, Any]], Dict[str, float]],
                      parallel: bool = True) -> ExperimentResult:
        """Run a complete research experiment."""
        
        start_time = time.time()
        self.logger.info(f"Starting experiment: {config.name}")
        
        # Design experiment
        experimental_runs = self.design_experiment(config)
        
        # Run experiments
        all_measurements = []
        
        if parallel and len(experimental_runs) > 1:
            measurements = self._run_parallel_experiment(
                experimental_runs, model_factory, evaluation_function
            )
        else:
            measurements = self._run_sequential_experiment(
                experimental_runs, model_factory, evaluation_function
            )
        
        all_measurements.extend(measurements)
        
        # Statistical analysis
        statistical_results = self._analyze_experiment_results(
            config.hypothesis, all_measurements
        )
        
        # Create result
        result = ExperimentResult(
            experiment_name=config.name,
            hypothesis_name=config.hypothesis.name,
            configuration=asdict(config),
            measurements=self._aggregate_measurements(all_measurements),
            statistical_tests=statistical_results["tests"],
            effect_sizes=statistical_results["effect_sizes"],
            confidence_intervals=statistical_results["confidence_intervals"],
            significance=statistical_results["significance"],
            metadata={
                "num_runs": len(experimental_runs),
                "successful_runs": len([m for m in all_measurements if m.get("error") is None]),
                "failed_runs": len([m for m in all_measurements if m.get("error") is not None]),
            },
            timestamp=datetime.now().isoformat(),
            execution_time=time.time() - start_time
        )
        
        # Store result
        self.experiments[config.name] = result
        
        # Save results
        self._save_experiment_result(result)
        
        self.logger.info(f"Completed experiment: {config.name} in {result.execution_time:.2f}s")
        return result
    
    def _run_sequential_experiment(self,
                                 runs: List[Dict[str, Any]],
                                 model_factory: Callable,
                                 evaluation_function: Callable) -> List[Dict[str, Any]]:
        """Run experiments sequentially."""
        measurements = []
        
        for i, run_config in enumerate(runs):
            self.logger.info(f"Running experiment {i+1}/{len(runs)}")
            
            try:
                # Create model with current configuration
                model = model_factory(run_config)
                
                # Evaluate model
                metrics = evaluation_function(model, run_config)
                
                # Store measurement
                measurement = {**run_config, **metrics}
                measurements.append(measurement)
                
            except Exception as e:
                self.logger.error(f"Run {i+1} failed: {e}")
                measurements.append({**run_config, "error": str(e)})
                
        return measurements
    
    def _run_parallel_experiment(self,
                               runs: List[Dict[str, Any]],
                               model_factory: Callable,
                               evaluation_function: Callable) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        measurements = []
        
        def run_single_experiment(run_config):
            try:
                model = model_factory(run_config)
                metrics = evaluation_function(model, run_config)
                return {**run_config, **metrics}
            except Exception as e:
                return {**run_config, "error": str(e)}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run_single_experiment, runs))
            measurements.extend(results)
            
        return measurements
    
    def _analyze_experiment_results(self,
                                  hypothesis: ResearchHypothesis,
                                  measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results statistically."""
        
        # Filter successful measurements
        valid_measurements = [m for m in measurements if m.get("error") is None]
        
        if not valid_measurements:
            return {
                "tests": {},
                "effect_sizes": {},
                "confidence_intervals": {},
                "significance": {}
            }
        
        results = {
            "tests": {},
            "effect_sizes": {},
            "confidence_intervals": {},
            "significance": {}
        }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(valid_measurements)
        
        # Analyze each dependent variable
        for dependent_var in hypothesis.dependent_variables:
            if dependent_var not in df.columns:
                continue
                
            # Group by independent variables
            for independent_var in hypothesis.independent_variables:
                if independent_var not in df.columns:
                    continue
                
                groups = df.groupby(independent_var)[dependent_var].apply(list).tolist()
                
                if len(groups) >= 2:
                    # Compare groups
                    if len(groups) == 2:
                        comparison = self.analyzer.compare_groups(groups[0], groups[1])
                    else:
                        comparison = self.analyzer.anova(groups)
                    
                    key = f"{dependent_var}_vs_{independent_var}"
                    results["tests"][key] = comparison
                    results["effect_sizes"][key] = comparison.get("cohens_d", comparison.get("eta_squared", 0))
                    results["confidence_intervals"][key] = comparison.get("mean_diff_ci", (0, 0))
                    results["significance"][key] = comparison["significant"]
        
        return results
    
    def _aggregate_measurements(self, measurements: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate measurements across all runs."""
        valid_measurements = [m for m in measurements if m.get("error") is None]
        
        if not valid_measurements:
            return {}
        
        df = pd.DataFrame(valid_measurements)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        aggregated = {}
        for col in numeric_columns:
            aggregated[f"{col}_mean"] = df[col].mean()
            aggregated[f"{col}_std"] = df[col].std()
            aggregated[f"{col}_min"] = df[col].min()
            aggregated[f"{col}_max"] = df[col].max()
            
        return aggregated
    
    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_{result.experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert result to JSON-serializable format
        result_dict = asdict(result)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Saved experiment result to {filepath}")
    
    def generate_research_report(self, experiment_names: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        
        if experiment_names is None:
            experiments = list(self.experiments.values())
        else:
            experiments = [self.experiments[name] for name in experiment_names if name in self.experiments]
        
        if not experiments:
            return {"error": "No experiments found"}
        
        report = {
            "summary": {
                "total_experiments": len(experiments),
                "total_hypotheses": len(set(exp.hypothesis_name for exp in experiments)),
                "experiment_names": [exp.experiment_name for exp in experiments],
                "date_range": (
                    min(exp.timestamp for exp in experiments),
                    max(exp.timestamp for exp in experiments)
                )
            },
            "hypothesis_testing": {},
            "effect_sizes": {},
            "recommendations": [],
            "meta_analysis": {}
        }
        
        # Analyze each hypothesis
        hypothesis_groups = {}
        for exp in experiments:
            if exp.hypothesis_name not in hypothesis_groups:
                hypothesis_groups[exp.hypothesis_name] = []
            hypothesis_groups[exp.hypothesis_name].append(exp)
        
        for hypothesis_name, hypothesis_experiments in hypothesis_groups.items():
            hypothesis = self.hypotheses.get(hypothesis_name)
            
            # Combine results across experiments
            all_significance = {}
            all_effect_sizes = {}
            
            for exp in hypothesis_experiments:
                all_significance.update(exp.significance)
                all_effect_sizes.update(exp.effect_sizes)
            
            # Meta-analysis (simple combination for now)
            significant_results = sum(all_significance.values())
            total_tests = len(all_significance)
            
            report["hypothesis_testing"][hypothesis_name] = {
                "hypothesis": asdict(hypothesis) if hypothesis else {},
                "experiments_count": len(hypothesis_experiments),
                "significant_results": significant_results,
                "total_tests": total_tests,
                "proportion_significant": significant_results / total_tests if total_tests > 0 else 0,
                "mean_effect_size": np.mean(list(all_effect_sizes.values())) if all_effect_sizes else 0,
                "evidence_strength": self._assess_evidence_strength(
                    significant_results / total_tests if total_tests > 0 else 0,
                    np.mean(list(all_effect_sizes.values())) if all_effect_sizes else 0
                )
            }
        
        # Generate recommendations
        report["recommendations"] = self._generate_research_recommendations(experiments)
        
        return report
    
    def _assess_evidence_strength(self, proportion_significant: float, mean_effect_size: float) -> str:
        """Assess the strength of evidence for a hypothesis."""
        
        if proportion_significant >= 0.8 and abs(mean_effect_size) >= 0.5:
            return "Strong evidence"
        elif proportion_significant >= 0.6 and abs(mean_effect_size) >= 0.3:
            return "Moderate evidence"
        elif proportion_significant >= 0.4 or abs(mean_effect_size) >= 0.2:
            return "Weak evidence"
        else:
            return "Insufficient evidence"
    
    def _generate_research_recommendations(self, experiments: List[ExperimentResult]) -> List[str]:
        """Generate research recommendations based on experiment results."""
        
        recommendations = []
        
        # Analyze success rates
        total_tests = sum(len(exp.significance) for exp in experiments)
        significant_tests = sum(sum(exp.significance.values()) for exp in experiments)
        
        if total_tests > 0:
            success_rate = significant_tests / total_tests
            
            if success_rate < 0.3:
                recommendations.append(
                    "Low significance rate suggests need for larger sample sizes or different experimental design"
                )
            elif success_rate > 0.8:
                recommendations.append(
                    "High significance rate indicates robust findings suitable for publication"
                )
        
        # Analyze effect sizes
        all_effect_sizes = []
        for exp in experiments:
            all_effect_sizes.extend(exp.effect_sizes.values())
        
        if all_effect_sizes:
            mean_effect = np.mean([abs(es) for es in all_effect_sizes])
            if mean_effect < 0.2:
                recommendations.append(
                    "Small effect sizes suggest practical significance may be limited"
                )
            elif mean_effect > 0.8:
                recommendations.append(
                    "Large effect sizes indicate practically significant findings"
                )
        
        # Analyze execution efficiency
        mean_execution_time = np.mean([exp.execution_time for exp in experiments])
        if mean_execution_time > 3600:  # 1 hour
            recommendations.append(
                "Long execution times suggest need for computational optimization"
            )
        
        return recommendations
    
    def plot_experiment_results(self, experiment_name: str, save_plots: bool = True):
        """Generate comprehensive plots for experiment results."""
        
        if experiment_name not in self.experiments:
            self.logger.error(f"Experiment {experiment_name} not found")
            return
        
        experiment = self.experiments[experiment_name]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Results for Experiment: {experiment_name}", fontsize=16)
        
        # Plot 1: Effect sizes
        if experiment.effect_sizes:
            ax1 = axes[0, 0]
            effects = list(experiment.effect_sizes.values())
            labels = list(experiment.effect_sizes.keys())
            
            colors = ['green' if abs(e) >= 0.5 else 'orange' if abs(e) >= 0.2 else 'red' for e in effects]
            bars = ax1.barh(range(len(effects)), effects, color=colors)
            ax1.set_yticks(range(len(labels)))
            ax1.set_yticklabels([l.replace('_', ' ').title() for l in labels])
            ax1.set_xlabel("Effect Size (Cohen's d)")
            ax1.set_title("Effect Sizes by Comparison")
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
            ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium')
            ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large')
            ax1.legend()
        
        # Plot 2: Statistical significance
        if experiment.significance:
            ax2 = axes[0, 1]
            sig_values = list(experiment.significance.values())
            sig_labels = list(experiment.significance.keys())
            
            significant_count = sum(sig_values)
            non_significant_count = len(sig_values) - significant_count
            
            ax2.pie([significant_count, non_significant_count], 
                   labels=['Significant', 'Not Significant'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
            ax2.set_title("Statistical Significance")
        
        # Plot 3: Measurement distributions (if available in metadata)
        ax3 = axes[1, 0]
        measurements = experiment.measurements
        numeric_measurements = {k: v for k, v in measurements.items() 
                              if isinstance(v, (int, float)) and 'mean' in k}
        
        if numeric_measurements:
            keys = list(numeric_measurements.keys())[:5]  # Top 5 measurements
            values = [numeric_measurements[k] for k in keys]
            
            ax3.bar(range(len(keys)), values)
            ax3.set_xticks(range(len(keys)))
            ax3.set_xticklabels([k.replace('_', ' ').title() for k in keys], rotation=45)
            ax3.set_title("Key Measurements")
        else:
            ax3.text(0.5, 0.5, "No numeric measurements available", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Key Measurements")
        
        # Plot 4: Confidence intervals
        ax4 = axes[1, 1]
        if experiment.confidence_intervals:
            ci_data = experiment.confidence_intervals
            ci_labels = list(ci_data.keys())
            ci_values = list(ci_data.values())
            
            means = [np.mean(ci) for ci in ci_values]
            errors = [[np.mean(ci) - ci[0], ci[1] - np.mean(ci)] for ci in ci_values]
            
            ax4.errorbar(range(len(ci_labels)), means, 
                        yerr=np.array(errors).T, fmt='o', capsize=5)
            ax4.set_xticks(range(len(ci_labels)))
            ax4.set_xticklabels([l.replace('_', ' ').title() for l in ci_labels], rotation=45)
            ax4.set_title("Confidence Intervals")
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No confidence intervals available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Confidence Intervals")
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"experiment_plot_{experiment_name}_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plots saved to {plot_path}")
        
        plt.show()


# Example usage for neuromorphic research
def create_research_examples():
    """Create example research hypotheses and experiments."""
    
    # Research Framework
    framework = ResearchFramework("neuromorphic_research")
    
    # Example Hypothesis 1: Spike sparsity vs Energy efficiency
    sparsity_hypothesis = ResearchHypothesis(
        name="sparsity_energy_correlation",
        description="Higher spike sparsity leads to better energy efficiency in neuromorphic models",
        independent_variables=["spike_sparsity"],
        dependent_variables=["energy_per_inference", "inference_accuracy"],
        expected_outcome={
            "energy_per_inference": "decreases",
            "correlation_direction": "negative",
            "effect_size": "medium_to_large"
        },
        significance_level=0.05,
        effect_size_threshold=0.3
    )
    
    # Example Hypothesis 2: Timesteps vs Accuracy trade-off
    timestep_hypothesis = ResearchHypothesis(
        name="timestep_accuracy_tradeoff",
        description="Increasing timesteps improves accuracy but with diminishing returns",
        independent_variables=["timesteps"],
        dependent_variables=["accuracy", "computational_cost"],
        expected_outcome={
            "accuracy": "logarithmic_increase",
            "computational_cost": "linear_increase"
        },
        significance_level=0.05,
        effect_size_threshold=0.4
    )
    
    # Experiment Configuration
    sparsity_config = ExperimentConfig(
        name="spike_sparsity_energy_experiment",
        hypothesis=sparsity_hypothesis,
        parameter_space={
            "threshold": [0.5, 1.0, 1.5, 2.0],
            "timesteps": [16, 32, 64],
            "neuron_type": ["LIF", "ALIF", "PLIF"]
        },
        baseline_config={"threshold": 1.0, "timesteps": 32, "neuron_type": "LIF"},
        sample_size=5,
        power_analysis=True
    )
    
    return framework, sparsity_config


if __name__ == "__main__":
    # Create research framework and example
    framework, config = create_research_examples()
    
    print(f"Research framework initialized with output directory: {framework.output_dir}")
    print(f"Registered hypothesis: {config.hypothesis.name}")
    print(f"Parameter space: {config.parameter_space}")
    
    # Example of how to use:
    # runs = framework.design_experiment(config)
    # result = framework.run_experiment(config, model_factory, evaluation_function)
    # report = framework.generate_research_report()
    # framework.plot_experiment_results(config.name)