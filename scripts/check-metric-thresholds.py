#!/usr/bin/env python3
"""
Metric Threshold Checker for Spikeformer
Checks collected metrics against configured thresholds and generates alerts
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class ThresholdChecker:
    """Check metrics against configured thresholds"""
    
    def __init__(self, metrics_data: Dict[str, Any], config_data: Dict[str, Any]):
        """Initialize with metrics and configuration data"""
        self.metrics = metrics_data
        self.config = config_data
        self.violations = []
    
    def check_all_thresholds(self) -> List[Dict[str, Any]]:
        """Check all metrics against their thresholds"""
        self.violations = []
        
        # Check code quality thresholds
        self._check_code_quality_thresholds()
        
        # Check performance thresholds
        self._check_performance_thresholds()
        
        # Check security thresholds
        self._check_security_thresholds()
        
        # Check reliability thresholds
        self._check_reliability_thresholds()
        
        # Check neuromorphic-specific thresholds
        self._check_neuromorphic_thresholds()
        
        return self.violations
    
    def _check_code_quality_thresholds(self):
        """Check code quality metrics against thresholds"""
        cq_metrics = self.metrics.get("code_quality", {})
        cq_thresholds = self.config.get("metrics", {}).get("code_quality", {})
        
        # Coverage threshold
        coverage = cq_metrics.get("coverage")
        coverage_target = cq_thresholds.get("coverage_target")
        if coverage is not None and coverage_target is not None:
            if coverage < coverage_target:
                self.violations.append({
                    "category": "code_quality",
                    "metric": "test_coverage",
                    "current": coverage,
                    "threshold": coverage_target,
                    "threshold_type": "minimum",
                    "severity": "medium",
                    "message": f"Test coverage ({coverage:.1f}%) is below target ({coverage_target}%)"
                })
        
        # Complexity threshold
        complexity = cq_metrics.get("complexity")
        complexity_threshold = cq_thresholds.get("complexity_threshold")
        if complexity is not None and complexity_threshold is not None:
            if complexity > complexity_threshold:
                self.violations.append({
                    "category": "code_quality", 
                    "metric": "code_complexity",
                    "current": complexity,
                    "threshold": complexity_threshold,
                    "threshold_type": "maximum",
                    "severity": "medium",
                    "message": f"Code complexity ({complexity:.1f}) exceeds threshold ({complexity_threshold})"
                })
        
        # Maintainability threshold
        maintainability = cq_metrics.get("maintainability")
        maintainability_min = cq_thresholds.get("maintainability_index_min")
        if maintainability is not None and maintainability_min is not None:
            if maintainability < maintainability_min:
                self.violations.append({
                    "category": "code_quality",
                    "metric": "maintainability_index", 
                    "current": maintainability,
                    "threshold": maintainability_min,
                    "threshold_type": "minimum",
                    "severity": "low",
                    "message": f"Maintainability index ({maintainability:.1f}) is below minimum ({maintainability_min})"
                })
        
        # Technical debt threshold
        technical_debt = cq_metrics.get("technical_debt")
        debt_max = cq_thresholds.get("technical_debt_ratio_max")
        if technical_debt is not None and debt_max is not None:
            if technical_debt > debt_max:
                self.violations.append({
                    "category": "code_quality",
                    "metric": "technical_debt_ratio",
                    "current": technical_debt,
                    "threshold": debt_max,
                    "threshold_type": "maximum",
                    "severity": "medium",
                    "message": f"Technical debt ratio ({technical_debt:.3f}) exceeds maximum ({debt_max})"
                })
        
        # Duplication threshold
        duplication = cq_metrics.get("duplication")
        duplication_max = cq_thresholds.get("duplication_percentage_max")
        if duplication is not None and duplication_max is not None:
            if duplication > duplication_max:
                self.violations.append({
                    "category": "code_quality",
                    "metric": "code_duplication",
                    "current": duplication,
                    "threshold": duplication_max, 
                    "threshold_type": "maximum",
                    "severity": "low",
                    "message": f"Code duplication ({duplication:.1f}%) exceeds maximum ({duplication_max}%)"
                })
    
    def _check_performance_thresholds(self):
        """Check performance metrics against thresholds"""
        perf_metrics = self.metrics.get("performance", {})
        perf_thresholds = self.config.get("metrics", {}).get("performance", {})
        
        # Build time threshold
        build_time = perf_metrics.get("build_time")
        build_time_target = perf_thresholds.get("build_time_target_minutes", 0) * 60  # Convert to seconds
        if build_time is not None and build_time_target > 0:
            if build_time > build_time_target:
                self.violations.append({
                    "category": "performance",
                    "metric": "build_time",
                    "current": build_time,
                    "threshold": build_time_target,
                    "threshold_type": "maximum",
                    "severity": "medium",
                    "message": f"Build time ({build_time:.1f}s) exceeds target ({build_time_target}s)"
                })
        
        # Test runtime threshold
        test_time = perf_metrics.get("test_runtime")
        test_time_target = perf_thresholds.get("test_suite_runtime_target_minutes", 0) * 60
        if test_time is not None and test_time_target > 0:
            if test_time > test_time_target:
                self.violations.append({
                    "category": "performance",
                    "metric": "test_runtime",
                    "current": test_time,
                    "threshold": test_time_target,
                    "threshold_type": "maximum",
                    "severity": "medium",
                    "message": f"Test runtime ({test_time:.1f}s) exceeds target ({test_time_target}s)"
                })
        
        # Model conversion time threshold
        conversion_time = perf_metrics.get("model_conversion_time")
        conversion_target = perf_thresholds.get("model_conversion_time_target_seconds")
        if conversion_time is not None and conversion_target is not None:
            if conversion_time > conversion_target:
                self.violations.append({
                    "category": "performance",
                    "metric": "model_conversion_time",
                    "current": conversion_time,
                    "threshold": conversion_target,
                    "threshold_type": "maximum",
                    "severity": "medium",
                    "message": f"Model conversion time ({conversion_time:.1f}s) exceeds target ({conversion_target}s)"
                })
        
        # Energy efficiency threshold
        energy_efficiency = perf_metrics.get("energy_efficiency", {})
        if isinstance(energy_efficiency, dict):
            efficiency_improvement = energy_efficiency.get("efficiency_improvement")
            efficiency_min = perf_thresholds.get("energy_efficiency_improvement_min")
            if efficiency_improvement is not None and efficiency_min is not None:
                if efficiency_improvement < efficiency_min:
                    self.violations.append({
                        "category": "performance",
                        "metric": "energy_efficiency_improvement",
                        "current": efficiency_improvement,
                        "threshold": efficiency_min,
                        "threshold_type": "minimum",
                        "severity": "high",
                        "message": f"Energy efficiency improvement ({efficiency_improvement:.1f}x) is below minimum ({efficiency_min}x)"
                    })
        
        # Accuracy retention threshold
        accuracy_metrics = perf_metrics.get("accuracy_retention", {})
        if isinstance(accuracy_metrics, dict):
            avg_accuracy = accuracy_metrics.get("average_retention")
            accuracy_min = perf_thresholds.get("accuracy_retention_min")
            if avg_accuracy is not None and accuracy_min is not None:
                if avg_accuracy < accuracy_min:
                    self.violations.append({
                        "category": "performance",
                        "metric": "accuracy_retention",
                        "current": avg_accuracy,
                        "threshold": accuracy_min,
                        "threshold_type": "minimum",
                        "severity": "high",
                        "message": f"Accuracy retention ({avg_accuracy:.1%}) is below minimum ({accuracy_min:.1%})"
                    })
    
    def _check_security_thresholds(self):
        """Check security metrics against thresholds"""
        sec_metrics = self.metrics.get("security", {})
        sec_thresholds = self.config.get("metrics", {}).get("security", {})
        
        # Vulnerability thresholds
        vulnerabilities = sec_metrics.get("vulnerabilities", {})
        
        critical_vulns = vulnerabilities.get("critical", 0)
        critical_max = sec_thresholds.get("critical_vulnerabilities_max", 0)
        if critical_vulns > critical_max:
            self.violations.append({
                "category": "security",
                "metric": "critical_vulnerabilities",
                "current": critical_vulns,
                "threshold": critical_max,
                "threshold_type": "maximum",
                "severity": "critical",
                "message": f"Critical vulnerabilities ({critical_vulns}) exceed maximum allowed ({critical_max})"
            })
        
        high_vulns = vulnerabilities.get("high", 0)
        high_max = sec_thresholds.get("high_vulnerabilities_max", 5)
        if high_vulns > high_max:
            self.violations.append({
                "category": "security",
                "metric": "high_vulnerabilities",
                "current": high_vulns,
                "threshold": high_max,
                "threshold_type": "maximum",
                "severity": "high",
                "message": f"High vulnerabilities ({high_vulns}) exceed maximum allowed ({high_max})"
            })
        
        medium_vulns = vulnerabilities.get("medium", 0)
        medium_max = sec_thresholds.get("medium_vulnerabilities_max", 20)
        if medium_vulns > medium_max:
            self.violations.append({
                "category": "security",
                "metric": "medium_vulnerabilities",
                "current": medium_vulns,
                "threshold": medium_max,
                "threshold_type": "maximum",
                "severity": "medium",
                "message": f"Medium vulnerabilities ({medium_vulns}) exceed maximum allowed ({medium_max})"
            })
        
        # Secrets detection
        secrets_scan = sec_metrics.get("secrets_scan", {})
        secrets_detected = secrets_scan.get("secrets_detected", False)
        secrets_max = sec_thresholds.get("secrets_detected_max", 0)
        if secrets_detected and secrets_max == 0:
            self.violations.append({
                "category": "security",
                "metric": "secrets_detected",
                "current": 1,
                "threshold": secrets_max,
                "threshold_type": "maximum",
                "severity": "critical",
                "message": "Secrets detected in codebase - immediate action required"
            })
        
        # License compliance
        license_compliance = sec_metrics.get("license_compliance", {})
        if not license_compliance.get("compliant", True) and sec_thresholds.get("license_compliance_required", True):
            self.violations.append({
                "category": "security",
                "metric": "license_compliance",
                "current": "non-compliant",
                "threshold": "compliant",
                "threshold_type": "required",
                "severity": "medium",
                "message": "License compliance violations detected"
            })
    
    def _check_reliability_thresholds(self):
        """Check reliability metrics against thresholds"""
        # This would be implemented when reliability metrics are collected
        # For now, it's a placeholder for future reliability monitoring
        pass
    
    def _check_neuromorphic_thresholds(self):
        """Check neuromorphic-specific metrics against thresholds"""
        neuro_metrics = self.metrics.get("neuromorphic", {})
        neuro_thresholds = self.config.get("metrics", {}).get("neuromorphic_specific", {})
        
        # Model performance thresholds
        model_performance = neuro_metrics.get("model_performance", {})
        model_thresholds = neuro_thresholds.get("model_performance", {})
        
        # Conversion success rate
        conversion_rate = model_performance.get("conversion_success_rate")
        conversion_min = model_thresholds.get("conversion_success_rate_min")
        if conversion_rate is not None and conversion_min is not None:
            if conversion_rate < conversion_min:
                self.violations.append({
                    "category": "neuromorphic",
                    "metric": "model_conversion_success_rate",
                    "current": conversion_rate,
                    "threshold": conversion_min,
                    "threshold_type": "minimum",
                    "severity": "high",
                    "message": f"Model conversion success rate ({conversion_rate:.1%}) is below minimum ({conversion_min:.1%})"
                })
        
        # Hardware utilization
        hardware_util = model_performance.get("hardware_utilization_target")
        util_target = model_thresholds.get("hardware_utilization_target")
        if hardware_util is not None and util_target is not None:
            # This would typically check if utilization is too low (inefficient) or too high (overloaded)
            if hardware_util < util_target * 0.5:  # Less than 50% of target is concerning
                self.violations.append({
                    "category": "neuromorphic",
                    "metric": "hardware_utilization",
                    "current": hardware_util,
                    "threshold": util_target * 0.5,
                    "threshold_type": "minimum",
                    "severity": "medium",
                    "message": f"Hardware utilization ({hardware_util:.1%}) is significantly below target"
                })
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of violations by severity"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for violation in self.violations:
            severity = violation.get("severity", "medium")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def has_violations(self) -> bool:
        """Check if there are any violations"""
        return len(self.violations) > 0
    
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations"""
        return any(v.get("severity") == "critical" for v in self.violations)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of threshold check results"""
        severity_counts = self.get_severity_counts()
        
        return {
            "total_violations": len(self.violations),
            "severity_breakdown": severity_counts,
            "has_critical": self.has_critical_violations(),
            "categories_affected": list(set(v.get("category") for v in self.violations)),
            "timestamp": self.metrics.get("metadata", {}).get("collection_time"),
            "violations": self.violations
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Check Spikeformer metrics against thresholds")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to collected metrics JSON file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to metrics configuration JSON file"
    )
    parser.add_argument(
        "--output",
        help="Output file for violations (JSON format)"
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with non-zero code if critical violations found"
    )
    parser.add_argument(
        "--fail-on-any",
        action="store_true", 
        help="Exit with non-zero code if any violations found"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load metrics data
    try:
        with open(args.metrics) as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Metrics file not found: {args.metrics}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metrics file: {e}")
        sys.exit(1)
    
    # Load configuration
    try:
        with open(args.config) as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)
    
    # Check thresholds
    checker = ThresholdChecker(metrics_data, config_data)
    violations = checker.check_all_thresholds()
    summary = checker.generate_summary()
    
    # Output results
    if args.verbose or not args.output:
        print(f"Threshold Check Results:")
        print(f"  Total violations: {summary['total_violations']}")
        print(f"  Critical: {summary['severity_breakdown']['critical']}")
        print(f"  High: {summary['severity_breakdown']['high']}")
        print(f"  Medium: {summary['severity_breakdown']['medium']}")
        print(f"  Low: {summary['severity_breakdown']['low']}")
        
        if violations and args.verbose:
            print("\nViolation Details:")
            for violation in violations:
                print(f"  [{violation['severity'].upper()}] {violation['message']}")
    
    # Save detailed results if output file specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if args.verbose:
            print(f"\nDetailed results saved to: {output_path}")
    
    # Set GitHub Actions output
    print(f"##[set-output name=violations;]{summary['has_critical'] or len(violations) > 0}")
    print(f"##[set-output name=critical_violations;]{summary['has_critical']}")
    print(f"##[set-output name=total_violations;]{len(violations)}")
    
    # Exit with appropriate code
    exit_code = 0
    if args.fail_on_critical and checker.has_critical_violations():
        exit_code = 1
    elif args.fail_on_any and checker.has_violations():
        exit_code = 1
    
    if exit_code != 0:
        print(f"\nExiting with code {exit_code} due to threshold violations")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()