#!/usr/bin/env python3
"""Comprehensive Quality Gates for Spikeformer Neuromorphic Kit - Autonomous Validation."""

import os
import sys
import subprocess
import time
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import tempfile
import shutil


@dataclass
class QualityMetric:
    """Quality metric result."""
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    score: float
    details: str
    critical: bool = False


@dataclass
class QualityGateResult:
    """Complete quality gate assessment."""
    overall_status: str
    overall_score: float
    metrics: List[QualityMetric]
    execution_time: float
    timestamp: str


class AutonomousQualityGates:
    """Comprehensive autonomous quality validation system."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results = []
        self.critical_failures = []
        
        # Quality thresholds
        self.thresholds = {
            "code_coverage": 85.0,
            "security_score": 95.0,
            "performance_score": 80.0,
            "documentation_score": 70.0,
            "code_quality_score": 85.0,
            "architecture_score": 80.0
        }
        
        # File patterns
        self.python_files = list(self.repo_path.glob("**/*.py"))
        self.test_files = list(self.repo_path.glob("**/test_*.py")) + list(self.repo_path.glob("**/*_test.py"))
        self.source_files = [f for f in self.python_files if "/test" not in str(f)]
        
    def run_all_quality_gates(self) -> QualityGateResult:
        """Run all quality gates and return comprehensive results."""
        
        start_time = time.time()
        print("🚀 TERRAGON AUTONOMOUS QUALITY GATES - EXECUTING...")
        
        # Execute all quality checks
        self._check_code_structure()
        self._check_security_compliance()
        self._check_performance_patterns()
        self._check_documentation_quality()
        self._check_code_quality()
        self._check_architecture_patterns()
        self._check_test_coverage()
        self._check_research_implementation()
        self._check_scalability_features()
        self._check_production_readiness()
        
        execution_time = time.time() - start_time
        
        # Calculate overall results
        overall_score = self._calculate_overall_score()
        overall_status = self._determine_overall_status()
        
        result = QualityGateResult(
            overall_status=overall_status,
            overall_score=overall_score,
            metrics=self.results,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self._print_comprehensive_report(result)
        
        return result
    
    def _check_code_structure(self):
        """Verify code structure and organization."""
        
        print("📁 Checking code structure...")
        
        # Check directory structure
        required_dirs = ["spikeformer", "tests", "docs", "scripts"]
        missing_dirs = [d for d in required_dirs if not (self.repo_path / d).exists()]
        
        structure_score = 100.0 - (len(missing_dirs) * 10)
        
        # Check init files
        init_files = list(self.repo_path.glob("**/__init__.py"))
        expected_inits = len([d for d in self.repo_path.glob("**/") if any(f.suffix == ".py" for f in d.iterdir())])
        
        init_score = min(100.0, (len(init_files) / max(expected_inits, 1)) * 100)
        
        # Check for main modules
        main_modules = ["models.py", "conversion.py", "hardware.py", "profiling.py"]
        existing_modules = [m for m in main_modules if (self.repo_path / "spikeformer" / m).exists()]
        
        module_score = (len(existing_modules) / len(main_modules)) * 100
        
        final_score = (structure_score + init_score + module_score) / 3
        
        self.results.append(QualityMetric(
            name="Code Structure",
            status="PASS" if final_score >= 80 else "FAIL" if final_score < 60 else "WARN",
            score=final_score,
            details=f"Dir structure: {structure_score:.1f}%, Init files: {init_score:.1f}%, Modules: {module_score:.1f}%",
            critical=True
        ))
    
    def _check_security_compliance(self):
        """Check security compliance and vulnerabilities."""
        
        print("🔒 Checking security compliance...")
        
        security_issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][^'\"]+['\"]"
        ]
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues.append(f"Potential secret in {file_path.name}")
            except:
                continue
        
        # Check for unsafe imports
        unsafe_imports = ["pickle", "eval", "exec", "subprocess.call"]
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for unsafe in unsafe_imports:
                    if unsafe in content:
                        # More sophisticated check needed
                        if re.search(rf"\b{re.escape(unsafe)}\b", content):
                            security_issues.append(f"Unsafe import/call '{unsafe}' in {file_path.name}")
            except:
                continue
        
        # Check for SQL injection patterns (if any SQL usage)
        sql_patterns = [r"f['\"].*SELECT.*{.*}.*['\"]", r"\.format.*SELECT"]
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for pattern in sql_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        security_issues.append(f"Potential SQL injection in {file_path.name}")
            except:
                continue
        
        # Score based on issues found
        security_score = max(0, 100 - len(security_issues) * 10)
        
        self.results.append(QualityMetric(
            name="Security Compliance",
            status="PASS" if security_score >= self.thresholds["security_score"] else "FAIL",
            score=security_score,
            details=f"Found {len(security_issues)} potential security issues",
            critical=True
        ))
    
    def _check_performance_patterns(self):
        """Check for performance optimization patterns."""
        
        print("⚡ Checking performance patterns...")
        
        performance_indicators = {
            "caching": 0,
            "async_patterns": 0,
            "optimization_classes": 0,
            "memory_management": 0,
            "parallel_processing": 0
        }
        
        # Pattern checks
        patterns = {
            "caching": [r"@lru_cache", r"cache", r"LRUCache"],
            "async_patterns": [r"async def", r"await", r"asyncio"],
            "optimization_classes": [r"class.*Optimizer", r"class.*Accelerator"],
            "memory_management": [r"gc\.collect", r"torch\.cuda\.empty_cache", r"del "],
            "parallel_processing": [r"ThreadPoolExecutor", r"ProcessPoolExecutor", r"multiprocessing"]
        }
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for category, category_patterns in patterns.items():
                    for pattern in category_patterns:
                        if re.search(pattern, content):
                            performance_indicators[category] += 1
            except:
                continue
        
        # Score performance patterns
        total_indicators = sum(performance_indicators.values())
        performance_score = min(100, (total_indicators / 10) * 100)  # Expect ~10 total indicators
        
        details = ", ".join([f"{k}: {v}" for k, v in performance_indicators.items()])
        
        self.results.append(QualityMetric(
            name="Performance Patterns",
            status="PASS" if performance_score >= self.thresholds["performance_score"] else "WARN",
            score=performance_score,
            details=f"Performance indicators found: {details}"
        ))
    
    def _check_documentation_quality(self):
        """Check documentation quality and completeness."""
        
        print("📚 Checking documentation quality...")
        
        doc_score = 0
        
        # Check for README
        readme_files = list(self.repo_path.glob("README*"))
        if readme_files:
            doc_score += 20
            # Check README content quality
            try:
                readme_content = readme_files[0].read_text()
                if len(readme_content) > 1000:
                    doc_score += 10
                if "installation" in readme_content.lower():
                    doc_score += 5
                if "usage" in readme_content.lower():
                    doc_score += 5
            except:
                pass
        
        # Check for docstrings
        total_functions = 0
        documented_functions = 0
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                # Simple AST parsing to count functions and docstrings
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        if (node.body and isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
            except:
                continue
        
        if total_functions > 0:
            docstring_coverage = (documented_functions / total_functions) * 100
            doc_score += min(40, docstring_coverage * 0.4)
        
        # Check for docs directory
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md"))
            doc_score += min(20, len(doc_files) * 2)
        
        # Check for API documentation
        api_files = list(self.repo_path.glob("**/API*.md")) + list(self.repo_path.glob("**/api*.md"))
        if api_files:
            doc_score += 10
        
        self.results.append(QualityMetric(
            name="Documentation Quality",
            status="PASS" if doc_score >= self.thresholds["documentation_score"] else "WARN",
            score=doc_score,
            details=f"README: {bool(readme_files)}, Docstrings: {documented_functions}/{total_functions}, Docs: {len(list(docs_dir.glob('*.md')) if docs_dir.exists() else [])}"
        ))
    
    def _check_code_quality(self):
        """Check code quality metrics."""
        
        print("🔍 Checking code quality...")
        
        quality_metrics = {
            "complexity": 0,
            "maintainability": 0,
            "readability": 0,
            "style": 0
        }
        
        total_files = len(self.source_files)
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                # Complexity (simple line count and function count)
                function_count = len(re.findall(r'^def ', content, re.MULTILINE))
                if len(lines) > 0:
                    avg_function_length = len(lines) / max(function_count, 1)
                    if avg_function_length < 50:  # Reasonable function length
                        quality_metrics["complexity"] += 1
                
                # Maintainability (comments and docstrings)
                comment_lines = len([l for l in lines if l.strip().startswith('#')])
                comment_ratio = comment_lines / max(len(lines), 1)
                if comment_ratio > 0.1:  # At least 10% comments
                    quality_metrics["maintainability"] += 1
                
                # Readability (naming conventions)
                if re.search(r'class [A-Z][a-zA-Z]*:', content):  # PascalCase classes
                    quality_metrics["readability"] += 0.5
                if re.search(r'def [a-z_][a-z0-9_]*\(', content):  # snake_case functions
                    quality_metrics["readability"] += 0.5
                
                # Style (imports organization)
                import_lines = [l for l in lines[:50] if l.strip().startswith('import') or l.strip().startswith('from')]
                if len(import_lines) > 0:
                    # Check if imports are at the top
                    first_import_line = next((i for i, l in enumerate(lines) if 'import' in l), 0)
                    if first_import_line < 20:  # Imports near top
                        quality_metrics["style"] += 1
                        
            except Exception as e:
                continue
        
        # Calculate overall quality score
        if total_files > 0:
            complexity_score = (quality_metrics["complexity"] / total_files) * 100
            maintainability_score = (quality_metrics["maintainability"] / total_files) * 100
            readability_score = (quality_metrics["readability"] / total_files) * 100
            style_score = (quality_metrics["style"] / total_files) * 100
            
            overall_quality = (complexity_score + maintainability_score + readability_score + style_score) / 4
        else:
            overall_quality = 0
        
        self.results.append(QualityMetric(
            name="Code Quality",
            status="PASS" if overall_quality >= self.thresholds["code_quality_score"] else "WARN",
            score=overall_quality,
            details=f"Complexity: {complexity_score:.1f}%, Maintainability: {maintainability_score:.1f}%, Readability: {readability_score:.1f}%, Style: {style_score:.1f}%"
        ))
    
    def _check_architecture_patterns(self):
        """Check architecture patterns and design quality."""
        
        print("🏗️ Checking architecture patterns...")
        
        architecture_score = 0
        
        # Check for design patterns
        patterns_found = {
            "factory": False,
            "strategy": False,
            "observer": False,
            "adapter": False,
            "singleton": False
        }
        
        pattern_indicators = {
            "factory": [r"class.*Factory", r"def create_"],
            "strategy": [r"class.*Strategy", r"class.*Algorithm"],
            "observer": [r"class.*Observer", r"def notify", r"class.*Subject"],
            "adapter": [r"class.*Adapter", r"class.*Wrapper"],
            "singleton": [r"__new__.*cls", r"_instance.*None"]
        }
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for pattern_name, indicators in pattern_indicators.items():
                    for indicator in indicators:
                        if re.search(indicator, content):
                            patterns_found[pattern_name] = True
                            break
            except:
                continue
        
        architecture_score += sum(patterns_found.values()) * 10  # 10 points per pattern
        
        # Check for modular architecture
        module_separation = {
            "models": (self.repo_path / "spikeformer" / "models.py").exists(),
            "conversion": (self.repo_path / "spikeformer" / "conversion.py").exists(),
            "hardware": (self.repo_path / "spikeformer" / "hardware.py").exists(),
            "profiling": (self.repo_path / "spikeformer" / "profiling.py").exists(),
            "training": (self.repo_path / "spikeformer" / "training.py").exists()
        }
        
        architecture_score += sum(module_separation.values()) * 8  # 8 points per module
        
        # Check for interfaces and abstractions
        abstract_classes = 0
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                abstract_classes += len(re.findall(r'class.*\(.*ABC.*\):', content))
                abstract_classes += len(re.findall(r'@abstractmethod', content))
            except:
                continue
        
        architecture_score += min(20, abstract_classes * 5)  # 5 points per abstract class/method
        
        self.results.append(QualityMetric(
            name="Architecture Patterns",
            status="PASS" if architecture_score >= self.thresholds["architecture_score"] else "WARN",
            score=architecture_score,
            details=f"Design patterns: {sum(patterns_found.values())}, Modules: {sum(module_separation.values())}, Abstractions: {abstract_classes}"
        ))
    
    def _check_test_coverage(self):
        """Check test coverage and quality."""
        
        print("🧪 Checking test coverage...")
        
        # Count test files and functions
        test_functions = 0
        test_classes = 0
        
        for test_file in self.test_files:
            try:
                content = test_file.read_text()
                test_functions += len(re.findall(r'def test_', content))
                test_classes += len(re.findall(r'class Test.*:', content))
            except:
                continue
        
        # Count source functions for coverage estimation
        source_functions = 0
        for source_file in self.source_files:
            try:
                content = source_file.read_text()
                source_functions += len(re.findall(r'def [^_]', content))  # Non-private functions
            except:
                continue
        
        # Estimate coverage
        if source_functions > 0:
            estimated_coverage = min(100, (test_functions / source_functions) * 100)
        else:
            estimated_coverage = 0
        
        # Check for test types
        test_types = {
            "unit": 0,
            "integration": 0,
            "performance": 0,
            "hardware": 0
        }
        
        for test_file in self.test_files:
            try:
                content = test_file.read_text()
                if "unit" in test_file.name or "test_" in test_file.name:
                    test_types["unit"] += 1
                if "integration" in test_file.name:
                    test_types["integration"] += 1
                if "performance" in test_file.name or "benchmark" in content:
                    test_types["performance"] += 1
                if "hardware" in test_file.name or "loihi" in content or "spinnaker" in content:
                    test_types["hardware"] += 1
            except:
                continue
        
        test_diversity_score = min(100, sum(test_types.values()) * 20)
        final_test_score = (estimated_coverage + test_diversity_score) / 2
        
        self.results.append(QualityMetric(
            name="Test Coverage",
            status="PASS" if final_test_score >= self.thresholds["code_coverage"] else "FAIL",
            score=final_test_score,
            details=f"Functions: {test_functions}, Coverage: {estimated_coverage:.1f}%, Types: {dict(test_types)}",
            critical=True
        ))
    
    def _check_research_implementation(self):
        """Check research and novel algorithm implementation."""
        
        print("🔬 Checking research implementation...")
        
        research_indicators = {
            "quantum_algorithms": 0,
            "adaptive_systems": 0,
            "meta_learning": 0,
            "neuromorphic_patterns": 0,
            "self_improvement": 0
        }
        
        research_patterns = {
            "quantum_algorithms": [r"quantum", r"superposition", r"entanglement", r"coherence"],
            "adaptive_systems": [r"adaptive", r"self.adapt", r"adaptation", r"homeostasis"],
            "meta_learning": [r"meta.*learn", r"learn.*learn", r"meta.*optim"],
            "neuromorphic_patterns": [r"spike", r"neuron", r"membrane", r"threshold"],
            "self_improvement": [r"self.*improv", r"autonomous.*optim", r"auto.*adapt"]
        }
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text().lower()
                for category, patterns in research_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content):
                            research_indicators[category] += 1
            except:
                continue
        
        # Check for specific research modules
        research_modules = ["research.py", "adaptive.py", "self_improving.py", "quantum_scaling.py"]
        research_module_score = sum(1 for module in research_modules if (self.repo_path / "spikeformer" / module).exists())
        
        research_score = min(100, (sum(research_indicators.values()) / 20 + research_module_score / 4) * 50)
        
        self.results.append(QualityMetric(
            name="Research Implementation",
            status="PASS" if research_score >= 70 else "WARN",
            score=research_score,
            details=f"Indicators: {dict(research_indicators)}, Modules: {research_module_score}/4"
        ))
    
    def _check_scalability_features(self):
        """Check scalability and performance features."""
        
        print("⚡ Checking scalability features...")
        
        scalability_features = {
            "distributed_processing": 0,
            "parallel_execution": 0,
            "resource_optimization": 0,
            "load_balancing": 0,
            "auto_scaling": 0
        }
        
        scalability_patterns = {
            "distributed_processing": [r"distributed", r"multi.*node", r"cluster"],
            "parallel_execution": [r"parallel", r"concurrent", r"thread", r"process"],
            "resource_optimization": [r"memory.*optim", r"cache", r"pool"],
            "load_balancing": [r"load.*balanc", r"worker.*pool", r"queue"],
            "auto_scaling": [r"auto.*scal", r"scale.*resour", r"dynamic.*scal"]
        }
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text().lower()
                for category, patterns in scalability_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, content):
                            scalability_features[category] += 1
            except:
                continue
        
        # Check for scaling modules
        scaling_modules = ["scaling.py", "quantum_scaling.py", "concurrency.py", "performance.py"]
        scaling_module_score = sum(1 for module in scaling_modules if (self.repo_path / "spikeformer" / module).exists())
        
        scalability_score = min(100, (sum(scalability_features.values()) / 15 + scaling_module_score / 4) * 50)
        
        self.results.append(QualityMetric(
            name="Scalability Features",
            status="PASS" if scalability_score >= 70 else "WARN",
            score=scalability_score,
            details=f"Features: {dict(scalability_features)}, Modules: {scaling_module_score}/4"
        ))
    
    def _check_production_readiness(self):
        """Check production readiness indicators."""
        
        print("🚀 Checking production readiness...")
        
        production_score = 0
        
        # Check configuration management
        config_files = list(self.repo_path.glob("*.toml")) + list(self.repo_path.glob("*.yaml")) + list(self.repo_path.glob("*.yml"))
        if config_files:
            production_score += 15
        
        # Check dependency management
        if (self.repo_path / "requirements.txt").exists() or (self.repo_path / "pyproject.toml").exists():
            production_score += 15
        
        # Check deployment configuration
        deployment_indicators = ["dockerfile", "docker-compose", "kubernetes", "deployment"]
        deployment_files = []
        for indicator in deployment_indicators:
            deployment_files.extend(list(self.repo_path.glob(f"**/*{indicator}*")))
        
        if deployment_files:
            production_score += 20
        
        # Check monitoring and logging
        monitoring_patterns = [r"logging", r"logger", r"metrics", r"monitor"]
        monitoring_count = 0
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text().lower()
                for pattern in monitoring_patterns:
                    monitoring_count += len(re.findall(pattern, content))
            except:
                continue
        
        if monitoring_count > 10:
            production_score += 20
        
        # Check error handling
        error_handling_patterns = [r"try:", r"except", r"raise", r"error"]
        error_handling_count = 0
        
        for file_path in self.source_files:
            try:
                content = file_path.read_text()
                for pattern in error_handling_patterns:
                    error_handling_count += len(re.findall(pattern, content))
            except:
                continue
        
        if error_handling_count > 20:
            production_score += 15
        
        # Check CI/CD setup
        ci_files = list(self.repo_path.glob(".github/workflows/*")) + list(self.repo_path.glob(".gitlab-ci.yml"))
        if ci_files:
            production_score += 15
        
        self.results.append(QualityMetric(
            name="Production Readiness",
            status="PASS" if production_score >= 80 else "WARN",
            score=production_score,
            details=f"Config: {bool(config_files)}, Deploy: {len(deployment_files)}, Monitoring: {monitoring_count}, Errors: {error_handling_count}"
        ))
    
    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall score."""
        
        if not self.results:
            return 0.0
        
        # Weight critical metrics more heavily
        total_weighted_score = 0
        total_weight = 0
        
        for metric in self.results:
            weight = 2 if metric.critical else 1
            total_weighted_score += metric.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_overall_status(self) -> str:
        """Determine overall status based on critical failures and scores."""
        
        # Check for critical failures
        critical_failures = [m for m in self.results if m.critical and m.status == "FAIL"]
        if critical_failures:
            return "FAIL"
        
        overall_score = self._calculate_overall_score()
        
        if overall_score >= 85:
            return "PASS"
        elif overall_score >= 70:
            return "WARN"
        else:
            return "FAIL"
    
    def _print_comprehensive_report(self, result: QualityGateResult):
        """Print comprehensive quality gate report."""
        
        print("\n" + "="*80)
        print("🎯 TERRAGON AUTONOMOUS QUALITY GATES - FINAL REPORT")
        print("="*80)
        
        # Overall status
        status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        print(f"\n📊 OVERALL STATUS: {status_emoji.get(result.overall_status, '❓')} {result.overall_status}")
        print(f"📈 OVERALL SCORE: {result.overall_score:.1f}/100")
        print(f"⏱️ EXECUTION TIME: {result.execution_time:.2f}s")
        print(f"📅 TIMESTAMP: {result.timestamp}")
        
        # Detailed metrics
        print(f"\n📋 DETAILED METRICS ({len(result.metrics)} checks):")
        print("-" * 80)
        
        for metric in result.metrics:
            status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
            critical_marker = " [CRITICAL]" if metric.critical else ""
            
            print(f"{status_emoji.get(metric.status, '❓')} {metric.name:<25} {metric.score:>6.1f}% {critical_marker}")
            print(f"   └─ {metric.details}")
        
        # Critical issues summary
        critical_issues = [m for m in result.metrics if m.critical and m.status in ["FAIL", "WARN"]]
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   • {issue.name}: {issue.status} ({issue.score:.1f}%)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        failing_metrics = [m for m in result.metrics if m.status == "FAIL"]
        warning_metrics = [m for m in result.metrics if m.status == "WARN"]
        
        if failing_metrics:
            print("   🔴 IMMEDIATE ACTION REQUIRED:")
            for metric in failing_metrics:
                print(f"      • Fix {metric.name} (Score: {metric.score:.1f}%)")
        
        if warning_metrics:
            print("   🟡 IMPROVEMENTS RECOMMENDED:")
            for metric in warning_metrics:
                print(f"      • Enhance {metric.name} (Score: {metric.score:.1f}%)")
        
        if result.overall_status == "PASS":
            print("   🎉 EXCELLENT! All quality gates passed successfully.")
            print("   🚀 Ready for production deployment.")
        
        print("\n" + "="*80)
        print("🔬 TERRAGON QUALITY VALIDATION COMPLETE")
        print("="*80)
        
        # Save results
        self._save_results(result)
    
    def _save_results(self, result: QualityGateResult):
        """Save quality gate results to file."""
        
        results_file = self.repo_path / "quality_gate_results.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            print(f"📄 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")


def main():
    """Main execution function."""
    
    # Check if running in correct directory
    if not Path("/root/repo").exists():
        print("❌ Repository not found at /root/repo")
        sys.exit(1)
    
    # Initialize and run quality gates
    quality_gates = AutonomousQualityGates("/root/repo")
    result = quality_gates.run_all_quality_gates()
    
    # Exit with appropriate code
    if result.overall_status == "FAIL":
        sys.exit(1)
    elif result.overall_status == "WARN":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()