#!/usr/bin/env python3
"""
Comprehensive Quality Gates - SDLC Validation Suite
Testing, security, performance, documentation, and deployment readiness validation
"""

import sys
import os
import json
import logging
import time
import hashlib
import subprocess
import threading
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
import ast
import traceback
import gc
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import resource
import urllib.request
import urllib.error

# Import our implementations
try:
    from basic_functionality_demo import BasicSpikingNetwork, BasicEnergyProfiler
    from robust_spikeformer_enhanced import RobustSpikingNetwork, SecurityConfig, SecurityLevel
    from scaling_spikeformer_optimized import ScalingSpikingNetwork, ScalingStrategy
except ImportError as e:
    print(f"⚠️ Warning: Could not import some modules: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quality_gates.log')
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics"""
    test_coverage_percent: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    code_quality_score: float = 0.0
    documentation_score: float = 0.0
    deployment_readiness_score: float = 0.0
    overall_quality_score: float = 0.0
    
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    passed_gates: List[str] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)

class SecurityScanner:
    """Comprehensive security scanning and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'unsafe_functions': [
                r'\beval\s*\(',
                r'\bexec\s*\(',
                r'\binput\s*\(',
                r'os\.system\s*\(',
                r'subprocess\.call\s*\('
            ],
            'sql_injection_risk': [
                r'execute\s*\(\s*["\'][^"\']*%[^"\']*["\']',
                r'cursor\.execute\s*\(\s*["\'][^"\']*\+[^"\']*["\']'
            ],
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\s*\(',
                r'random\.random\s*\('
            ]
        }
        
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for security issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'category': category,
                            'pattern': pattern,
                            'line': line_num,
                            'file': str(file_path),
                            'match': match.group(0)
                        })
            
            return {
                'file': str(file_path),
                'issues': issues,
                'lines_scanned': len(content.splitlines()),
                'scan_time': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to scan file {file_path}: {e}")
            return {
                'file': str(file_path),
                'issues': [],
                'error': str(e),
                'scan_time': time.time()
            }
    
    def scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan all Python files in directory"""
        start_time = time.time()
        all_issues = []
        files_scanned = 0
        
        python_files = list(directory.glob("**/*.py"))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.scan_file, file_path) for file_path in python_files]
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    all_issues.extend(result.get('issues', []))
                    files_scanned += 1
                except TimeoutError:
                    self.logger.warning("File scan timeout")
                except Exception as e:
                    self.logger.error(f"Scan error: {e}")
        
        # Calculate security score
        total_lines = sum(len(open(f, 'r').readlines()) for f in python_files if f.exists())
        issue_density = len(all_issues) / max(total_lines, 1)
        security_score = max(0, 100 - (issue_density * 1000))  # Scale appropriately
        
        return {
            'scan_time': time.time() - start_time,
            'files_scanned': files_scanned,
            'total_issues': len(all_issues),
            'issues_by_category': self._categorize_issues(all_issues),
            'security_score': security_score,
            'all_issues': all_issues
        }
    
    def _categorize_issues(self, issues: List[Dict]) -> Dict[str, int]:
        """Categorize security issues"""
        categories = {}
        for issue in issues:
            category = issue.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories

class PerformanceTester:
    """Comprehensive performance testing and benchmarking"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def test_basic_performance(self) -> Dict[str, Any]:
        """Test basic spikeformer performance"""
        self.logger.info("Testing basic spikeformer performance")
        
        try:
            network = BasicSpikingNetwork(input_size=5, hidden_size=10, output_size=3)
            test_inputs = [0.8, 0.3, 0.9, 0.1, 0.7]
            
            # Warmup
            for _ in range(5):
                network.forward(test_inputs, timesteps=10)
            
            # Performance test
            times = []
            for _ in range(20):
                start = time.time()
                result = network.forward(test_inputs, timesteps=20)
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            return {
                'test_name': 'basic_performance',
                'avg_time_ms': avg_time * 1000,
                'min_time_ms': min_time * 1000,
                'max_time_ms': max_time * 1000,
                'throughput_ops_per_sec': 1.0 / avg_time,
                'passed': avg_time < 0.01,  # Should be under 10ms
                'score': max(0, 100 - (avg_time * 10000))  # Scale score
            }
            
        except Exception as e:
            self.logger.error(f"Basic performance test failed: {e}")
            return {
                'test_name': 'basic_performance',
                'error': str(e),
                'passed': False,
                'score': 0
            }
    
    def test_robust_performance(self) -> Dict[str, Any]:
        """Test robust spikeformer performance"""
        self.logger.info("Testing robust spikeformer performance")
        
        try:
            security_config = SecurityConfig(level=SecurityLevel.HIGH)
            network = RobustSpikingNetwork(
                input_size=5, hidden_size=8, output_size=3,
                security_config=security_config
            )
            test_inputs = [0.8, 0.3, 0.9, 0.1, 0.7]
            
            # Performance test with error scenarios
            times = []
            errors = 0
            
            for _ in range(15):
                try:
                    start = time.time()
                    result = network.forward(test_inputs, timesteps=20)
                    times.append(time.time() - start)
                except Exception:
                    errors += 1
            
            if not times:
                return {
                    'test_name': 'robust_performance',
                    'error': 'No successful executions',
                    'passed': False,
                    'score': 0
                }
            
            avg_time = sum(times) / len(times)
            error_rate = errors / 15
            
            return {
                'test_name': 'robust_performance',
                'avg_time_ms': avg_time * 1000,
                'error_rate': error_rate,
                'successful_runs': len(times),
                'passed': avg_time < 0.02 and error_rate < 0.1,
                'score': max(0, 100 - (avg_time * 5000) - (error_rate * 50))
            }
            
        except Exception as e:
            self.logger.error(f"Robust performance test failed: {e}")
            return {
                'test_name': 'robust_performance',
                'error': str(e),
                'passed': False,
                'score': 0
            }
    
    def test_scaling_performance(self) -> Dict[str, Any]:
        """Test scaling spikeformer performance"""
        self.logger.info("Testing scaling spikeformer performance")
        
        try:
            network = ScalingSpikingNetwork(
                input_size=5, hidden_size=10, output_size=3,
                scaling_strategy=ScalingStrategy.QUANTUM
            )
            test_inputs = [0.8, 0.3, 0.9, 0.1, 0.7]
            
            # Single inference test
            start = time.time()
            result = network.forward(test_inputs, timesteps=20)
            single_time = time.time() - start
            
            # Batch test
            batch_inputs = [test_inputs for _ in range(10)]
            start = time.time()
            batch_results = network.parallel_forward(batch_inputs, timesteps=20)
            batch_time = time.time() - start
            
            batch_throughput = len(batch_inputs) / batch_time
            
            # Get scaling diagnostics
            diagnostics = network.get_scaling_diagnostics()
            network.shutdown()
            
            return {
                'test_name': 'scaling_performance',
                'single_time_ms': single_time * 1000,
                'batch_time_ms': batch_time * 1000,
                'batch_throughput': batch_throughput,
                'memory_usage_mb': diagnostics['performance']['estimated_memory_mb'],
                'cache_hit_ratio': diagnostics['caching']['hit_ratio'],
                'passed': single_time < 0.005 and batch_throughput > 100,
                'score': min(100, batch_throughput / 10)  # Scale by throughput
            }
            
        except Exception as e:
            self.logger.error(f"Scaling performance test failed: {e}")
            return {
                'test_name': 'scaling_performance',
                'error': str(e),
                'passed': False,
                'score': 0
            }
    
    def test_memory_efficiency(self) -> Dict[str, Any]:
        """Test memory efficiency and garbage collection"""
        self.logger.info("Testing memory efficiency")
        
        try:
            # Use resource module to get memory usage
            initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB (Linux)
            
            # Create and destroy networks to test memory leaks
            networks = []
            for i in range(10):
                network = BasicSpikingNetwork(input_size=5, hidden_size=10, output_size=3)
                networks.append(network)
                
                # Use the network
                test_inputs = [0.8, 0.3, 0.9, 0.1, 0.7]
                network.forward(test_inputs, timesteps=10)
            
            mid_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
            
            # Clean up
            del networks
            gc.collect()
            
            final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
            
            memory_growth = final_memory - initial_memory
            memory_cleaned = mid_memory - final_memory
            
            return {
                'test_name': 'memory_efficiency',
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': mid_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': memory_growth,
                'memory_cleaned_mb': memory_cleaned,
                'passed': memory_growth < 50,  # Less than 50MB growth
                'score': max(0, 100 - memory_growth * 2)
            }
            
        except Exception as e:
            self.logger.error(f"Memory efficiency test failed: {e}")
            return {
                'test_name': 'memory_efficiency',
                'error': str(e),
                'passed': False,
                'score': 0
            }

class CodeQualityAnalyzer:
    """Code quality analysis and metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Calculate metrics
            lines = content.splitlines()
            total_lines = len(lines)
            blank_lines = sum(1 for line in lines if not line.strip())
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            code_lines = total_lines - blank_lines - comment_lines
            
            # Count functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Calculate complexity (simplified)
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    complexity += 1
            
            # Check for docstrings
            docstrings = 0
            for func in functions:
                if (ast.get_docstring(func) is not None):
                    docstrings += 1
            for cls in classes:
                if (ast.get_docstring(cls) is not None):
                    docstrings += 1
            
            documentation_ratio = docstrings / max(len(functions) + len(classes), 1)
            
            return {
                'file': str(file_path),
                'total_lines': total_lines,
                'code_lines': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'functions': len(functions),
                'classes': len(classes),
                'complexity': complexity,
                'documentation_ratio': documentation_ratio,
                'avg_function_length': code_lines / max(len(functions), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}: {e}")
            return {
                'file': str(file_path),
                'error': str(e)
            }
    
    def analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """Analyze all Python files in directory"""
        start_time = time.time()
        
        python_files = list(directory.glob("**/*.py"))
        if not python_files:
            return {
                'error': 'No Python files found',
                'score': 0
            }
        
        file_analyses = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.analyze_file, file_path) for file_path in python_files]
            
            for future in futures:
                try:
                    result = future.result(timeout=10)
                    if 'error' not in result:
                        file_analyses.append(result)
                except Exception as e:
                    self.logger.error(f"File analysis error: {e}")
        
        if not file_analyses:
            return {
                'error': 'No files successfully analyzed',
                'score': 0
            }
        
        # Aggregate metrics
        total_code_lines = sum(f['code_lines'] for f in file_analyses)
        total_functions = sum(f['functions'] for f in file_analyses)
        total_classes = sum(f['classes'] for f in file_analyses)
        avg_complexity = sum(f['complexity'] for f in file_analyses) / len(file_analyses)
        avg_documentation = sum(f['documentation_ratio'] for f in file_analyses) / len(file_analyses)
        
        # Calculate quality score
        complexity_score = max(0, 100 - avg_complexity * 2)  # Lower complexity is better
        documentation_score = avg_documentation * 100
        structure_score = min(100, (total_functions + total_classes) / max(len(file_analyses), 1) * 10)
        
        quality_score = (complexity_score + documentation_score + structure_score) / 3
        
        return {
            'analysis_time': time.time() - start_time,
            'files_analyzed': len(file_analyses),
            'total_code_lines': total_code_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'avg_complexity': avg_complexity,
            'avg_documentation_ratio': avg_documentation,
            'quality_score': quality_score,
            'complexity_score': complexity_score,
            'documentation_score': documentation_score,
            'structure_score': structure_score
        }

class DocumentationChecker:
    """Documentation completeness and quality checker"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def check_readme(self, directory: Path) -> Dict[str, Any]:
        """Check README file quality"""
        readme_files = list(directory.glob("README*"))
        
        if not readme_files:
            return {
                'found': False,
                'score': 0,
                'issues': ['No README file found']
            }
        
        readme_path = readme_files[0]
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential sections
            essential_sections = [
                'installation', 'usage', 'example', 'api', 'license',
                'getting started', 'quick start', 'features'
            ]
            
            found_sections = []
            for section in essential_sections:
                if re.search(rf'#{1,6}\s*{section}', content, re.IGNORECASE):
                    found_sections.append(section)
            
            # Check for code examples
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            
            # Calculate score
            section_score = (len(found_sections) / len(essential_sections)) * 60
            code_score = min(40, code_blocks * 10)
            length_score = min(20, len(content) / 100)  # Reward comprehensive docs
            
            total_score = section_score + code_score + length_score
            
            return {
                'found': True,
                'file': str(readme_path),
                'length': len(content),
                'sections_found': found_sections,
                'code_blocks': code_blocks,
                'score': total_score,
                'section_score': section_score,
                'code_score': code_score
            }
            
        except Exception as e:
            return {
                'found': True,
                'file': str(readme_path),
                'error': str(e),
                'score': 0
            }
    
    def check_api_documentation(self, directory: Path) -> Dict[str, Any]:
        """Check for API documentation"""
        api_files = list(directory.glob("**/API*")) + list(directory.glob("**/api*"))
        docs_dir = directory / "docs"
        
        score = 0
        files_found = []
        
        if api_files:
            files_found.extend([str(f) for f in api_files])
            score += 30
        
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("**/*.md")) + list(docs_dir.glob("**/*.rst"))
            files_found.extend([str(f) for f in doc_files])
            score += len(doc_files) * 10
        
        return {
            'api_files_found': len(api_files),
            'docs_directory_exists': docs_dir.exists(),
            'total_doc_files': len(files_found),
            'files_found': files_found,
            'score': min(100, score)
        }

class DeploymentReadinessChecker:
    """Deployment readiness and production validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def check_docker_support(self, directory: Path) -> Dict[str, Any]:
        """Check for Docker deployment support"""
        dockerfile = directory / "Dockerfile"
        docker_compose = directory / "docker-compose.yml"
        dockerignore = directory / ".dockerignore"
        
        score = 0
        files_found = []
        
        if dockerfile.exists():
            files_found.append("Dockerfile")
            score += 40
            
            # Check Dockerfile quality
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                if 'FROM' in content:
                    score += 10
                if 'COPY' in content or 'ADD' in content:
                    score += 10
                if 'RUN' in content:
                    score += 10
                if 'CMD' in content or 'ENTRYPOINT' in content:
                    score += 10
                    
            except Exception as e:
                self.logger.error(f"Error reading Dockerfile: {e}")
        
        if docker_compose.exists():
            files_found.append("docker-compose.yml")
            score += 20
        
        if dockerignore.exists():
            files_found.append(".dockerignore")
            score += 10
        
        return {
            'dockerfile_exists': dockerfile.exists(),
            'docker_compose_exists': docker_compose.exists(),
            'dockerignore_exists': dockerignore.exists(),
            'files_found': files_found,
            'score': score
        }
    
    def check_requirements(self, directory: Path) -> Dict[str, Any]:
        """Check for dependency management files"""
        requirements_files = [
            "requirements.txt", "requirements-dev.txt", "pyproject.toml",
            "setup.py", "Pipfile", "poetry.lock"
        ]
        
        found_files = []
        score = 0
        
        for req_file in requirements_files:
            file_path = directory / req_file
            if file_path.exists():
                found_files.append(req_file)
                score += 20 if req_file in ["requirements.txt", "pyproject.toml"] else 10
        
        return {
            'requirements_files_found': found_files,
            'count': len(found_files),
            'score': min(100, score)
        }
    
    def check_ci_cd_setup(self, directory: Path) -> Dict[str, Any]:
        """Check for CI/CD configuration"""
        ci_paths = [
            ".github/workflows", ".gitlab-ci.yml", ".travis.yml",
            "Jenkinsfile", ".circleci", "azure-pipelines.yml"
        ]
        
        found_configs = []
        score = 0
        
        for ci_path in ci_paths:
            path = directory / ci_path
            if path.exists():
                found_configs.append(ci_path)
                score += 30 if "github" in ci_path else 20
        
        return {
            'ci_configs_found': found_configs,
            'count': len(found_configs),
            'score': min(100, score)
        }

class ComprehensiveQualityGates:
    """Master quality gates orchestrator"""
    
    def __init__(self, project_directory: Path):
        self.project_directory = project_directory
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize all checkers
        self.security_scanner = SecurityScanner()
        self.performance_tester = PerformanceTester()
        self.code_analyzer = CodeQualityAnalyzer()
        self.doc_checker = DocumentationChecker()
        self.deployment_checker = DeploymentReadinessChecker()
        
        self.start_time = time.time()
        
    def run_all_gates(self) -> QualityMetrics:
        """Run all quality gates and return comprehensive metrics"""
        self.logger.info("Starting comprehensive quality gates validation")
        
        metrics = QualityMetrics()
        
        # Gate 1: Security Scanning
        self.logger.info("🔒 Running security scanning...")
        security_results = self.security_scanner.scan_directory(self.project_directory)
        metrics.security_score = security_results.get('security_score', 0)
        
        if security_results.get('total_issues', 0) == 0:
            metrics.passed_gates.append("Security Scanning")
        else:
            metrics.failed_gates.append("Security Scanning")
            metrics.issues_found.extend([
                f"Security: {issue['category']} in {issue['file']}:{issue['line']}"
                for issue in security_results.get('all_issues', [])[:5]  # Show first 5
            ])
        
        # Gate 2: Performance Testing
        self.logger.info("⚡ Running performance testing...")
        perf_tests = [
            self.performance_tester.test_basic_performance(),
            self.performance_tester.test_robust_performance(),
            self.performance_tester.test_scaling_performance(),
            self.performance_tester.test_memory_efficiency()
        ]
        
        perf_scores = [test.get('score', 0) for test in perf_tests if test.get('passed', False)]
        metrics.performance_score = sum(perf_scores) / len(perf_scores) if perf_scores else 0
        
        passed_perf_tests = [test['test_name'] for test in perf_tests if test.get('passed', False)]
        if len(passed_perf_tests) >= 3:
            metrics.passed_gates.append("Performance Testing")
        else:
            metrics.failed_gates.append("Performance Testing")
            failed_tests = [test['test_name'] for test in perf_tests if not test.get('passed', False)]
            metrics.issues_found.extend([f"Performance: {test} failed" for test in failed_tests])
        
        # Gate 3: Code Quality Analysis
        self.logger.info("📊 Running code quality analysis...")
        code_results = self.code_analyzer.analyze_directory(self.project_directory)
        metrics.code_quality_score = code_results.get('quality_score', 0)
        
        if metrics.code_quality_score >= 70:
            metrics.passed_gates.append("Code Quality")
        else:
            metrics.failed_gates.append("Code Quality")
            metrics.issues_found.append(f"Code Quality: Score {metrics.code_quality_score:.1f} below threshold")
        
        # Gate 4: Documentation Check
        self.logger.info("📚 Running documentation validation...")
        readme_results = self.doc_checker.check_readme(self.project_directory)
        api_results = self.doc_checker.check_api_documentation(self.project_directory)
        
        doc_score = (readme_results.get('score', 0) + api_results.get('score', 0)) / 2
        metrics.documentation_score = doc_score
        
        if doc_score >= 60:
            metrics.passed_gates.append("Documentation")
        else:
            metrics.failed_gates.append("Documentation")
            if not readme_results.get('found', False):
                metrics.issues_found.append("Documentation: No README file found")
            if readme_results.get('score', 0) < 50:
                metrics.issues_found.append("Documentation: README quality insufficient")
        
        # Gate 5: Deployment Readiness
        self.logger.info("🚀 Running deployment readiness check...")
        docker_results = self.deployment_checker.check_docker_support(self.project_directory)
        req_results = self.deployment_checker.check_requirements(self.project_directory)
        ci_results = self.deployment_checker.check_ci_cd_setup(self.project_directory)
        
        deployment_score = (
            docker_results.get('score', 0) * 0.4 +
            req_results.get('score', 0) * 0.4 +
            ci_results.get('score', 0) * 0.2
        )
        metrics.deployment_readiness_score = deployment_score
        
        if deployment_score >= 60:
            metrics.passed_gates.append("Deployment Readiness")
        else:
            metrics.failed_gates.append("Deployment Readiness")
            if not docker_results.get('dockerfile_exists', False):
                metrics.issues_found.append("Deployment: No Dockerfile found")
            if len(req_results.get('requirements_files_found', [])) == 0:
                metrics.issues_found.append("Deployment: No requirements files found")
        
        # Calculate overall quality score
        metrics.overall_quality_score = (
            metrics.security_score * 0.25 +
            metrics.performance_score * 0.25 +
            metrics.code_quality_score * 0.20 +
            metrics.documentation_score * 0.15 +
            metrics.deployment_readiness_score * 0.15
        )
        
        # Generate recommendations
        self._generate_recommendations(metrics)
        
        # Calculate test coverage (simplified)
        metrics.test_coverage_percent = self._estimate_test_coverage()
        
        total_time = time.time() - self.start_time
        self.logger.info(f"Quality gates completed in {total_time:.2f} seconds")
        
        return metrics
    
    def _generate_recommendations(self, metrics: QualityMetrics):
        """Generate actionable recommendations"""
        if metrics.security_score < 90:
            metrics.recommendations.append("Review and fix security issues in code")
        
        if metrics.performance_score < 80:
            metrics.recommendations.append("Optimize performance bottlenecks")
        
        if metrics.code_quality_score < 70:
            metrics.recommendations.append("Improve code structure and add documentation")
        
        if metrics.documentation_score < 60:
            metrics.recommendations.append("Enhance README and API documentation")
        
        if metrics.deployment_readiness_score < 60:
            metrics.recommendations.append("Add Docker support and CI/CD configuration")
        
        if len(metrics.passed_gates) < 4:
            metrics.recommendations.append("Address failing quality gates before production deployment")
    
    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on test files vs source files"""
        try:
            source_files = list(self.project_directory.glob("**/*.py"))
            test_files = list(self.project_directory.glob("**/test_*.py")) + \
                        list(self.project_directory.glob("**/*_test.py")) + \
                        list(self.project_directory.glob("**/tests/**/*.py"))
            
            # Remove test files from source count
            source_files = [f for f in source_files if f not in test_files]
            
            if not source_files:
                return 0.0
            
            coverage_ratio = len(test_files) / len(source_files)
            return min(100, coverage_ratio * 80)  # Scale appropriately
            
        except Exception:
            return 0.0

def run_quality_gates():
    """Main function to run all quality gates"""
    print("🎯 COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 70)
    
    try:
        project_dir = Path("/root/repo")
        quality_gates = ComprehensiveQualityGates(project_dir)
        
        # Run all quality gates
        metrics = quality_gates.run_all_gates()
        
        # Display results
        print(f"\n📊 QUALITY GATES RESULTS")
        print("=" * 50)
        
        print(f"🔒 Security Score: {metrics.security_score:.1f}/100")
        print(f"⚡ Performance Score: {metrics.performance_score:.1f}/100")
        print(f"📊 Code Quality Score: {metrics.code_quality_score:.1f}/100")
        print(f"📚 Documentation Score: {metrics.documentation_score:.1f}/100")
        print(f"🚀 Deployment Readiness: {metrics.deployment_readiness_score:.1f}/100")
        print(f"🧪 Test Coverage: {metrics.test_coverage_percent:.1f}%")
        print(f"\n🎯 OVERALL QUALITY SCORE: {metrics.overall_quality_score:.1f}/100")
        
        print(f"\n✅ PASSED GATES ({len(metrics.passed_gates)}):")
        for gate in metrics.passed_gates:
            print(f"   ✓ {gate}")
        
        if metrics.failed_gates:
            print(f"\n❌ FAILED GATES ({len(metrics.failed_gates)}):")
            for gate in metrics.failed_gates:
                print(f"   ✗ {gate}")
        
        if metrics.issues_found:
            print(f"\n⚠️ ISSUES FOUND ({len(metrics.issues_found)}):")
            for issue in metrics.issues_found[:10]:  # Show first 10
                print(f"   • {issue}")
        
        if metrics.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in metrics.recommendations:
                print(f"   → {rec}")
        
        # Determine if quality gates passed
        gates_passed = len(metrics.passed_gates) >= 4 and metrics.overall_quality_score >= 75
        
        if gates_passed:
            print(f"\n🎉 QUALITY GATES PASSED - PRODUCTION READY!")
        else:
            print(f"\n⚠️ QUALITY GATES REQUIRE ATTENTION")
        
        # Save results
        results = {
            'overall_score': metrics.overall_quality_score,
            'security_score': metrics.security_score,
            'performance_score': metrics.performance_score,
            'code_quality_score': metrics.code_quality_score,
            'documentation_score': metrics.documentation_score,
            'deployment_readiness_score': metrics.deployment_readiness_score,
            'test_coverage_percent': metrics.test_coverage_percent,
            'passed_gates': metrics.passed_gates,
            'failed_gates': metrics.failed_gates,
            'issues_found': metrics.issues_found,
            'recommendations': metrics.recommendations,
            'gates_passed': gates_passed,
            'timestamp': time.time()
        }
        
        with open('/root/repo/quality_gates_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: quality_gates_results.json")
        
        return gates_passed
        
    except Exception as e:
        print(f"❌ QUALITY GATES FAILED: {e}")
        logger.critical(f"Quality gates execution failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)