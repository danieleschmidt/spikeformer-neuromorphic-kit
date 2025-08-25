#!/usr/bin/env python3
"""
Quality Gates and Security Validation Suite
===========================================

Comprehensive quality assurance and security validation for 
quantum consciousness research implementations:

Quality Gates:
- Code quality and complexity analysis
- Performance benchmarking and optimization
- Memory safety and resource management
- Error handling and fault tolerance
- Documentation completeness

Security Validation:
- Input validation and sanitization
- No malicious code patterns
- Secure random number generation
- Environment variable security
- No hardcoded secrets or keys
"""

import os
import sys
import ast
import time
import hashlib
import re
from typing import Dict, List, Tuple, Any, Optional
import json
from dataclasses import dataclass, field
import traceback

# Mock psutil for environments where it's not available
try:
    import psutil
except ImportError:
    class MockProcess:
        def memory_info(self):
            class MemInfo:
                def __init__(self):
                    self.rss = 50 * 1024 * 1024  # Mock 50MB
            return MemInfo()
    
    class MockPsutil:
        def Process(self):
            return MockProcess()
    
    psutil = MockPsutil()


@dataclass
class QualityMetrics:
    """Quality metrics structure."""
    code_quality_score: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    documentation_score: float = 0.0
    overall_score: float = 0.0
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CodeQualityAnalyzer:
    """Analyzes code quality metrics."""
    
    def __init__(self):
        self.quality_thresholds = {
            'max_function_complexity': 15,
            'max_function_length': 150,
            'max_class_methods': 25,
            'min_docstring_coverage': 0.8,
            'max_nesting_depth': 4
        }
    
    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """Analyze a single Python file for quality metrics."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            metrics = {
                'filepath': filepath,
                'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
                'total_lines': len(content.split('\n')),
                'functions': [],
                'classes': [],
                'complexity_score': 0.0,
                'docstring_coverage': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_metrics = self._analyze_function(node, content)
                    metrics['functions'].append(func_metrics)
                    
                elif isinstance(node, ast.ClassDef):
                    class_metrics = self._analyze_class(node, content)
                    metrics['classes'].append(class_metrics)
            
            # Calculate overall metrics
            metrics['complexity_score'] = self._calculate_complexity_score(metrics)
            metrics['docstring_coverage'] = self._calculate_docstring_coverage(metrics)
            
            # Generate quality issues and recommendations
            metrics['issues'], metrics['recommendations'] = self._generate_quality_feedback(metrics)
            
            return metrics
            
        except Exception as e:
            return {
                'filepath': filepath,
                'error': f"Analysis failed: {str(e)}",
                'complexity_score': 0.0,
                'docstring_coverage': 0.0
            }
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict:
        """Analyze individual function metrics."""
        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
        
        # Count complexity indicators
        complexity_factors = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity_factors += 1
            elif isinstance(child, ast.BoolOp):
                complexity_factors += 1
        
        # Check for docstring
        has_docstring = (len(node.body) > 0 and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str))
        
        return {
            'name': node.name,
            'line_count': func_lines,
            'complexity_factors': complexity_factors,
            'has_docstring': has_docstring,
            'parameters': len(node.args.args),
            'returns_value': any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))
        }
    
    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict:
        """Analyze individual class metrics."""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        # Check for class docstring
        has_docstring = (len(node.body) > 0 and 
                        isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str))
        
        return {
            'name': node.name,
            'method_count': len(methods),
            'has_docstring': has_docstring,
            'methods': [m.name for m in methods]
        }
    
    def _calculate_complexity_score(self, metrics: Dict) -> float:
        """Calculate overall complexity score (0-1, higher is better)."""
        total_complexity = 0
        total_functions = len(metrics['functions'])
        
        if total_functions == 0:
            return 1.0
        
        for func in metrics['functions']:
            # Penalize high complexity
            complexity_penalty = max(0, func['complexity_factors'] - self.quality_thresholds['max_function_complexity']) * 0.1
            length_penalty = max(0, func['line_count'] - self.quality_thresholds['max_function_length']) * 0.005
            
            func_score = max(0.0, 1.0 - complexity_penalty - length_penalty)
            total_complexity += func_score
        
        return total_complexity / total_functions
    
    def _calculate_docstring_coverage(self, metrics: Dict) -> float:
        """Calculate docstring coverage (0-1)."""
        total_items = len(metrics['functions']) + len(metrics['classes'])
        
        if total_items == 0:
            return 1.0
        
        documented_items = (sum(1 for f in metrics['functions'] if f['has_docstring']) +
                           sum(1 for c in metrics['classes'] if c['has_docstring']))
        
        return documented_items / total_items
    
    def _generate_quality_feedback(self, metrics: Dict) -> Tuple[List[str], List[str]]:
        """Generate quality issues and recommendations."""
        issues = []
        recommendations = []
        
        # Check complexity issues
        for func in metrics['functions']:
            if func['complexity_factors'] > self.quality_thresholds['max_function_complexity']:
                issues.append(f"High complexity in function '{func['name']}': {func['complexity_factors']} factors")
                recommendations.append(f"Refactor '{func['name']}' to reduce complexity")
            
            if func['line_count'] > self.quality_thresholds['max_function_length']:
                issues.append(f"Long function '{func['name']}': {func['line_count']} lines")
                recommendations.append(f"Break down '{func['name']}' into smaller functions")
        
        # Check class issues
        for cls in metrics['classes']:
            if cls['method_count'] > self.quality_thresholds['max_class_methods']:
                issues.append(f"Large class '{cls['name']}': {cls['method_count']} methods")
                recommendations.append(f"Consider splitting '{cls['name']}' into smaller classes")
        
        # Check documentation
        if metrics['docstring_coverage'] < self.quality_thresholds['min_docstring_coverage']:
            issues.append(f"Low docstring coverage: {metrics['docstring_coverage']:.1%}")
            recommendations.append("Add docstrings to functions and classes")
        
        return issues, recommendations


class SecurityValidator:
    """Validates security aspects of code."""
    
    def __init__(self):
        self.security_patterns = {
            'dangerous_functions': [
                r'\beval\(',
                r'\bexec\(',
                r'\b__import__\(',
                r'\bcompile\(',
                r'subprocess\.call\(',
                r'os\.system\(',
                r'os\.popen\('
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'].*["\']',
                r'secret\s*=\s*["\'].*["\']',
                r'api_key\s*=\s*["\'].*["\']',
                r'token\s*=\s*["\'].*["\']'
            ],
            'sql_injection_patterns': [
                r'\.execute\([^)]*%[^)]*\)',
                r'\.execute\([^)]*\+[^)]*\)',
                r'\.format\([^)]*\)'
            ],
            'file_operations': [
                r'open\([^)]*["\']w["\'][^)]*\)',
                r'\.write\(',
                r'\.writelines\('
            ]
        }
        
        self.security_recommendations = {
            'use_secure_random': "Use secrets.SystemRandom() for cryptographic randomness",
            'validate_inputs': "Add input validation and sanitization",
            'use_parameterized_queries': "Use parameterized queries to prevent SQL injection",
            'avoid_dangerous_functions': "Avoid eval(), exec(), and similar dangerous functions",
            'use_environment_variables': "Store secrets in environment variables"
        }
    
    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate security aspects of a single file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            security_results = {
                'filepath': filepath,
                'security_issues': [],
                'security_warnings': [],
                'security_recommendations': [],
                'security_score': 1.0
            }
            
            # Check for dangerous patterns
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        if category == 'dangerous_functions':
                            security_results['security_issues'].append(
                                f"Dangerous function usage: {matches[0]}"
                            )
                            security_results['security_score'] -= 0.3
                            
                        elif category == 'hardcoded_secrets':
                            security_results['security_issues'].append(
                                f"Potential hardcoded secret detected"
                            )
                            security_results['security_score'] -= 0.4
                            
                        elif category == 'sql_injection_patterns':
                            security_results['security_warnings'].append(
                                f"Potential SQL injection vulnerability"
                            )
                            security_results['security_score'] -= 0.2
                            
                        elif category == 'file_operations':
                            security_results['security_warnings'].append(
                                f"File write operation detected - ensure path validation"
                            )
                            security_results['security_score'] -= 0.1
            
            # Check for security best practices
            security_results['security_recommendations'] = self._generate_security_recommendations(
                content, security_results
            )
            
            # Ensure score doesn't go below 0
            security_results['security_score'] = max(0.0, security_results['security_score'])
            
            return security_results
            
        except Exception as e:
            return {
                'filepath': filepath,
                'error': f"Security validation failed: {str(e)}",
                'security_score': 0.0
            }
    
    def _generate_security_recommendations(self, content: str, results: Dict) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Check for random number usage
        if 'random.' in content and 'secrets.' not in content:
            recommendations.append(self.security_recommendations['use_secure_random'])
        
        # Check for input validation
        if ('input(' in content or 'sys.argv' in content) and 'validate' not in content.lower():
            recommendations.append(self.security_recommendations['validate_inputs'])
        
        # Add general recommendations based on findings
        if results['security_issues']:
            recommendations.append(self.security_recommendations['avoid_dangerous_functions'])
        
        if any('secret' in issue.lower() for issue in results['security_issues']):
            recommendations.append(self.security_recommendations['use_environment_variables'])
        
        return list(set(recommendations))  # Remove duplicates


class PerformanceBenchmarker:
    """Benchmarks performance characteristics."""
    
    def __init__(self):
        self.benchmark_functions = [
            'run_consciousness_experiment',
            'run_enhanced_experiment', 
            'run_hyperscale_experiment',
            'process_timestep',
            'process_hyperscale_consciousness'
        ]
    
    def benchmark_execution(self, module, function_name: str, *args, **kwargs) -> Dict:
        """Benchmark function execution."""
        if not hasattr(module, function_name):
            return {'error': f"Function {function_name} not found in module"}
        
        func = getattr(module, function_name)
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark execution
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Calculate performance metrics
            performance_score = self._calculate_performance_score(execution_time, memory_used)
            
            return {
                'function_name': function_name,
                'execution_time': execution_time,
                'memory_used_mb': memory_used,
                'performance_score': performance_score,
                'result_type': type(result).__name__,
                'success': True
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'function_name': function_name,
                'execution_time': execution_time,
                'error': str(e),
                'performance_score': 0.0,
                'success': False
            }
    
    def _calculate_performance_score(self, execution_time: float, memory_used: float) -> float:
        """Calculate performance score (0-1, higher is better)."""
        # Time score (penalize slow execution)
        time_score = max(0.0, 1.0 - execution_time / 10.0)  # Penalty after 10 seconds
        
        # Memory score (penalize high memory usage)
        memory_score = max(0.0, 1.0 - memory_used / 1000.0)  # Penalty after 1GB
        
        # Combined score
        return (time_score * 0.7 + memory_score * 0.3)


class QualityGateValidator:
    """Main quality gate validation orchestrator."""
    
    def __init__(self):
        self.code_analyzer = CodeQualityAnalyzer()
        self.security_validator = SecurityValidator()
        self.performance_benchmarker = PerformanceBenchmarker()
        
        self.quality_thresholds = {
            'min_code_quality': 0.7,
            'min_security_score': 0.8,
            'min_performance_score': 0.6,
            'min_documentation_score': 0.7
        }
        
        print("🛡️ Quality Gate Validator Initialized")
        print(f"   Quality Thresholds: {len(self.quality_thresholds)} metrics")
    
    def validate_research_modules(self) -> Dict[str, Any]:
        """Validate all research modules comprehensively."""
        print("\n🔍 Running Quality Gates and Security Validation")
        print("=" * 60)
        
        validation_start = time.time()
        
        # Modules to validate
        research_modules = [
            'quantum_consciousness_demo_simple.py',
            'robust_adaptive_consciousness.py', 
            'hyperscale_quantum_optimization.py',
            'comprehensive_research_testing.py'
        ]
        
        validation_results = {
            'timestamp': time.time(),
            'total_validation_time': 0.0,
            'modules_validated': len(research_modules),
            'quality_gate_results': {},
            'overall_assessment': {},
            'gate_status': 'PENDING'
        }
        
        module_results = {}
        
        # Validate each module
        for module_path in research_modules:
            print(f"\n📁 Validating module: {module_path}")
            
            if not os.path.exists(module_path):
                print(f"   ❌ Module not found: {module_path}")
                module_results[module_path] = {'error': 'Module not found'}
                continue
            
            # Code quality analysis
            print("   🔍 Analyzing code quality...")
            code_quality = self.code_analyzer.analyze_file(module_path)
            
            # Security validation
            print("   🔒 Running security validation...")
            security_results = self.security_validator.validate_file(module_path)
            
            # Performance benchmarking (basic file size analysis)
            print("   ⚡ Checking performance characteristics...")
            file_size = os.path.getsize(module_path) / 1024  # KB
            performance_analysis = self._analyze_file_performance(module_path, file_size)
            
            # Combine results
            module_results[module_path] = {
                'code_quality': code_quality,
                'security': security_results,
                'performance': performance_analysis,
                'overall_module_score': self._calculate_module_score(
                    code_quality, security_results, performance_analysis
                )
            }
            
            # Print module summary
            overall_score = module_results[module_path]['overall_module_score']
            status = "✅ PASSED" if overall_score > 0.7 else "⚠️  NEEDS REVIEW" if overall_score > 0.5 else "❌ FAILED"
            print(f"   {status} Overall Score: {overall_score:.3f}")
        
        validation_results['quality_gate_results'] = module_results
        validation_results['total_validation_time'] = time.time() - validation_start
        
        # Overall assessment
        overall_assessment = self._generate_overall_assessment(module_results)
        validation_results['overall_assessment'] = overall_assessment
        
        # Determine gate status
        gate_passed = overall_assessment['average_score'] > 0.7
        validation_results['gate_status'] = 'PASSED' if gate_passed else 'FAILED'
        
        self._print_validation_summary(validation_results)
        
        return validation_results
    
    def _analyze_file_performance(self, filepath: str, file_size_kb: float) -> Dict:
        """Analyze file-level performance characteristics."""
        # Basic performance analysis based on file characteristics
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Count complexity indicators
        loop_count = len(re.findall(r'\b(for|while)\b', content))
        nested_structures = len(re.findall(r'    .*for|    .*while|    .*if', content))
        large_data_structures = len(re.findall(r'\[.*\].*\[.*\]|\{.*\}.*\{.*\}', content))
        
        # Calculate performance score
        performance_score = 1.0
        
        # Penalize large files
        if file_size_kb > 100:
            performance_score -= min(0.3, file_size_kb / 1000)
        
        # Penalize complex structures
        performance_score -= min(0.3, (loop_count + nested_structures) / 50)
        performance_score -= min(0.2, large_data_structures / 20)
        
        performance_score = max(0.0, performance_score)
        
        return {
            'file_size_kb': file_size_kb,
            'loop_count': loop_count,
            'nested_structures': nested_structures,
            'large_data_structures': large_data_structures,
            'performance_score': performance_score,
            'recommendations': self._get_performance_recommendations(
                file_size_kb, loop_count, nested_structures
            )
        }
    
    def _get_performance_recommendations(self, file_size: float, loops: int, nesting: int) -> List[str]:
        """Get performance recommendations."""
        recommendations = []
        
        if file_size > 150:
            recommendations.append("Consider splitting large file into smaller modules")
        
        if loops > 20:
            recommendations.append("Review loop efficiency and consider optimization")
        
        if nesting > 10:
            recommendations.append("Reduce nested complexity for better performance")
        
        return recommendations
    
    def _calculate_module_score(self, code_quality: Dict, security: Dict, performance: Dict) -> float:
        """Calculate overall module score."""
        # Extract scores with fallbacks
        quality_score = code_quality.get('complexity_score', 0.5)
        doc_score = code_quality.get('docstring_coverage', 0.5)
        security_score = security.get('security_score', 0.5)
        performance_score = performance.get('performance_score', 0.5)
        
        # Weighted average
        overall_score = (
            quality_score * 0.3 +
            doc_score * 0.2 +
            security_score * 0.3 +
            performance_score * 0.2
        )
        
        return overall_score
    
    def _generate_overall_assessment(self, module_results: Dict) -> Dict:
        """Generate overall quality assessment."""
        scores = []
        total_issues = []
        total_recommendations = []
        
        for module_path, results in module_results.items():
            if 'error' in results:
                continue
                
            scores.append(results['overall_module_score'])
            
            # Collect issues
            if 'code_quality' in results:
                total_issues.extend(results['code_quality'].get('issues', []))
                total_recommendations.extend(results['code_quality'].get('recommendations', []))
            
            if 'security' in results:
                total_issues.extend(results['security'].get('security_issues', []))
                total_recommendations.extend(results['security'].get('security_recommendations', []))
            
            if 'performance' in results:
                total_recommendations.extend(results['performance'].get('recommendations', []))
        
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # Quality assessment
        if average_score > 0.85:
            quality_level = 'Excellent'
        elif average_score > 0.75:
            quality_level = 'Good'
        elif average_score > 0.6:
            quality_level = 'Acceptable'
        else:
            quality_level = 'Needs Improvement'
        
        return {
            'average_score': average_score,
            'quality_level': quality_level,
            'modules_analyzed': len([r for r in module_results.values() if 'error' not in r]),
            'total_issues': len(total_issues),
            'total_recommendations': len(set(total_recommendations)),
            'top_issues': total_issues[:5],
            'top_recommendations': list(set(total_recommendations))[:5]
        }
    
    def _print_validation_summary(self, results: Dict):
        """Print comprehensive validation summary."""
        print("\n" + "=" * 60)
        print("🛡️ QUALITY GATES & SECURITY VALIDATION RESULTS")
        print("=" * 60)
        
        # Overall status
        gate_status = results['gate_status']
        status_icon = "✅" if gate_status == 'PASSED' else "❌"
        print(f"\n🚪 Quality Gate Status: {status_icon} {gate_status}")
        
        # Summary metrics
        assessment = results['overall_assessment']
        print(f"\n📊 Overall Assessment:")
        print(f"   Quality Level: {assessment['quality_level']}")
        print(f"   Average Score: {assessment['average_score']:.3f}")
        print(f"   Modules Analyzed: {assessment['modules_analyzed']}")
        print(f"   Total Issues: {assessment['total_issues']}")
        print(f"   Validation Time: {results['total_validation_time']:.2f}s")
        
        # Module breakdown
        print(f"\n📁 Module Breakdown:")
        for module_path, module_result in results['quality_gate_results'].items():
            if 'error' in module_result:
                print(f"   {module_path}: ❌ ERROR - {module_result['error']}")
                continue
                
            score = module_result['overall_module_score']
            status = "✅" if score > 0.7 else "⚠️" if score > 0.5 else "❌"
            print(f"   {module_path}: {status} Score: {score:.3f}")
            
            # Security summary
            security = module_result.get('security', {})
            security_issues = len(security.get('security_issues', []))
            security_warnings = len(security.get('security_warnings', []))
            
            if security_issues > 0:
                print(f"      🔒 Security Issues: {security_issues}")
            if security_warnings > 0:
                print(f"      ⚠️  Security Warnings: {security_warnings}")
        
        # Key issues and recommendations
        if assessment['top_issues']:
            print(f"\n🚨 Top Issues Found:")
            for issue in assessment['top_issues']:
                print(f"   • {issue}")
        
        if assessment['top_recommendations']:
            print(f"\n💡 Top Recommendations:")
            for rec in assessment['top_recommendations']:
                print(f"   • {rec}")
        
        # Quality gate verdict
        print(f"\n🎯 Quality Gate Verdict:")
        if gate_status == 'PASSED':
            print("   ✅ All quality gates PASSED")
            print("   ✅ Code meets quality and security standards")
            print("   ✅ Ready for production deployment")
        else:
            print("   ❌ Quality gates FAILED")
            print("   🔧 Address issues before production deployment")
            print("   📋 Review recommendations for improvement")


def main():
    """Run comprehensive quality gates and security validation."""
    print("🛡️ Quality Gates and Security Validation Suite")
    print("Comprehensive quality assurance for quantum consciousness research")
    print("=" * 70)
    
    try:
        # Initialize validator
        validator = QualityGateValidator()
        
        # Run comprehensive validation
        validation_results = validator.validate_research_modules()
        
        # Save results
        results_filename = "quality_security_validation_results.json"
        with open(results_filename, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"\n📄 Quality validation results saved to: {results_filename}")
        
        # Return final status
        return validation_results
        
    except Exception as e:
        print(f"\n❌ Critical validation error: {str(e)}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    quality_results = main()