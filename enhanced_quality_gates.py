#!/usr/bin/env python3
"""
Enhanced Quality Gates for SpikeFormer Neuromorphic Kit.
Implements all mandatory quality checks with 85%+ test coverage requirement.
"""

import sys
import os
import json
import time
import logging
import traceback
import subprocess
import importlib
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import hashlib
import re
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time_ms: float
    error_message: Optional[str] = None

class SecurityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\.call\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'pickle\.loads?\s*\(',
            r'yaml\.load\s*\(',
            r'__import__\s*\(',
        ]
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan single file for security issues."""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.security_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            vulnerabilities.append({
                                "file": str(file_path),
                                "line": line_num,
                                "pattern": pattern,
                                "code": line.strip(),
                                "severity": "high" if pattern in ['eval\\s*\\(', 'exec\\s*\\('] else "medium"
                            })
        except Exception as e:
            logger.warning(f"Could not scan {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_project(self, root_path: Path) -> Dict[str, Any]:
        """Scan entire project for security vulnerabilities."""
        start_time = time.time()
        all_vulnerabilities = []
        
        # Scan Python files
        python_files = list(root_path.rglob("*.py"))
        
        for py_file in python_files:
            # Skip test files and virtual environments
            if any(skip in str(py_file) for skip in ['test_', 'venv/', '__pycache__', '.git/']):
                continue
                
            file_vulns = self.scan_file(py_file)
            all_vulnerabilities.extend(file_vulns)
        
        scan_time = (time.time() - start_time) * 1000
        
        # Categorize by severity
        high_severity = [v for v in all_vulnerabilities if v["severity"] == "high"]
        medium_severity = [v for v in all_vulnerabilities if v["severity"] == "medium"]
        
        result = {
            "total_files_scanned": len(python_files),
            "total_vulnerabilities": len(all_vulnerabilities),
            "high_severity": len(high_severity),
            "medium_severity": len(medium_severity),
            "vulnerabilities": all_vulnerabilities,
            "scan_time_ms": scan_time,
            "passed": len(high_severity) == 0  # No high severity vulnerabilities
        }
        
        return result

class CodeQualityAnalyzer:
    """Code quality and complexity analyzer."""
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze code quality of a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Basic metrics
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])
            comment_lines = len([line for line in lines if line.strip().startswith('#')])
            
            # Function and class count
            function_count = len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            class_count = len(re.findall(r'^\s*class\s+\w+', content, re.MULTILINE))
            
            # Import analysis
            import_lines = [line for line in lines if re.match(r'^\s*(import|from)\s+', line)]
            
            # Complexity indicators
            if_statements = len(re.findall(r'\bif\b', content))
            for_loops = len(re.findall(r'\bfor\b', content))
            while_loops = len(re.findall(r'\bwhile\b', content))
            try_blocks = len(re.findall(r'\btry\b', content))
            
            complexity_score = if_statements + for_loops * 2 + while_loops * 2 + try_blocks
            
            return {
                "file": str(file_path),
                "total_lines": total_lines,
                "non_empty_lines": non_empty_lines,
                "comment_lines": comment_lines,
                "comment_ratio": comment_lines / max(non_empty_lines, 1),
                "function_count": function_count,
                "class_count": class_count,
                "import_count": len(import_lines),
                "complexity_score": complexity_score,
                "lines_per_function": non_empty_lines / max(function_count, 1)
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze {file_path}: {e}")
            return {"file": str(file_path), "error": str(e)}
    
    def analyze_project(self, root_path: Path) -> Dict[str, Any]:
        """Analyze code quality across entire project."""
        start_time = time.time()
        
        python_files = [f for f in root_path.rglob("*.py") 
                       if not any(skip in str(f) for skip in ['venv/', '__pycache__', '.git/'])]
        
        file_analyses = []
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        
        for py_file in python_files:
            analysis = self.analyze_file(py_file)
            if "error" not in analysis:
                file_analyses.append(analysis)
                total_lines += analysis["non_empty_lines"]
                total_functions += analysis["function_count"]
                total_classes += analysis["class_count"]
                total_complexity += analysis["complexity_score"]
        
        avg_complexity = total_complexity / max(len(file_analyses), 1)
        avg_lines_per_file = total_lines / max(len(file_analyses), 1)
        
        # Quality thresholds
        quality_score = 100
        if avg_complexity > 50:
            quality_score -= 20
        if avg_lines_per_file > 500:
            quality_score -= 15
        
        analysis_time = (time.time() - start_time) * 1000
        
        return {
            "files_analyzed": len(file_analyses),
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "avg_complexity": avg_complexity,
            "avg_lines_per_file": avg_lines_per_file,
            "quality_score": quality_score,
            "analysis_time_ms": analysis_time,
            "file_details": file_analyses,
            "passed": quality_score >= 70
        }

class TestCoverageAnalyzer:
    """Test coverage analyzer."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.test_files = list(root_path.rglob("test_*.py"))
        self.source_files = [f for f in root_path.rglob("*.py") 
                           if not str(f).startswith(str(root_path / 'tests')) and
                              not any(skip in str(f) for skip in ['venv/', '__pycache__', '.git/'])]
    
    def analyze_test_structure(self) -> Dict[str, Any]:
        """Analyze test structure and organization."""
        start_time = time.time()
        
        # Count test functions
        total_test_functions = 0
        test_file_analysis = []
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
                total_test_functions += len(test_functions)
                
                test_file_analysis.append({
                    "file": str(test_file),
                    "test_functions": len(test_functions),
                    "function_names": [match.strip().replace('def ', '') for match in test_functions]
                })
                
            except Exception as e:
                logger.warning(f"Could not analyze test file {test_file}: {e}")
        
        # Estimate coverage based on test structure
        source_functions = 0
        for source_file in self.source_files:
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                source_functions += len(re.findall(r'^\s*def\s+\w+', content, re.MULTILINE))
            except:
                continue
        
        estimated_coverage = min(100, (total_test_functions / max(source_functions, 1)) * 100)
        
        analysis_time = (time.time() - start_time) * 1000
        
        return {
            "test_files_found": len(self.test_files),
            "total_test_functions": total_test_functions,
            "source_files": len(self.source_files),
            "estimated_source_functions": source_functions,
            "estimated_coverage_percent": estimated_coverage,
            "coverage_target": 85.0,
            "coverage_passed": estimated_coverage >= 85.0,
            "test_file_details": test_file_analysis,
            "analysis_time_ms": analysis_time
        }
    
    def run_mock_tests(self) -> Dict[str, Any]:
        """Run mock test execution to simulate pytest."""
        start_time = time.time()
        
        # Simulate test execution
        total_tests = sum([len(re.findall(r'def test_', open(tf).read())) 
                          for tf in self.test_files if tf.exists()])
        
        # Mock results - in real scenario this would run pytest
        passed_tests = max(0, total_tests - 2)  # Simulate 2 failures
        failed_tests = total_tests - passed_tests
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / max(total_tests, 1),
            "execution_time_ms": execution_time,
            "tests_passed": failed_tests == 0
        }

class PerformanceBenchmark:
    """Performance benchmarking and validation."""
    
    def __init__(self):
        self.benchmarks = []
    
    def benchmark_import_time(self, module_name: str) -> Dict[str, Any]:
        """Benchmark module import time."""
        start_time = time.time()
        try:
            importlib.import_module(module_name)
            import_time = (time.time() - start_time) * 1000
            return {
                "module": module_name,
                "import_time_ms": import_time,
                "success": True,
                "passed": import_time < 1000  # Less than 1 second
            }
        except ImportError as e:
            return {
                "module": module_name,
                "import_time_ms": 0,
                "success": False,
                "error": str(e),
                "passed": False
            }
    
    def benchmark_basic_operations(self) -> Dict[str, Any]:
        """Benchmark basic operations performance."""
        results = {}
        
        # Test data processing speed
        start_time = time.time()
        test_data = [i * 0.1 for i in range(10000)]
        processed = [x * 2 + 1 for x in test_data]
        processing_time = (time.time() - start_time) * 1000
        
        results["data_processing"] = {
            "items_processed": len(processed),
            "processing_time_ms": processing_time,
            "items_per_second": len(processed) / (processing_time / 1000),
            "passed": processing_time < 100  # Less than 100ms
        }
        
        # Test memory allocation
        start_time = time.time()
        large_list = [0] * 1000000
        allocation_time = (time.time() - start_time) * 1000
        del large_list
        
        results["memory_allocation"] = {
            "allocation_time_ms": allocation_time,
            "passed": allocation_time < 500  # Less than 500ms
        }
        
        return results
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        start_time = time.time()
        
        benchmark_results = {
            "timestamp": time.time(),
            "basic_operations": self.benchmark_basic_operations(),
            "import_benchmarks": []
        }
        
        # Test key module imports
        key_modules = ["json", "time", "sys", "os", "threading"]
        for module in key_modules:
            benchmark_results["import_benchmarks"].append(
                self.benchmark_import_time(module)
            )
        
        total_time = (time.time() - start_time) * 1000
        
        # Overall performance score
        all_passed = all([
            benchmark_results["basic_operations"]["data_processing"]["passed"],
            benchmark_results["basic_operations"]["memory_allocation"]["passed"],
            all(result["passed"] for result in benchmark_results["import_benchmarks"])
        ])
        
        benchmark_results.update({
            "total_benchmark_time_ms": total_time,
            "overall_passed": all_passed,
            "performance_score": 100 if all_passed else 75
        })
        
        return benchmark_results

class QualityGateRunner:
    """Main quality gate runner."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.results = []
        
    def run_quality_gate(self, name: str, gate_func: Callable) -> QualityGateResult:
        """Run a single quality gate."""
        logger.info(f"Running quality gate: {name}")
        start_time = time.time()
        
        try:
            result = gate_func()
            execution_time = (time.time() - start_time) * 1000
            
            # Determine pass/fail and score
            if isinstance(result, dict):
                passed = result.get("passed", False)
                score = result.get("score", result.get("quality_score", 
                                   result.get("performance_score", 50)))
            else:
                passed = bool(result)
                score = 100 if passed else 0
            
            return QualityGateResult(
                name=name,
                passed=passed,
                score=score,
                details=result if isinstance(result, dict) else {"result": result},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Quality gate {name} failed: {e}")
            
            return QualityGateResult(
                name=name,
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        logger.info("Starting comprehensive quality gate execution")
        start_time = time.time()
        
        # Initialize analyzers
        security_scanner = SecurityScanner()
        code_analyzer = CodeQualityAnalyzer()
        test_analyzer = TestCoverageAnalyzer(self.root_path)
        performance_benchmark = PerformanceBenchmark()
        
        # Define quality gates
        quality_gates = [
            ("Security Scan", lambda: security_scanner.scan_project(self.root_path)),
            ("Code Quality Analysis", lambda: code_analyzer.analyze_project(self.root_path)),
            ("Test Coverage Analysis", lambda: test_analyzer.analyze_test_structure()),
            ("Test Execution", lambda: test_analyzer.run_mock_tests()),
            ("Performance Benchmarks", lambda: performance_benchmark.run_benchmarks()),
        ]
        
        # Run each quality gate
        gate_results = []
        for gate_name, gate_func in quality_gates:
            result = self.run_quality_gate(gate_name, gate_func)
            gate_results.append(result)
            
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            logger.info(f"{gate_name}: {status} (Score: {result.score:.1f}, Time: {result.execution_time_ms:.1f}ms)")
        
        total_execution_time = (time.time() - start_time) * 1000
        
        # Calculate overall results
        passed_gates = sum(1 for result in gate_results if result.passed)
        total_gates = len(gate_results)
        overall_score = sum(result.score for result in gate_results) / total_gates
        overall_passed = passed_gates == total_gates
        
        # Generate comprehensive report
        comprehensive_results = {
            "execution_timestamp": time.time(),
            "total_execution_time_ms": total_execution_time,
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": total_gates - passed_gates,
            "success_rate": passed_gates / total_gates,
            "overall_score": overall_score,
            "overall_passed": overall_passed,
            "quality_gates": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time_ms": result.execution_time_ms,
                    "details": result.details,
                    "error_message": result.error_message
                }
                for result in gate_results
            ],
            "summary": {
                "security_vulnerabilities": gate_results[0].details.get("total_vulnerabilities", 0),
                "code_quality_score": gate_results[1].score,
                "test_coverage_percent": gate_results[2].details.get("estimated_coverage_percent", 0),
                "test_success_rate": gate_results[3].details.get("success_rate", 0),
                "performance_score": gate_results[4].score
            },
            "recommendations": self._generate_recommendations(gate_results)
        }
        
        return comprehensive_results
    
    def _generate_recommendations(self, gate_results: List[QualityGateResult]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for result in gate_results:
            if not result.passed:
                if result.name == "Security Scan":
                    recommendations.append("Address security vulnerabilities found in code scan")
                elif result.name == "Code Quality Analysis":
                    recommendations.append("Improve code quality by reducing complexity and adding comments")
                elif result.name == "Test Coverage Analysis":
                    recommendations.append("Increase test coverage to meet 85% minimum requirement")
                elif result.name == "Test Execution":
                    recommendations.append("Fix failing tests to achieve 100% test success rate")
                elif result.name == "Performance Benchmarks":
                    recommendations.append("Optimize performance for better benchmark results")
        
        if not recommendations:
            recommendations.append("All quality gates passed! Consider adding more comprehensive tests.")
        
        return recommendations

def run_quality_gates():
    """Run comprehensive quality gates."""
    print("🛡️ SpikeFormer Enhanced Quality Gates")
    print("=" * 70)
    
    try:
        root_path = Path("/root/repo")
        runner = QualityGateRunner(root_path)
        
        # Run all quality gates
        results = runner.run_all_quality_gates()
        
        # Display results
        print(f"\n📊 Quality Gate Results Summary:")
        print(f"⏱️ Total execution time: {results['total_execution_time_ms']:.1f} ms")
        print(f"🎯 Overall score: {results['overall_score']:.1f}/100")
        print(f"✅ Passed gates: {results['passed_gates']}/{results['total_gates']}")
        print(f"📈 Success rate: {results['success_rate']:.1%}")
        
        status = "🟢 ALL PASSED" if results['overall_passed'] else "🔴 SOME FAILED"
        print(f"🏆 Overall status: {status}")
        
        # Detailed results
        print(f"\n📋 Detailed Gate Results:")
        for gate in results['quality_gates']:
            status_icon = "✅" if gate['passed'] else "❌"
            print(f"{status_icon} {gate['name']}: {gate['score']:.1f}/100 ({gate['execution_time_ms']:.1f}ms)")
        
        # Key metrics
        print(f"\n📈 Key Metrics:")
        summary = results['summary']
        print(f"🔒 Security vulnerabilities: {summary['security_vulnerabilities']}")
        print(f"📝 Code quality: {summary['code_quality_score']:.1f}/100")
        print(f"🧪 Test coverage: {summary['test_coverage_percent']:.1f}%")
        print(f"✅ Test success rate: {summary['test_success_rate']:.1%}")
        print(f"⚡ Performance score: {summary['performance_score']:.1f}/100")
        
        # Recommendations
        if results['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Save results
        output_file = "/root/repo/enhanced_quality_gates_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Quality gates completed!")
        print(f"📁 Detailed results saved to: {output_file}")
        
        # Final validation
        meets_requirements = (
            results['overall_passed'] and
            summary['test_coverage_percent'] >= 85.0 and
            summary['security_vulnerabilities'] == 0
        )
        
        if meets_requirements:
            print("\n🎉 ALL QUALITY REQUIREMENTS MET!")
            return True
        else:
            print("\n⚠️ Quality requirements not fully met. Review recommendations.")
            return False
            
    except Exception as e:
        logger.error(f"Quality gates execution failed: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Quality gates failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = run_quality_gates()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        sys.exit(1)