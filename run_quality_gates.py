#!/usr/bin/env python3
"""Quality gates for neuromorphic computing research and deployment."""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: Optional[float]
    threshold: Optional[float]
    message: str
    details: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None


class QualityGate:
    """Base class for quality gate implementations."""
    
    def __init__(self, name: str, threshold: Optional[float] = None):
        self.name = name
        self.threshold = threshold
        self.logger = logging.getLogger(f"QualityGate.{name}")
    
    def check(self, project_path: Path) -> QualityGateResult:
        """Execute the quality gate check."""
        start_time = time.time()
        
        try:
            score, details = self._execute_check(project_path)
            passed = self._evaluate_result(score)
            message = self._generate_message(score, passed)
            
            return QualityGateResult(
                gate_name=self.name,
                passed=passed,
                score=score,
                threshold=self.threshold,
                message=message,
                details=details,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Quality gate {self.name} failed: {e}")
            return QualityGateResult(
                gate_name=self.name,
                passed=False,
                score=None,
                threshold=self.threshold,
                message=f"Quality gate execution failed: {str(e)}",
                details={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _execute_check(self, project_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
        """Execute the specific check logic. Override in subclasses."""
        raise NotImplementedError
    
    def _evaluate_result(self, score: Optional[float]) -> bool:
        """Evaluate if the result passes the quality gate."""
        if score is None or self.threshold is None:
            return True
        return score >= self.threshold
    
    def _generate_message(self, score: Optional[float], passed: bool) -> str:
        """Generate human-readable message for the result."""
        status = "PASSED" if passed else "FAILED"
        if score is not None and self.threshold is not None:
            return f"{self.name}: {status} (Score: {score:.3f}, Threshold: {self.threshold})"
        else:
            return f"{self.name}: {status}"


class CodeQualityGate(QualityGate):
    """Quality gate for code quality metrics."""
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("Code Quality", threshold)
    
    def _execute_check(self, project_path: Path) -> Tuple[float, Dict[str, Any]]:
        """Check code quality using various tools."""
        details = {}
        total_score = 0.0
        max_score = 0.0
        
        # Check if we can run linting tools
        python_files = list(project_path.rglob("*.py"))
        if not python_files:
            return 1.0, {"message": "No Python files found"}
        
        details["python_files_count"] = len(python_files)
        
        # Simulate code quality checks (since we can't run actual tools)
        
        # 1. Basic syntax check (weight: 0.4)
        syntax_score = self._check_python_syntax(python_files)
        details["syntax_score"] = syntax_score
        total_score += syntax_score * 0.4
        max_score += 0.4
        
        # 2. Import structure check (weight: 0.3)
        import_score = self._check_imports(python_files)
        details["import_score"] = import_score
        total_score += import_score * 0.3
        max_score += 0.3
        
        # 3. Documentation check (weight: 0.3)
        doc_score = self._check_documentation(python_files)
        details["documentation_score"] = doc_score
        total_score += doc_score * 0.3
        max_score += 0.3
        
        final_score = total_score / max_score if max_score > 0 else 0.0
        details["final_score"] = final_score
        
        return final_score, details
    
    def _check_python_syntax(self, python_files: List[Path]) -> float:
        """Check Python syntax validity."""
        valid_files = 0
        total_files = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(file_path), 'exec')
                valid_files += 1
            except SyntaxError:
                continue
            except Exception:
                continue
        
        return valid_files / total_files if total_files > 0 else 0.0
    
    def _check_imports(self, python_files: List[Path]) -> float:
        """Check import organization and validity."""
        good_import_files = 0
        total_files = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                import_lines = [line.strip() for line in lines if line.strip().startswith(('import ', 'from '))]
                
                # Simple heuristics for good imports
                has_good_structure = True
                
                # Check for relative imports in package
                if 'spikeformer/' in str(file_path):
                    relative_imports = [line for line in import_lines if line.startswith('from .')]
                    if relative_imports:
                        has_good_structure = True
                
                # Check for reasonable import count
                if len(import_lines) > 50:  # Too many imports
                    has_good_structure = False
                
                if has_good_structure:
                    good_import_files += 1
                    
            except Exception:
                continue
        
        return good_import_files / total_files if total_files > 0 else 0.0
    
    def _check_documentation(self, python_files: List[Path]) -> float:
        """Check documentation coverage."""
        documented_files = 0
        total_files = len(python_files)
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for docstrings
                has_module_docstring = '"""' in content[:1000]  # Check first 1000 chars
                has_function_docs = 'def ' in content and '"""' in content
                has_class_docs = 'class ' in content and '"""' in content
                
                if has_module_docstring or has_function_docs or has_class_docs:
                    documented_files += 1
                    
            except Exception:
                continue
        
        return documented_files / total_files if total_files > 0 else 0.0


class ResearchQualityGate(QualityGate):
    """Quality gate for research-specific requirements."""
    
    def __init__(self, threshold: float = 0.8):
        super().__init__("Research Quality", threshold)
    
    def _execute_check(self, project_path: Path) -> Tuple[float, Dict[str, Any]]:
        """Check research quality indicators."""
        details = {}
        research_score = 0.0
        max_score = 0.0
        
        # Check for research documentation
        research_docs = list(project_path.rglob("*.md")) + list(project_path.rglob("*.rst"))
        details["documentation_files"] = len(research_docs)
        
        if research_docs:
            research_score += 0.2
        max_score += 0.2
        
        # Check for benchmarking code
        benchmark_files = [f for f in project_path.rglob("*.py") if 'benchmark' in str(f).lower()]
        details["benchmark_files"] = len(benchmark_files)
        
        if benchmark_files:
            research_score += 0.2
        max_score += 0.2
        
        # Check for experimental code
        exp_indicators = ['experiment', 'eval', 'metric', 'result', 'analysis']
        has_experiments = False
        
        for py_file in project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(indicator in content for indicator in exp_indicators):
                        has_experiments = True
                        break
            except:
                continue
        
        if has_experiments:
            research_score += 0.2
        max_score += 0.2
        details["has_experiments"] = has_experiments
        
        # Check for data handling
        data_files = [f for f in project_path.rglob("*.py") if any(word in str(f).lower() for word in ['data', 'dataset', 'loader'])]
        details["data_files"] = len(data_files)
        
        if data_files:
            research_score += 0.2
        max_score += 0.2
        
        # Check for reproducibility features
        repro_files = [f for f in project_path.rglob("*.py") if any(word in str(f).lower() for word in ['config', 'seed', 'random'])]
        details["reproducibility_files"] = len(repro_files)
        
        if repro_files:
            research_score += 0.2
        max_score += 0.2
        
        final_score = research_score / max_score if max_score > 0 else 0.0
        details["final_score"] = final_score
        
        return final_score, details


class QualityGateRunner:
    """Main runner for all quality gates."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.logger = logging.getLogger("QualityGateRunner")
        
        # Initialize quality gates
        self.gates = [
            CodeQualityGate(threshold=0.8),
            ResearchQualityGate(threshold=0.8),
        ]
    
    def run_all_gates(self) -> List[QualityGateResult]:
        """Run all quality gates."""
        results = []
        
        self.logger.info(f"Running {len(self.gates)} quality gates on {self.project_path}")
        
        for gate in self.gates:
            self.logger.info(f"Running quality gate: {gate.name}")
            result = gate.check(self.project_path)
            results.append(result)
            
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            self.logger.info(f"{gate.name}: {status}")
            
            if result.error_message:
                self.logger.error(f"{gate.name} error: {result.error_message}")
        
        return results
    
    def generate_report(self, results: List[QualityGateResult]) -> Dict[str, Any]:
        """Generate comprehensive quality gate report."""
        passed_gates = [r for r in results if r.passed]
        failed_gates = [r for r in results if not r.passed]
        
        overall_pass_rate = len(passed_gates) / len(results) if results else 0.0
        
        report = {
            "summary": {
                "total_gates": len(results),
                "passed": len(passed_gates),
                "failed": len(failed_gates),
                "pass_rate": overall_pass_rate,
                "overall_status": "PASSED" if overall_pass_rate >= 0.8 else "FAILED"
            },
            "gates": [
                {
                    "name": result.gate_name,
                    "passed": result.passed,
                    "score": result.score,
                    "threshold": result.threshold,
                    "message": result.message,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "error": result.error_message
                }
                for result in results
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return report


def main():
    """Main entry point for quality gate runner."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get project path from command line or use current directory
    project_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    if not project_path.exists():
        print(f"Error: Project path {project_path} does not exist")
        sys.exit(1)
    
    # Run quality gates
    runner = QualityGateRunner(project_path)
    results = runner.run_all_gates()
    
    # Generate and print report
    report = runner.generate_report(results)
    print("\n" + "="*60)
    print("QUALITY GATES SUMMARY")
    print("="*60)
    print(f"Total Gates: {report['summary']['total_gates']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    
    print("\nDetailed Results:")
    for gate_result in report['gates']:
        status = "✓ PASSED" if gate_result['passed'] else "✗ FAILED"
        score_info = f" (Score: {gate_result['score']:.3f})" if gate_result['score'] is not None else ""
        print(f"  {status} {gate_result['name']}{score_info}")
    
    # Exit with non-zero code if gates failed
    if report['summary']['overall_status'] == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()