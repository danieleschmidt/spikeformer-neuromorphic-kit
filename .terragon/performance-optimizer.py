#!/usr/bin/env python3
"""
Terragon Autonomous Performance Optimizer
Analyzes code for performance opportunities and suggests optimizations
"""

import ast
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
import re

class PerformanceAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code for performance opportunities."""
    
    def __init__(self):
        self.issues = []
        self.current_file = ""
        self.function_complexity = {}
        self.current_function = ""
        
    def analyze_file(self, file_path: Path) -> List[Dict]:
        """Analyze a Python file for performance issues."""
        self.current_file = str(file_path)
        self.issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            self.visit(tree)
            
        except SyntaxError as e:
            self.issues.append({
                "type": "syntax_error",
                "severity": "high",
                "line": e.lineno,
                "message": f"Syntax error: {e.msg}",
                "wsjf_score": 5.0
            })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        return self.issues
    
    def visit_For(self, node):
        """Detect potentially inefficient loops."""
        # Check for nested loops
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)) and child != node:
                self.issues.append({
                    "type": "nested_loop",
                    "severity": "medium",
                    "line": node.lineno,
                    "message": "Nested loop detected - consider vectorization or algorithmic optimization",
                    "wsjf_score": self._calculate_performance_wsjf("nested_loop", node.lineno),
                    "suggestion": "Consider using NumPy vectorization or DataFrame operations for better performance"
                })
                
        # Check for inefficient string concatenation in loops
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                if isinstance(child.target, ast.Name):
                    self.issues.append({
                        "type": "string_concat_loop",
                        "severity": "medium", 
                        "line": child.lineno,
                        "message": "String concatenation in loop - use join() instead",
                        "wsjf_score": self._calculate_performance_wsjf("string_concat", child.lineno),
                        "suggestion": "Use ''.join(list) or f-strings for better performance"
                    })
        
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        """Analyze list comprehensions for optimization opportunities."""
        # Check for nested list comprehensions
        for comp in ast.walk(node):
            if isinstance(comp, (ast.ListComp, ast.SetComp, ast.DictComp)) and comp != node:
                self.issues.append({
                    "type": "nested_comprehension",
                    "severity": "medium",
                    "line": node.lineno,
                    "message": "Nested comprehension - consider generator expressions or NumPy operations",
                    "wsjf_score": self._calculate_performance_wsjf("nested_comprehension", node.lineno),
                    "suggestion": "Use generator expressions for memory efficiency or NumPy for numerical operations"
                })
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Detect potentially expensive function calls."""
        if isinstance(node.func, ast.Attribute):
            # Check for inefficient pandas operations
            if hasattr(node.func, 'attr'):
                if node.func.attr in ['apply', 'iterrows', 'itertuples']:
                    self.issues.append({
                        "type": "inefficient_pandas",
                        "severity": "high",
                        "line": node.lineno,
                        "message": f"Inefficient pandas operation: {node.func.attr}",
                        "wsjf_score": self._calculate_performance_wsjf("pandas_inefficient", node.lineno),
                        "suggestion": "Use vectorized operations instead of apply() or iteration"
                    })
                    
        # Check for repeated expensive operations
        if isinstance(node.func, ast.Name):
            expensive_funcs = ['len', 'sorted', 'min', 'max']
            if node.func.id in expensive_funcs:
                # This is a simplified check - would need more context in real implementation
                pass
                
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Analyze function definitions for complexity and performance issues."""
        self.current_function = node.name
        
        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)
        self.function_complexity[node.name] = complexity
        
        if complexity > 15:
            self.issues.append({
                "type": "high_complexity",
                "severity": "medium",
                "line": node.lineno,
                "message": f"Function '{node.name}' has high complexity ({complexity})",
                "wsjf_score": self._calculate_performance_wsjf("complexity", node.lineno, complexity),
                "suggestion": "Consider breaking this function into smaller, more focused functions"
            })
            
        # Check for missing type hints (affects performance with mypy/type checkers)
        if not node.returns and len(node.args.args) > 0:
            missing_hints = [arg.arg for arg in node.args.args if not arg.annotation]
            if missing_hints:
                self.issues.append({
                    "type": "missing_type_hints",
                    "severity": "low",
                    "line": node.lineno,
                    "message": f"Function '{node.name}' missing type hints",
                    "wsjf_score": self._calculate_performance_wsjf("type_hints", node.lineno),
                    "suggestion": "Add type hints for better static analysis and potential performance gains"
                })
        
        self.generic_visit(node)
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def _calculate_performance_wsjf(self, issue_type: str, line: int, complexity: int = 0) -> float:
        """Calculate WSJF score for performance issues."""
        # Base impact scores
        impact_scores = {
            "nested_loop": 8,
            "string_concat": 6,
            "nested_comprehension": 5,
            "pandas_inefficient": 9,
            "complexity": min(10, complexity // 2),
            "type_hints": 2
        }
        
        user_business_value = impact_scores.get(issue_type, 3)
        time_criticality = 3  # Performance issues accumulate over time
        risk_reduction = 4    # Performance issues can impact user experience
        opportunity_enablement = 2  # Better performance enables other features
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        
        # Job size (effort to fix)
        effort_estimates = {
            "nested_loop": 4,
            "string_concat": 1,
            "nested_comprehension": 3,
            "pandas_inefficient": 2,
            "complexity": min(8, complexity // 3),
            "type_hints": 1
        }
        
        job_size = effort_estimates.get(issue_type, 2)
        
        return round(cost_of_delay / job_size, 2)

class PerformanceOptimizer:
    """Main performance optimization system."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.results_path = repo_path / ".terragon" / "performance-analysis.json"
        
    def analyze_repository(self) -> Dict:
        """Analyze entire repository for performance opportunities."""
        print("⚡ Analyzing repository for performance opportunities...")
        
        results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_files": 0,
                "total_issues": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0
            },
            "files": {},
            "backlog_items": []
        }
        
        analyzer = PerformanceAnalyzer()
        
        # Find all Python files
        python_files = list(self.repo_path.rglob("*.py"))
        
        # Exclude test files and virtual environments
        python_files = [
            f for f in python_files 
            if not any(part in str(f) for part in ['test', '__pycache__', '.venv', 'venv'])
        ]
        
        results["summary"]["total_files"] = len(python_files)
        
        for py_file in python_files:
            try:
                issues = analyzer.analyze_file(py_file)
                if issues:
                    rel_path = str(py_file.relative_to(self.repo_path))
                    results["files"][rel_path] = issues
                    
                    for issue in issues:
                        results["summary"]["total_issues"] += 1
                        severity = issue.get("severity", "low")
                        if severity == "high":
                            results["summary"]["high_priority"] += 1
                        elif severity == "medium":
                            results["summary"]["medium_priority"] += 1
                        else:
                            results["summary"]["low_priority"] += 1
                            
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        # Generate backlog items
        results["backlog_items"] = self._generate_backlog_items(results)
        
        # Add dependency performance analysis
        results["dependency_analysis"] = self._analyze_dependencies()
        
        return results
    
    def _generate_backlog_items(self, analysis_results: Dict) -> List[Dict]:
        """Generate prioritized backlog items from analysis."""
        items = []
        item_id = 1
        
        # Collect all issues and sort by WSJF score
        all_issues = []
        for file_path, issues in analysis_results["files"].items():
            for issue in issues:
                issue["file"] = file_path
                all_issues.append(issue)
        
        all_issues.sort(key=lambda x: x.get("wsjf_score", 0), reverse=True)
        
        # Generate backlog items for top issues
        for issue in all_issues[:10]:  # Top 10 performance issues
            items.append({
                "id": f"PERF-OPT-{item_id:03d}",
                "title": f"Optimize {issue['type'].replace('_', ' ')} in {Path(issue['file']).name}",
                "description": issue["message"],
                "category": "Performance",
                "subcategory": issue["type"],
                "wsjf_score": issue["wsjf_score"],
                "estimated_hours": self._estimate_fix_effort(issue["type"]),
                "priority": issue["severity"],
                "file": issue["file"],
                "line": issue.get("line", 0),
                "suggestion": issue.get("suggestion", "")
            })
            item_id += 1
            
        return items
    
    def _estimate_fix_effort(self, issue_type: str) -> float:
        """Estimate effort in hours to fix performance issue."""
        effort_map = {
            "nested_loop": 2.0,
            "string_concat_loop": 0.5,
            "nested_comprehension": 1.5,
            "inefficient_pandas": 1.0,
            "high_complexity": 4.0,
            "missing_type_hints": 0.5
        }
        return effort_map.get(issue_type, 1.0)
    
    def _analyze_dependencies(self) -> Dict:
        """Analyze dependencies for performance opportunities."""
        analysis = {
            "slow_imports": [],
            "heavy_dependencies": [],
            "optimization_opportunities": []
        }
        
        # Check for potentially slow imports
        heavy_packages = [
            "tensorflow", "torch", "transformers", "scipy", "sklearn",
            "pandas", "numpy", "matplotlib", "seaborn"
        ]
        
        try:
            with open(self.repo_path / "requirements.txt", "r") as f:
                requirements = f.read()
                
            for package in heavy_packages:
                if package in requirements:
                    analysis["heavy_dependencies"].append({
                        "package": package,
                        "suggestion": f"Consider lazy loading {package} to improve startup time"
                    })
                    
        except FileNotFoundError:
            pass
            
        return analysis
    
    def save_results(self, results: Dict) -> None:
        """Save analysis results."""
        self.results_path.parent.mkdir(exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
    
    def run_analysis(self) -> Dict:
        """Run complete performance analysis."""
        results = self.analyze_repository()
        self.save_results(results)
        
        print(f"✅ Performance analysis complete:")
        print(f"   Files analyzed: {results['summary']['total_files']}")
        print(f"   Issues found: {results['summary']['total_issues']}")
        print(f"   High priority: {results['summary']['high_priority']}")
        print(f"   Medium priority: {results['summary']['medium_priority']}")
        print(f"   Low priority: {results['summary']['low_priority']}")
        
        return results

def main():
    """Main entry point."""
    optimizer = PerformanceOptimizer()
    results = optimizer.run_analysis()
    
    # Print top 3 performance opportunities
    if results["backlog_items"]:
        print("\n⚡ Top 3 performance optimization opportunities:")
        for i, item in enumerate(results["backlog_items"][:3], 1):
            print(f"{i}. {item['title']} (WSJF: {item['wsjf_score']})")
            print(f"   {item['description']}")
            if item['suggestion']:
                print(f"   💡 {item['suggestion']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())