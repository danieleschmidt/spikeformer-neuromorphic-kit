#!/usr/bin/env python3
"""Comprehensive security scanning script for SpikeFormer."""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any


class SecurityScanner:
    """Comprehensive security scanner for the codebase."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results = {}
        
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scan."""
        print("🔍 Running Bandit security scan...")
        
        try:
            cmd = [
                "bandit", 
                "-r", str(self.repo_root / "spikeformer"),
                "-f", "json",
                "-o", str(self.repo_root / "bandit-report.json"),
                "--severity-level", "medium",
                "--confidence-level", "medium"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Bandit scan completed successfully")
                return {"status": "success", "issues": 0}
            else:
                # Bandit returns non-zero when issues are found
                try:
                    with open(self.repo_root / "bandit-report.json", 'r') as f:
                        report = json.load(f)
                        issue_count = len(report.get("results", []))
                        print(f"⚠️  Bandit found {issue_count} security issues")
                        return {"status": "issues_found", "issues": issue_count, "report": report}
                except Exception as e:
                    print(f"❌ Failed to parse Bandit report: {e}")
                    return {"status": "error", "error": str(e)}
                    
        except FileNotFoundError:
            print("❌ Bandit not installed. Install with: pip install bandit")
            return {"status": "error", "error": "Bandit not found"}
        except Exception as e:
            print(f"❌ Bandit scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety dependency vulnerability scan."""
        print("🔍 Running Safety dependency scan...")
        
        try:
            cmd = ["safety", "check", "--json", "--output", "safety-report.json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                print("✅ Safety scan completed - no known vulnerabilities")
                return {"status": "success", "vulnerabilities": 0}
            else:
                try:
                    with open(self.repo_root / "safety-report.json", 'r') as f:
                        report = json.load(f)
                        vuln_count = len(report)
                        print(f"⚠️  Safety found {vuln_count} vulnerabilities")
                        return {"status": "vulnerabilities_found", "vulnerabilities": vuln_count, "report": report}
                except Exception as e:
                    print(f"❌ Failed to parse Safety report: {e}")
                    return {"status": "error", "error": str(e)}
                    
        except FileNotFoundError:
            print("❌ Safety not installed. Install with: pip install safety")
            return {"status": "error", "error": "Safety not found"}
        except Exception as e:
            print(f"❌ Safety scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_pip_audit_scan(self) -> Dict[str, Any]:
        """Run pip-audit for additional dependency scanning."""
        print("🔍 Running pip-audit scan...")
        
        try:
            cmd = ["pip-audit", "--format=json", "--output=pip-audit-report.json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0:
                print("✅ pip-audit scan completed - no vulnerabilities")
                return {"status": "success", "vulnerabilities": 0}
            else:
                try:
                    with open(self.repo_root / "pip-audit-report.json", 'r') as f:
                        report = json.load(f)
                        vuln_count = len(report.get("vulnerabilities", []))
                        print(f"⚠️  pip-audit found {vuln_count} vulnerabilities")
                        return {"status": "vulnerabilities_found", "vulnerabilities": vuln_count, "report": report}
                except Exception as e:
                    print(f"❌ Failed to parse pip-audit report: {e}")
                    return {"status": "error", "error": str(e)}
                    
        except FileNotFoundError:
            print("⚠️  pip-audit not installed. Install with: pip install pip-audit")
            return {"status": "skipped", "reason": "pip-audit not available"}
        except Exception as e:
            print(f"❌ pip-audit scan failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_secrets(self) -> Dict[str, Any]:
        """Check for potential secrets in the codebase."""
        print("🔍 Checking for potential secrets...")
        
        # Simple regex patterns for common secrets
        secret_patterns = [
            (r'api[_-]?key[_-]?[=:]?\s*["\']?([a-zA-Z0-9]{20,})', "API Key"),
            (r'secret[_-]?key[_-]?[=:]?\s*["\']?([a-zA-Z0-9]{20,})', "Secret Key"),
            (r'password[_-]?[=:]?\s*["\']?([a-zA-Z0-9]{8,})', "Password"),
            (r'token[_-]?[=:]?\s*["\']?([a-zA-Z0-9]{20,})', "Token"),
            (r'-----BEGIN RSA PRIVATE KEY-----', "RSA Private Key"),
            (r'-----BEGIN DSA PRIVATE KEY-----', "DSA Private Key"),
            (r'-----BEGIN EC PRIVATE KEY-----', "EC Private Key"),
        ]
        
        issues = []
        
        # Scan Python files
        for py_file in self.repo_root.rglob("*.py"):
            if any(exclude in str(py_file) for exclude in [".venv", "venv", "__pycache__", ".git"]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.splitlines(), 1):
                    for pattern, secret_type in secret_patterns:
                        import re
                        if re.search(pattern, line, re.IGNORECASE):
                            issues.append({
                                "file": str(py_file.relative_to(self.repo_root)),
                                "line": line_num,
                                "type": secret_type,
                                "content": line.strip()[:100] + "..." if len(line.strip()) > 100 else line.strip()
                            })
            except Exception as e:
                print(f"⚠️  Could not scan {py_file}: {e}")
        
        if issues:
            print(f"⚠️  Found {len(issues)} potential secrets")
            return {"status": "secrets_found", "secrets": len(issues), "details": issues}
        else:
            print("✅ No obvious secrets detected")
            return {"status": "success", "secrets": 0}
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check for files with overly permissive permissions."""
        print("🔍 Checking file permissions...")
        
        issues = []
        
        # Check for world-writable files
        for file_path in self.repo_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = stat.st_mode
                    
                    # Check if world-writable (dangerous)
                    if mode & 0o002:
                        issues.append({
                            "file": str(file_path.relative_to(self.repo_root)),
                            "issue": "World-writable",
                            "permissions": oct(mode)
                        })
                    
                    # Check for executable files that shouldn't be
                    if file_path.suffix in ['.py', '.txt', '.md', '.json', '.yml', '.yaml']:
                        if mode & 0o111:  # Any execute bit set
                            issues.append({
                                "file": str(file_path.relative_to(self.repo_root)),
                                "issue": "Unnecessarily executable",
                                "permissions": oct(mode)
                            })
                            
                except Exception as e:
                    continue
        
        if issues:
            print(f"⚠️  Found {len(issues)} permission issues")
            return {"status": "issues_found", "issues": len(issues), "details": issues}
        else:
            print("✅ File permissions look good")
            return {"status": "success", "issues": 0}
    
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans and return comprehensive results."""
        print("🛡️  Running comprehensive security scan...\n")
        
        results = {
            "bandit": self.run_bandit_scan(),
            "safety": self.run_safety_scan(),
            "pip_audit": self.run_pip_audit_scan(),
            "secrets": self.check_secrets(),
            "permissions": self.check_file_permissions()
        }
        
        # Generate summary
        total_issues = 0
        critical_issues = 0
        
        for scan_name, scan_result in results.items():
            if scan_result["status"] in ["issues_found", "vulnerabilities_found", "secrets_found"]:
                if scan_name == "bandit":
                    total_issues += scan_result.get("issues", 0)
                elif scan_name in ["safety", "pip_audit"]:
                    vulns = scan_result.get("vulnerabilities", 0)
                    total_issues += vulns
                    critical_issues += vulns  # Treat dependency vulns as critical
                elif scan_name == "secrets":
                    secrets = scan_result.get("secrets", 0)
                    total_issues += secrets
                    critical_issues += secrets  # Treat secrets as critical
                elif scan_name == "permissions":
                    total_issues += scan_result.get("issues", 0)
        
        summary = {
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "overall_status": "PASS" if critical_issues == 0 else "FAIL",
            "recommendations": []
        }
        
        # Add recommendations
        if results["bandit"]["status"] == "issues_found":
            summary["recommendations"].append("Review Bandit security findings and address high-confidence issues")
        
        if results["safety"]["status"] == "vulnerabilities_found":
            summary["recommendations"].append("Update dependencies to fix known vulnerabilities")
            
        if results["secrets"]["status"] == "secrets_found":
            summary["recommendations"].append("Remove or secure potential secrets found in code")
        
        if results["permissions"]["status"] == "issues_found":
            summary["recommendations"].append("Fix file permission issues")
        
        results["summary"] = summary
        
        print(f"\n📊 Security Scan Summary:")
        print(f"   Overall Status: {summary['overall_status']}")
        print(f"   Total Issues: {summary['total_issues']}")
        print(f"   Critical Issues: {summary['critical_issues']}")
        
        if summary["recommendations"]:
            print(f"\n📋 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"   • {rec}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run security scans on SpikeFormer codebase")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--fail-on-issues", action="store_true", 
                       help="Exit with non-zero code if issues found")
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    scanner = SecurityScanner(repo_root)
    
    results = scanner.run_all_scans()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Exit with appropriate code
    if args.fail_on_issues and results["summary"]["critical_issues"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()