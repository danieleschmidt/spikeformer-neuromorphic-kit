#!/usr/bin/env python3
"""
Hardware Security Scanner for SpikeFormer

This script performs security checks specific to neuromorphic hardware
integration and identifies potential security risks.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Security check results
SECURITY_CHECKS = {
    "hardware_config_validation": {
        "description": "Validate hardware configuration files for security",
        "severity": "high",
        "checks": []
    },
    "privilege_escalation": {
        "description": "Check for potential privilege escalation in hardware access",
        "severity": "critical", 
        "checks": []
    },
    "data_exposure": {
        "description": "Check for potential data exposure through hardware interfaces",
        "severity": "high",
        "checks": []
    },
    "hardware_authentication": {
        "description": "Verify hardware authentication mechanisms",
        "severity": "medium",
        "checks": []
    },
    "side_channel_protection": {
        "description": "Check for side-channel attack protections",
        "severity": "medium",
        "checks": []
    }
}

def check_hardware_config_security(repo_path: Path) -> List[Dict[str, Any]]:
    """Check hardware configuration files for security issues."""
    findings = []
    
    # Check for hardcoded credentials
    config_patterns = [
        "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"
    ]
    
    suspicious_patterns = [
        "password", "secret", "key", "token", "credential",
        "api_key", "auth", "private", "cert", "ssl"
    ]
    
    for pattern in config_patterns:
        for config_file in repo_path.rglob(pattern):
            if config_file.is_file():
                try:
                    content = config_file.read_text().lower()
                    for suspicious in suspicious_patterns:
                        if suspicious in content:
                            findings.append({
                                "file": str(config_file.relative_to(repo_path)),
                                "issue": f"Potential credential in configuration: {suspicious}",
                                "severity": "high",
                                "line": "unknown"
                            })
                except Exception as e:
                    findings.append({
                        "file": str(config_file.relative_to(repo_path)),
                        "issue": f"Failed to read configuration file: {e}",
                        "severity": "low",
                        "line": "N/A"
                    })
    
    return findings

def check_privilege_escalation(repo_path: Path) -> List[Dict[str, Any]]:
    """Check for potential privilege escalation vulnerabilities."""
    findings = []
    
    # Check Python files for privilege escalation patterns
    dangerous_imports = [
        "subprocess", "os.system", "eval", "exec", "compile",
        "importlib", "__import__"
    ]
    
    dangerous_functions = [
        "system(", "popen(", "spawn(", "# exec() removed for security", "# eval() removed for security",
        "compile(", "setuid(", "setgid(", "chmod("
    ]
    
    for py_file in repo_path.rglob("*.py"):
        if py_file.is_file():
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    # Check for dangerous imports
                    for danger_import in dangerous_imports:
                        if f"import {danger_import}" in line_lower:
                            findings.append({
                                "file": str(py_file.relative_to(repo_path)),
                                "issue": f"Potentially dangerous import: {danger_import}",
                                "severity": "medium",
                                "line": i
                            })
                    
                    # Check for dangerous function calls
                    for danger_func in dangerous_functions:
                        if danger_func in line_lower:
                            findings.append({
                                "file": str(py_file.relative_to(repo_path)),
                                "issue": f"Potentially dangerous function call: {danger_func}",
                                "severity": "high", 
                                "line": i
                            })
                            
            except Exception as e:
                findings.append({
                    "file": str(py_file.relative_to(repo_path)),
                    "issue": f"Failed to analyze file: {e}",
                    "severity": "low",
                    "line": "N/A"
                })
    
    return findings

def check_data_exposure(repo_path: Path) -> List[Dict[str, Any]]:
    """Check for potential data exposure through hardware interfaces."""
    findings = []
    
    # Check for data logging without sanitization
    logging_patterns = [
        "print(", "log.", "logger.", "logging.", "debug(",
        "info(", "warn(", "error(", "exception("
    ]
    
    sensitive_data_patterns = [
        "password", "token", "key", "secret", "credential",
        "private", "sensitive", "confidential"
    ]
    
    for py_file in repo_path.rglob("*.py"):
        if py_file.is_file():
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    # Check for logging of sensitive data
                    has_logging = any(pattern in line_lower for pattern in logging_patterns)
                    has_sensitive = any(pattern in line_lower for pattern in sensitive_data_patterns)
                    
                    if has_logging and has_sensitive:
                        findings.append({
                            "file": str(py_file.relative_to(repo_path)),
                            "issue": "Potential logging of sensitive data",
                            "severity": "high",
                            "line": i
                        })
                        
            except Exception as e:
                findings.append({
                    "file": str(py_file.relative_to(repo_path)),
                    "issue": f"Failed to analyze file: {e}",
                    "severity": "low",
                    "line": "N/A"
                })
    
    return findings

def check_hardware_authentication(repo_path: Path) -> List[Dict[str, Any]]:
    """Check hardware authentication mechanisms."""
    findings = []
    
    # Check for hardware authentication in code
    auth_patterns = [
        "authenticate", "verify", "validate", "authorize",
        "certificate", "signature", "handshake"
    ]
    
    hardware_files = []
    for hw_dir in ["hardware", "loihi2", "spinnaker", "edge"]:
        hw_path = repo_path / "spikeformer" / hw_dir
        if hw_path.exists():
            hardware_files.extend(hw_path.rglob("*.py"))
    
    for py_file in hardware_files:
        if py_file.is_file():
            try:
                content = py_file.read_text()
                lines = content.split('\n')
                
                has_auth = False
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    if any(pattern in line_lower for pattern in auth_patterns):
                        has_auth = True
                        break
                
                if not has_auth:
                    findings.append({
                        "file": str(py_file.relative_to(repo_path)),
                        "issue": "Hardware interface may lack authentication",
                        "severity": "medium",
                        "line": "N/A"
                    })
                    
            except Exception as e:
                findings.append({
                    "file": str(py_file.relative_to(repo_path)),
                    "issue": f"Failed to analyze file: {e}",
                    "severity": "low",
                    "line": "N/A"
                })
    
    return findings

def check_side_channel_protection(repo_path: Path) -> List[Dict[str, Any]]:
    """Check for side-channel attack protections."""
    findings = []
    
    # Check for timing attack protections
    timing_patterns = [
        "time.sleep", "constant_time", "secure_compare",
        "timing_safe", "random.randint"
    ]
    
    crypto_files = []
    for pattern in ["*crypto*", "*security*", "*auth*", "*hash*"]:
        crypto_files.extend(repo_path.rglob(pattern))
    
    for py_file in crypto_files:
        if py_file.is_file() and py_file.suffix == ".py":
            try:
                content = py_file.read_text()
                
                has_timing_protection = any(
                    pattern in content for pattern in timing_patterns
                )
                
                # Check for comparison operations that might be timing vulnerable
                if "==" in content and not has_timing_protection:
                    findings.append({
                        "file": str(py_file.relative_to(repo_path)),
                        "issue": "Potential timing attack vulnerability in comparisons",
                        "severity": "medium",
                        "line": "N/A"
                    })
                    
            except Exception as e:
                findings.append({
                    "file": str(py_file.relative_to(repo_path)),
                    "issue": f"Failed to analyze file: {e}",
                    "severity": "low",
                    "line": "N/A"
                })
    
    return findings

def generate_security_report(repo_path: Path) -> Dict[str, Any]:
    """Generate comprehensive hardware security report."""
    report = {
        "scan_timestamp": datetime.utcnow().isoformat(),
        "repository_path": str(repo_path),
        "scanner_version": "1.0.0",
        "summary": {
            "total_checks": len(SECURITY_CHECKS),
            "total_findings": 0,
            "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        },
        "checks": {}
    }
    
    # Run all security checks
    check_functions = {
        "hardware_config_validation": check_hardware_config_security,
        "privilege_escalation": check_privilege_escalation,
        "data_exposure": check_data_exposure,
        "hardware_authentication": check_hardware_authentication,
        "side_channel_protection": check_side_channel_protection
    }
    
    for check_name, check_func in check_functions.items():
        print(f"Running {check_name}...")
        findings = check_func(repo_path)
        
        report["checks"][check_name] = {
            "description": SECURITY_CHECKS[check_name]["description"],
            "severity": SECURITY_CHECKS[check_name]["severity"],
            "findings_count": len(findings),
            "findings": findings
        }
        
        # Update summary
        report["summary"]["total_findings"] += len(findings)
        for finding in findings:
            severity = finding.get("severity", "low")
            report["summary"]["severity_counts"][severity] += 1
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description="Hardware Security Scanner for SpikeFormer"
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path.cwd(),
        help="Path to repository root (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="hardware-security-report.json",
        help="Output file for security report"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format"
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with error code if critical issues found"
    )
    
    args = parser.parse_args()
    
    # Validate repository path
    if not args.repo_path.exists():
        print(f"Error: Repository path does not exist: {args.repo_path}")
        sys.exit(1)
    
    print(f"🔍 Starting hardware security scan of {args.repo_path}")
    
    # Generate security report
    report = generate_security_report(args.repo_path)
    
    # Output report
    if args.format == "json":
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Security report saved to {args.output}")
    else:
        # Text format output
        print("\n" + "="*60)
        print("HARDWARE SECURITY SCAN REPORT")
        print("="*60)
        print(f"Scan Timestamp: {report['scan_timestamp']}")
        print(f"Repository: {report['repository_path']}")
        print(f"Total Findings: {report['summary']['total_findings']}")
        print(f"Critical: {report['summary']['severity_counts']['critical']}")
        print(f"High: {report['summary']['severity_counts']['high']}")
        print(f"Medium: {report['summary']['severity_counts']['medium']}")
        print(f"Low: {report['summary']['severity_counts']['low']}")
        
        for check_name, check_data in report["checks"].items():
            print(f"\n{check_name.upper()}")
            print("-" * 40)
            print(f"Description: {check_data['description']}")
            print(f"Findings: {check_data['findings_count']}")
            
            for finding in check_data["findings"]:
                print(f"  • [{finding['severity'].upper()}] {finding['file']}:{finding['line']}")
                print(f"    {finding['issue']}")
    
    # Check for critical issues
    critical_count = report["summary"]["severity_counts"]["critical"]
    if args.fail_on_critical and critical_count > 0:
        print(f"\n❌ {critical_count} critical security issues found!")
        sys.exit(1)
    
    print(f"\n✅ Hardware security scan completed")
    print(f"Total findings: {report['summary']['total_findings']}")

if __name__ == "__main__":
    main()