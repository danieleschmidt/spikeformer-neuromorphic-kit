#!/usr/bin/env python3
"""
Container security scanning script using multiple tools.
Performs comprehensive security analysis of Docker images.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class ContainerSecurityScanner:
    """Container security scanner using multiple tools."""
    
    def __init__(self, image_name: str = "spikeformer:latest"):
        self.image_name = image_name
        self.results_dir = Path("security-reports")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    def check_tool_availability(self, tool: str) -> bool:
        """Check if a security tool is available."""
        try:
            subprocess.run([tool, "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def run_trivy_scan(self) -> Optional[Dict[str, Any]]:
        """Run Trivy vulnerability scanner."""
        if not self.check_tool_availability("trivy"):
            print("⚠️  Trivy not available - skipping vulnerability scan")
            return None
        
        print("🔍 Running Trivy vulnerability scan...")
        
        output_file = self.results_dir / f"trivy-{self.timestamp}.json"
        
        try:
            # Scan for vulnerabilities
            cmd = [
                "trivy", "image",
                "--format", "json",
                "--output", str(output_file),
                "--severity", "HIGH,CRITICAL",
                "--ignore-unfixed",
                self.image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                with open(output_file) as f:
                    trivy_data = json.load(f)
                
                print(f"✓ Trivy scan completed: {output_file}")
                return trivy_data
            else:
                print(f"❌ Trivy scan failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Trivy scan error: {e}")
            return None
    
    def run_docker_bench_security(self) -> Optional[Dict[str, Any]]:
        """Run Docker Bench Security checks."""
        print("🔍 Running Docker Bench Security checks...")
        
        output_file = self.results_dir / f"docker-bench-{self.timestamp}.json"
        
        try:
            # Check if docker-bench-security is available
            bench_script = "/usr/local/bin/docker-bench-security.sh"
            if not os.path.exists(bench_script):
                print("⚠️  Docker Bench Security not available - skipping")
                return None
            
            cmd = [bench_script, "-l", str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                with open(output_file) as f:
                    bench_data = json.load(f)
                
                print(f"✓ Docker Bench Security completed: {output_file}")
                return bench_data
            else:
                print("⚠️  Docker Bench Security scan incomplete")
                return None
                
        except Exception as e:
            print(f"❌ Docker Bench Security error: {e}")
            return None
    
    def run_hadolint_scan(self) -> Optional[Dict[str, Any]]:
        """Run Hadolint Dockerfile linter."""
        if not self.check_tool_availability("hadolint"):
            print("⚠️  Hadolint not available - skipping Dockerfile analysis")
            return None
        
        print("🔍 Running Hadolint Dockerfile analysis...")
        
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            print("⚠️  No Dockerfile found - skipping Hadolint")
            return None
        
        output_file = self.results_dir / f"hadolint-{self.timestamp}.json"
        
        try:
            cmd = [
                "hadolint", 
                "--format", "json",
                str(dockerfile_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                hadolint_data = json.loads(result.stdout)
                
                with open(output_file, 'w') as f:
                    json.dump(hadolint_data, f, indent=2)
                
                print(f"✓ Hadolint analysis completed: {output_file}")
                return hadolint_data
            else:
                print("✓ Hadolint found no issues")
                return {"issues": []}
                
        except Exception as e:
            print(f"❌ Hadolint analysis error: {e}")
            return None
    
    def run_grype_scan(self) -> Optional[Dict[str, Any]]:
        """Run Grype vulnerability scanner."""
        if not self.check_tool_availability("grype"):
            print("⚠️  Grype not available - skipping additional vulnerability scan")
            return None
        
        print("🔍 Running Grype vulnerability scan...")
        
        output_file = self.results_dir / f"grype-{self.timestamp}.json"
        
        try:
            cmd = [
                "grype", 
                self.image_name,
                "-o", "json",
                "--file", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                with open(output_file) as f:
                    grype_data = json.load(f)
                
                print(f"✓ Grype scan completed: {output_file}")
                return grype_data
            else:
                print(f"❌ Grype scan failed: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Grype scan error: {e}")
            return None
    
    def analyze_results(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and summarize scan results."""
        summary = {
            "scan_timestamp": datetime.utcnow().isoformat() + "Z",
            "image_name": self.image_name,
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": 0,
            "low_vulnerabilities": 0,
            "dockerfile_issues": 0,
            "security_benchmarks": "not_available",
            "recommendations": []
        }
        
        # Analyze Trivy results
        if scan_results.get("trivy"):
            trivy = scan_results["trivy"]
            if "Results" in trivy:
                for result in trivy["Results"]:
                    if "Vulnerabilities" in result:
                        for vuln in result["Vulnerabilities"]:
                            summary["total_vulnerabilities"] += 1
                            severity = vuln.get("Severity", "").upper()
                            if severity == "CRITICAL":
                                summary["critical_vulnerabilities"] += 1
                            elif severity == "HIGH":
                                summary["high_vulnerabilities"] += 1
                            elif severity == "MEDIUM":
                                summary["medium_vulnerabilities"] += 1
                            elif severity == "LOW":
                                summary["low_vulnerabilities"] += 1
        
        # Analyze Hadolint results
        if scan_results.get("hadolint"):
            hadolint = scan_results["hadolint"]
            if isinstance(hadolint, list):
                summary["dockerfile_issues"] = len(hadolint)
            elif isinstance(hadolint, dict) and "issues" in hadolint:
                summary["dockerfile_issues"] = len(hadolint["issues"])
        
        # Generate recommendations
        if summary["critical_vulnerabilities"] > 0:
            summary["recommendations"].append(
                "URGENT: Address critical vulnerabilities immediately"
            )
        
        if summary["high_vulnerabilities"] > 5:
            summary["recommendations"].append(
                "HIGH: Multiple high-severity vulnerabilities detected"
            )
        
        if summary["dockerfile_issues"] > 0:
            summary["recommendations"].append(
                "MEDIUM: Dockerfile security issues found - review and fix"
            )
        
        if summary["total_vulnerabilities"] == 0:
            summary["recommendations"].append(
                "GOOD: No vulnerabilities detected in current scan scope"
            )
        
        return summary
    
    def generate_report(self, scan_results: Dict[str, Any], summary: Dict[str, Any]):
        """Generate comprehensive security report."""
        report_file = self.results_dir / f"security-report-{self.timestamp}.json"
        
        report = {
            "metadata": {
                "report_version": "1.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "scanner_version": "spikeformer-security-scanner-1.0",
                "image_analyzed": self.image_name
            },
            "summary": summary,
            "detailed_results": scan_results,
            "mitigation_guide": {
                "critical_actions": [
                    "Update base image to latest patched version",
                    "Update vulnerable packages to secure versions",
                    "Review and implement Dockerfile security best practices"
                ],
                "monitoring": [
                    "Set up automated vulnerability scanning in CI/CD",
                    "Enable container runtime security monitoring",
                    "Implement security policy enforcement"
                ],
                "resources": [
                    "https://docs.docker.com/develop/security-best-practices/",
                    "https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html",
                    "https://kubernetes.io/docs/concepts/security/"
                ]
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Comprehensive security report: {report_file}")
        
        return report_file
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print security scan summary."""
        print("\n" + "="*60)
        print("🛡️  CONTAINER SECURITY SCAN SUMMARY")
        print("="*60)
        print(f"Image: {summary['image_name']}")
        print(f"Scan Time: {summary['scan_timestamp']}")
        print()
        
        print("📊 Vulnerability Summary:")
        print(f"  Total: {summary['total_vulnerabilities']}")
        print(f"  Critical: {summary['critical_vulnerabilities']}")
        print(f"  High: {summary['high_vulnerabilities']}")
        print(f"  Medium: {summary['medium_vulnerabilities']}")
        print(f"  Low: {summary['low_vulnerabilities']}")
        print()
        
        print(f"📋 Dockerfile Issues: {summary['dockerfile_issues']}")
        print()
        
        if summary['recommendations']:
            print("🔧 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("="*60)


def main():
    """Main function to run container security scan."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Container Security Scanner for Spikeformer"
    )
    parser.add_argument(
        "--image", "-i",
        default="spikeformer:latest",
        help="Docker image name to scan"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="security-reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    print("🛡️  Starting container security scan...")
    print(f"Target image: {args.image}")
    
    scanner = ContainerSecurityScanner(args.image)
    
    # Run all available scans
    scan_results = {}
    
    trivy_result = scanner.run_trivy_scan()
    if trivy_result:
        scan_results["trivy"] = trivy_result
    
    hadolint_result = scanner.run_hadolint_scan()
    if hadolint_result:
        scan_results["hadolint"] = hadolint_result
    
    grype_result = scanner.run_grype_scan()
    if grype_result:
        scan_results["grype"] = grype_result
    
    bench_result = scanner.run_docker_bench_security()
    if bench_result:
        scan_results["docker_bench"] = bench_result
    
    # Analyze results
    summary = scanner.analyze_results(scan_results)
    
    # Generate report
    report_file = scanner.generate_report(scan_results, summary)
    
    # Print summary
    scanner.print_summary(summary)
    
    # Exit with appropriate code
    if summary["critical_vulnerabilities"] > 0:
        print("\n❌ CRITICAL vulnerabilities found!")
        sys.exit(1)
    elif summary["high_vulnerabilities"] > 10:
        print("\n⚠️  Many HIGH vulnerabilities found!")
        sys.exit(1)
    else:
        print("\n✅ Security scan completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()