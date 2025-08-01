#!/usr/bin/env python3
"""
Terragon Autonomous Dependency Security Scanner
Continuously monitors and prioritizes dependency security vulnerabilities
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class DependencyScanner:
    """Advanced dependency scanner with WSJF scoring for vulnerabilities."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.results_path = repo_path / ".terragon" / "security-scan-results.json"
        
    def scan_python_dependencies(self) -> Dict:
        """Scan Python dependencies for vulnerabilities using multiple tools."""
        results = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "vulnerabilities": [],
            "summary": {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        # Run safety scan
        try:
            safety_result = subprocess.run(
                ["safety", "check", "--json"], 
                capture_output=True, text=True, check=False
            )
            if safety_result.stdout:
                safety_data = json.loads(safety_result.stdout)
                for vuln in safety_data:
                    results["vulnerabilities"].append({
                        "source": "safety",
                        "package": vuln.get("package", "unknown"),
                        "vulnerability_id": vuln.get("id", "unknown"),
                        "severity": self._normalize_severity(vuln.get("severity", "unknown")),
                        "description": vuln.get("vulnerability", "No description"),
                        "affected_versions": vuln.get("affected_versions", "unknown"),
                        "more_info_url": vuln.get("more_info_url", ""),
                        "wsjf_score": self._calculate_vuln_wsjf(vuln)
                    })
        except Exception as e:
            print(f"Safety scan failed: {e}")
            
        # Run pip-audit if available
        try:
            audit_result = subprocess.run(
                ["pip-audit", "--format=json"], 
                capture_output=True, text=True, check=False
            )
            if audit_result.stdout:
                audit_data = json.loads(audit_result.stdout)
                for vuln in audit_data.get("vulnerabilities", []):
                    results["vulnerabilities"].append({
                        "source": "pip-audit",
                        "package": vuln.get("package", "unknown"),
                        "vulnerability_id": vuln.get("id", "unknown"),
                        "severity": self._normalize_severity(vuln.get("severity", "unknown")),
                        "description": vuln.get("description", "No description"),
                        "affected_versions": str(vuln.get("specs", [])),
                        "more_info_url": vuln.get("link", ""),
                        "wsjf_score": self._calculate_vuln_wsjf(vuln)
                    })
        except Exception as e:
            print(f"pip-audit scan failed: {e}")
            
        # Update summary
        for vuln in results["vulnerabilities"]:
            severity = vuln["severity"].lower()
            results["summary"]["total"] += 1
            if severity in results["summary"]:
                results["summary"][severity] += 1
                
        # Sort by WSJF score (highest priority first)
        results["vulnerabilities"].sort(key=lambda x: x["wsjf_score"], reverse=True)
        
        return results
    
    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across different tools."""
        severity = severity.lower()
        if severity in ["critical"]:
            return "critical"
        elif severity in ["high", "important"]:
            return "high"
        elif severity in ["medium", "moderate"]:
            return "medium"
        elif severity in ["low", "minor"]:
            return "low"
        else:
            return "medium"  # Default to medium if unknown
    
    def _calculate_vuln_wsjf(self, vuln: Dict) -> float:
        """Calculate WSJF score for vulnerability prioritization."""
        # Cost of Delay factors
        severity_impact = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2
        }
        
        severity = self._normalize_severity(vuln.get("severity", "medium"))
        user_business_value = severity_impact.get(severity, 4)
        
        # Time criticality (security vulnerabilities are time-sensitive)
        time_criticality = severity_impact.get(severity, 4) * 1.5
        
        # Risk reduction (high for security issues)
        risk_reduction = severity_impact.get(severity, 4) * 2.0
        
        # Opportunity enablement (security enables other features)
        opportunity_enablement = 3
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        
        # Job size (effort estimation)
        # Dependency updates are typically low effort
        job_size = 2  # Small effort for most dependency updates
        
        # Special cases for complex dependencies
        package_name = vuln.get("package", "").lower()
        if any(complex_pkg in package_name for complex_pkg in ["torch", "tensorflow", "transformers"]):
            job_size = 4  # Higher effort for complex ML dependencies
            
        wsjf_score = cost_of_delay / job_size
        
        # Security boost
        wsjf_score *= 1.8  # Security multiplier
        
        return round(wsjf_score, 2)
    
    def generate_backlog_items(self, scan_results: Dict) -> List[Dict]:
        """Generate backlog items from scan results."""
        items = []
        
        for i, vuln in enumerate(scan_results["vulnerabilities"][:10]):  # Top 10 most critical
            items.append({
                "id": f"SEC-DEP-{i+1:03d}",
                "title": f"Fix {vuln['severity']} vulnerability in {vuln['package']}",
                "description": f"{vuln['description'][:100]}...",
                "category": "Security",
                "subcategory": "Dependency Vulnerability",
                "wsjf_score": vuln["wsjf_score"],
                "estimated_hours": 0.5 if vuln["severity"] in ["low", "medium"] else 1.5,
                "priority": "high" if vuln["severity"] in ["critical", "high"] else "medium",
                "package": vuln["package"],
                "vulnerability_id": vuln["vulnerability_id"],
                "more_info": vuln["more_info_url"]
            })
            
        return items
    
    def save_results(self, results: Dict) -> None:
        """Save scan results to file."""
        self.results_path.parent.mkdir(exist_ok=True)
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)
    
    def run_full_scan(self) -> Dict:
        """Run complete dependency security scan."""
        print("🔍 Running comprehensive dependency security scan...")
        
        results = self.scan_python_dependencies()
        
        # Add package update analysis
        results["package_updates"] = self._analyze_package_updates()
        
        # Generate backlog items
        results["backlog_items"] = self.generate_backlog_items(results)
        
        # Save results
        self.save_results(results)
        
        print(f"✅ Scan complete: {results['summary']['total']} vulnerabilities found")
        print(f"   Critical: {results['summary']['critical']}")
        print(f"   High: {results['summary']['high']}")
        print(f"   Medium: {results['summary']['medium']}")
        print(f"   Low: {results['summary']['low']}")
        
        return results
    
    def _analyze_package_updates(self) -> Dict:
        """Analyze available package updates."""
        updates = {"total": 0, "major": 0, "minor": 0, "patch": 0}
        
        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, check=False
            )
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                updates["total"] = len(outdated)
                
                for pkg in outdated:
                    current = pkg.get("version", "0.0.0")
                    latest = pkg.get("latest_version", "0.0.0")
                    
                    # Simple version comparison
                    current_parts = current.split(".")
                    latest_parts = latest.split(".")
                    
                    if len(current_parts) >= 1 and len(latest_parts) >= 1:
                        if current_parts[0] != latest_parts[0]:
                            updates["major"] += 1
                        elif len(current_parts) >= 2 and len(latest_parts) >= 2 and current_parts[1] != latest_parts[1]:
                            updates["minor"] += 1
                        else:
                            updates["patch"] += 1
                            
        except Exception as e:
            print(f"Package update analysis failed: {e}")
            
        return updates

def main():
    """Main entry point for dependency scanner."""
    scanner = DependencyScanner()
    results = scanner.run_full_scan()
    
    # Print top 3 most critical vulnerabilities
    if results["vulnerabilities"]:
        print("\n🚨 Top 3 most critical vulnerabilities:")
        for i, vuln in enumerate(results["vulnerabilities"][:3], 1):
            print(f"{i}. {vuln['package']} ({vuln['severity']}) - WSJF: {vuln['wsjf_score']}")
            print(f"   {vuln['description'][:80]}...")
    
    return 0 if results["summary"]["critical"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())