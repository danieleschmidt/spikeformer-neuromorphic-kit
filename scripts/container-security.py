#!/usr/bin/env python3
"""Container security scanning and analysis tool for Spikeformer."""

import json
import subprocess
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityScanner:
    """Container security scanner using multiple tools."""
    
    def __init__(self, output_dir: str = "security-reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def check_tool_availability(self) -> Dict[str, bool]:
        """Check which security tools are available."""
        tools = {
            'trivy': self._check_command('trivy'),
            'grype': self._check_command('grype'),
            'syft': self._check_command('syft'),
            'docker': self._check_command('docker'),
            'hadolint': self._check_command('hadolint'),
        }
        
        logger.info(f"Tool availability: {tools}")
        return tools
    
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([command, '--help'], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def scan_image_vulnerabilities(self, image: str, scanner: str = 'trivy') -> Dict[str, Any]:
        """Scan container image for vulnerabilities."""
        logger.info(f"Scanning {image} for vulnerabilities using {scanner}")
        
        output_file = self.output_dir / f"vulnerabilities-{image.replace(':', '-')}-{self.timestamp}.json"
        
        if scanner == 'trivy':
            return self._scan_with_trivy(image, output_file)
        elif scanner == 'grype':
            return self._scan_with_grype(image, output_file)
        else:
            raise ValueError(f"Unsupported scanner: {scanner}")
    
    def _scan_with_trivy(self, image: str, output_file: Path) -> Dict[str, Any]:
        """Scan with Trivy."""
        cmd = [
            'trivy', 'image',
            '--format', 'json',
            '--output', str(output_file),
            '--severity', 'UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL',
            '--quiet',
            image
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results
            with open(output_file, 'r') as f:
                scan_data = json.load(f)
            
            return self._process_trivy_results(scan_data, image)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Trivy scan failed: {e.stderr}")
            return {'error': str(e)}
    
    def _scan_with_grype(self, image: str, output_file: Path) -> Dict[str, Any]:
        """Scan with Grype."""
        cmd = [
            'grype', image,
            '-o', 'json',
            '--file', str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results
            with open(output_file, 'r') as f:
                scan_data = json.load(f)
            
            return self._process_grype_results(scan_data, image)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Grype scan failed: {e.stderr}")
            return {'error': str(e)}
    
    def _process_trivy_results(self, scan_data: Dict, image: str) -> Dict[str, Any]:
        """Process Trivy scan results."""
        results = {
            'image': image,
            'scanner': 'trivy',
            'timestamp': self.timestamp,
            'summary': {
                'total_vulnerabilities': 0,
                'by_severity': {
                    'CRITICAL': 0,
                    'HIGH': 0,
                    'MEDIUM': 0,
                    'LOW': 0,
                    'UNKNOWN': 0
                }
            },
            'vulnerabilities': []
        }
        
        if 'Results' in scan_data:
            for result in scan_data['Results']:
                if 'Vulnerabilities' in result:
                    for vuln in result['Vulnerabilities']:
                        severity = vuln.get('Severity', 'UNKNOWN')
                        results['summary']['by_severity'][severity] += 1
                        results['summary']['total_vulnerabilities'] += 1
                        
                        results['vulnerabilities'].append({
                            'id': vuln.get('VulnerabilityID'),
                            'severity': severity,
                            'package': vuln.get('PkgName'),
                            'version': vuln.get('InstalledVersion'),
                            'fixed_version': vuln.get('FixedVersion'),
                            'title': vuln.get('Title'),
                            'description': vuln.get('Description', '')[:200] + '...' if len(vuln.get('Description', '')) > 200 else vuln.get('Description', ''),
                            'references': vuln.get('References', [])
                        })
        
        return results
    
    def _process_grype_results(self, scan_data: Dict, image: str) -> Dict[str, Any]:
        """Process Grype scan results."""
        results = {
            'image': image,
            'scanner': 'grype',
            'timestamp': self.timestamp,
            'summary': {
                'total_vulnerabilities': 0,
                'by_severity': {
                    'Critical': 0,
                    'High': 0,
                    'Medium': 0,
                    'Low': 0,
                    'Negligible': 0
                }
            },
            'vulnerabilities': []
        }
        
        if 'matches' in scan_data:
            for match in scan_data['matches']:
                vulnerability = match.get('vulnerability', {})
                severity = vulnerability.get('severity', 'Unknown')
                
                if severity in results['summary']['by_severity']:
                    results['summary']['by_severity'][severity] += 1
                    results['summary']['total_vulnerabilities'] += 1
                
                results['vulnerabilities'].append({
                    'id': vulnerability.get('id'),
                    'severity': severity,
                    'package': match.get('artifact', {}).get('name'),
                    'version': match.get('artifact', {}).get('version'),
                    'fixed_version': vulnerability.get('fix', {}).get('versions', []),
                    'description': vulnerability.get('description', '')
                })
        
        return results
    
    def generate_sbom(self, image: str) -> Dict[str, Any]:
        """Generate Software Bill of Materials (SBOM)."""
        logger.info(f"Generating SBOM for {image}")
        
        output_file = self.output_dir / f"sbom-{image.replace(':', '-')}-{self.timestamp}.json"
        
        cmd = [
            'syft', image,
            '-o', f'spdx-json={output_file}'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            with open(output_file, 'r') as f:
                sbom_data = json.load(f)
            
            return {
                'image': image,
                'sbom_file': str(output_file),
                'packages_count': len(sbom_data.get('packages', [])),
                'timestamp': self.timestamp
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SBOM generation failed: {e}")
            return {'error': str(e)}
    
    def scan_dockerfile(self, dockerfile_path: str = "Dockerfile") -> Dict[str, Any]:
        """Scan Dockerfile for security issues using hadolint."""
        logger.info(f"Scanning {dockerfile_path} for security issues")
        
        output_file = self.output_dir / f"dockerfile-scan-{self.timestamp}.json"
        
        cmd = [
            'hadolint',
            '--format', 'json',
            dockerfile_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Hadolint returns non-zero on findings, so don't check=True
            if result.stdout:
                findings = json.loads(result.stdout)
                
                with open(output_file, 'w') as f:
                    json.dump(findings, f, indent=2)
                
                # Process findings
                summary = {
                    'total_issues': len(findings),
                    'by_level': {
                        'error': 0,
                        'warning': 0,
                        'info': 0,
                        'style': 0
                    }
                }
                
                for finding in findings:
                    level = finding.get('level', 'info')
                    if level in summary['by_level']:
                        summary['by_level'][level] += 1
                
                return {
                    'dockerfile': dockerfile_path,
                    'summary': summary,
                    'findings': findings,
                    'output_file': str(output_file)
                }
            else:
                return {
                    'dockerfile': dockerfile_path,
                    'summary': {'total_issues': 0},
                    'findings': []
                }
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Dockerfile scan failed: {e}")
            return {'error': str(e)}
    
    def analyze_image_layers(self, image: str) -> Dict[str, Any]:
        """Analyze image layers for size and efficiency."""
        logger.info(f"Analyzing layers for {image}")
        
        cmd = ['docker', 'history', '--format', 'json', '--no-trunc', image]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            layers = []
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    layer = json.loads(line)
                    layers.append({
                        'id': layer.get('ID', 'N/A'),
                        'created': layer.get('CreatedAt', 'N/A'),
                        'created_by': layer.get('CreatedBy', 'N/A'),
                        'size': layer.get('Size', 'N/A'),
                        'comment': layer.get('Comment', 'N/A')
                    })
            
            # Get image info
            inspect_cmd = ['docker', 'inspect', image]
            inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True, check=True)
            image_info = json.loads(inspect_result.stdout)[0]
            
            return {
                'image': image,
                'total_size': image_info.get('Size', 0),
                'layer_count': len(layers),
                'layers': layers,
                'architecture': image_info.get('Architecture', 'N/A'),
                'os': image_info.get('Os', 'N/A')
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Layer analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_security_report(self, scan_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive security report."""
        report_file = self.output_dir / f"security-report-{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Spikeformer Container Security Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            total_vulns = sum(result.get('summary', {}).get('total_vulnerabilities', 0) 
                            for result in scan_results if 'summary' in result)
            f.write(f"- **Total Images Scanned:** {len(scan_results)}\n")
            f.write(f"- **Total Vulnerabilities Found:** {total_vulns}\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            
            for result in scan_results:
                if 'error' in result:
                    continue
                    
                image = result.get('image', 'Unknown')
                f.write(f"### {image}\n\n")
                
                if 'summary' in result:
                    summary = result['summary']
                    f.write("| Severity | Count |\n")
                    f.write("|----------|-------|\n")
                    
                    for severity, count in summary.get('by_severity', {}).items():
                        f.write(f"| {severity} | {count} |\n")
                    
                    f.write("\n")
                
                # Top vulnerabilities
                if 'vulnerabilities' in result:
                    high_crit_vulns = [v for v in result['vulnerabilities'] 
                                     if v.get('severity', '').upper() in ['HIGH', 'CRITICAL']]
                    
                    if high_crit_vulns:
                        f.write("#### Critical/High Severity Vulnerabilities\n\n")
                        for vuln in high_crit_vulns[:5]:  # Top 5
                            f.write(f"- **{vuln.get('id', 'N/A')}** ({vuln.get('severity', 'N/A')})\n")
                            f.write(f"  - Package: {vuln.get('package', 'N/A')} {vuln.get('version', 'N/A')}\n")
                            f.write(f"  - Fix: {vuln.get('fixed_version', 'N/A')}\n")
                            if vuln.get('title'):
                                f.write(f"  - Title: {vuln['title']}\n")
                            f.write("\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Update base images** to latest versions with security patches\n")
            f.write("2. **Pin package versions** to avoid dependency confusion\n")
            f.write("3. **Regular scanning** as part of CI/CD pipeline\n")
            f.write("4. **Multi-stage builds** to reduce attack surface\n")
            f.write("5. **Non-root user** for container execution\n")
            f.write("6. **Minimal base images** (distroless, alpine) when possible\n\n")
        
        logger.info(f"Security report generated: {report_file}")
        return str(report_file)


def main():
    parser = argparse.ArgumentParser(description="Container security scanner for Spikeformer")
    parser.add_argument('images', nargs='*', help='Container images to scan')
    parser.add_argument('--scanner', choices=['trivy', 'grype'], default='trivy',
                       help='Vulnerability scanner to use')
    parser.add_argument('--output-dir', default='security-reports',
                       help='Output directory for reports')
    parser.add_argument('--dockerfile', default='Dockerfile',
                       help='Dockerfile to scan')
    parser.add_argument('--generate-sbom', action='store_true',
                       help='Generate Software Bill of Materials')
    parser.add_argument('--scan-dockerfile', action='store_true',
                       help='Scan Dockerfile for security issues')
    parser.add_argument('--analyze-layers', action='store_true',
                       help='Analyze image layers')
    parser.add_argument('--all-checks', action='store_true',
                       help='Run all security checks')
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.output_dir)
    
    # Check tool availability
    tools = scanner.check_tool_availability()
    
    if not tools['docker']:
        logger.error("Docker is required but not available")
        sys.exit(1)
    
    if args.scanner == 'trivy' and not tools['trivy']:
        logger.error("Trivy scanner requested but not available")
        sys.exit(1)
    
    if args.scanner == 'grype' and not tools['grype']:
        logger.error("Grype scanner requested but not available")
        sys.exit(1)
    
    results = []
    
    # Scan Dockerfile if requested
    if args.scan_dockerfile or args.all_checks:
        if tools['hadolint']:
            dockerfile_result = scanner.scan_dockerfile(args.dockerfile)
            results.append(dockerfile_result)
            logger.info(f"Dockerfile scan completed: {dockerfile_result.get('summary', {}).get('total_issues', 0)} issues found")
        else:
            logger.warning("hadolint not available, skipping Dockerfile scan")
    
    # Scan images
    if args.images:
        for image in args.images:
            logger.info(f"Processing image: {image}")
            
            # Vulnerability scan
            vuln_result = scanner.scan_image_vulnerabilities(image, args.scanner)
            results.append(vuln_result)
            
            if 'error' not in vuln_result:
                total_vulns = vuln_result.get('summary', {}).get('total_vulnerabilities', 0)
                critical_high = (vuln_result.get('summary', {}).get('by_severity', {}).get('CRITICAL', 0) + 
                               vuln_result.get('summary', {}).get('by_severity', {}).get('HIGH', 0))
                logger.info(f"Vulnerability scan completed: {total_vulns} total, {critical_high} critical/high")
            
            # Generate SBOM if requested
            if (args.generate_sbom or args.all_checks) and tools['syft']:
                sbom_result = scanner.generate_sbom(image)
                if 'error' not in sbom_result:
                    logger.info(f"SBOM generated: {sbom_result.get('packages_count', 0)} packages")
            
            # Analyze layers if requested
            if args.analyze_layers or args.all_checks:
                layer_result = scanner.analyze_image_layers(image)
                if 'error' not in layer_result:
                    logger.info(f"Layer analysis completed: {layer_result.get('layer_count', 0)} layers, {layer_result.get('total_size', 0)} bytes")
    
    # Generate comprehensive report
    if results:
        report_file = scanner.generate_security_report(results)
        logger.info(f"Comprehensive security report: {report_file}")
        
        # Exit with error code if critical vulnerabilities found
        critical_vulns = sum(result.get('summary', {}).get('by_severity', {}).get('CRITICAL', 0) 
                           for result in results if 'summary' in result)
        
        if critical_vulns > 0:
            logger.warning(f"Found {critical_vulns} critical vulnerabilities")
            sys.exit(1)
    
    logger.info("Security scanning completed successfully")


if __name__ == "__main__":
    main()