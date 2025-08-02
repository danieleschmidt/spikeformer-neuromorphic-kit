#!/usr/bin/env python3
"""
Metrics Report Generator for Spikeformer
Generates comprehensive metrics reports from collected data
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class MetricsReporter:
    """Generate comprehensive metrics reports"""
    
    def __init__(self, metrics_data: Dict[str, Any]):
        """Initialize with metrics data"""
        self.metrics = metrics_data
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary"""
        summary = []
        summary.append("# Spikeformer Metrics Report")
        summary.append(f"**Generated:** {self.timestamp}")
        summary.append("")
        
        # Project overview
        project = self.metrics.get("metadata", {}).get("project", {})
        summary.append("## Project Overview")
        summary.append(f"- **Name:** {project.get('name', 'N/A')}")
        summary.append(f"- **Repository:** {project.get('repository', 'N/A')}")
        summary.append(f"- **Domain:** {project.get('domain', 'N/A')}")
        summary.append("")
        
        # Key metrics at a glance
        summary.append("## Key Metrics Summary")
        
        # Code quality
        cq = self.metrics.get("code_quality", {})
        if cq.get("coverage") is not None:
            coverage_status = "✅" if cq["coverage"] >= 85 else "⚠️" if cq["coverage"] >= 70 else "❌"
            summary.append(f"- **Test Coverage:** {cq['coverage']:.1f}% {coverage_status}")
        
        if cq.get("complexity") is not None:
            complexity_status = "✅" if cq["complexity"] <= 10 else "⚠️" if cq["complexity"] <= 15 else "❌"
            summary.append(f"- **Code Complexity:** {cq['complexity']:.1f} {complexity_status}")
        
        # Performance
        perf = self.metrics.get("performance", {})
        if perf.get("build_time") is not None:
            build_status = "✅" if perf["build_time"] <= 900 else "⚠️" if perf["build_time"] <= 1800 else "❌"
            summary.append(f"- **Build Time:** {perf['build_time']:.1f}s {build_status}")
        
        # Security
        sec = self.metrics.get("security", {})
        vulns = sec.get("vulnerabilities", {})
        critical_vulns = vulns.get("critical", 0)
        high_vulns = vulns.get("high", 0)
        security_status = "✅" if critical_vulns == 0 and high_vulns <= 5 else "❌"
        summary.append(f"- **Security Status:** {critical_vulns} critical, {high_vulns} high {security_status}")
        
        # GitHub metrics
        github = self.metrics.get("github", {})
        if not github.get("collection_failed", False):
            summary.append(f"- **GitHub Stars:** {github.get('stars', 0)}")
            summary.append(f"- **Open Issues:** {github.get('open_issues', 0)}")
        
        summary.append("")
        return "\n".join(summary)
    
    def generate_detailed_analysis(self) -> str:
        """Generate detailed metrics analysis"""
        analysis = []
        
        # Code Quality Analysis
        analysis.append("## Code Quality Analysis")
        cq = self.metrics.get("code_quality", {})
        
        if cq:
            analysis.append("### Test Coverage")
            coverage = cq.get("coverage")
            if coverage is not None:
                analysis.append(f"- **Current Coverage:** {coverage:.1f}%")
                analysis.append(f"- **Target:** 85%")
                analysis.append(f"- **Status:** {'✅ Meets target' if coverage >= 85 else '⚠️ Below target'}")
            else:
                analysis.append("- Coverage data not available")
            analysis.append("")
            
            analysis.append("### Code Complexity")
            complexity = cq.get("complexity")
            if complexity is not None:
                analysis.append(f"- **Average Complexity:** {complexity:.1f}")
                analysis.append(f"- **Target:** ≤ 10")
                analysis.append(f"- **Status:** {'✅ Within limits' if complexity <= 10 else '⚠️ Exceeds target'}")
                
                if complexity > 10:
                    analysis.append("- **Recommendation:** Refactor complex functions to improve maintainability")
            else:
                analysis.append("- Complexity data not available")
            analysis.append("")
            
            analysis.append("### Maintainability")
            maintainability = cq.get("maintainability")
            if maintainability is not None:
                analysis.append(f"- **Maintainability Index:** {maintainability:.1f}")
                analysis.append(f"- **Target:** ≥ 70")
                analysis.append(f"- **Status:** {'✅ Good' if maintainability >= 70 else '⚠️ Needs attention'}")
            else:
                analysis.append("- Maintainability data not available")
            analysis.append("")
        
        # Performance Analysis
        analysis.append("## Performance Analysis")
        perf = self.metrics.get("performance", {})
        
        if perf:
            analysis.append("### Build Performance")
            build_time = perf.get("build_time")
            if build_time is not None:
                analysis.append(f"- **Build Time:** {build_time:.1f} seconds")
                analysis.append(f"- **Target:** ≤ 15 minutes (900s)")
                analysis.append(f"- **Status:** {'✅ Fast' if build_time <= 900 else '⚠️ Slow'}")
            else:
                analysis.append("- Build time data not available")
            analysis.append("")
            
            analysis.append("### Test Performance")
            test_time = perf.get("test_runtime")
            if test_time is not None:
                analysis.append(f"- **Test Runtime:** {test_time:.1f} seconds")
                analysis.append(f"- **Target:** ≤ 30 minutes (1800s)")
                analysis.append(f"- **Status:** {'✅ Fast' if test_time <= 1800 else '⚠️ Slow'}")
            else:
                analysis.append("- Test runtime data not available")
            analysis.append("")
            
            # Energy efficiency (neuromorphic specific)
            energy = perf.get("energy_efficiency", {})
            if isinstance(energy, dict) and energy:
                analysis.append("### Energy Efficiency")
                efficiency = energy.get("efficiency_improvement")
                if efficiency is not None:
                    analysis.append(f"- **Energy Efficiency Improvement:** {efficiency:.1f}x")
                    analysis.append(f"- **Target:** ≥ 5.0x")
                    analysis.append(f"- **Status:** {'✅ Excellent' if efficiency >= 5.0 else '⚠️ Below target'}")
                analysis.append("")
        
        # Security Analysis
        analysis.append("## Security Analysis")
        sec = self.metrics.get("security", {})
        
        if sec:
            analysis.append("### Vulnerability Assessment")
            vulns = sec.get("vulnerabilities", {})
            if vulns:
                analysis.append(f"- **Critical:** {vulns.get('critical', 0)}")
                analysis.append(f"- **High:** {vulns.get('high', 0)}")
                analysis.append(f"- **Medium:** {vulns.get('medium', 0)}")
                analysis.append(f"- **Low:** {vulns.get('low', 0)}")
                
                total_high_priority = vulns.get('critical', 0) + vulns.get('high', 0)
                if total_high_priority == 0:
                    analysis.append("- **Status:** ✅ No high-priority vulnerabilities")
                else:
                    analysis.append(f"- **Status:** ❌ {total_high_priority} high-priority vulnerabilities require attention")
            else:
                analysis.append("- Vulnerability data not available")
            analysis.append("")
            
            analysis.append("### Secrets Detection")
            secrets = sec.get("secrets_scan", {})
            if secrets.get("scan_completed", False):
                secrets_found = secrets.get("secrets_detected", False)
                analysis.append(f"- **Secrets Detected:** {'❌ Yes' if secrets_found else '✅ None'}")
                if secrets_found:
                    analysis.append("- **Action Required:** Remove exposed secrets immediately")
            else:
                analysis.append("- Secrets scan not completed")
            analysis.append("")
            
            analysis.append("### License Compliance")
            license_info = sec.get("license_compliance", {})
            if license_info:
                compliant = license_info.get("compliant", True)
                analysis.append(f"- **Status:** {'✅ Compliant' if compliant else '❌ Non-compliant'}")
                analysis.append(f"- **Dependencies Checked:** {license_info.get('total_dependencies', 0)}")
            else:
                analysis.append("- License compliance data not available")
            analysis.append("")
        
        # Neuromorphic Specific Analysis
        analysis.append("## Neuromorphic AI Analysis")
        neuro = self.metrics.get("neuromorphic", {})
        
        if neuro:
            analysis.append("### Hardware Availability")
            hardware = neuro.get("hardware_availability", {})
            if hardware:
                analysis.append(f"- **Loihi 2:** {'✅ Available' if hardware.get('loihi2', False) else '❌ Not available'}")
                analysis.append(f"- **SpiNNaker:** {'✅ Available' if hardware.get('spinnaker', False) else '❌ Not available'}")
                analysis.append(f"- **Simulation Mode:** ✅ Available")
            analysis.append("")
            
            analysis.append("### Model Performance")
            model_perf = neuro.get("model_performance", {})
            if model_perf:
                success_rate = model_perf.get("conversion_success_rate")
                if success_rate is not None:
                    analysis.append(f"- **Conversion Success Rate:** {success_rate:.1%}")
                    analysis.append(f"- **Target:** ≥ 95%")
                    analysis.append(f"- **Status:** {'✅ Good' if success_rate >= 0.95 else '⚠️ Below target'}")
                
                accuracy = model_perf.get("average_accuracy")
                if accuracy is not None:
                    analysis.append(f"- **Average Accuracy:** {accuracy:.1%}")
                    analysis.append(f"- **Target:** ≥ 90%")
                    analysis.append(f"- **Status:** {'✅ Good' if accuracy >= 0.9 else '⚠️ Below target'}")
            analysis.append("")
        
        return "\n".join(analysis)
    
    def generate_recommendations(self) -> str:
        """Generate actionable recommendations"""
        recommendations = []
        recommendations.append("## Recommendations")
        
        # Code quality recommendations
        cq = self.metrics.get("code_quality", {})
        if cq.get("coverage", 100) < 85:
            recommendations.append("### Code Coverage")
            recommendations.append("- Increase test coverage to meet the 85% target")
            suggestions = [
                "Add unit tests for uncovered functions",
                "Implement integration tests for complex workflows",
                "Add property-based tests for neuromorphic algorithms"
            ]
            for suggestion in suggestions:
                recommendations.append(f"  - {suggestion}")
            recommendations.append("")
        
        if cq.get("complexity", 0) > 10:
            recommendations.append("### Code Complexity")
            recommendations.append("- Reduce code complexity to improve maintainability")
            suggestions = [
                "Break down complex functions into smaller units",
                "Use design patterns to simplify neuromorphic model handling",
                "Extract common spike processing logic into utilities"
            ]
            for suggestion in suggestions:
                recommendations.append(f"  - {suggestion}")
            recommendations.append("")
        
        # Performance recommendations
        perf = self.metrics.get("performance", {})
        if perf.get("build_time", 0) > 900:
            recommendations.append("### Build Performance")
            recommendations.append("- Optimize build process to reduce time")
            suggestions = [
                "Use build caching for dependencies",
                "Parallelize compilation steps",
                "Optimize Docker layer caching"
            ]
            for suggestion in suggestions:
                recommendations.append(f"  - {suggestion}")
            recommendations.append("")
        
        # Security recommendations
        sec = self.metrics.get("security", {})
        vulns = sec.get("vulnerabilities", {})
        if vulns.get("critical", 0) > 0 or vulns.get("high", 0) > 5:
            recommendations.append("### Security")
            recommendations.append("- Address high-priority security vulnerabilities")
            suggestions = [
                "Update dependencies with known vulnerabilities",
                "Implement automated security scanning in CI/CD",
                "Regular security audits of neuromorphic hardware interfaces"
            ]
            for suggestion in suggestions:
                recommendations.append(f"  - {suggestion}")
            recommendations.append("")
        
        # Neuromorphic-specific recommendations
        neuro = self.metrics.get("neuromorphic", {})
        model_perf = neuro.get("model_performance", {})
        if model_perf.get("conversion_success_rate", 1.0) < 0.95:
            recommendations.append("### Neuromorphic Performance")
            recommendations.append("- Improve model conversion success rate")
            suggestions = [
                "Add more robust error handling in conversion pipeline",
                "Implement fallback strategies for unsupported operations",
                "Enhance testing with diverse model architectures"
            ]
            for suggestion in suggestions:
                recommendations.append(f"  - {suggestion}")
            recommendations.append("")
        
        if not recommendations or len(recommendations) <= 2:
            recommendations.append("### Overall Status")
            recommendations.append("✅ All metrics are within acceptable ranges. Continue monitoring and maintaining current practices.")
            recommendations.append("")
        
        return "\n".join(recommendations)
    
    def generate_trend_indicators(self) -> str:
        """Generate trend analysis section"""
        trends = []
        trends.append("## Trend Analysis")
        trends.append("*Note: Trend analysis requires historical data from multiple collection runs.*")
        trends.append("")
        
        # This would be enhanced with actual historical data
        trends.append("### Key Trends to Monitor")
        trends.append("- Test coverage progression over time")
        trends.append("- Build time performance trends")
        trends.append("- Security vulnerability emergence patterns")
        trends.append("- Neuromorphic model conversion accuracy trends")
        trends.append("- Energy efficiency improvements")
        trends.append("")
        
        return "\n".join(trends)
    
    def generate_full_report(self) -> str:
        """Generate complete metrics report"""
        report_sections = [
            self.generate_executive_summary(),
            self.generate_detailed_analysis(),
            self.generate_recommendations(),
            self.generate_trend_indicators(),
            self._generate_appendix()
        ]
        
        return "\n".join(report_sections)
    
    def _generate_appendix(self) -> str:
        """Generate appendix with raw data summary"""
        appendix = []
        appendix.append("## Appendix")
        appendix.append("")
        
        appendix.append("### Data Collection Details")
        metadata = self.metrics.get("metadata", {})
        appendix.append(f"- **Collection Time:** {metadata.get('collection_time', 'Unknown')}")
        appendix.append(f"- **Collector Version:** {metadata.get('collector_version', 'Unknown')}")
        appendix.append("")
        
        appendix.append("### Metrics Categories Collected")
        categories = [key for key in self.metrics.keys() if key != "metadata"]
        for category in categories:
            appendix.append(f"- {category.replace('_', ' ').title()}")
        appendix.append("")
        
        appendix.append("### Configuration Reference")
        appendix.append("This report is based on thresholds and targets defined in `.github/project-metrics.json`")
        appendix.append("")
        
        return "\n".join(appendix)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Spikeformer metrics report")
    parser.add_argument(
        "--input",
        required=True,
        help="Input metrics JSON file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output report file (markdown)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Load metrics data
    try:
        with open(args.input) as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    
    # Generate report
    reporter = MetricsReporter(metrics_data)
    
    if args.format == "markdown":
        report = reporter.generate_full_report()
    elif args.format == "json":
        # Generate structured summary for JSON format
        report = json.dumps({
            "summary": {
                "timestamp": reporter.timestamp,
                "overall_status": "analyzed",
                "key_metrics": {
                    "coverage": metrics_data.get("code_quality", {}).get("coverage"),
                    "complexity": metrics_data.get("code_quality", {}).get("complexity"),
                    "build_time": metrics_data.get("performance", {}).get("build_time"),
                    "vulnerabilities": metrics_data.get("security", {}).get("vulnerabilities", {})
                }
            },
            "full_data": metrics_data
        }, indent=2)
    else:  # HTML format would be implemented here
        report = reporter.generate_full_report()
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Metrics report generated: {output_path}")


if __name__ == "__main__":
    main()