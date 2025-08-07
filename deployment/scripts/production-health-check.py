#!/usr/bin/env python3
"""
Production health check and monitoring script for Neuromorphic Platform.
Performs comprehensive health checks across all system components.
"""

import asyncio
import aiohttp
import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: str
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    total_response_time_ms: float
    checks: List[HealthCheckResult]
    timestamp: datetime

class HealthChecker:
    """Comprehensive health checker for the neuromorphic platform."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_api_health(self) -> HealthCheckResult:
        """Check API health endpoint."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'healthy':
                        status = "healthy"
                        message = "API is healthy"
                    else:
                        status = "degraded"
                        message = f"API reports status: {data.get('status', 'unknown')}"
                        
                    return HealthCheckResult(
                        component="API Health",
                        status=status,
                        response_time_ms=response_time,
                        message=message,
                        details=data,
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="API Health",
                        status="unhealthy",
                        response_time_ms=response_time,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="API Health",
                status="unhealthy",
                response_time_ms=response_time,
                message=f"Connection failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def check_api_readiness(self) -> HealthCheckResult:
        """Check API readiness endpoint."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/ready") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    return HealthCheckResult(
                        component="API Readiness",
                        status="healthy" if data.get('ready', False) else "degraded",
                        response_time_ms=response_time,
                        message="API is ready" if data.get('ready', False) else "API not ready",
                        details=data,
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="API Readiness",
                        status="unhealthy",
                        response_time_ms=response_time,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="API Readiness",
                status="unhealthy",
                response_time_ms=response_time,
                message=f"Connection failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def check_inference_endpoint(self) -> HealthCheckResult:
        """Check inference endpoint with sample data."""
        start_time = time.time()
        
        try:
            # Sample inference request
            sample_data = {
                "model": "spiking_transformer_test",
                "inputs": [[1, 2, 3, 4, 5]],  # Simple token sequence
                "parameters": {
                    "timesteps": 16,
                    "threshold": 1.0
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/inference", 
                json=sample_data
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "result" in data and data.get("success", False):
                        status = "healthy"
                        message = "Inference endpoint is working"
                    else:
                        status = "degraded"
                        message = f"Inference failed: {data.get('error', 'Unknown error')}"
                        
                    return HealthCheckResult(
                        component="Inference Engine",
                        status=status,
                        response_time_ms=response_time,
                        message=message,
                        details={
                            "response_time_ms": response_time,
                            "chip_id": data.get("chip_id"),
                            "model_used": data.get("model_used")
                        },
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="Inference Engine",
                        status="unhealthy",
                        response_time_ms=response_time,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="Inference Engine",
                status="unhealthy",
                response_time_ms=response_time,
                message=f"Inference request failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def check_cluster_status(self) -> HealthCheckResult:
        """Check neuromorphic cluster status."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/cluster/status") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    cluster_stats = data.get("cluster_stats", {})
                    available_chips = cluster_stats.get("available_chips", 0)
                    total_chips = cluster_stats.get("total_chips", 0)
                    cluster_health = cluster_stats.get("cluster_health", "unknown")
                    
                    if cluster_health == "healthy" and available_chips > 0:
                        status = "healthy"
                        message = f"Cluster healthy: {available_chips}/{total_chips} chips available"
                    elif available_chips > 0:
                        status = "degraded"
                        message = f"Cluster degraded: {available_chips}/{total_chips} chips available"
                    else:
                        status = "unhealthy"
                        message = "No chips available"
                        
                    return HealthCheckResult(
                        component="Neuromorphic Cluster",
                        status=status,
                        response_time_ms=response_time,
                        message=message,
                        details=cluster_stats,
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="Neuromorphic Cluster",
                        status="unhealthy",
                        response_time_ms=response_time,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="Neuromorphic Cluster",
                status="unhealthy",
                response_time_ms=response_time,
                message=f"Cluster status check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def check_metrics_endpoint(self) -> HealthCheckResult:
        """Check metrics endpoint (Prometheus)."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    metrics_text = await response.text()
                    
                    # Basic validation of metrics format
                    if "neuromorphic_" in metrics_text and "# HELP" in metrics_text:
                        status = "healthy"
                        message = "Metrics endpoint is working"
                        metric_count = len([line for line in metrics_text.split('\n') 
                                         if line.startswith('neuromorphic_')])
                    else:
                        status = "degraded"
                        message = "Metrics endpoint responding but format unclear"
                        metric_count = 0
                        
                    return HealthCheckResult(
                        component="Metrics Endpoint",
                        status=status,
                        response_time_ms=response_time,
                        message=message,
                        details={"metric_count": metric_count},
                        timestamp=datetime.now()
                    )
                else:
                    return HealthCheckResult(
                        component="Metrics Endpoint",
                        status="unhealthy",
                        response_time_ms=response_time,
                        message=f"HTTP {response.status}",
                        details={"status_code": response.status},
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="Metrics Endpoint",
                status="unhealthy",
                response_time_ms=response_time,
                message=f"Metrics check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all health checks and return overall system health."""
        checks = await asyncio.gather(
            self.check_api_health(),
            self.check_api_readiness(),
            self.check_inference_endpoint(),
            self.check_cluster_status(),
            self.check_metrics_endpoint(),
            return_exceptions=True
        )
        
        # Filter out exceptions and convert to HealthCheckResult list
        valid_checks = []
        for check in checks:
            if isinstance(check, HealthCheckResult):
                valid_checks.append(check)
            elif isinstance(check, Exception):
                logger.error(f"Health check failed with exception: {check}")
                valid_checks.append(HealthCheckResult(
                    component="Unknown",
                    status="unhealthy",
                    response_time_ms=0,
                    message=f"Check failed: {str(check)}",
                    details={"error": str(check)},
                    timestamp=datetime.now()
                ))
        
        # Calculate overall health
        healthy_count = len([c for c in valid_checks if c.status == "healthy"])
        degraded_count = len([c for c in valid_checks if c.status == "degraded"])
        unhealthy_count = len([c for c in valid_checks if c.status == "unhealthy"])
        
        total_response_time = sum(c.response_time_ms for c in valid_checks)
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return SystemHealth(
            overall_status=overall_status,
            healthy_components=healthy_count,
            degraded_components=degraded_count,
            unhealthy_components=unhealthy_count,
            total_response_time_ms=total_response_time,
            checks=valid_checks,
            timestamp=datetime.now()
        )

def print_health_report(health: SystemHealth, verbose: bool = False):
    """Print a formatted health report."""
    status_colors = {
        "healthy": "\033[92m",    # Green
        "degraded": "\033[93m",   # Yellow
        "unhealthy": "\033[91m",  # Red
    }
    reset_color = "\033[0m"
    
    # Overall status
    color = status_colors.get(health.overall_status, "")
    print(f"\n{'='*60}")
    print(f"NEUROMORPHIC PLATFORM HEALTH REPORT")
    print(f"{'='*60}")
    print(f"Overall Status: {color}{health.overall_status.upper()}{reset_color}")
    print(f"Timestamp: {health.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Total Response Time: {health.total_response_time_ms:.2f}ms")
    print(f"\nComponent Summary:")
    print(f"  ✅ Healthy: {health.healthy_components}")
    print(f"  ⚠️  Degraded: {health.degraded_components}")
    print(f"  ❌ Unhealthy: {health.unhealthy_components}")
    
    # Individual component status
    print(f"\n{'='*60}")
    print("COMPONENT STATUS")
    print(f"{'='*60}")
    
    for check in health.checks:
        color = status_colors.get(check.status, "")
        status_icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(check.status, "❓")
        
        print(f"\n{status_icon} {check.component}")
        print(f"   Status: {color}{check.status.upper()}{reset_color}")
        print(f"   Response Time: {check.response_time_ms:.2f}ms")
        print(f"   Message: {check.message}")
        
        if verbose and check.details:
            print(f"   Details: {json.dumps(check.details, indent=2)}")

async def main():
    parser = argparse.ArgumentParser(
        description="Neuromorphic Platform Production Health Check"
    )
    parser.add_argument(
        "--url", 
        default="http://localhost:8080",
        help="Base URL of the neuromorphic platform API"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--output", 
        help="Output file for health report (JSON format)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed check information"
    )
    parser.add_argument(
        "--continuous", 
        type=int,
        help="Run checks continuously every N seconds"
    )
    parser.add_argument(
        "--fail-on-degraded", 
        action="store_true",
        help="Exit with code 1 if any component is degraded"
    )
    
    args = parser.parse_args()
    
    async def run_health_check():
        async with HealthChecker(args.url, args.timeout) as checker:
            health = await checker.run_all_checks()
            
            print_health_report(health, args.verbose)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(asdict(health), f, indent=2, default=str)
                print(f"\nHealth report saved to: {args.output}")
            
            # Determine exit code
            if health.overall_status == "unhealthy":
                return 2
            elif health.overall_status == "degraded" and args.fail_on_degraded:
                return 1
            else:
                return 0
    
    if args.continuous:
        logger.info(f"Running continuous health checks every {args.continuous} seconds...")
        exit_code = 0
        
        try:
            while True:
                current_exit_code = await run_health_check()
                exit_code = max(exit_code, current_exit_code)  # Keep worst exit code
                
                print(f"\nNext check in {args.continuous} seconds...")
                await asyncio.sleep(args.continuous)
                
        except KeyboardInterrupt:
            print("\nHealth monitoring stopped.")
            sys.exit(exit_code)
    else:
        exit_code = await run_health_check()
        sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())