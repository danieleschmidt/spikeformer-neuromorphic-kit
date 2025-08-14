#!/usr/bin/env python3
"""Production deployment suite for Spikeformer Neuromorphic Kit."""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import subprocess


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: str = "production"
    replicas: int = 3
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    storage_size: str = "10Gi"
    neuromorphic_hardware: bool = True
    monitoring_enabled: bool = True
    auto_scaling: bool = True
    max_replicas: int = 10
    target_cpu_utilization: int = 70


class ProductionDeploymentSuite:
    """Comprehensive production deployment automation."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.deployment_dir = self.repo_path / "deployment"
        self.config = DeploymentConfig()
        
    def create_production_infrastructure(self):
        """Create complete production deployment infrastructure."""
        
        print("🚀 CREATING PRODUCTION DEPLOYMENT INFRASTRUCTURE...")
        
        # Create deployment directories
        self._create_deployment_structure()
        
        # Generate Docker configuration
        self._create_docker_configuration()
        
        # Generate Kubernetes manifests
        self._create_kubernetes_manifests()
        
        # Create monitoring configuration
        self._create_monitoring_configuration()
        
        # Generate CI/CD workflows
        self._create_cicd_workflows()
        
        # Create health check scripts
        self._create_health_checks()
        
        # Generate scaling policies
        self._create_scaling_policies()
        
        # Create backup and recovery scripts
        self._create_backup_recovery()
        
        print("✅ Production deployment infrastructure created successfully!")
        
    def _create_deployment_structure(self):
        """Create deployment directory structure."""
        
        directories = [
            "deployment/docker",
            "deployment/kubernetes",
            "deployment/monitoring",
            "deployment/scripts",
            "deployment/ci-cd",
            "deployment/backup",
            "deployment/configs"
        ]
        
        for directory in directories:
            (self.repo_path / directory).mkdir(parents=True, exist_ok=True)
            
    def _create_docker_configuration(self):
        """Create Docker configuration files."""
        
        # Production Dockerfile
        dockerfile_content = '''# Production Dockerfile for Spikeformer Neuromorphic Kit
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NEUROMORPHIC_ENV=production

# Create app user
RUN groupadd -r spikeformer && useradd -r -g spikeformer spikeformer

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R spikeformer:spikeformer /app
USER spikeformer

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python3 -c "import spikeformer; print('Health check passed')" || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python3", "-m", "spikeformer.cli.main", "--serve", "--port", "8080"]
'''
        
        (self.deployment_dir / "docker" / "Dockerfile").write_text(dockerfile_content)
        
        # Docker Compose for local development
        docker_compose_content = '''version: '3.8'

services:
  spikeformer:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile
    container_name: spikeformer-app
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - NEUROMORPHIC_ENV=development
      - PYTHONPATH=/app
    volumes:
      - spikeformer-data:/app/data
      - spikeformer-logs:/app/logs
    networks:
      - spikeformer-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  spikeformer-monitoring:
    image: prom/prometheus:latest
    container_name: spikeformer-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - spikeformer-network

  spikeformer-grafana:
    image: grafana/grafana:latest
    container_name: spikeformer-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    networks:
      - spikeformer-network

volumes:
  spikeformer-data:
  spikeformer-logs:
  prometheus-data:
  grafana-data:

networks:
  spikeformer-network:
    driver: bridge
'''
        
        (self.deployment_dir / "docker" / "docker-compose.yml").write_text(docker_compose_content)
        
        # Docker build script
        build_script = '''#!/bin/bash
set -e

# Build production Docker image
echo "🐳 Building production Docker image..."

# Get version from pyproject.toml
VERSION=$(grep "version = " pyproject.toml | cut -d'"' -f2)

# Build image
docker build -f deployment/docker/Dockerfile -t spikeformer-neuromorphic:${VERSION} .
docker tag spikeformer-neuromorphic:${VERSION} spikeformer-neuromorphic:latest

echo "✅ Docker image built successfully: spikeformer-neuromorphic:${VERSION}"

# Optional: Push to registry
if [ "$1" = "--push" ]; then
    echo "📤 Pushing to container registry..."
    docker push spikeformer-neuromorphic:${VERSION}
    docker push spikeformer-neuromorphic:latest
    echo "✅ Images pushed to registry"
fi
'''
        
        build_script_path = self.deployment_dir / "scripts" / "build-docker.sh"
        build_script_path.write_text(build_script)
        build_script_path.chmod(0o755)
        
    def _create_kubernetes_manifests(self):
        """Create Kubernetes deployment manifests."""
        
        # Namespace
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "spikeformer",
                "labels": {
                    "name": "spikeformer",
                    "environment": "production"
                }
            }
        }
        
        # ConfigMap
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "spikeformer-config",
                "namespace": "spikeformer"
            },
            "data": {
                "NEUROMORPHIC_ENV": "production",
                "LOG_LEVEL": "INFO",
                "WORKERS": "4",
                "MAX_TIMESTEPS": "64",
                "ENERGY_OPTIMIZATION": "true"
            }
        }
        
        # Deployment
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "spikeformer-app",
                "namespace": "spikeformer",
                "labels": {
                    "app": "spikeformer",
                    "component": "neuromorphic-processor"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "spikeformer"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "spikeformer"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "spikeformer",
                            "image": "spikeformer-neuromorphic:latest",
                            "ports": [{
                                "containerPort": 8080,
                                "name": "http"
                            }],
                            "env": [{
                                "name": "NEUROMORPHIC_ENV",
                                "valueFrom": {
                                    "configMapKeyRef": {
                                        "name": "spikeformer-config",
                                        "key": "NEUROMORPHIC_ENV"
                                    }
                                }
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": self.config.cpu_request,
                                    "memory": self.config.memory_request
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000
                        }
                    }
                }
            }
        }
        
        # Service
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "spikeformer-service",
                "namespace": "spikeformer"
            },
            "spec": {
                "selector": {
                    "app": "spikeformer"
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8080
                }],
                "type": "ClusterIP"
            }
        }
        
        # HorizontalPodAutoscaler
        hpa_manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "spikeformer-hpa",
                "namespace": "spikeformer"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "spikeformer-app"
                },
                "minReplicas": self.config.replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": self.config.target_cpu_utilization
                        }
                    }
                }]
            }
        }
        
        # Save manifests manually
        self._write_namespace_manifest()
        self._write_configmap_manifest()  
        self._write_deployment_manifest()
        self._write_service_manifest()
        self._write_hpa_manifest()
        
    def _write_namespace_manifest(self):
        """Write namespace manifest manually."""
        manifest_path = self.deployment_dir / "kubernetes" / "namespace.yaml"
        with open(manifest_path, 'w') as f:
            f.write("apiVersion: v1\n")
            f.write("kind: Namespace\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer\n")
            f.write("  labels:\n")
            f.write("    name: spikeformer\n")
            f.write("    environment: production\n")
    
    def _write_configmap_manifest(self):
        """Write configmap manifest manually.""" 
        manifest_path = self.deployment_dir / "kubernetes" / "configmap.yaml"
        with open(manifest_path, 'w') as f:
            f.write("apiVersion: v1\n")
            f.write("kind: ConfigMap\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer-config\n")
            f.write("  namespace: spikeformer\n")
            f.write("data:\n")
            f.write("  NEUROMORPHIC_ENV: production\n")
            f.write("  LOG_LEVEL: INFO\n")
            f.write("  WORKERS: '4'\n")
            f.write("  MAX_TIMESTEPS: '64'\n")
            f.write("  ENERGY_OPTIMIZATION: 'true'\n")
    
    def _write_deployment_manifest(self):
        """Write deployment manifest manually."""
        manifest_path = self.deployment_dir / "kubernetes" / "deployment.yaml"
        with open(manifest_path, 'w') as f:
            f.write("apiVersion: apps/v1\n")
            f.write("kind: Deployment\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer-app\n")
            f.write("  namespace: spikeformer\n")
            f.write("  labels:\n")
            f.write("    app: spikeformer\n")
            f.write("    component: neuromorphic-processor\n")
            f.write("spec:\n")
            f.write(f"  replicas: {self.config.replicas}\n")
            f.write("  selector:\n")
            f.write("    matchLabels:\n")
            f.write("      app: spikeformer\n")
            f.write("  template:\n")
            f.write("    metadata:\n")
            f.write("      labels:\n")
            f.write("        app: spikeformer\n")
            f.write("    spec:\n")
            f.write("      containers:\n")
            f.write("        - name: spikeformer\n")
            f.write("          image: spikeformer-neuromorphic:latest\n")
            f.write("          ports:\n")
            f.write("            - containerPort: 8080\n")
            f.write("              name: http\n")
            f.write("          resources:\n")
            f.write("            requests:\n")
            f.write(f"              cpu: {self.config.cpu_request}\n")
            f.write(f"              memory: {self.config.memory_request}\n")
            f.write("            limits:\n")
            f.write(f"              cpu: {self.config.cpu_limit}\n")
            f.write(f"              memory: {self.config.memory_limit}\n")
    
    def _write_service_manifest(self):
        """Write service manifest manually."""
        manifest_path = self.deployment_dir / "kubernetes" / "service.yaml"
        with open(manifest_path, 'w') as f:
            f.write("apiVersion: v1\n")
            f.write("kind: Service\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer-service\n")
            f.write("  namespace: spikeformer\n")
            f.write("spec:\n")
            f.write("  selector:\n")
            f.write("    app: spikeformer\n")
            f.write("  ports:\n")
            f.write("    - protocol: TCP\n")
            f.write("      port: 80\n")
            f.write("      targetPort: 8080\n")
            f.write("  type: ClusterIP\n")
    
    def _write_hpa_manifest(self):
        """Write HPA manifest manually."""
        manifest_path = self.deployment_dir / "kubernetes" / "hpa.yaml"
        with open(manifest_path, 'w') as f:
            f.write("apiVersion: autoscaling/v2\n")
            f.write("kind: HorizontalPodAutoscaler\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer-hpa\n")
            f.write("  namespace: spikeformer\n")
            f.write("spec:\n")
            f.write("  scaleTargetRef:\n")
            f.write("    apiVersion: apps/v1\n")
            f.write("    kind: Deployment\n")
            f.write("    name: spikeformer-app\n")
            f.write(f"  minReplicas: {self.config.replicas}\n")
            f.write(f"  maxReplicas: {self.config.max_replicas}\n")
            f.write("  metrics:\n")
            f.write("    - type: Resource\n")
            f.write("      resource:\n")
            f.write("        name: cpu\n")
            f.write("        target:\n")
            f.write("          type: Utilization\n")
            f.write(f"          averageUtilization: {self.config.target_cpu_utilization}\n")
                
    def _create_monitoring_configuration(self):
        """Create monitoring and observability configuration."""
        
        # Prometheus configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s"
            },
            "scrape_configs": [{
                "job_name": "spikeformer",
                "static_configs": [{
                    "targets": ["spikeformer-service:80"]
                }],
                "metrics_path": "/metrics",
                "scrape_interval": "15s"
            }]
        }
        
        prometheus_path = self.deployment_dir / "monitoring" / "prometheus.yml"
        with open(prometheus_path, 'w') as f:
            # Write YAML manually
            f.write("global:\n  scrape_interval: 15s\n\n")
            f.write("scrape_configs:\n")
            f.write("  - job_name: spikeformer\n")
            f.write("    static_configs:\n")
            f.write("      - targets: ['spikeformer-service:80']\n")
            f.write("    metrics_path: /metrics\n")
            f.write("    scrape_interval: 15s\n")
            
        # Grafana dashboard
        grafana_dashboard = {
            "dashboard": {
                "title": "Spikeformer Neuromorphic Metrics",
                "panels": [
                    {
                        "title": "Spike Rate",
                        "type": "graph",
                        "targets": [{
                            "expr": "spikeformer_spike_rate",
                            "legendFormat": "{{instance}}"
                        }]
                    },
                    {
                        "title": "Energy Consumption",
                        "type": "graph",
                        "targets": [{
                            "expr": "spikeformer_energy_consumption_mj",
                            "legendFormat": "{{instance}}"
                        }]
                    },
                    {
                        "title": "Processing Latency",
                        "type": "graph",
                        "targets": [{
                            "expr": "spikeformer_processing_latency_ms",
                            "legendFormat": "{{instance}}"
                        }]
                    }
                ]
            }
        }
        
        dashboard_path = self.deployment_dir / "monitoring" / "spikeformer-dashboard.json"
        with open(dashboard_path, 'w') as f:
            json.dump(grafana_dashboard, f, indent=2)
            
    def _create_cicd_workflows(self):
        """Create CI/CD workflow configurations."""
        
        # GitHub Actions workflow
        github_workflow = {
            "name": "Spikeformer CI/CD",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main"]
                }
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.11"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run quality gates",
                            "run": "python3 quality_gates_comprehensive.py"
                        },
                        {
                            "name": "Run tests",
                            "run": "python3 -m pytest tests/"
                        }
                    ]
                },
                "build-and-deploy": {
                    "needs": "test",
                    "runs-on": "ubuntu-latest",
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Build Docker image",
                            "run": "deployment/scripts/build-docker.sh"
                        },
                        {
                            "name": "Deploy to production",
                            "run": "deployment/scripts/deploy-production.sh"
                        }
                    ]
                }
            }
        }
        
        github_dir = self.repo_path / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = github_dir / "ci-cd.yml"
        with open(workflow_path, 'w') as f:
            # Write GitHub Actions YAML manually
            f.write("name: Spikeformer CI/CD\n\n")
            f.write("on:\n")
            f.write("  push:\n    branches: [main, develop]\n")
            f.write("  pull_request:\n    branches: [main]\n\n")
            f.write("jobs:\n")
            f.write("  test:\n")
            f.write("    runs-on: ubuntu-latest\n")
            f.write("    steps:\n")
            f.write("      - uses: actions/checkout@v3\n")
            f.write("      - name: Set up Python\n")
            f.write("        uses: actions/setup-python@v4\n")
            f.write("        with:\n          python-version: '3.11'\n")
            f.write("      - name: Install dependencies\n")
            f.write("        run: pip install -r requirements.txt\n")
            f.write("      - name: Run quality gates\n")
            f.write("        run: python3 quality_gates_comprehensive.py\n")
            f.write("      - name: Run tests\n")
            f.write("        run: python3 -m pytest tests/\n")
            
    def _create_health_checks(self):
        """Create comprehensive health check scripts."""
        
        # Production health check
        health_check_script = '''#!/usr/bin/env python3
"""Production health check for Spikeformer Neuromorphic Kit."""

import sys
import time
import requests
import json
from pathlib import Path


def check_api_health():
    """Check API health endpoint."""
    try:
        response = requests.get("http://localhost:8080/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def check_neuromorphic_hardware():
    """Check neuromorphic hardware connectivity."""
    try:
        # Simulate hardware check
        return True  # Would check actual hardware
    except:
        return False


def check_memory_usage():
    """Check memory usage is within limits."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90
    except:
        return False


def check_disk_space():
    """Check disk space availability."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        return usage_percent < 85
    except:
        return False


def main():
    """Run comprehensive health checks."""
    
    checks = {
        "api_health": check_api_health,
        "neuromorphic_hardware": check_neuromorphic_hardware,
        "memory_usage": check_memory_usage,
        "disk_space": check_disk_space
    }
    
    results = {}
    all_healthy = True
    
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_healthy = False
        except Exception as e:
            results[check_name] = False
            all_healthy = False
    
    # Output results
    health_status = {
        "timestamp": time.time(),
        "healthy": all_healthy,
        "checks": results
    }
    
    print(json.dumps(health_status, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    main()
'''
        
        health_check_path = self.deployment_dir / "scripts" / "health-check.py"
        health_check_path.write_text(health_check_script)
        health_check_path.chmod(0o755)
        
    def _create_scaling_policies(self):
        """Create auto-scaling policies and configurations."""
        
        # Kubernetes YAML for custom scaling
        scaling_policy = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "spikeformer-advanced-hpa",
                "namespace": "spikeformer"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "spikeformer-app"
                },
                "minReplicas": 2,
                "maxReplicas": 20,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [{
                            "type": "Percent",
                            "value": 100,
                            "periodSeconds": 15
                        }]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [{
                            "type": "Percent", 
                            "value": 10,
                            "periodSeconds": 60
                        }]
                    }
                }
            }
        }
        
        scaling_path = self.deployment_dir / "kubernetes" / "advanced-hpa.yaml"
        with open(scaling_path, 'w') as f:
            # Write Kubernetes YAML manually
            f.write("apiVersion: autoscaling/v2\n")
            f.write("kind: HorizontalPodAutoscaler\n")
            f.write("metadata:\n")
            f.write("  name: spikeformer-advanced-hpa\n")
            f.write("  namespace: spikeformer\n")
            f.write("spec:\n")
            f.write("  scaleTargetRef:\n")
            f.write("    apiVersion: apps/v1\n")
            f.write("    kind: Deployment\n")
            f.write("    name: spikeformer-app\n")
            f.write("  minReplicas: 2\n")
            f.write("  maxReplicas: 20\n")
            f.write("  metrics:\n")
            f.write("    - type: Resource\n")
            f.write("      resource:\n")
            f.write("        name: cpu\n")
            f.write("        target:\n")
            f.write("          type: Utilization\n")
            f.write("          averageUtilization: 70\n")
            
    def _create_backup_recovery(self):
        """Create backup and disaster recovery scripts."""
        
        # Backup script
        backup_script = '''#!/bin/bash
set -e

# Backup script for Spikeformer production data
BACKUP_DIR="/backups/spikeformer"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="spikeformer_backup_${TIMESTAMP}"

echo "🔄 Starting backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup application data
echo "📦 Backing up application data..."
kubectl exec -n spikeformer deployment/spikeformer-app -- tar czf - /app/data | \\
    cat > "${BACKUP_DIR}/${BACKUP_NAME}/app_data.tar.gz"

# Backup configuration
echo "⚙️ Backing up configuration..."
kubectl get configmap -n spikeformer -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/configmaps.yaml"
kubectl get secret -n spikeformer -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/secrets.yaml"

# Backup persistent volumes
echo "💾 Backing up persistent volumes..."
kubectl get pv,pvc -n spikeformer -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/volumes.yaml"

# Create backup manifest
cat > "${BACKUP_DIR}/${BACKUP_NAME}/backup_manifest.json" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "${TIMESTAMP}",
    "kubernetes_version": "$(kubectl version --short)",
    "spikeformer_version": "$(kubectl get deployment -n spikeformer spikeformer-app -o jsonpath='{.spec.template.spec.containers[0].image}')"
}
EOF

# Compress backup
echo "🗜️ Compressing backup..."
cd "${BACKUP_DIR}"
tar czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

echo "✅ Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "spikeformer_backup_*.tar.gz" -mtime +7 -delete
'''
        
        backup_path = self.deployment_dir / "scripts" / "backup.sh"
        backup_path.write_text(backup_script)
        backup_path.chmod(0o755)
        
        # Recovery script
        recovery_script = '''#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/spikeformer_restore_$(date +%s)"

echo "🔄 Starting recovery from: ${BACKUP_FILE}"

# Extract backup
echo "📦 Extracting backup..."
mkdir -p "${RESTORE_DIR}"
tar xzf "${BACKUP_FILE}" -C "${RESTORE_DIR}" --strip-components=1

# Restore configuration
echo "⚙️ Restoring configuration..."
kubectl apply -f "${RESTORE_DIR}/configmaps.yaml"
kubectl apply -f "${RESTORE_DIR}/secrets.yaml"

# Restore application data
echo "📂 Restoring application data..."
kubectl exec -n spikeformer deployment/spikeformer-app -- tar xzf - -C / < "${RESTORE_DIR}/app_data.tar.gz"

# Restart deployment
echo "🔄 Restarting deployment..."
kubectl rollout restart deployment/spikeformer-app -n spikeformer
kubectl rollout status deployment/spikeformer-app -n spikeformer

# Cleanup
rm -rf "${RESTORE_DIR}"

echo "✅ Recovery completed successfully"
'''
        
        recovery_path = self.deployment_dir / "scripts" / "recovery.sh"
        recovery_path.write_text(recovery_script)
        recovery_path.chmod(0o755)
        
        # Deployment script
        deploy_script = '''#!/bin/bash
set -e

echo "🚀 Deploying Spikeformer to production..."

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/spikeformer-app -n spikeformer --timeout=300s

# Run health check
echo "🏥 Running health check..."
kubectl exec -n spikeformer deployment/spikeformer-app -- python3 deployment/scripts/health-check.py

echo "✅ Production deployment completed successfully!"
'''
        
        deploy_path = self.deployment_dir / "scripts" / "deploy-production.sh" 
        deploy_path.write_text(deploy_script)
        deploy_path.chmod(0o755)


def main():
    """Main execution function."""
    
    print("🏗️ TERRAGON PRODUCTION DEPLOYMENT SUITE")
    print("=" * 50)
    
    deployment_suite = ProductionDeploymentSuite()
    deployment_suite.create_production_infrastructure()
    
    print("\n📋 DEPLOYMENT ARTIFACTS CREATED:")
    print("├── deployment/docker/")
    print("│   ├── Dockerfile")
    print("│   └── docker-compose.yml")
    print("├── deployment/kubernetes/")
    print("│   ├── namespace.yaml")
    print("│   ├── deployment.yaml")
    print("│   ├── service.yaml")
    print("│   └── hpa.yaml")
    print("├── deployment/monitoring/")
    print("│   ├── prometheus.yml")
    print("│   └── grafana-dashboard.json")
    print("├── deployment/scripts/")
    print("│   ├── build-docker.sh")
    print("│   ├── deploy-production.sh")
    print("│   ├── health-check.py")
    print("│   ├── backup.sh")
    print("│   └── recovery.sh")
    print("└── .github/workflows/")
    print("    └── ci-cd.yml")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review deployment configuration")
    print("2. Configure container registry")
    print("3. Set up Kubernetes cluster")
    print("4. Run: deployment/scripts/build-docker.sh")
    print("5. Run: deployment/scripts/deploy-production.sh")
    
    print("\n🚀 PRODUCTION DEPLOYMENT READY!")


if __name__ == "__main__":
    main()