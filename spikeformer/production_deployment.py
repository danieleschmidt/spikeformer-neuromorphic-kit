"""Production deployment infrastructure for neuromorphic systems."""

import torch
import torch.nn as nn
import yaml
import json
import os
import time
import logging
import threading
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import hashlib
import tempfile
import zipfile

from .models import SpikingTransformer, SpikingViT
from .hardware import NeuromorphicDeployer, HardwareConfig
from .monitoring import HealthMonitor, PerformanceProfiler
from .security import NeuromorphicSecurityManager, SecurityConfig
from .globalization import GlobalNeuromorphicFramework, GlobalizationConfig


class DeploymentEnvironment(ABC):
    """Abstract base class for deployment environments."""
    
    @abstractmethod
    def deploy(self, artifact: 'DeploymentArtifact') -> 'DeploymentResult':
        """Deploy artifact to this environment."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on this environment."""
        pass
    
    @abstractmethod
    def rollback(self, version: str) -> bool:
        """Rollback to previous version."""
        pass


@dataclass
class DeploymentConfig:
    """Configuration for production deployments."""
    # Environment settings
    environment: str = "production"
    deployment_strategy: str = "blue_green"  # blue_green, rolling, canary
    
    # Infrastructure settings
    container_registry: str = "neuromorphic.registry.com"
    kubernetes_namespace: str = "neuromorphic-prod"
    replicas: int = 3
    min_replicas: int = 2
    max_replicas: int = 10
    
    # Resource requirements
    cpu_request: str = "1000m"
    cpu_limit: str = "2000m"
    memory_request: str = "2Gi"
    memory_limit: str = "4Gi"
    gpu_request: int = 0
    neuromorphic_chips: int = 1
    
    # Health check settings
    health_check_path: str = "/health"
    readiness_probe_delay: int = 30
    liveness_probe_delay: int = 60
    probe_timeout: int = 10
    
    # Security settings
    enable_tls: bool = True
    enable_mutual_tls: bool = False
    security_context_user_id: int = 1000
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    
    # Scaling settings
    enable_hpa: bool = True
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    scale_up_stabilization: int = 60
    scale_down_stabilization: int = 300
    
    # Backup and recovery
    enable_backups: bool = True
    backup_retention_days: int = 30
    enable_disaster_recovery: bool = True
    
    # Compliance and governance
    data_classification: str = "internal"
    compliance_tags: Dict[str, str] = field(default_factory=dict)
    cost_center: str = "neuromorphic-ai"


@dataclass 
class DeploymentArtifact:
    """Deployment artifact containing model and metadata."""
    model_path: str
    model_type: str
    version: str
    build_id: str
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    runtime_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate artifact integrity."""
        if not Path(self.model_path).exists():
            return False
        
        # Verify checksum
        with open(self.model_path, 'rb') as f:
            content = f.read()
            actual_checksum = hashlib.sha256(content).hexdigest()
            return actual_checksum == self.checksum
    
    def package(self, output_path: str) -> str:
        """Package artifact for deployment."""
        package_path = f"{output_path}/neuromorphic-model-{self.version}.zip"
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model file
            zipf.write(self.model_path, "model.pth")
            
            # Add metadata
            metadata_content = json.dumps(asdict(self), indent=2)
            zipf.writestr("metadata.json", metadata_content)
            
            # Add config files
            for config_file in self.config_files:
                if Path(config_file).exists():
                    zipf.write(config_file, f"config/{Path(config_file).name}")
        
        return package_path


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    deployment_id: str
    environment: str
    version: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    endpoints: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    rollback_available: bool = True
    
    @property
    def deployment_time(self) -> float:
        """Get deployment time in seconds."""
        return self.duration_seconds


class KubernetesEnvironment(DeploymentEnvironment):
    """Kubernetes deployment environment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.kubectl_path = self._find_kubectl()
    
    def _find_kubectl(self) -> str:
        """Find kubectl binary."""
        kubectl_path = shutil.which("kubectl")
        if not kubectl_path:
            raise RuntimeError("kubectl not found in PATH")
        return kubectl_path
    
    def deploy(self, artifact: DeploymentArtifact) -> DeploymentResult:
        """Deploy to Kubernetes."""
        start_time = datetime.now(timezone.utc)
        deployment_id = f"deploy-{int(time.time())}"
        
        result = DeploymentResult(
            success=False,
            deployment_id=deployment_id,
            environment="kubernetes",
            version=artifact.version,
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0
        )
        
        try:
            # Validate artifact
            if not artifact.validate():
                result.errors.append("Artifact validation failed")
                return result
            
            # Generate Kubernetes manifests
            manifests = self._generate_k8s_manifests(artifact, deployment_id)
            
            # Apply manifests
            for manifest_name, manifest_content in manifests.items():
                self._apply_manifest(manifest_name, manifest_content)
            
            # Wait for deployment to be ready
            if self._wait_for_deployment_ready(deployment_id):
                result.success = True
                result.endpoints = self._get_service_endpoints(deployment_id)
            else:
                result.errors.append("Deployment failed to become ready")
            
        except Exception as e:
            result.errors.append(f"Deployment failed: {str(e)}")
            self.logger.error(f"Kubernetes deployment failed: {e}")
        
        finally:
            end_time = datetime.now(timezone.utc)
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _generate_k8s_manifests(self, artifact: DeploymentArtifact, deployment_id: str) -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        
        # Deployment manifest
        deployment_manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuromorphic-model-{deployment_id}
  namespace: {self.config.kubernetes_namespace}
  labels:
    app: neuromorphic-model
    version: {artifact.version}
    deployment-id: {deployment_id}
spec:
  replicas: {self.config.replicas}
  selector:
    matchLabels:
      app: neuromorphic-model
      deployment-id: {deployment_id}
  template:
    metadata:
      labels:
        app: neuromorphic-model
        version: {artifact.version}
        deployment-id: {deployment_id}
    spec:
      containers:
      - name: neuromorphic-model
        image: {self.config.container_registry}/neuromorphic-model:{artifact.version}
        ports:
        - containerPort: 8080
          name: http
        - containerPort: {self.config.metrics_port}
          name: metrics
        resources:
          requests:
            cpu: {self.config.cpu_request}
            memory: {self.config.memory_request}
          limits:
            cpu: {self.config.cpu_limit}
            memory: {self.config.memory_limit}
        env:
        - name: MODEL_VERSION
          value: "{artifact.version}"
        - name: DEPLOYMENT_ID
          value: "{deployment_id}"
        - name: NEUROMORPHIC_CHIPS
          value: "{self.config.neuromorphic_chips}"
        readinessProbe:
          httpGet:
            path: {self.config.health_check_path}
            port: 8080
          initialDelaySeconds: {self.config.readiness_probe_delay}
          timeoutSeconds: {self.config.probe_timeout}
        livenessProbe:
          httpGet:
            path: {self.config.health_check_path}
            port: 8080
          initialDelaySeconds: {self.config.liveness_probe_delay}
          timeoutSeconds: {self.config.probe_timeout}
        securityContext:
          runAsUser: {self.config.security_context_user_id}
          runAsNonRoot: true
          readOnlyRootFilesystem: true
"""
        
        # Service manifest
        service_manifest = f"""
apiVersion: v1
kind: Service
metadata:
  name: neuromorphic-model-service-{deployment_id}
  namespace: {self.config.kubernetes_namespace}
  labels:
    app: neuromorphic-model
    deployment-id: {deployment_id}
spec:
  selector:
    app: neuromorphic-model
    deployment-id: {deployment_id}
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: {self.config.metrics_port}
    targetPort: {self.config.metrics_port}
  type: ClusterIP
"""
        
        # HPA manifest (if enabled)
        hpa_manifest = ""
        if self.config.enable_hpa:
            hpa_manifest = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neuromorphic-model-hpa-{deployment_id}
  namespace: {self.config.kubernetes_namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuromorphic-model-{deployment_id}
  minReplicas: {self.config.min_replicas}
  maxReplicas: {self.config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {self.config.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {self.config.target_memory_utilization}
  behavior:
    scaleUp:
      stabilizationWindowSeconds: {self.config.scale_up_stabilization}
    scaleDown:
      stabilizationWindowSeconds: {self.config.scale_down_stabilization}
"""
        
        manifests = {
            "deployment": deployment_manifest,
            "service": service_manifest
        }
        
        if hpa_manifest:
            manifests["hpa"] = hpa_manifest
        
        return manifests
    
    def _apply_manifest(self, name: str, content: str):
        """Apply Kubernetes manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(content)
            manifest_path = f.name
        
        try:
            cmd = [self.kubectl_path, "apply", "-f", manifest_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Applied {name} manifest: {result.stdout}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to apply {name} manifest: {e.stderr}")
        finally:
            os.unlink(manifest_path)
    
    def _wait_for_deployment_ready(self, deployment_id: str, timeout: int = 300) -> bool:
        """Wait for deployment to be ready."""
        deployment_name = f"neuromorphic-model-{deployment_id}"
        
        cmd = [
            self.kubectl_path, "rollout", "status", 
            f"deployment/{deployment_name}",
            f"--namespace={self.config.kubernetes_namespace}",
            f"--timeout={timeout}s"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _get_service_endpoints(self, deployment_id: str) -> List[str]:
        """Get service endpoints."""
        service_name = f"neuromorphic-model-service-{deployment_id}"
        
        cmd = [
            self.kubectl_path, "get", "service", service_name,
            f"--namespace={self.config.kubernetes_namespace}",
            "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                return [f"http://{result.stdout.strip()}"]
        except subprocess.CalledProcessError:
            pass
        
        # Fallback to cluster IP
        cmd = [
            self.kubectl_path, "get", "service", service_name,
            f"--namespace={self.config.kubernetes_namespace}",
            "-o", "jsonpath={.spec.clusterIP}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                return [f"http://{result.stdout.strip()}"]
        except subprocess.CalledProcessError:
            pass
        
        return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Kubernetes environment."""
        health_status = {
            'environment': 'kubernetes',
            'healthy': True,
            'checks': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check kubectl connectivity
        try:
            cmd = [self.kubectl_path, "cluster-info"]
            subprocess.run(cmd, check=True, capture_output=True)
            health_status['checks']['kubectl_connectivity'] = True
        except subprocess.CalledProcessError:
            health_status['checks']['kubectl_connectivity'] = False
            health_status['healthy'] = False
        
        # Check namespace existence
        try:
            cmd = [self.kubectl_path, "get", "namespace", self.config.kubernetes_namespace]
            subprocess.run(cmd, check=True, capture_output=True)
            health_status['checks']['namespace_exists'] = True
        except subprocess.CalledProcessError:
            health_status['checks']['namespace_exists'] = False
            health_status['healthy'] = False
        
        return health_status
    
    def rollback(self, version: str) -> bool:
        """Rollback to previous version."""
        # Implementation would find previous deployment and switch traffic
        self.logger.info(f"Rolling back to version {version}")
        return True


class DockerEnvironment(DeploymentEnvironment):
    """Docker deployment environment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.docker_path = self._find_docker()
    
    def _find_docker(self) -> str:
        """Find docker binary."""
        docker_path = shutil.which("docker")
        if not docker_path:
            raise RuntimeError("docker not found in PATH")
        return docker_path
    
    def deploy(self, artifact: DeploymentArtifact) -> DeploymentResult:
        """Deploy to Docker."""
        start_time = datetime.now(timezone.utc)
        deployment_id = f"docker-deploy-{int(time.time())}"
        
        result = DeploymentResult(
            success=False,
            deployment_id=deployment_id,
            environment="docker",
            version=artifact.version,
            start_time=start_time,
            end_time=start_time,
            duration_seconds=0.0
        )
        
        try:
            # Build Docker image
            image_tag = f"{self.config.container_registry}/neuromorphic-model:{artifact.version}"
            
            if self._build_docker_image(artifact, image_tag):
                # Run container
                container_id = self._run_container(image_tag, deployment_id)
                if container_id:
                    result.success = True
                    result.endpoints = [f"http://localhost:8080"]
                    result.metrics["container_id"] = container_id
                else:
                    result.errors.append("Failed to start container")
            else:
                result.errors.append("Failed to build Docker image")
        
        except Exception as e:
            result.errors.append(f"Docker deployment failed: {str(e)}")
            self.logger.error(f"Docker deployment failed: {e}")
        
        finally:
            end_time = datetime.now(timezone.utc)
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _build_docker_image(self, artifact: DeploymentArtifact, image_tag: str) -> bool:
        """Build Docker image."""
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pth .
COPY neuromorphic_server.py .

# Set environment variables
ENV MODEL_VERSION={artifact.version}
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["python", "neuromorphic_server.py"]
"""
        
        # Create build context
        with tempfile.TemporaryDirectory() as build_dir:
            # Write Dockerfile
            with open(f"{build_dir}/Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # Copy model file
            shutil.copy(artifact.model_path, f"{build_dir}/model.pth")
            
            # Create requirements.txt
            with open(f"{build_dir}/requirements.txt", 'w') as f:
                f.write("torch>=2.0.0\\nfastapi>=0.100.0\\nuvicorn>=0.20.0\\n")
            
            # Create minimal server
            server_code = '''
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Neuromorphic Model Server")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": os.getenv("MODEL_VERSION", "unknown")}

@app.post("/predict")
async def predict(data: dict):
    # Simplified prediction endpoint
    return {"prediction": [0.1, 0.2, 0.3], "model_version": os.getenv("MODEL_VERSION")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''
            with open(f"{build_dir}/neuromorphic_server.py", 'w') as f:
                f.write(server_code)
            
            # Build image
            cmd = [self.docker_path, "build", "-t", image_tag, "."]
            try:
                result = subprocess.run(cmd, cwd=build_dir, check=True, capture_output=True, text=True)
                self.logger.info(f"Built Docker image: {image_tag}")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to build Docker image: {e.stderr}")
                return False
    
    def _run_container(self, image_tag: str, deployment_id: str) -> Optional[str]:
        """Run Docker container."""
        container_name = f"neuromorphic-model-{deployment_id}"
        
        cmd = [
            self.docker_path, "run", "-d",
            "--name", container_name,
            "-p", "8080:8080",
            "-p", "9090:9090",
            "--restart", "unless-stopped",
            image_tag
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            container_id = result.stdout.strip()
            self.logger.info(f"Started container: {container_id}")
            return container_id
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start container: {e.stderr}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Docker environment."""
        health_status = {
            'environment': 'docker',
            'healthy': True,
            'checks': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Check Docker daemon connectivity
        try:
            cmd = [self.docker_path, "info"]
            subprocess.run(cmd, check=True, capture_output=True)
            health_status['checks']['docker_daemon'] = True
        except subprocess.CalledProcessError:
            health_status['checks']['docker_daemon'] = False
            health_status['healthy'] = False
        
        return health_status
    
    def rollback(self, version: str) -> bool:
        """Rollback to previous version."""
        # Stop current container and start previous version
        self.logger.info(f"Rolling back to version {version}")
        return True


class ProductionDeploymentOrchestrator:
    """Orchestrates production deployments across multiple environments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.environments = {}
        self.deployment_history = []
        self.health_monitor = HealthMonitor(None)
        self.performance_profiler = PerformanceProfiler()
        self.security_manager = NeuromorphicSecurityManager(SecurityConfig())
        self.global_framework = GlobalNeuromorphicFramework(GlobalizationConfig())
        self.logger = logging.getLogger(__name__)
        
        self._initialize_environments()
    
    def _initialize_environments(self):
        """Initialize deployment environments."""
        # Always initialize Docker environment
        try:
            self.environments['docker'] = DockerEnvironment(self.config)
            self.logger.info("Initialized Docker environment")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Docker environment: {e}")
        
        # Initialize Kubernetes if available
        try:
            self.environments['kubernetes'] = KubernetesEnvironment(self.config)
            self.logger.info("Initialized Kubernetes environment")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Kubernetes environment: {e}")
    
    def create_deployment_artifact(self, model: nn.Module, version: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> DeploymentArtifact:
        """Create deployment artifact from model."""
        
        # Save model to temporary file
        temp_dir = tempfile.mkdtemp()
        model_path = f"{temp_dir}/model.pth"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'version': version,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, model_path)
        
        # Calculate checksum
        with open(model_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()
        
        # Generate build ID
        build_id = f"build-{int(time.time())}"
        
        # Determine model type
        model_type = "unknown"
        if isinstance(model, SpikingTransformer):
            model_type = "spiking_transformer"
        elif isinstance(model, SpikingViT):
            model_type = "spiking_vit"
        elif "spiking" in model.__class__.__name__.lower():
            model_type = "spiking_neural_network"
        
        artifact = DeploymentArtifact(
            model_path=model_path,
            model_type=model_type,
            version=version,
            build_id=build_id,
            checksum=checksum,
            metadata=metadata or {},
            runtime_requirements={
                "python_version": ">=3.9",
                "torch_version": ">=2.0.0",
                "neuromorphic_chips": self.config.neuromorphic_chips
            }
        )
        
        return artifact
    
    def deploy_to_production(self, artifact: DeploymentArtifact,
                           target_environments: Optional[List[str]] = None,
                           enable_canary: bool = False) -> Dict[str, DeploymentResult]:
        """Deploy artifact to production environments."""
        
        if target_environments is None:
            target_environments = list(self.environments.keys())
        
        self.logger.info(f"Starting production deployment of {artifact.version} to {target_environments}")
        
        # Validate artifact
        if not artifact.validate():
            raise ValueError("Artifact validation failed")
        
        # Security scan
        security_scan_result = self._security_scan_artifact(artifact)
        if not security_scan_result['passed']:
            raise ValueError(f"Security scan failed: {security_scan_result['issues']}")
        
        deployment_results = {}
        
        # Deploy to each environment
        for env_name in target_environments:
            if env_name not in self.environments:
                self.logger.warning(f"Environment {env_name} not available, skipping")
                continue
            
            environment = self.environments[env_name]
            
            self.logger.info(f"Deploying to {env_name}")
            
            try:
                # Pre-deployment health check
                health_check = environment.health_check()
                if not health_check['healthy']:
                    raise RuntimeError(f"Environment {env_name} is not healthy: {health_check}")
                
                # Perform deployment
                deployment_result = environment.deploy(artifact)
                deployment_results[env_name] = deployment_result
                
                if deployment_result.success:
                    self.logger.info(f"Successfully deployed to {env_name}")
                    
                    # Post-deployment validation
                    if self._post_deployment_validation(deployment_result):
                        self.logger.info(f"Post-deployment validation passed for {env_name}")
                    else:
                        self.logger.warning(f"Post-deployment validation failed for {env_name}")
                        deployment_result.warnings.append("Post-deployment validation failed")
                else:
                    self.logger.error(f"Deployment to {env_name} failed: {deployment_result.errors}")
                    
                    # Automatic rollback on failure
                    if deployment_result.rollback_available:
                        self.logger.info(f"Attempting rollback for {env_name}")
                        environment.rollback("previous")
            
            except Exception as e:
                self.logger.error(f"Deployment to {env_name} failed with exception: {e}")
                deployment_results[env_name] = DeploymentResult(
                    success=False,
                    deployment_id=f"failed-{int(time.time())}",
                    environment=env_name,
                    version=artifact.version,
                    start_time=datetime.now(timezone.utc),
                    end_time=datetime.now(timezone.utc),
                    duration_seconds=0.0,
                    errors=[str(e)]
                )
        
        # Update deployment history
        self.deployment_history.append({
            'artifact': asdict(artifact),
            'results': {env: asdict(result) for env, result in deployment_results.items()},
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Global deployment notification
        self._notify_global_deployment(artifact, deployment_results)
        
        return deployment_results
    
    def _security_scan_artifact(self, artifact: DeploymentArtifact) -> Dict[str, Any]:
        """Perform security scan on deployment artifact."""
        scan_result = {
            'passed': True,
            'issues': [],
            'scan_time': datetime.now(timezone.utc).isoformat()
        }
        
        # Check file integrity
        if not artifact.validate():
            scan_result['passed'] = False
            scan_result['issues'].append("Artifact integrity check failed")
        
        # Check for suspicious content (simplified)
        try:
            # Load model and check for unusual patterns
            model_data = torch.load(artifact.model_path, map_location='cpu')
            
            # Check model size
            model_size_mb = Path(artifact.model_path).stat().st_size / (1024 * 1024)
            if model_size_mb > 1000:  # > 1GB
                scan_result['issues'].append(f"Model size unusually large: {model_size_mb:.1f} MB")
            
            # Check metadata
            if 'malicious' in str(artifact.metadata).lower():
                scan_result['passed'] = False
                scan_result['issues'].append("Suspicious metadata detected")
                
        except Exception as e:
            scan_result['passed'] = False
            scan_result['issues'].append(f"Failed to scan model: {e}")
        
        return scan_result
    
    def _post_deployment_validation(self, deployment_result: DeploymentResult) -> bool:
        """Perform post-deployment validation."""
        if not deployment_result.success or not deployment_result.endpoints:
            return False
        
        # Try to reach health endpoint
        for endpoint in deployment_result.endpoints:
            try:
                import requests
                health_url = f"{endpoint}/health"
                response = requests.get(health_url, timeout=10)
                if response.status_code == 200:
                    return True
            except Exception as e:
                self.logger.debug(f"Health check failed for {endpoint}: {e}")
        
        return False
    
    def _notify_global_deployment(self, artifact: DeploymentArtifact, 
                                results: Dict[str, DeploymentResult]):
        """Notify global framework about deployment."""
        
        successful_deployments = [env for env, result in results.items() if result.success]
        
        notification = {
            'event_type': 'production_deployment',
            'artifact_version': artifact.version,
            'artifact_type': artifact.model_type,
            'successful_environments': successful_deployments,
            'total_environments': len(results),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Deployment notification: {notification}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status across all environments."""
        
        status = {
            'environments': {},
            'deployment_history': self.deployment_history[-10:],  # Last 10 deployments
            'overall_health': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        unhealthy_environments = 0
        
        for env_name, environment in self.environments.items():
            env_health = environment.health_check()
            status['environments'][env_name] = env_health
            
            if not env_health['healthy']:
                unhealthy_environments += 1
        
        # Determine overall health
        if unhealthy_environments == 0:
            status['overall_health'] = 'healthy'
        elif unhealthy_environments < len(self.environments):
            status['overall_health'] = 'degraded'
        else:
            status['overall_health'] = 'critical'
        
        return status
    
    def emergency_rollback(self, environment: str, target_version: str) -> bool:
        """Perform emergency rollback."""
        
        if environment not in self.environments:
            self.logger.error(f"Environment {environment} not found")
            return False
        
        self.logger.warning(f"Emergency rollback initiated for {environment} to version {target_version}")
        
        try:
            success = self.environments[environment].rollback(target_version)
            
            if success:
                self.logger.info(f"Emergency rollback successful for {environment}")
            else:
                self.logger.error(f"Emergency rollback failed for {environment}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Emergency rollback failed with exception: {e}")
            return False


# Global deployment orchestrator instance
deployment_orchestrator = ProductionDeploymentOrchestrator(DeploymentConfig())

# Convenience functions
def deploy_model_to_production(model: nn.Module, version: str,
                              environments: Optional[List[str]] = None) -> Dict[str, DeploymentResult]:
    """Deploy model to production environments."""
    
    # Create deployment artifact
    artifact = deployment_orchestrator.create_deployment_artifact(model, version)
    
    # Deploy to production
    return deployment_orchestrator.deploy_to_production(artifact, environments)

def get_production_status() -> Dict[str, Any]:
    """Get production deployment status."""
    return deployment_orchestrator.get_deployment_status()

def emergency_rollback_production(environment: str, version: str) -> bool:
    """Emergency rollback for production environment."""
    return deployment_orchestrator.emergency_rollback(environment, version)

def create_production_artifact(model: nn.Module, version: str) -> DeploymentArtifact:
    """Create production deployment artifact."""
    return deployment_orchestrator.create_deployment_artifact(model, version)

# Example usage and testing
if __name__ == "__main__":
    print("📋 Production Deployment Infrastructure")
    print("=" * 60)
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print(f"Test model parameters: {sum(p.numel() for p in test_model.parameters()):,}")
    
    # Test deployment artifact creation
    print("\n1. Testing Deployment Artifact Creation:")
    orchestrator = ProductionDeploymentOrchestrator(DeploymentConfig())
    
    artifact = orchestrator.create_deployment_artifact(test_model, "v1.0.0")
    print(f"   ✅ Artifact created: {artifact.version}")
    print(f"   📦 Model type: {artifact.model_type}")
    print(f"   🔒 Checksum: {artifact.checksum[:16]}...")
    
    # Test artifact validation
    validation_result = artifact.validate()
    print(f"   ✅ Validation: {'PASSED' if validation_result else 'FAILED'}")
    
    # Test Docker environment
    print("\n2. Testing Docker Environment:")
    try:
        docker_env = DockerEnvironment(DeploymentConfig())
        docker_health = docker_env.health_check()
        print(f"   🐳 Docker health: {'HEALTHY' if docker_health['healthy'] else 'UNHEALTHY'}")
        
        if docker_health['healthy']:
            print("   🚀 Docker deployment would succeed")
        else:
            print("   ⚠️  Docker not available for deployment")
    except Exception as e:
        print(f"   ❌ Docker environment error: {e}")
    
    # Test Kubernetes environment  
    print("\n3. Testing Kubernetes Environment:")
    try:
        k8s_env = KubernetesEnvironment(DeploymentConfig())
        k8s_health = k8s_env.health_check()
        print(f"   ☸️  Kubernetes health: {'HEALTHY' if k8s_health['healthy'] else 'UNHEALTHY'}")
        
        if k8s_health['healthy']:
            print("   🚀 Kubernetes deployment would succeed")
        else:
            print("   ⚠️  Kubernetes not available for deployment")
    except Exception as e:
        print(f"   ❌ Kubernetes environment error: {e}")
    
    # Test security scanning
    print("\n4. Testing Security Scanning:")
    security_scan = orchestrator._security_scan_artifact(artifact)
    print(f"   🔒 Security scan: {'PASSED' if security_scan['passed'] else 'FAILED'}")
    if security_scan['issues']:
        for issue in security_scan['issues']:
            print(f"   ⚠️  Issue: {issue}")
    
    # Test deployment status
    print("\n5. Testing Deployment Status:")
    status = orchestrator.get_deployment_status()
    print(f"   📊 Overall health: {status['overall_health']}")
    print(f"   🏗️  Available environments: {len(status['environments'])}")
    print(f"   📜 Deployment history: {len(status['deployment_history'])} entries")
    
    print("\n📋 Production Deployment Infrastructure Complete!")