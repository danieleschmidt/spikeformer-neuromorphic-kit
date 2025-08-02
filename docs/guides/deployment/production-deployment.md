# Production Deployment Guide

## Overview

This guide covers deploying the Spikeformer Neuromorphic Kit in production environments, including cloud deployments, on-premises installations, and edge device deployments.

## Deployment Options

### 1. Cloud Deployment (AWS/GCP/Azure)

#### Prerequisites
- Container orchestration platform (Kubernetes/Docker Swarm)
- Access to neuromorphic hardware or simulation capabilities
- Monitoring and logging infrastructure
- CI/CD pipeline setup

#### Kubernetes Deployment

**Namespace Configuration**:
```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: spikeformer
  labels:
    name: spikeformer
    monitoring: enabled
```

**Application Deployment**:
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spikeformer-api
  namespace: spikeformer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spikeformer-api
  template:
    metadata:
      labels:
        app: spikeformer-api
    spec:
      containers:
      - name: spikeformer
        image: spikeformer:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: spikeformer-secrets
              key: database-url
        - name: HARDWARE_CONFIG
          value: "cloud-simulation"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service Configuration**:
```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: spikeformer-service
  namespace: spikeformer
spec:
  selector:
    app: spikeformer-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 2. On-Premises Deployment

#### Hardware Requirements

**Minimum Requirements**:
- CPU: 16 cores
- RAM: 64GB
- Storage: 1TB SSD
- GPU: NVIDIA RTX 4090 or equivalent (optional)
- Network: 10Gbps

**Recommended for Production**:
- CPU: 32+ cores
- RAM: 128GB+
- Storage: 2TB+ NVMe SSD
- GPU: Multiple NVIDIA A100/H100
- Network: 25Gbps+
- Neuromorphic Hardware: Intel Loihi 2, SpiNNaker2

#### Docker Compose Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  spikeformer-api:
    image: spikeformer:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - HARDWARE_CONFIG=on-premises
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - /dev/nxcore:/dev/nxcore  # Loihi 2 device access
    depends_on:
      - postgres
      - redis
      - prometheus
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '4'
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: spikeformer
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - spikeformer-api

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 3. Edge Deployment

#### Edge Device Configuration

**Supported Devices**:
- NVIDIA Jetson AGX Xavier/Orin
- Intel NUC with Loihi USB stick
- Custom neuromorphic edge devices
- ARM-based single board computers

**Deployment Script**:
```bash
#!/bin/bash
# deploy-edge.sh

# Configuration
DEVICE_TYPE=${1:-jetson}
MODEL_SIZE=${2:-small}
HARDWARE_TARGET=${3:-cpu}

echo "Deploying Spikeformer to $DEVICE_TYPE with $MODEL_SIZE models for $HARDWARE_TARGET"

# Install dependencies
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Download optimized container
docker pull spikeformer:edge-${DEVICE_TYPE}

# Create deployment directory
mkdir -p /opt/spikeformer
cd /opt/spikeformer

# Download edge configuration
curl -O https://releases.spikeformer.ai/edge/docker-compose.edge.yml
curl -O https://releases.spikeformer.ai/edge/config.${DEVICE_TYPE}.yml

# Configure hardware access
if [ "$HARDWARE_TARGET" = "loihi2" ]; then
    # Set up Loihi 2 USB access
    sudo usermod -a -G dialout $USER
    echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b01", MODE="0666"' | sudo tee /etc/udev/rules.d/99-loihi.rules
    sudo udevadm control --reload-rules
fi

# Start services
docker-compose -f docker-compose.edge.yml up -d

echo "Deployment complete. Access at http://localhost:8080"
```

## Configuration Management

### Environment Configuration

```python
# config/production.py
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # Application settings
    debug: bool = False
    testing: bool = False
    secret_key: str = os.getenv('SECRET_KEY', 'change-in-production')
    
    # Database configuration
    database_url: str = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/spikeformer')
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Hardware configuration
    hardware_config: str = os.getenv('HARDWARE_CONFIG', 'simulation')
    loihi2_devices: List[str] = os.getenv('LOIHI2_DEVICES', '').split(',')
    spinnaker_boards: List[str] = os.getenv('SPINNAKER_BOARDS', '').split(',')
    
    # Performance settings
    worker_processes: int = int(os.getenv('WORKER_PROCESSES', '4'))
    max_model_size_mb: int = int(os.getenv('MAX_MODEL_SIZE_MB', '1000'))
    conversion_timeout: int = int(os.getenv('CONVERSION_TIMEOUT', '1800'))
    
    # Monitoring settings
    prometheus_enabled: bool = os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true'
    jaeger_endpoint: Optional[str] = os.getenv('JAEGER_ENDPOINT')
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Security settings
    allowed_hosts: List[str] = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')
    cors_origins: List[str] = os.getenv('CORS_ORIGINS', '').split(',')
    rate_limit: str = os.getenv('RATE_LIMIT', '100/hour')
```

### Secrets Management

**Using Kubernetes Secrets**:
```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: spikeformer-secrets
  namespace: spikeformer
type: Opaque
data:
  database-url: <base64-encoded-database-url>
  secret-key: <base64-encoded-secret-key>
  api-key: <base64-encoded-api-key>
```

**Using HashiCorp Vault**:
```python
# utils/vault_config.py
import hvac

class VaultSecretManager:
    """Manage secrets with HashiCorp Vault."""
    
    def __init__(self, vault_url: str, token: str):
        self.client = hvac.Client(url=vault_url, token=token)
    
    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from Vault."""
        response = self.client.secrets.kv.v2.read_secret_version(path=path)
        return response['data']['data'][key]
    
    def get_database_config(self) -> dict:
        """Get database configuration from Vault."""
        return {
            'url': self.get_secret('spikeformer/db', 'url'),
            'username': self.get_secret('spikeformer/db', 'username'),
            'password': self.get_secret('spikeformer/db', 'password')
        }
```

## Scaling and Load Balancing

### Horizontal Scaling

**Auto-scaling Configuration**:
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spikeformer-hpa
  namespace: spikeformer
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spikeformer-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: conversion_queue_length
      target:
        type: AverageValue
        averageValue: "5"
```

### Load Balancing

**NGINX Configuration**:
```nginx
# nginx/nginx.conf
upstream spikeformer_backend {
    least_conn;
    server spikeformer-api-1:8000 weight=3;
    server spikeformer-api-2:8000 weight=3;
    server spikeformer-api-3:8000 weight=3;
    server spikeformer-api-4:8000 weight=1 backup;
}

server {
    listen 80;
    server_name api.spikeformer.ai;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.spikeformer.ai;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    location / {
        proxy_pass http://spikeformer_backend;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for long-running conversions
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 1800s; # 30 minutes for model conversion
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /health {
        proxy_pass http://spikeformer_backend;
        access_log off;
    }
    
    location /metrics {
        proxy_pass http://spikeformer_backend;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
    }
}
```

## Security Hardening

### Container Security

**Dockerfile Security Best Practices**:
```dockerfile
# Use specific version and minimal base image
FROM python:3.11-slim@sha256:specific-hash

# Create non-root user
RUN groupadd -r spikeformer && useradd -r -g spikeformer spikeformer

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=spikeformer:spikeformer . .

# Switch to non-root user
USER spikeformer

# Use specific port and disable debug
EXPOSE 8000
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:create_app()"]
```

### Network Security

**Network Policies**:
```yaml
# kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: spikeformer-network-policy
  namespace: spikeformer
spec:
  podSelector:
    matchLabels:
      app: spikeformer-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## Disaster Recovery

### Backup Strategy

**Database Backups**:
```bash
#!/bin/bash
# scripts/backup-database.sh

BACKUP_DIR="/backups/spikeformer"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="spikeformer_backup_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump $DATABASE_URL > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Upload to S3 (or other cloud storage)
aws s3 cp "$BACKUP_DIR/${BACKUP_FILE}.gz" s3://spikeformer-backups/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

**Model and Configuration Backups**:
```python
# scripts/backup_models.py
import os
import shutil
import datetime
from pathlib import Path

def backup_models():
    """Backup trained models and configurations."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"/backups/models/backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup model files
    models_dir = Path("/app/models")
    if models_dir.exists():
        shutil.copytree(models_dir, backup_dir / "models")
    
    # Backup configurations
    config_dir = Path("/app/config")
    if config_dir.exists():
        shutil.copytree(config_dir, backup_dir / "config")
    
    # Create archive
    archive_path = f"/backups/models_backup_{timestamp}.tar.gz"
    shutil.make_archive(archive_path.replace('.tar.gz', ''), 'gztar', backup_dir)
    
    # Clean up temporary directory
    shutil.rmtree(backup_dir)
    
    print(f"Models backup created: {archive_path}")

if __name__ == "__main__":
    backup_models()
```

### Recovery Procedures

**Database Recovery**:
```bash
#!/bin/bash
# scripts/restore-database.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Download backup from S3
aws s3 cp "s3://spikeformer-backups/$BACKUP_FILE" ./

# Decompress if needed
if [[ $BACKUP_FILE == *.gz ]]; then
    gunzip "$BACKUP_FILE"
    BACKUP_FILE=${BACKUP_FILE%.gz}
fi

# Stop application
docker-compose stop spikeformer-api

# Drop and recreate database
psql $DATABASE_URL -c "DROP DATABASE IF EXISTS spikeformer;"
psql $DATABASE_URL -c "CREATE DATABASE spikeformer;"

# Restore database
psql $DATABASE_URL < "$BACKUP_FILE"

# Start application
docker-compose start spikeformer-api

echo "Database restored from $BACKUP_FILE"
```

## Performance Optimization

### Resource Optimization

**CPU and Memory Tuning**:
```python
# config/performance.py
import multiprocessing
import psutil

class PerformanceConfig:
    """Performance optimization configuration."""
    
    @staticmethod
    def get_optimal_workers():
        """Calculate optimal number of worker processes."""
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Rule of thumb: 2 workers per CPU core, limited by memory
        max_workers_cpu = cpu_count * 2
        max_workers_memory = int(memory_gb / 2)  # 2GB per worker
        
        return min(max_workers_cpu, max_workers_memory, 16)  # Cap at 16
    
    @staticmethod
    def get_conversion_pool_size():
        """Get optimal conversion pool size."""
        return min(multiprocessing.cpu_count(), 8)
    
    @staticmethod
    def get_cache_size():
        """Get optimal cache size based on available memory."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        return min(int(memory_gb * 0.1), 4)  # 10% of RAM, max 4GB
```

### Caching Strategy

**Redis Caching**:
```python
# utils/cache.py
import redis
import pickle
import hashlib
from typing import Any, Optional

class ModelCache:
    """Cache for converted models and intermediate results."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def get_cache_key(self, model_id: str, config: dict) -> str:
        """Generate cache key for model and configuration."""
        config_hash = hashlib.sha256(str(sorted(config.items())).encode()).hexdigest()
        return f"model:{model_id}:{config_hash}"
    
    def get_converted_model(self, model_id: str, config: dict) -> Optional[Any]:
        """Retrieve converted model from cache."""
        key = self.get_cache_key(model_id, config)
        cached_data = self.redis.get(key)
        
        if cached_data:
            return pickle.loads(cached_data)
        return None
    
    def cache_converted_model(self, model_id: str, config: dict, model: Any, ttl: int = 3600):
        """Cache converted model."""
        key = self.get_cache_key(model_id, config)
        data = pickle.dumps(model)
        self.redis.setex(key, ttl, data)
    
    def invalidate_model_cache(self, model_id: str):
        """Invalidate all cache entries for a model."""
        pattern = f"model:{model_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

This production deployment guide provides comprehensive coverage of deployment scenarios, security considerations, scaling strategies, and operational procedures for running Spikeformer in production environments.