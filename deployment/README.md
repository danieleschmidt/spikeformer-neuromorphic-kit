# Neuromorphic Platform Deployment Guide

This directory contains comprehensive deployment configurations and automation for the Neuromorphic Computing Platform, supporting multiple environments and deployment strategies.

## 🚀 Quick Start

### Prerequisites
- Kubernetes cluster (v1.25+)
- kubectl configured
- Docker with registry access
- Helm (optional, for advanced deployments)

### One-Command Deployment
```bash
# Deploy to production
./deploy.sh production deploy

# Deploy to development
./deploy.sh dev deploy

# Check deployment status
./deploy.sh production status
```

## 📁 Directory Structure

```
deployment/
├── deploy.sh                 # Main deployment automation script
├── docker/                   # Docker configurations
│   ├── Dockerfile            # Multi-stage production Dockerfile
│   └── docker-compose.yml    # Local development stack
├── kubernetes/               # Kubernetes manifests
│   └── deployment.yaml       # Complete K8s deployment
├── monitoring/               # Monitoring and observability
│   └── prometheus-rules.yaml # Advanced monitoring rules
├── ci-cd/                    # CI/CD pipeline configurations
│   └── github-actions.yaml   # GitHub Actions workflow
├── scripts/                  # Utility scripts
│   └── production-health-check.py  # Health monitoring
└── production_config.py      # Environment configurations
```

## 🏗️ Architecture Overview

The deployment supports multiple environments with different characteristics:

### Environment Types

| Environment | Purpose | Scale | Features |
|-------------|---------|-------|----------|
| **Development** | Local testing | Single node | Hot reload, debug mode |
| **Staging** | Pre-production | 2-3 nodes | Full monitoring, testing |
| **Production** | Live system | 3+ nodes | HA, scaling, security |
| **Edge** | Neuromorphic hardware | 1 node | Low latency, optimized |
| **Research** | Experimentation | Variable | Research tools, logging |

### Components Deployed

1. **Neuromorphic API Service**
   - Multi-chip inference engine
   - Auto-scaling (HPA)
   - Health checks and monitoring

2. **Database Layer**
   - PostgreSQL (StatefulSet)
   - Persistent storage
   - Backup automation

3. **Cache Layer**
   - Redis cluster
   - Session management
   - Result caching

4. **Monitoring Stack**
   - Prometheus metrics
   - Custom neuromorphic alerts
   - Performance dashboards

5. **Security**
   - Network policies
   - Pod security standards
   - Secret management

## 🔧 Deployment Methods

### Method 1: Automated Script (Recommended)

```bash
# Full deployment
./deploy.sh production deploy

# Update existing deployment
./deploy.sh production update

# Rollback to previous version
./deploy.sh production rollback

# View logs
./deploy.sh production logs

# Health check
./deploy.sh production health
```

### Method 2: Manual Kubernetes

```bash
# Create namespace
kubectl create namespace neuromorphic

# Apply all manifests
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get all -n neuromorphic
```

### Method 3: CI/CD Pipeline

The GitHub Actions workflow automatically:
- Runs tests and security scans
- Builds multi-architecture Docker images
- Deploys to staging and production
- Performs health checks and rollback if needed

## 📊 Monitoring and Observability

### Health Checks

```bash
# Run comprehensive health check
python scripts/production-health-check.py --url https://neuromorphic.yourdomain.com

# Continuous monitoring
python scripts/production-health-check.py --continuous 60

# Detailed verbose output
python scripts/production-health-check.py --verbose --output health-report.json
```

### Key Metrics Monitored

- **Performance**: Latency, throughput, response times
- **Neuromorphic**: Spike sparsity, energy efficiency, chip utilization
- **System**: CPU, memory, disk, network
- **Business**: Cost per inference, SLA compliance
- **Research**: Model accuracy, experiment progress

### Custom Alerts

The deployment includes specialized alerts for:
- Neuromorphic chip failures
- Energy efficiency degradation
- Model accuracy drift
- SLA violations
- Cost efficiency issues

## 🔒 Security Features

### Network Security
- Network policies restrict inter-pod communication
- Ingress with TLS termination
- Service mesh ready (Istio compatible)

### Pod Security
- Non-root containers
- Read-only filesystems
- Security contexts enforced
- Resource limits and quotas

### Secrets Management
- Kubernetes secrets for sensitive data
- Base64 encoded configuration
- Automatic secret rotation support

## ⚡ Performance Optimization

### Resource Configuration

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| API Pods | 1-2 cores | 2-4 GB | Ephemeral |
| Database | 0.5-1 core | 1-2 GB | 20 GB SSD |
| Cache | 0.1-0.5 core | 0.5-1 GB | 5 GB SSD |

### Auto-scaling
- Horizontal Pod Autoscaler (HPA) configured
- Metrics: CPU (70%), Memory (80%)
- Scale: 2-10 replicas based on load

### Caching Strategy
- Redis for session and result caching
- Model caching for frequently used networks
- Connection pooling for database access

## 🧪 Testing and Validation

### Pre-deployment Checks
```bash
# Validate Kubernetes manifests
kubectl apply --dry-run=client -f kubernetes/deployment.yaml

# Test Docker build
docker build -t neuromorphic-test -f docker/Dockerfile .

# Run unit tests
python -m pytest tests/unit/

# Performance benchmarks
python -m pytest tests/performance/ --benchmark-only
```

### Post-deployment Validation
```bash
# Smoke tests
curl -f https://neuromorphic.yourdomain.com/health

# Load testing
python scripts/load_test.py --url https://neuromorphic.yourdomain.com

# Health monitoring
python scripts/production-health-check.py --fail-on-degraded
```

## 📈 Scaling Guide

### Horizontal Scaling
```bash
# Scale API pods
kubectl scale deployment neuromorphic-api -n neuromorphic --replicas=5

# Check HPA status
kubectl get hpa -n neuromorphic
```

### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment neuromorphic-api -n neuromorphic -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "neuromorphic-api",
          "resources": {
            "limits": {"cpu": "4", "memory": "8Gi"},
            "requests": {"cpu": "2", "memory": "4Gi"}
          }
        }]
      }
    }
  }
}'
```

### Multi-Region Deployment
For global deployment:
1. Deploy to multiple Kubernetes clusters
2. Use external load balancer (e.g., AWS ALB, GCP Load Balancer)
3. Implement cross-region database replication
4. Configure DNS-based routing

## 🔧 Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl describe pod -n neuromorphic <pod-name>

# View logs
kubectl logs -n neuromorphic <pod-name>

# Check resource constraints
kubectl top pods -n neuromorphic
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it -n neuromorphic <api-pod> -- nc -zv neuromorphic-postgres 5432

# Check database logs
kubectl logs -n neuromorphic <postgres-pod>
```

#### Performance Issues
```bash
# Check resource utilization
kubectl top pods -n neuromorphic

# Review HPA status
kubectl describe hpa neuromorphic-api-hpa -n neuromorphic

# Analyze metrics
curl https://neuromorphic.yourdomain.com/metrics
```

### Debug Mode
Enable debug mode for development:
```bash
# Deploy with debug configuration
IMAGE_TAG=debug ./deploy.sh dev deploy

# Port forward for direct access
kubectl port-forward -n neuromorphic svc/neuromorphic-api 8080:8080
```

## 📚 Advanced Configuration

### Custom Environment Variables
Modify `kubernetes/deployment.yaml` ConfigMap:
```yaml
data:
  NEUROMORPHIC_LOG_LEVEL: "DEBUG"
  ENABLE_RESEARCH_MODE: "true"
  CUSTOM_HARDWARE_CONFIG: "loihi2_optimized"
```

### Hardware-Specific Deployments

#### Loihi 2 Optimization
```yaml
resources:
  limits:
    intel.com/loihi2: "1"
nodeSelector:
  hardware.neuromorphic/type: "loihi2"
```

#### SpiNNaker Configuration
```yaml
nodeSelector:
  hardware.neuromorphic/type: "spinnaker"
tolerations:
- key: "spinnaker-dedicated"
  operator: "Equal"
  effect: "NoSchedule"
```

## 🔄 Backup and Recovery

### Database Backups
Automated daily backups with PostgreSQL:
```bash
# Manual backup
kubectl exec -n neuromorphic <postgres-pod> -- pg_dump -U neuromorphic_user neuromorphic > backup.sql

# Restore from backup
kubectl exec -i -n neuromorphic <postgres-pod> -- psql -U neuromorphic_user neuromorphic < backup.sql
```

### Disaster Recovery
1. Backup persistent volumes
2. Export Kubernetes configurations
3. Document external dependencies
4. Test recovery procedures regularly

## 📞 Support

### Logs and Diagnostics
```bash
# Collect all logs
./deploy.sh production logs > production-logs.txt

# Generate health report
python scripts/production-health-check.py --output health-report.json

# System diagnostics
kubectl describe all -n neuromorphic > system-status.txt
```

### Performance Monitoring
- Prometheus metrics: `https://neuromorphic.yourdomain.com/metrics`
- Grafana dashboards: Custom neuromorphic dashboards available
- Application logs: Structured JSON logging with correlation IDs

For issues and support, please refer to the project documentation or create an issue in the repository.

## 🚀 Next Steps

After successful deployment:
1. Configure monitoring dashboards
2. Set up alerting rules
3. Implement backup strategies
4. Plan capacity scaling
5. Optimize for your specific neuromorphic hardware
6. Enable research and experimentation features

The Neuromorphic Platform is now ready for production workloads with enterprise-grade reliability, monitoring, and scalability.