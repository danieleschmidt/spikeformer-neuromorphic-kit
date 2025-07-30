# 🚀 SpikeFormer Deployment Guide

## Overview

This guide covers deployment strategies for SpikeFormer across different neuromorphic hardware platforms and environments.

## Deployment Targets

### 1. Intel Loihi 2 Deployment

#### Prerequisites
```bash
# Install Intel NxSDK (requires Intel partnership)
pip install nxsdk>=1.0.0

# Verify hardware access
python -c "import nxsdk; print('Loihi 2 access:', nxsdk.test_connection())"
```

#### Basic Deployment
```python
from spikeformer.hardware import Loihi2Deployer

deployer = Loihi2Deployer(
    num_chips=2,
    partition_strategy="layer_wise",
    optimization_level=3
)

deployed_model = deployer.deploy(spiking_model)
```

#### Production Configuration
```yaml
# loihi2_config.yaml
hardware:
  platform: loihi2
  chips: 4
  cores_per_chip: 128
  
optimization:
  level: 3
  memory_optimization: true
  power_gating: true
  
monitoring:
  energy_profiling: true
  thermal_monitoring: true
  error_correction: true
```

#### Monitoring Setup
```python
from spikeformer.monitoring import HardwareMonitor

monitor = HardwareMonitor("loihi2")
with monitor.session() as session:
    results = deployed_model.run(inputs)
    metrics = session.get_metrics()
```

### 2. SpiNNaker2 Deployment

#### Prerequisites  
```bash
# Install SpiNNaker tools
pip install spynnaker>=6.0.0 spalloc>=6.0.0

# Configure board access
spinnaker_machine_config
```

#### Deployment Process
```python
from spikeformer.hardware import SpiNNakerDeployer

deployer = SpiNNakerDeployer(
    board_config="spin2-48chip",
    routing_algorithm="neighbour_aware"
)

spinn_model = deployer.deploy(spiking_model)
```

#### Real-time Operation
```python
# Configure for real-time operation
spinn_model.configure_realtime(
    time_scale_factor=1000,
    live_input=True,
    live_output=True
)

# Stream processing
for batch in input_stream:
    output = spinn_model.process_realtime(batch)
    yield output
```

### 3. Edge Neuromorphic Deployment

#### Supported Platforms
- Intel Loihi (mobile)
- Akida 1000/2000
- GrAI Matter VIP
- Mythic M1076

#### Edge Compilation
```python
from spikeformer.edge import EdgeCompiler

compiler = EdgeCompiler(
    target="akida_2000",
    quantization_bits=4,
    memory_budget_mb=16
)

edge_model = compiler.compile(
    spiking_model,
    optimize_for="power"
)
```

#### Deployment Package
```bash
# Generate deployment package
spikeformer-deploy --target edge \
  --model path/to/model.pth \
  --output edge_deployment/ \
  --include-runtime \
  --optimize power
```

## Container Deployment

### Docker Deployment
```dockerfile
# Multi-stage build for production
FROM python:3.10-slim as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app

# Hardware-specific configurations
ENV PYTHONPATH=/app:$PYTHONPATH
ENV NXSDK_PATH=/opt/nxsdk
ENV SPINNAKER_CONFIG=/app/config/spinnaker.cfg

EXPOSE 8080
CMD ["python", "-m", "spikeformer.serve", "--host", "0.0.0.0"]
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spikeformer-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spikeformer
  template:
    metadata:
      labels:
        app: spikeformer
    spec:
      containers:
      - name: spikeformer
        image: spikeformer:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: HARDWARE_BACKEND
          value: "loihi2"
        - name: ENERGY_MONITORING
          value: "true"
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: hardware-config
          mountPath: /config
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: hardware-config
        configMap:
          name: hardware-config
```

## Cloud Deployment

### AWS Deployment
```yaml
# aws-deployment.yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  SpikeFormerInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: f1.2xlarge  # FPGA instance
      ImageId: ami-neuromorphic-optimized
      SecurityGroups:
        - !Ref SpikeFormerSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          pip install spikeformer-neuromorphic-kit
          systemctl start spikeformer-service
```

### Azure ML Deployment
```python
from azureml.core import Workspace, Environment, Model
from azureml.core.webservice import AciWebservice

# Register model
model = Model.register(
    workspace=ws,
    model_path="spikeformer_model.pkl",
    model_name="spikeformer-v1"
)

# Deploy to container instance
service = Model.deploy(
    workspace=ws,
    name="spikeformer-service",
    models=[model],
    environment=Environment.from_conda_specification(
        name="spikeformer-env",
        file_path="environment.yml"
    ),
    deployment_config=AciWebservice.deploy_configuration(
        cpu_cores=2,
        memory_gb=4,
        enable_app_insights=True
    )
)
```

## Production Considerations

### Security
```yaml
# security-config.yaml
security:
  encryption:
    at_rest: true
    in_transit: true
  authentication:
    enabled: true
    method: "oauth2"
  access_control:
    rbac: true
    hardware_isolation: true
```

### Monitoring & Observability
```yaml
# monitoring-config.yaml
monitoring:
  metrics:
    - energy_consumption
    - inference_latency
    - model_accuracy
    - hardware_utilization
    - error_rates
  
  alerting:
    energy_threshold: 100  # mJ
    latency_threshold: 50  # ms
    accuracy_threshold: 90  # %
  
  logging:
    level: INFO
    format: json
    destination: elasticsearch
```

### Performance Optimization
```python
# performance tuning
optimizer = DeploymentOptimizer()
optimizer.tune_hardware_mapping(model, target_hardware)
optimizer.optimize_memory_usage(batch_size=32)
optimizer.enable_power_gating(idle_threshold=100)
```

### Disaster Recovery
```yaml
# dr-config.yaml
disaster_recovery:
  backup:
    models: true
    configurations: true
    frequency: daily
  
  failover:
    automatic: true
    target_platform: cpu_simulation
    max_downtime: 60  # seconds
  
  health_checks:
    interval: 30  # seconds
    timeout: 10   # seconds
    retries: 3
```

## Troubleshooting

### Common Issues

#### Hardware Connection Issues
```bash
# Check hardware status
spikeformer-diagnose --hardware loihi2
spikeformer-diagnose --hardware spinnaker

# Reset hardware connection
sudo systemctl restart loihi-daemon
sudo systemctl restart spinnaker-service
```

#### Memory Issues
```python
# Memory optimization
import gc
torch.cuda.empty_cache()  # If using GPU simulation
gc.collect()

# Reduce batch size
model.set_batch_size(16)  # Reduce from 32
```

#### Performance Issues
```bash
# Profile deployment
spikeformer-profile --deployment production \
  --hardware loihi2 \
  --duration 60s \
  --output profile.json

# Analyze bottlenecks
spikeformer-analyze profile.json
```

### Support Resources
- [Hardware-specific Troubleshooting](./troubleshooting/hardware.md)
- [Performance Tuning Guide](./troubleshooting/performance.md)
- [Security Best Practices](./security/deployment.md)
- [Monitoring Setup](./monitoring/setup.md)

## Deployment Checklist

### Pre-deployment
- [ ] Hardware compatibility verified
- [ ] Dependencies installed and tested
- [ ] Security configurations applied
- [ ] Monitoring setup completed
- [ ] Backup strategy implemented

### Deployment
- [ ] Model conversion validated
- [ ] Hardware deployment successful
- [ ] Performance benchmarks meet requirements
- [ ] Security scan passed
- [ ] Documentation updated

### Post-deployment
- [ ] Health checks passing
- [ ] Monitoring data flowing
- [ ] Performance within SLA
- [ ] Disaster recovery tested
- [ ] Team training completed

For additional support, contact the SpikeFormer team or consult the [support documentation](../SUPPORT.md).