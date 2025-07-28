# Deployment Runbook

## Overview

This runbook covers deployment procedures for SpikeFormer across different environments and hardware platforms.

## Pre-deployment Checklist

### Code Quality
- [ ] All tests pass
- [ ] Code coverage >= 80%
- [ ] Security scans pass
- [ ] Performance benchmarks meet targets
- [ ] Documentation updated

### Infrastructure
- [ ] Target environment accessible
- [ ] Hardware requirements verified
- [ ] Network connectivity confirmed
- [ ] Storage requirements met
- [ ] Monitoring configured

### Security
- [ ] Secrets properly configured
- [ ] SSL certificates valid
- [ ] Firewall rules configured
- [ ] Access controls verified

## Deployment Environments

### Development Environment

```bash
# Deploy to development
docker-compose -f docker-compose.yml up -d

# Verify deployment
curl http://localhost:8000/health

# Check logs
docker-compose logs -f spikeformer-dev
```

### Staging Environment

```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run smoke tests
python scripts/smoke_tests.py --env staging

# Verify metrics
curl http://staging.your-org.com/metrics
```

### Production Environment

```bash
# Deploy to production (requires approval)
docker-compose -f docker-compose.prod.yml up -d

# Run health checks
python scripts/health_check.py --env production

# Monitor deployment
python scripts/monitor_deployment.py --duration 300
```

## Hardware-Specific Deployments

### Loihi 2 Deployment

#### Prerequisites
- Intel NxSDK installed
- Loihi 2 hardware accessible
- Proper permissions configured

#### Deployment Steps

```bash
# 1. Verify hardware access
python -c "import nxsdk; print('NxSDK available')"

# 2. Build Loihi2-specific image
make build-docker-loihi2

# 3. Deploy with hardware access
docker run --privileged \
  -v /dev:/dev \
  -e NXSDK_ROOT=/opt/nxsdk \
  spikeformer:loihi2

# 4. Verify deployment
curl http://localhost:8000/health/hardware
```

#### Troubleshooting
- **Hardware not detected**: Check permissions and device access
- **Compilation fails**: Verify NxSDK version compatibility
- **Performance issues**: Monitor chip utilization

### SpiNNaker Deployment

#### Prerequisites
- sPyNNaker installed
- SpiNNaker board accessible
- Network configuration correct

#### Deployment Steps

```bash
# 1. Verify board connection
ping $SPINNAKER_IP

# 2. Build SpiNNaker-specific image
make build-docker-spinnaker

# 3. Deploy with network access
docker run \
  -e SPYNNAKER_IP=$SPINNAKER_IP \
  -e SPINN_DIRS=/opt/spinnaker \
  spikeformer:spinnaker

# 4. Test deployment
python scripts/test_spinnaker_connection.py
```

#### Troubleshooting
- **Connection timeout**: Check network configuration
- **Board allocation fails**: Verify board availability
- **Communication errors**: Check firewall settings

## Database Deployment

### PostgreSQL Setup

```bash
# 1. Deploy database
docker-compose up -d postgres

# 2. Initialize schema
python scripts/init_database.py

# 3. Run migrations
python scripts/migrate_database.py

# 4. Seed initial data
python scripts/seed_database.py --env production
```

### Redis Setup

```bash
# Deploy Redis
docker-compose up -d redis

# Verify connection
redis-cli ping
```

## Monitoring Setup

### Prometheus Deployment

```bash
# 1. Deploy Prometheus
docker-compose up -d prometheus

# 2. Verify targets
curl http://localhost:9090/api/v1/targets

# 3. Test queries
curl 'http://localhost:9090/api/v1/query?query=up'
```

### Grafana Deployment

```bash
# 1. Deploy Grafana
docker-compose up -d grafana

# 2. Access dashboard
open http://localhost:3001

# 3. Import dashboards
python scripts/import_dashboards.py
```

## Scaling Procedures

### Horizontal Scaling

```bash
# Scale service replicas
docker-compose up -d --scale spikeformer-cpu=3

# Verify load distribution
curl http://localhost:8000/metrics | grep requests_total
```

### Vertical Scaling

```bash
# Update resource limits
docker-compose -f docker-compose.prod.yml \
  --scale spikeformer-cpu=1 \
  up -d --force-recreate
```

## Rollback Procedures

### Standard Rollback

```bash
# 1. Stop current deployment
docker-compose down

# 2. Deploy previous version
docker-compose -f docker-compose.yml \
  -f docker-compose.prod.yml \
  -e IMAGE_TAG=v0.1.0 \
  up -d

# 3. Verify rollback
curl http://localhost:8000/health
```

### Database Rollback

```bash
# 1. Stop application
docker-compose stop spikeformer-*

# 2. Restore database backup
pg_restore -d spikeformer backup_20250128.sql

# 3. Restart application
docker-compose start spikeformer-*
```

## Backup Procedures

### Application Backup

```bash
# Create configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  docker-compose*.yml \
  monitoring/ \
  .env*

# Upload to backup storage
aws s3 cp config_backup_*.tar.gz s3://backup-bucket/
```

### Database Backup

```bash
# Create database backup
pg_dump spikeformer > backup_$(date +%Y%m%d_%H%M%S).sql

# Compress and upload
gzip backup_*.sql
aws s3 cp backup_*.sql.gz s3://db-backups/
```

## Disaster Recovery

### Complete Service Recovery

1. **Assess Damage**
   - Check infrastructure status
   - Identify failed components
   - Estimate recovery time

2. **Restore Infrastructure**
   ```bash
   # Restore from infrastructure as code
   terraform apply -var-file=production.tfvars
   ```

3. **Restore Data**
   ```bash
   # Restore latest database backup
   aws s3 cp s3://db-backups/latest.sql.gz .
   gunzip latest.sql.gz
   pg_restore -d spikeformer latest.sql
   ```

4. **Redeploy Services**
   ```bash
   # Deploy latest stable version
   docker-compose -f docker-compose.prod.yml up -d
   ```

5. **Verify Recovery**
   ```bash
   # Run comprehensive health checks
   python scripts/disaster_recovery_test.py
   ```

## Performance Tuning

### Application Tuning

```bash
# Monitor resource usage
docker stats

# Optimize JVM (if applicable)
export JAVA_OPTS="-Xmx4g -XX:+UseG1GC"

# Tune database connections
export DB_POOL_SIZE=20
```

### Hardware Tuning

```bash
# CPU affinity for better performance
docker run --cpuset-cpus="0-3" spikeformer:latest

# Memory optimization
docker run --memory=8g --memory-swap=8g spikeformer:latest
```

## Security Hardening

### Container Security

```bash
# Run as non-root user
docker run --user 1000:1000 spikeformer:latest

# Read-only filesystem
docker run --read-only spikeformer:latest

# Drop capabilities
docker run --cap-drop=ALL spikeformer:latest
```

### Network Security

```bash
# Use custom network
docker network create --driver bridge secure-network

# Deploy with network isolation
docker-compose -f docker-compose.secure.yml up -d
```

## Maintenance Windows

### Planned Maintenance

1. **Pre-maintenance**
   - Notify users
   - Create backups
   - Prepare rollback plan

2. **During Maintenance**
   - Apply updates
   - Test functionality
   - Monitor performance

3. **Post-maintenance**
   - Verify all services
   - Update documentation
   - Notify completion

### Emergency Maintenance

```bash
# Quick service restart
docker-compose restart spikeformer-*

# Apply critical patch
docker pull spikeformer:patch-version
docker-compose up -d --force-recreate
```

## Troubleshooting Guide

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose logs spikeformer-cpu
   
   # Check resource usage
   docker system df
   ```

2. **High Memory Usage**
   ```bash
   # Monitor memory
   docker stats --no-stream
   
   # Adjust limits
   docker update --memory=4g container_name
   ```

3. **Database Connection Issues**
   ```bash
   # Check database status
   docker-compose exec postgres psql -U spikeformer -c "\l"
   
   # Test connectivity
   telnet postgres_host 5432
   ```

### Emergency Contacts

- **On-call Engineer**: +1-555-0123
- **DevOps Team**: devops@your-org.com
- **Security Team**: security@your-org.com
- **Hardware Support**: hardware@your-org.com

## Post-Deployment Tasks

1. **Verify Deployment**
   - Run smoke tests
   - Check monitoring dashboards
   - Verify all endpoints

2. **Update Documentation**
   - Record any configuration changes
   - Update runbook if needed
   - Document any issues encountered

3. **Notify Stakeholders**
   - Send deployment notification
   - Update status page
   - Share performance metrics

## Automation Scripts

All deployment procedures can be automated using the provided scripts:

- `scripts/deploy.py` - Main deployment script
- `scripts/health_check.py` - Health verification
- `scripts/rollback.py` - Automated rollback
- `scripts/backup.py` - Backup automation
- `scripts/monitor.py` - Deployment monitoring