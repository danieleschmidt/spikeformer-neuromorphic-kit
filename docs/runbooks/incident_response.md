# Incident Response Runbook

## Overview

This runbook provides procedures for responding to incidents in the Spikeformer Neuromorphic Kit infrastructure and services.

## Incident Classification

### Severity Levels

#### SEV-1: Critical
- **Definition**: Complete service outage or data loss
- **Response Time**: 15 minutes
- **Examples**: 
  - All model inference endpoints down
  - Hardware systems completely unavailable
  - Data corruption or loss

#### SEV-2: High
- **Definition**: Major functionality impaired
- **Response Time**: 1 hour
- **Examples**:
  - Single hardware backend unavailable
  - Significant performance degradation (>50% slower)
  - Security vulnerability in production

#### SEV-3: Medium
- **Definition**: Minor functionality impaired
- **Response Time**: 4 hours
- **Examples**:
  - Non-critical features unavailable
  - Moderate performance impact
  - Documentation or UI issues

#### SEV-4: Low
- **Definition**: Cosmetic or enhancement
- **Response Time**: 24 hours
- **Examples**:
  - Minor UI inconsistencies
  - Enhancement requests
  - Non-urgent improvements

## Incident Response Process

### 1. Detection and Alert

#### Automatic Detection
- Monitor dashboards and alerts
- Health check failures
- Performance metric thresholds
- Error rate increases

#### Manual Detection
- User reports
- Team member observations
- External monitoring services

### 2. Initial Response (< 5 minutes)

1. **Acknowledge the Incident**
   ```bash
   # Acknowledge in monitoring system
   curl -X POST alertmanager:9093/api/v1/silences \
     -d '{"matchers":[{"name":"alertname","value":"<ALERT_NAME>"}],"startsAt":"<NOW>","endsAt":"<NOW+4h>","comment":"Investigating incident"}'
   ```

2. **Assess Severity**
   - Determine impact scope
   - Classify severity level
   - Escalate if necessary

3. **Communicate**
   - Post in incident channel: `#incident-response`
   - Update status page if customer-facing
   - Notify stakeholders per severity

### 3. Investigation and Diagnosis

#### Common Investigation Steps

1. **Check System Health**
   ```bash
   # Check service status
   kubectl get pods -n spikeformer
   
   # Check resource usage
   kubectl top nodes
   kubectl top pods -n spikeformer
   
   # Check recent deployments
   kubectl rollout history deployment/spikeformer-api
   ```

2. **Analyze Logs**
   ```bash
   # Application logs
   kubectl logs -f deployment/spikeformer-api -n spikeformer
   
   # System logs
   journalctl -u spikeformer-service -f
   
   # Hardware logs (if applicable)
   tail -f /var/log/loihi2/system.log
   ```

3. **Review Metrics**
   - Open Grafana dashboards
   - Check Prometheus metrics
   - Analyze performance trends

#### Hardware-Specific Diagnosis

**Loihi 2 Issues**
```bash
# Check hardware status
python -c "import nxsdk; print(nxsdk.get_board_status())"

# Check temperature
sensors | grep -A5 "loihi"

# Check power consumption
cat /sys/class/power_supply/loihi2/energy_now
```

**SpiNNaker Issues**
```bash
# Check board connectivity
spalloc-where-is --machine spinn-1

# Check network status
ping spinnaker-board.local

# Check job queue
spalloc-job --list
```

### 4. Mitigation and Resolution

#### Immediate Mitigation Strategies

1. **Service Restart**
   ```bash
   kubectl rollout restart deployment/spikeformer-api
   ```

2. **Scale Resources**
   ```bash
   kubectl scale deployment/spikeformer-api --replicas=5
   ```

3. **Rollback Deployment**
   ```bash
   kubectl rollout undo deployment/spikeformer-api
   ```

4. **Enable Maintenance Mode**
   ```bash
   kubectl patch ingress spikeformer-ingress -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/maintenance-mode":"true"}}}'
   ```

#### Hardware-Specific Mitigation

**Hardware Overheating**
1. Reduce workload intensity
2. Increase cooling
3. Temporarily disable affected hardware
4. Route traffic to backup systems

**Memory Issues**
1. Clear caches
2. Restart memory-intensive services
3. Scale down non-essential processes
4. Implement memory limits

### 5. Communication Templates

#### Initial Incident Report
```
🚨 INCIDENT ALERT - SEV-{LEVEL}

Summary: {Brief description}
Impact: {User/system impact}
Start Time: {Timestamp}
Status: Investigating

Updates will be posted every 30 minutes.
Lead: @{incident_commander}
```

#### Progress Updates
```
📊 INCIDENT UPDATE - SEV-{LEVEL}

Summary: {Current status}
Progress: {What has been done}
Next Steps: {What's being worked on}
ETA: {Expected resolution time}

Last Updated: {Timestamp}
```

#### Resolution Notice
```
✅ INCIDENT RESOLVED - SEV-{LEVEL}

Summary: {Final resolution}
Root Cause: {What caused the issue}
Resolution: {How it was fixed}
Duration: {Total incident time}

Post-mortem meeting: {Date/time}
```

## Specific Incident Scenarios

### Model Inference Failures

#### Symptoms
- 500 errors from inference endpoints
- High latency or timeouts
- Model loading failures

#### Investigation Steps
1. Check model file integrity
2. Verify hardware availability
3. Check memory and GPU usage
4. Review recent model updates

#### Resolution
```bash
# Check model files
ls -la /models/
md5sum /models/spikeformer-vit-base.pth

# Restart model service
kubectl rollout restart deployment/model-inference

# Check hardware allocation
nvidia-smi  # For GPU models
python scripts/check_hardware_status.py
```

### Hardware Overheating

#### Symptoms
- Hardware temperature alerts
- Performance degradation
- Hardware protection shutdowns

#### Investigation Steps
1. Check temperature sensors
2. Verify cooling system operation
3. Review workload intensity
4. Check for hardware faults

#### Resolution
```bash
# Check temperatures
sensors | grep -E "(temp|thermal)"

# Reduce workload
kubectl scale deployment/spikeformer-worker --replicas=2

# Enable throttling
echo "1" > /sys/class/thermal/thermal_zone0/trip_point_0_temp
```

### Database Performance Issues

#### Symptoms
- Slow query responses
- Connection timeouts
- High database CPU usage

#### Investigation Steps
1. Check active connections
2. Review slow query logs
3. Analyze query performance
4. Check database resources

#### Resolution
```bash
# Check connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Kill long-running queries
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Restart database connection pool
kubectl rollout restart deployment/pgbouncer
```

## Post-Incident Activities

### 1. Recovery Verification
- Confirm all systems are operational
- Run smoke tests
- Monitor metrics for stability
- Verify customer-facing features

### 2. Post-Mortem Process
**Timeline: Within 48 hours of resolution**

1. **Schedule Meeting**
   - Include all involved team members
   - Block 1-2 hours for thorough discussion
   - Record session for documentation

2. **Prepare Materials**
   - Timeline of events
   - Metrics and logs
   - Actions taken
   - Impact assessment

3. **Conduct Analysis**
   - Root cause analysis
   - Process improvement opportunities
   - Action items with owners
   - Prevention strategies

### 3. Follow-up Actions
- Implement identified improvements
- Update monitoring and alerting
- Enhance documentation
- Training if needed

## Contact Information

### Escalation Path
1. **On-Call Engineer**: +1-555-ONCALL
2. **Engineering Manager**: +1-555-ENGMGR
3. **CTO**: +1-555-CTO (SEV-1 only)

### Team Contacts
- **Infrastructure**: @infrastructure-team
- **ML Platform**: @ml-platform-team
- **Hardware**: @hardware-team
- **Security**: @security-team

### External Contacts
- **Cloud Provider**: [Support case system]
- **Hardware Vendor**: Intel support, SpiNNaker support
- **Security Partner**: [Security vendor contact]

## Tools and Resources

### Monitoring
- **Grafana**: https://grafana.your-org.com
- **Prometheus**: https://prometheus.your-org.com
- **AlertManager**: https://alerts.your-org.com

### Communication
- **Slack**: #incident-response, #alerts-critical
- **Status Page**: https://status.your-org.com
- **Zoom Room**: https://zoom.us/j/emergency-room

### Documentation
- **Runbooks**: /docs/runbooks/
- **Architecture**: /docs/ARCHITECTURE.md
- **Deployment**: /docs/runbooks/deployment.md

---

*This runbook is reviewed monthly and updated after each major incident.*