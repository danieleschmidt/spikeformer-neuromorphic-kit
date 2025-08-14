#!/bin/bash
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
kubectl exec -n spikeformer deployment/spikeformer-app -- tar czf - /app/data | \
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
