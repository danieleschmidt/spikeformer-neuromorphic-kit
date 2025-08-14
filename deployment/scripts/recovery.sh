#!/bin/bash
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
