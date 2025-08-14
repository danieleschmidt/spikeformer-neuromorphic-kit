#!/bin/bash
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
