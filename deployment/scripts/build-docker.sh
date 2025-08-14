#!/bin/bash
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
