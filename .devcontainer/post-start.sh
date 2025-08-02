#!/bin/bash
# Post-start script for devcontainer

echo "🚀 Starting Spikeformer development environment..."

# Activate virtual environment if it exists
if [ -d "/workspace/.venv" ]; then
    source /workspace/.venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Start monitoring services in background
if [ -f "/workspace/monitoring/docker-compose.yml" ]; then
    echo "📊 Starting monitoring stack..."
    cd /workspace
    docker-compose -f monitoring/docker-compose.yml up -d --quiet-pull
    echo "✓ Monitoring services started"
fi

# Check hardware availability
echo "🔧 Checking hardware availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "ℹ️ No NVIDIA GPU detected - using CPU mode"
fi

# Check neuromorphic hardware
if [ -d "/opt/nxsdk" ]; then
    echo "✓ Intel NxSDK available"
    export NXSDK_ROOT="/opt/nxsdk"
else
    echo "ℹ️ Intel NxSDK not found"
fi

if command -v spynnaker &> /dev/null; then
    echo "✓ SpiNNaker tools available"
else
    echo "ℹ️ SpiNNaker tools not found"
fi

# Start Jupyter Lab if not already running
if ! pgrep -f "jupyter-lab" > /dev/null; then
    echo "📓 Starting Jupyter Lab..."
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root > /tmp/jupyter.log 2>&1 &
    echo "✓ Jupyter Lab started on port 8888"
fi

echo "🎉 Development environment ready!"
echo ""
echo "Services available:"
echo "  📓 Jupyter Lab: http://localhost:8888 (token: spikeformer-dev)"
echo "  📊 Prometheus: http://localhost:9090"
echo "  📈 Grafana: http://localhost:3000"
echo ""
echo "Quick start:"
echo "  📝 Open workspace: code /workspace"
echo "  🧪 Run tests: npm run test"
echo "  🏗️ Build project: npm run build"