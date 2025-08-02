#!/bin/bash
# Post-attach script for devcontainer

echo "🔗 Attaching to Spikeformer development environment..."

# Set up environment variables
export PYTHONPATH="/workspace:$PYTHONPATH"
export NEUROMORPHIC_ENV="development"

# Activate virtual environment
if [ -d "/workspace/.venv" ]; then
    source /workspace/.venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Show project status
echo ""
echo "📁 Project: Spikeformer Neuromorphic Kit"
echo "🌟 Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
echo "📦 Python: $(python --version 2>/dev/null || echo 'not found')"
echo "🔧 Environment: $(echo $NEUROMORPHIC_ENV)"
echo ""

# Show useful aliases
echo "💡 Useful commands:"
echo "  npm run test     - Run test suite"
echo "  npm run lint     - Check code quality" 
echo "  npm run format   - Format code"
echo "  npm run dev      - Start development server"
echo "  npm run benchmark - Run performance benchmarks"
echo ""

# Check if services are running
if curl -s http://localhost:8888 > /dev/null; then
    echo "✓ Jupyter Lab is running"
else
    echo "ℹ️ Jupyter Lab not running - use 'npm run dev' to start"
fi

if curl -s http://localhost:9090 > /dev/null; then
    echo "✓ Prometheus is running"
else
    echo "ℹ️ Prometheus not running"
fi

echo ""
echo "🚀 Ready for development!"