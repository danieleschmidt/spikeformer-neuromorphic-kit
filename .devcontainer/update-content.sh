#!/bin/bash
# Update content script for devcontainer

echo "🔄 Updating development environment content..."

# Update dependencies if requirements changed
if [[ "requirements.txt" -nt "/workspace/.venv/pyvenv.cfg" ]] || [[ "pyproject.toml" -nt "/workspace/.venv/pyvenv.cfg" ]]; then
    echo "📦 Updating Python dependencies..."
    source /workspace/.venv/bin/activate
    
    if command -v uv &> /dev/null; then
        uv pip install -e ".[dev]"
    else
        pip install -e ".[dev]"
    fi
    echo "✓ Dependencies updated"
fi

# Update pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Updating pre-commit hooks..."
    pre-commit autoupdate
    echo "✓ Pre-commit hooks updated"
fi

# Update Node dependencies if package.json changed
if [[ "package.json" -nt "node_modules/.package-lock.json" ]] && [ -f "package.json" ]; then
    echo "📦 Updating Node.js dependencies..."
    npm install
    echo "✓ Node.js dependencies updated"
fi

# Refresh monitoring containers if compose file changed
if [ -f "monitoring/docker-compose.yml" ]; then
    echo "📊 Refreshing monitoring containers..."
    docker-compose -f monitoring/docker-compose.yml pull --quiet
    echo "✓ Monitoring containers refreshed"
fi

echo "✅ Environment update complete!"