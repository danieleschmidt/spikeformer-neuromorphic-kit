#!/bin/bash
set -e

echo "🚀 Setting up Spikeformer development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "📥 Installing spikeformer in development mode..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p {logs,data,models,outputs,experiments}
mkdir -p {.benchmarks,htmlcov,security-reports,sbom}

# Set up git configuration
echo "🔧 Configuring git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Initialize Jupyter Lab extensions (if needed)
echo "🧪 Setting up Jupyter Lab..."
jupyter lab build --dev-build=False --minimize=False || echo "Jupyter lab build failed, continuing..."

# Download sample datasets (lightweight)
echo "📊 Setting up sample data..."
mkdir -p data/samples
echo "Sample datasets would be downloaded here in a real setup" > data/samples/README.md

# Create example notebooks
echo "📓 Creating example notebooks..."
mkdir -p notebooks/examples
cat > notebooks/examples/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Spikeformer Getting Started\n",
    "\n",
    "This notebook demonstrates basic usage of the Spikeformer toolkit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Import spikeformer modules\n",
    "# from spikeformer import SpikeformerConverter\n",
    "print('Welcome to Spikeformer development environment!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set up environment file
echo "🔐 Creating environment configuration..."
cat > .env.example << 'EOF'
# Development environment variables
NEUROMORPHIC_ENV=development
PYTHONPATH=/workspace

# Jupyter configuration
JUPYTER_ENABLE_LAB=yes
JUPYTER_TOKEN=spikeformer-dev

# MLflow configuration
MLFLOW_TRACKING_URI=http://mlflow:5000

# Weights & Biases (offline mode for development)
WANDB_MODE=offline

# Database configuration
DATABASE_URL=postgresql://spikeformer:dev_password@postgres:5432/spikeformer_dev

# Redis configuration
REDIS_URL=redis://redis:6379/0

# Hardware configuration (development mode)
LOIHI2_ENABLED=false
SPINNAKER_ENABLED=false
HARDWARE_SIMULATION=true
EOF

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
python -m pytest tests/ -x --tb=short -q || echo "Some tests failed - this is expected in development setup"

# Run security scan
echo "🔒 Running security check..."
bandit -r spikeformer/ -f json -o security-reports/initial-scan.json || echo "Security scan completed with warnings"

# Generate initial documentation
echo "📚 Building documentation..."
if [ -d "docs/" ]; then
    cd docs && make html || echo "Documentation build failed"
    cd ..
fi

# Check code quality
echo "✨ Running code quality checks..."
black --check spikeformer/ || echo "Code formatting issues found - run 'black spikeformer/' to fix"
ruff check spikeformer/ || echo "Linting issues found"

# Set up VS Code workspace
echo "💻 Configuring VS Code workspace..."
cat > spikeformer.code-workspace << 'EOF'
{
    "folders": [
        {
            "path": ".",
            "name": "Spikeformer Root"
        },
        {
            "path": "./spikeformer",
            "name": "Source Code"
        },
        {
            "path": "./tests",
            "name": "Tests"
        },
        {
            "path": "./docs",
            "name": "Documentation"
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "./.venv/bin/python",
        "python.terminal.activateEnvironment": true
    },
    "extensions": {
        "recommendations": [
            "ms-python.python",
            "charliermarsh.ruff",
            "ms-toolsai.jupyter"
        ]
    }
}
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Open VS Code: code ."
echo "  2. Start Jupyter Lab: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "  3. Run tests: pytest tests/"
echo "  4. Format code: black spikeformer/ && isort spikeformer/"
echo "  5. Check types: mypy spikeformer/"
echo ""
echo "🌐 Available services:"
echo "  - Jupyter Lab: http://localhost:8888 (token: spikeformer-dev)"
echo "  - Grafana: http://localhost:3000 (admin/dev_password)"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "Happy coding! 🚀"