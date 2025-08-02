#!/bin/bash
# Development Environment Setup Script for Spikeformer Neuromorphic Kit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.9"
VENV_NAME="spikeformer-dev"
USE_UV=${USE_UV:-true}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# Check if running in CI
is_ci() {
    [[ "${CI}" == "true" ]] || [[ -n "${GITHUB_ACTIONS}" ]] || [[ -n "${GITLAB_CI}" ]]
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║               Spikeformer Development Setup                  ║"
    echo "║            Neuromorphic AI Toolkit Environment              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_deps=()
    
    # Essential tools
    if ! check_command python3; then
        missing_deps+=("python3")
    fi
    
    if! check_command git; then
        missing_deps+=("git")
    fi
    
    # Check Python version
    if check_command python3; then
        PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$PYTHON_VER >= $PYTHON_VERSION" | bc -l) -eq 0 ]]; then
            print_error "Python $PYTHON_VERSION or higher required. Found: $PYTHON_VER"
            exit 1
        fi
        print_success "Python $PYTHON_VER detected"
    fi
    
    # Optional but recommended tools
    if ! check_command docker; then
        print_warning "Docker not found - container features will be unavailable"
    fi
    
    if ! check_command node; then
        print_warning "Node.js not found - some development tools may be unavailable"
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install them and run this script again."
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

setup_python_environment() {
    print_status "Setting up Python environment..."
    
    if [[ "$USE_UV" == "true" ]] && check_command uv; then
        print_status "Using UV for fast dependency management..."
        
        # Install UV if not present
        if ! check_command uv; then
            print_status "Installing UV..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            source $HOME/.cargo/env
        fi
        
        # Create virtual environment with UV
        print_status "Creating virtual environment with UV..."
        uv venv --python $PYTHON_VERSION $VENV_NAME
        
        # Activate environment
        source $VENV_NAME/bin/activate
        
        # Install dependencies
        print_status "Installing dependencies with UV..."
        uv pip install -e ".[dev]"
        
    else
        print_status "Using standard pip for dependency management..."
        
        # Create virtual environment
        python3 -m venv $VENV_NAME
        source $VENV_NAME/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip setuptools wheel
        
        # Install dependencies
        print_status "Installing development dependencies..."
        pip install -e ".[dev]"
    fi
    
    print_success "Python environment configured"
}

setup_pre_commit() {
    print_status "Setting up pre-commit hooks..."
    
    if ! check_command pre-commit; then
        print_error "pre-commit not found in environment"
        return 1
    fi
    
    # Install pre-commit hooks
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Update hooks to latest versions
    pre-commit autoupdate
    
    print_success "Pre-commit hooks installed"
}

setup_git_config() {
    print_status "Configuring Git for development..."
    
    # Set up Git aliases for common operations
    git config --local alias.co checkout
    git config --local alias.br branch
    git config --local alias.ci commit
    git config --local alias.st status
    git config --local alias.unstage 'reset HEAD --'
    git config --local alias.last 'log -1 HEAD'
    git config --local alias.visual '!gitk'
    
    # Set up Git hooks directory
    git config --local core.hooksPath .githooks
    mkdir -p .githooks
    
    print_success "Git configuration complete"
}

setup_ide_configuration() {
    print_status "Setting up IDE configurations..."
    
    # Create .vscode directory if it doesn't exist
    mkdir -p .vscode
    
    # Jupyter configuration
    mkdir -p ~/.jupyter
    
    if [[ ! -f ~/.jupyter/jupyter_lab_config.py ]]; then
        cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# Jupyter Lab configuration for Spikeformer development
c.ServerApp.token = 'spikeformer-dev'
c.ServerApp.password = ''
c.ServerApp.open_browser = False
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.allow_root = True
c.ServerApp.notebook_dir = '/workspace'
EOF
    fi
    
    print_success "IDE configuration complete"
}

install_hardware_dependencies() {
    print_status "Checking hardware dependencies..."
    
    # Check for CUDA
    if check_command nvidia-smi; then
        print_success "NVIDIA GPU detected"
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
        print_status "CUDA Version: $CUDA_VERSION"
        
        # Install PyTorch with CUDA support
        if [[ "$USE_UV" == "true" ]]; then
            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        else
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        print_warning "No NVIDIA GPU detected - using CPU-only PyTorch"
    fi
    
    # Check for Intel hardware
    if [[ -d "/opt/nxsdk" ]]; then
        print_success "Intel NxSDK detected for Loihi 2 support"
        export NXSDK_ROOT="/opt/nxsdk"
    else
        print_warning "Intel NxSDK not found - Loihi 2 support unavailable"
    fi
    
    # Check for SpiNNaker tools
    if check_command spynnaker; then
        print_success "SpiNNaker tools detected"
    else
        print_warning "SpiNNaker tools not found - SpiNNaker support unavailable"
    fi
}

create_development_scripts() {
    print_status "Creating development convenience scripts..."
    
    # Create run-tests script
    cat > scripts/run-tests.sh << 'EOF'
#!/bin/bash
# Convenience script for running tests

set -e

echo "Running test suite..."

# Activate virtual environment
source spikeformer-dev/bin/activate

# Run different test categories based on argument
case "${1:-all}" in
    "unit")
        pytest tests/unit/ -v
        ;;
    "integration")
        pytest tests/integration/ -v
        ;;
    "hardware")
        pytest tests/hardware/ -v --hardware
        ;;
    "performance")
        pytest tests/performance/ -v --benchmark-only
        ;;
    "coverage")
        pytest tests/ --cov=spikeformer --cov-report=html --cov-report=term
        ;;
    "all")
        pytest tests/ -v
        ;;
    *)
        echo "Usage: $0 [unit|integration|hardware|performance|coverage|all]"
        exit 1
        ;;
esac
EOF
    
    # Create development server script
    cat > scripts/dev-server.sh << 'EOF'
#!/bin/bash
# Start development server with hot reload

set -e

# Activate virtual environment
source spikeformer-dev/bin/activate

# Start Jupyter Lab in background
echo "Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
JUPYTER_PID=$!

# Start API server with auto-reload
echo "Starting API server..."
uvicorn spikeformer.api.app:app --reload --host 0.0.0.0 --port 5000 &
API_PID=$!

# Start monitoring stack
echo "Starting monitoring..."
docker-compose -f monitoring/docker-compose.yml up -d

echo "Development environment started:"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - API Server: http://localhost:5000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap "kill $JUPYTER_PID $API_PID; docker-compose -f monitoring/docker-compose.yml down; exit" INT
wait
EOF
    
    # Make scripts executable
    chmod +x scripts/run-tests.sh scripts/dev-server.sh
    
    print_success "Development scripts created"
}

verify_installation() {
    print_status "Verifying installation..."
    
    # Activate environment
    source $VENV_NAME/bin/activate
    
    # Test imports
    python -c "import spikeformer; print('✓ Spikeformer package importable')" || print_error "Failed to import spikeformer"
    python -c "import torch; print(f'✓ PyTorch {torch.__version__} available')" || print_error "Failed to import PyTorch"
    python -c "import transformers; print(f'✓ Transformers {transformers.__version__} available')" || print_error "Failed to import transformers"
    
    # Run a simple test
    if [[ -f "tests/unit/test_conversion.py" ]]; then
        python -m pytest tests/unit/test_conversion.py -v --tb=short || print_warning "Some tests failed"
    fi
    
    print_success "Installation verification complete"
}

print_completion_message() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete!                          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Your Spikeformer development environment is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment: source $VENV_NAME/bin/activate"
    echo "  2. Start development: ./scripts/dev-server.sh"
    echo "  3. Run tests: ./scripts/run-tests.sh"
    echo "  4. Open VS Code: code ."
    echo ""
    echo "Useful commands:"
    echo "  - Run all tests: npm run test"
    echo "  - Format code: npm run format"
    echo "  - Type check: npm run typecheck"
    echo "  - Build docs: npm run docs"
    echo ""
    echo "Hardware support status:"
    if [[ -d "/opt/nxsdk" ]]; then
        echo "  ✓ Intel Loihi 2 (NxSDK found)"
    else
        echo "  ✗ Intel Loihi 2 (install NxSDK for support)"
    fi
    if check_command spynnaker; then
        echo "  ✓ SpiNNaker"
    else
        echo "  ✗ SpiNNaker (install spynnaker for support)"
    fi
    if check_command nvidia-smi; then
        echo "  ✓ NVIDIA GPU"
    else
        echo "  ✗ NVIDIA GPU (CPU-only mode)"
    fi
    echo ""
}

main() {
    print_banner
    
    # Skip interactive parts in CI
    if is_ci; then
        print_status "Running in CI mode - skipping interactive setup"
    fi
    
    check_prerequisites
    setup_python_environment
    setup_pre_commit
    setup_git_config
    setup_ide_configuration
    install_hardware_dependencies
    create_development_scripts
    verify_installation
    print_completion_message
}

# Run main function
main "$@"