#!/bin/bash
# Comprehensive build and security scanning script for Spikeformer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${REGISTRY:-"spikeformer"}
VERSION=${VERSION:-"latest"}
SCAN_ENABLED=${SCAN_ENABLED:-true}
PUSH_ENABLED=${PUSH_ENABLED:-false}
PARALLEL_BUILDS=${PARALLEL_BUILDS:-true}

# Build targets
TARGETS=(
    "development"
    "production-cpu" 
    "production-gpu"
    "loihi2"
    "spinnaker"
    "edge"
)

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

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           Spikeformer Docker Build & Security Scan          ║"
    echo "║                Neuromorphic AI Container Pipeline            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_dependencies() {
    print_status "Checking build dependencies..."
    
    local missing_deps=()
    
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check for security scanning tools
    if [[ "$SCAN_ENABLED" == "true" ]]; then
        if ! command -v trivy &> /dev/null && ! command -v grype &> /dev/null; then
            print_warning "No container security scanner found (trivy or grype)"
            print_status "Installing Trivy..."
            install_trivy
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "All dependencies satisfied"
}

install_trivy() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Install Trivy on Linux
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Install Trivy on macOS
        if command -v brew &> /dev/null; then
            brew install trivy
        else
            print_error "Please install Trivy manually or install Homebrew"
            exit 1
        fi
    else
        print_warning "Unsupported OS for automatic Trivy installation"
    fi
}

build_image() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    
    print_status "Building ${tag}..."
    
    # Build with buildkit for better performance and features
    DOCKER_BUILDKIT=1 docker build \
        --target "${target}" \
        --tag "${tag}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        .
    
    if [[ $? -eq 0 ]]; then
        print_success "Successfully built ${tag}"
        return 0
    else
        print_error "Failed to build ${tag}"
        return 1
    fi
}

scan_image() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    local scan_report="security-reports/scan-${target}-$(date +%Y%m%d-%H%M%S).json"
    
    print_status "Scanning ${tag} for security vulnerabilities..."
    
    # Create reports directory
    mkdir -p security-reports
    
    # Run security scan with Trivy
    if command -v trivy &> /dev/null; then
        trivy image \
            --format json \
            --output "${scan_report}" \
            --severity HIGH,CRITICAL \
            --exit-code 1 \
            "${tag}"
        
        local scan_result=$?
        
        # Generate human-readable report
        trivy image \
            --format table \
            --severity HIGH,CRITICAL \
            "${tag}" > "security-reports/scan-${target}-summary.txt"
        
        if [[ $scan_result -eq 0 ]]; then
            print_success "Security scan passed for ${tag}"
            return 0
        else
            print_error "Security vulnerabilities found in ${tag}"
            print_status "Detailed report: ${scan_report}"
            return 1
        fi
    else
        print_warning "Security scanner not available, skipping scan for ${tag}"
        return 0
    fi
}

test_image() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    
    print_status "Testing ${tag} functionality..."
    
    # Basic functionality tests
    case $target in
        "development")
            # Test development image
            docker run --rm "${tag}" python -c "import spikeformer; print('✓ Package import successful')"
            docker run --rm "${tag}" jupyter --version > /dev/null && echo "✓ Jupyter available"
            ;;
        "production-cpu"|"production-gpu")
            # Test production images
            docker run --rm "${tag}" python -c "import spikeformer; print('✓ Package import successful')"
            docker run --rm "${tag}" spikeformer --version > /dev/null && echo "✓ CLI available"
            ;;
        "loihi2")
            # Test Loihi2 specific image
            docker run --rm "${tag}" python -c "import spikeformer; print('✓ Loihi2 image functional')"
            ;;
        "spinnaker")
            # Test SpiNNaker specific image
            docker run --rm "${tag}" python -c "import spikeformer; print('✓ SpiNNaker image functional')"
            ;;
        "edge")
            # Test edge image
            docker run --rm "${tag}" python -c "import spikeformer; print('✓ Edge image functional')"
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        print_success "Functionality test passed for ${tag}"
        return 0
    else
        print_error "Functionality test failed for ${tag}"
        return 1
    fi
}

analyze_image_size() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    
    print_status "Analyzing image size for ${tag}..."
    
    local size=$(docker images "${tag}" --format "table {{.Size}}" | tail -n 1)
    local size_bytes=$(docker inspect "${tag}" --format='{{.Size}}')
    local size_mb=$((size_bytes / 1024 / 1024))
    
    echo "Image Size Analysis for ${tag}:"
    echo "  - Human readable: ${size}"
    echo "  - Size in MB: ${size_mb} MB"
    
    # Size warnings based on target
    case $target in
        "edge")
            if [[ $size_mb -gt 200 ]]; then
                print_warning "Edge image is larger than recommended (${size_mb}MB > 200MB)"
            fi
            ;;
        "production-cpu"|"production-gpu")
            if [[ $size_mb -gt 2000 ]]; then
                print_warning "Production image is quite large (${size_mb}MB > 2GB)"
            fi
            ;;
        "development")
            if [[ $size_mb -gt 5000 ]]; then
                print_warning "Development image is very large (${size_mb}MB > 5GB)"
            fi
            ;;
    esac
    
    # Layer analysis
    docker history "${tag}" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
}

generate_sbom() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    local sbom_file="security-reports/sbom-${target}-$(date +%Y%m%d-%H%M%S).json"
    
    print_status "Generating SBOM for ${tag}..."
    
    if command -v syft &> /dev/null; then
        syft "${tag}" -o spdx-json="${sbom_file}"
        print_success "SBOM generated: ${sbom_file}"
    else
        print_warning "Syft not available, skipping SBOM generation"
    fi
}

push_image() {
    local target=$1
    local tag="${REGISTRY}:${target}"
    
    if [[ "$PUSH_ENABLED" == "true" ]]; then
        print_status "Pushing ${tag} to registry..."
        docker push "${tag}"
        
        if [[ $? -eq 0 ]]; then
            print_success "Successfully pushed ${tag}"
        else
            print_error "Failed to push ${tag}"
            return 1
        fi
    else
        print_status "Skipping push for ${tag} (PUSH_ENABLED=false)"
    fi
}

build_single_target() {
    local target=$1
    local failed=false
    
    print_status "Processing target: ${target}"
    
    # Build
    if ! build_image "${target}"; then
        failed=true
    fi
    
    # Test
    if [[ "$failed" == "false" ]] && ! test_image "${target}"; then
        failed=true
    fi
    
    # Analyze
    if [[ "$failed" == "false" ]]; then
        analyze_image_size "${target}"
    fi
    
    # Security scan
    if [[ "$failed" == "false" ]] && [[ "$SCAN_ENABLED" == "true" ]]; then
        if ! scan_image "${target}"; then
            print_warning "Security scan failed for ${target}, but continuing..."
        fi
        generate_sbom "${target}"
    fi
    
    # Push
    if [[ "$failed" == "false" ]]; then
        push_image "${target}"
    fi
    
    if [[ "$failed" == "true" ]]; then
        print_error "Failed to process target: ${target}"
        return 1
    else
        print_success "Successfully processed target: ${target}"
        return 0
    fi
}

build_all_targets() {
    local failed_targets=()
    local successful_targets=()
    
    if [[ "$PARALLEL_BUILDS" == "true" ]]; then
        print_status "Building targets in parallel..."
        
        # Build in parallel using background processes
        local pids=()
        
        for target in "${TARGETS[@]}"; do
            build_single_target "${target}" &
            pids+=($!)
        done
        
        # Wait for all builds to complete
        for i in "${!pids[@]}"; do
            if wait "${pids[$i]}"; then
                successful_targets+=("${TARGETS[$i]}")
            else
                failed_targets+=("${TARGETS[$i]}")
            fi
        done
    else
        # Sequential builds
        print_status "Building targets sequentially..."
        
        for target in "${TARGETS[@]}"; do
            if build_single_target "${target}"; then
                successful_targets+=("${target}")
            else
                failed_targets+=("${target}")
            fi
        done
    fi
    
    # Report results
    echo ""
    print_success "Build Summary:"
    echo "  Successful targets: ${#successful_targets[@]}"
    for target in "${successful_targets[@]}"; do
        echo "    ✓ ${target}"
    done
    
    if [[ ${#failed_targets[@]} -gt 0 ]]; then
        echo "  Failed targets: ${#failed_targets[@]}"
        for target in "${failed_targets[@]}"; do
            echo "    ✗ ${target}"
        done
        return 1
    fi
    
    return 0
}

generate_build_report() {
    print_status "Generating build report..."
    
    local report_file="build-reports/build-report-$(date +%Y%m%d-%H%M%S).md"
    mkdir -p build-reports
    
    cat > "${report_file}" << EOF
# Spikeformer Docker Build Report

**Build Date:** $(date)
**Registry:** ${REGISTRY}
**Version:** ${VERSION}
**Security Scanning:** ${SCAN_ENABLED}

## Built Images

| Target | Status | Size | Security Scan |
|--------|--------|------|---------------|
EOF
    
    for target in "${TARGETS[@]}"; do
        local tag="${REGISTRY}:${target}"
        local status="❌ Failed"
        local size="N/A"
        local scan_status="❌ Failed"
        
        if docker images "${tag}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${tag}"; then
            status="✅ Success"
            size=$(docker images "${tag}" --format "{{.Size}}")
            
            if [[ -f "security-reports/scan-${target}-summary.txt" ]]; then
                scan_status="✅ Passed"
            fi
        fi
        
        echo "| ${target} | ${status} | ${size} | ${scan_status} |" >> "${report_file}"
    done
    
    cat >> "${report_file}" << EOF

## Security Reports

Security scan reports are available in the \`security-reports/\` directory.

## Next Steps

1. Review any security vulnerabilities found in the scans
2. Test images in your environment
3. Deploy to production when ready

---
Generated by Spikeformer Build Pipeline
EOF
    
    print_success "Build report generated: ${report_file}"
}

cleanup() {
    print_status "Cleaning up build artifacts..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove build cache if requested
    if [[ "${CLEAN_CACHE:-false}" == "true" ]]; then
        docker builder prune -f
    fi
    
    print_success "Cleanup completed"
}

main() {
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-scan)
                SCAN_ENABLED=false
                shift
                ;;
            --push)
                PUSH_ENABLED=true
                shift
                ;;
            --sequential)
                PARALLEL_BUILDS=false
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --target)
                TARGETS=("$2")
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --no-scan          Skip security scanning"
                echo "  --push             Push images to registry"
                echo "  --sequential       Build targets sequentially"
                echo "  --registry REG     Set registry name (default: spikeformer)"
                echo "  --version VER      Set version tag (default: latest)"
                echo "  --target TARGET    Build specific target only"
                echo "  --help             Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_status "Configuration:"
    echo "  Registry: ${REGISTRY}"
    echo "  Version: ${VERSION}"
    echo "  Security Scanning: ${SCAN_ENABLED}"
    echo "  Push Enabled: ${PUSH_ENABLED}"
    echo "  Parallel Builds: ${PARALLEL_BUILDS}"
    echo "  Targets: ${TARGETS[*]}"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Build all targets
    if build_all_targets; then
        print_success "All builds completed successfully!"
        generate_build_report
        cleanup
        exit 0
    else
        print_error "Some builds failed. Check the logs above."
        generate_build_report
        cleanup
        exit 1
    fi
}

# Run main function with all arguments
main "$@"