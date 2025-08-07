#!/bin/bash
# Deployment automation script for Neuromorphic Computing Platform
# Usage: ./deploy.sh [environment] [action]
# Environment: dev, staging, production, edge, research
# Action: deploy, update, rollback, status, logs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
NAMESPACE="neuromorphic"
APP_NAME="neuromorphic-api"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-neuromorphic}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "docker" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not connected to a cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

build_and_push_image() {
    local environment=$1
    local full_image_name="${DOCKER_REGISTRY}/spikeformer:${IMAGE_TAG}"
    
    log_info "Building Docker image for $environment environment..."
    
    # Build multi-stage image
    docker build \
        --file "$SCRIPT_DIR/docker/Dockerfile" \
        --target "$environment" \
        --tag "$full_image_name" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="${IMAGE_TAG}" \
        "$PROJECT_ROOT"
    
    log_info "Pushing image to registry..."
    docker push "$full_image_name"
    
    log_success "Image built and pushed: $full_image_name"
}

create_namespace() {
    log_info "Creating namespace if not exists..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for monitoring
    kubectl label namespace "$NAMESPACE" \
        app.kubernetes.io/name=neuromorphic \
        app.kubernetes.io/instance=production \
        --overwrite
    
    log_success "Namespace $NAMESPACE ready"
}

deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret neuromorphic-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist, skipping creation"
        return
    fi
    
    # Generate secrets if they don't exist
    DB_PASSWORD=$(openssl rand -base64 32)
    JWT_SECRET_KEY=$(openssl rand -base64 64)
    API_KEY=$(openssl rand -base64 32)
    
    kubectl create secret generic neuromorphic-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=DB_PASSWORD="$DB_PASSWORD" \
        --from-literal=JWT_SECRET_KEY="$JWT_SECRET_KEY" \
        --from-literal=API_KEY="$API_KEY"
    
    log_success "Secrets deployed"
}

deploy_configmap() {
    log_info "Deploying ConfigMap..."
    
    kubectl apply -f "$SCRIPT_DIR/kubernetes/deployment.yaml"
    
    log_success "ConfigMap deployed"
}

deploy_postgres() {
    log_info "Deploying PostgreSQL database..."
    
    # Create PostgreSQL configuration
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: $NAMESPACE
data:
  postgresql.conf: |
    # PostgreSQL configuration for neuromorphic workloads
    shared_buffers = 256MB
    effective_cache_size = 1GB
    maintenance_work_mem = 64MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 4MB
    min_wal_size = 1GB
    max_wal_size = 4GB
EOF
    
    # Deploy PostgreSQL StatefulSet
    kubectl apply -f "$SCRIPT_DIR/kubernetes/deployment.yaml"
    
    # Wait for PostgreSQL to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=database -n "$NAMESPACE" --timeout=300s
    
    log_success "PostgreSQL deployed and ready"
}

deploy_redis() {
    log_info "Deploying Redis cache..."
    
    kubectl apply -f "$SCRIPT_DIR/kubernetes/deployment.yaml"
    
    # Wait for Redis to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=cache -n "$NAMESPACE" --timeout=300s
    
    log_success "Redis deployed and ready"
}

deploy_application() {
    local environment=$1
    
    log_info "Deploying application for $environment environment..."
    
    # Update image tag in deployment
    kubectl patch deployment "$APP_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"template":{"spec":{"containers":[{"name":"neuromorphic-api","image":"'${DOCKER_REGISTRY}'/spikeformer:'${IMAGE_TAG}'"}]}}}}'
    
    # Wait for rollout to complete
    kubectl rollout status deployment/"$APP_NAME" -n "$NAMESPACE" --timeout=600s
    
    log_success "Application deployed successfully"
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Apply monitoring resources
    kubectl apply -f "$SCRIPT_DIR/kubernetes/deployment.yaml"
    
    log_success "Monitoring stack deployed"
}

setup_ingress() {
    log_info "Setting up ingress..."
    
    # Check if cert-manager is available
    if kubectl get crd certificates.cert-manager.io &> /dev/null; then
        log_info "cert-manager detected, SSL certificates will be automatically managed"
    else
        log_warning "cert-manager not found, SSL certificates need manual setup"
    fi
    
    kubectl apply -f "$SCRIPT_DIR/kubernetes/deployment.yaml"
    
    log_success "Ingress configured"
}

wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for all pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neuromorphic -n "$NAMESPACE" --timeout=600s
    
    # Check deployment status
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=neuromorphic
    
    log_success "Deployment is ready"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Port forward to access health endpoint
    kubectl port-forward -n "$NAMESPACE" svc/neuromorphic-api 8080:8080 &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    # Health check
    if curl -sf http://localhost:8080/health > /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    # Readiness check
    if curl -sf http://localhost:8080/ready > /dev/null; then
        log_success "Readiness check passed"
    else
        log_error "Readiness check failed"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_success "All health checks passed"
}

show_status() {
    log_info "Deployment status:"
    
    echo "Namespace:"
    kubectl get namespace "$NAMESPACE"
    
    echo -e "\nPods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo -e "\nServices:"
    kubectl get services -n "$NAMESPACE"
    
    echo -e "\nIngress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress found"
    
    echo -e "\nHPA Status:"
    kubectl get hpa -n "$NAMESPACE" 2>/dev/null || echo "No HPA found"
    
    echo -e "\nPVC Status:"
    kubectl get pvc -n "$NAMESPACE"
}

show_logs() {
    local pod_selector=${1:-"app.kubernetes.io/component=api"}
    
    log_info "Showing logs for pods matching: $pod_selector"
    
    kubectl logs -n "$NAMESPACE" -l "$pod_selector" --tail=100 -f
}

rollback_deployment() {
    local revision=${1:-""}
    
    log_warning "Rolling back deployment..."
    
    if [[ -n "$revision" ]]; then
        kubectl rollout undo deployment/"$APP_NAME" -n "$NAMESPACE" --to-revision="$revision"
    else
        kubectl rollout undo deployment/"$APP_NAME" -n "$NAMESPACE"
    fi
    
    kubectl rollout status deployment/"$APP_NAME" -n "$NAMESPACE" --timeout=300s
    
    log_success "Rollback completed"
}

cleanup_deployment() {
    log_warning "Cleaning up deployment..."
    
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Main deployment function
deploy_full_stack() {
    local environment=$1
    
    log_info "Starting full deployment for $environment environment..."
    
    check_prerequisites
    build_and_push_image "$environment"
    create_namespace
    deploy_secrets
    deploy_configmap
    deploy_postgres
    deploy_redis
    deploy_application "$environment"
    deploy_monitoring
    setup_ingress
    wait_for_deployment
    run_health_checks
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at: https://neuromorphic.yourdomain.com"
}

# Update deployment
update_deployment() {
    local environment=$1
    
    log_info "Updating deployment for $environment environment..."
    
    build_and_push_image "$environment"
    deploy_application "$environment"
    wait_for_deployment
    run_health_checks
    
    log_success "Update completed successfully!"
}

# Main script logic
main() {
    local environment=${1:-"production"}
    local action=${2:-"deploy"}
    
    case "$action" in
        deploy)
            deploy_full_stack "$environment"
            ;;
        update)
            update_deployment "$environment"
            ;;
        rollback)
            rollback_deployment "${3:-}"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${3:-}"
            ;;
        cleanup)
            cleanup_deployment
            ;;
        health)
            run_health_checks
            ;;
        *)
            log_error "Unknown action: $action"
            log_info "Usage: $0 [environment] [action]"
            log_info "Environment: dev, staging, production, edge, research"
            log_info "Actions: deploy, update, rollback, status, logs, cleanup, health"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"