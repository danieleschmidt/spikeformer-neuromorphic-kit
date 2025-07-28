.PHONY: help install install-dev test test-coverage lint format typecheck clean build build-docker run-dev docs security benchmark

# Default target
help: ## Show this help message
	@echo "SpikeFormer Neuromorphic Kit - Available Commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Development setup
install: ## Install package in development mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-all: ## Install package with all dependencies
	pip install -e ".[all]"

# Testing
test: ## Run unit tests
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-hardware: ## Run hardware tests (requires hardware)
	pytest tests/hardware/ -v --hardware

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=spikeformer --cov-report=html --cov-report=term --cov-report=xml

test-all: ## Run all tests
	pytest tests/ -v

# Code quality
lint: ## Run linting
	ruff check spikeformer/ tests/

lint-fix: ## Run linting with auto-fix
	ruff check spikeformer/ tests/ --fix

format: ## Format code
	black spikeformer/ tests/
	isort spikeformer/ tests/

typecheck: ## Run type checking
	mypy spikeformer/

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

# Security
security: ## Run security checks
	bandit -r spikeformer/ -f json -o security-report.json
	safety check

# Cleaning
clean: ## Clean build artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*.pyd' -delete
	find . -name '.coverage' -delete
	find . -name '*.orig' -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

clean-all: clean ## Clean everything including caches
	rm -rf .cache/
	rm -rf logs/
	rm -rf outputs/
	rm -rf wandb/
	rm -rf mlruns/

# Building
build: ## Build Python package
	python -m build

build-docker: ## Build Docker images
	docker-compose build

build-docker-dev: ## Build development Docker image
	docker build --target development -t spikeformer:dev .

build-docker-prod: ## Build production Docker images
	docker build --target production-cpu -t spikeformer:cpu .
	docker build --target production-gpu -t spikeformer:gpu .

build-docker-loihi2: ## Build Loihi2 specialized image
	docker build --target loihi2 -t spikeformer:loihi2 .

build-docker-spinnaker: ## Build SpiNNaker specialized image
	docker build --target spinnaker -t spikeformer:spinnaker .

build-docker-edge: ## Build edge deployment image
	docker build --target edge -t spikeformer:edge .

# Running
run-dev: ## Run development environment
	docker-compose up spikeformer-dev

run-prod-cpu: ## Run production CPU service
	docker-compose up spikeformer-cpu

run-prod-gpu: ## Run production GPU service
	docker-compose up spikeformer-gpu

run-monitoring: ## Start monitoring stack
	docker-compose up prometheus grafana

run-full-stack: ## Start complete development stack
	docker-compose up

# Documentation
docs: ## Build documentation
	sphinx-build -b html docs/ docs/_build/

docs-serve: ## Serve documentation with auto-reload
	sphinx-autobuild docs/ docs/_build/ --host=0.0.0.0 --port=8000

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

# Development tools
jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

profile: ## Profile model performance
	python -m cProfile -o profile.stats scripts/profile_model.py

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

# Model operations
convert-model: ## Convert transformer to SNN (requires model path)
	@if [ -z "$(MODEL)" ]; then echo "Usage: make convert-model MODEL=path/to/model"; exit 1; fi
	spikeformer-convert $(MODEL) --output converted_$(MODEL)

deploy-loihi2: ## Deploy model to Loihi 2 (requires model path)
	@if [ -z "$(MODEL)" ]; then echo "Usage: make deploy-loihi2 MODEL=path/to/model"; exit 1; fi
	spikeformer-deploy $(MODEL) --backend loihi2

deploy-spinnaker: ## Deploy model to SpiNNaker (requires model path)
	@if [ -z "$(MODEL)" ]; then echo "Usage: make deploy-spinnaker MODEL=path/to/model"; exit 1; fi
	spikeformer-deploy $(MODEL) --backend spinnaker

# Monitoring
monitor-energy: ## Start energy monitoring
	python scripts/monitoring.py --type energy

monitor-performance: ## Start performance monitoring
	python scripts/monitoring.py --type performance

# Data management
download-data: ## Download sample datasets
	python scripts/download_datasets.py

prepare-data: ## Prepare datasets for training
	python scripts/prepare_datasets.py

# CI/CD helpers
ci-test: ## Run CI test suite
	pytest tests/ --tb=short -x

ci-lint: ## Run CI linting
	ruff check spikeformer/ tests/
	black --check spikeformer/ tests/
	isort --check-only spikeformer/ tests/

ci-security: ## Run CI security checks
	bandit -r spikeformer/
	safety check

ci-all: ci-lint ci-security ci-test ## Run all CI checks

# Release
version: ## Show current version
	@python -c "import spikeformer; print(spikeformer.__version__)"

release-patch: ## Release patch version
	bump2version patch

release-minor: ## Release minor version
	bump2version minor

release-major: ## Release major version
	bump2version major

# Hardware testing
test-loihi2: ## Test Loihi 2 integration (requires hardware)
	pytest tests/hardware/test_loihi2.py -v --loihi2

test-spinnaker: ## Test SpiNNaker integration (requires hardware)
	pytest tests/hardware/test_spinnaker.py -v --spinnaker

# Environment
env-create: ## Create development environment
	python -m venv venv
	source venv/bin/activate && pip install -e ".[dev]"

env-update: ## Update development environment
	pip install --upgrade -e ".[dev]"

# Utilities
check-deps: ## Check for outdated dependencies
	pip list --outdated

check-security-deps: ## Check dependencies for security vulnerabilities
	safety check

update-deps: ## Update dependencies (use with caution)
	pip-compile --upgrade requirements.in

# Docker utilities
docker-clean: ## Clean Docker resources
	docker system prune -f
	docker volume prune -f

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-shell: ## Open shell in development container
	docker-compose exec spikeformer-dev bash

# Database operations
db-init: ## Initialize database
	docker-compose up -d postgres
	python scripts/init_database.py

db-migrate: ## Run database migrations
	python scripts/migrate_database.py

db-seed: ## Seed database with sample data
	python scripts/seed_database.py