# GitHub Actions Setup Guide

## Overview

This guide provides comprehensive instructions for setting up GitHub Actions workflows for the Spikeformer Neuromorphic Kit. Since GitHub Actions workflows require manual creation due to security restrictions, this document serves as a complete implementation guide.

## Prerequisites

### Repository Settings
1. **Branch Protection Rules** (Repository Settings → Branches)
   - Protect `main` branch
   - Require PR reviews (minimum 1)
   - Require status checks to pass
   - Dismiss stale reviews when new commits are pushed
   - Restrict pushes to matching branches

2. **Repository Permissions**
   - Enable "Allow GitHub Actions to create and approve pull requests"
   - Set "Fork pull request workflows from outside collaborators" to require approval

3. **Secrets Configuration** (Repository Settings → Secrets and variables → Actions)
   ```
   PYPI_API_TOKEN          # For package publishing
   DOCKER_USERNAME         # Docker Hub username
   DOCKER_PASSWORD         # Docker Hub password
   CODECOV_TOKEN          # Code coverage reporting
   SLACK_WEBHOOK_URL      # Notifications
   ```

## Workflow Files Structure

Create the following directory structure:
```
.github/
├── workflows/
│   ├── ci.yml                    # Main CI pipeline
│   ├── security.yml              # Security scanning
│   ├── release.yml               # Release automation
│   ├── deploy-staging.yml        # Staging deployment
│   ├── deploy-production.yml     # Production deployment
│   └── nightly.yml               # Nightly tasks
├── actions/
│   └── setup-neuromorphic/       # Custom action
└── ISSUE_TEMPLATE/
    └── bug_report.md             # Already exists
```

## Core Workflow Implementations

### 1. Main CI Pipeline (.github/workflows/ci.yml)

```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

jobs:
  # Security and Quality Gates (Run in Parallel)
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install bandit safety semgrep
          pip install -e ".[dev]"
      
      - name: Run Bandit security scan
        run: |
          bandit -r spikeformer/ -f json -o bandit-report.json
          bandit -r spikeformer/ -f txt
        continue-on-error: true
      
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check
        continue-on-error: true
      
      - name: Run Semgrep
        run: |
          semgrep --config=auto --json --output=semgrep-report.json spikeformer/
          semgrep --config=auto spikeformer/
        continue-on-error: true
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            bandit-report.json
            safety-report.json
            semgrep-report.json

  code-quality:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run Ruff
        run: |
          ruff check spikeformer/ --output-format=github
          ruff format --check spikeformer/
      
      - name: Run Black
        run: black --check --diff spikeformer/
      
      - name: Run isort
        run: isort --check-only --diff spikeformer/
      
      - name: Run MyPy
        run: mypy spikeformer/ --junit-xml mypy-results.xml
      
      - name: Upload MyPy results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: mypy-results
          path: mypy-results.xml

  # Testing Pipeline (Sequential)
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: [security-scan, code-quality]
    
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=spikeformer \
            --cov-report=xml \
            --cov-report=html \
            --junit-xml=junit-results.xml \
            --maxfail=5 \
            -v
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.11'
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage.xml
          fail_ci_if_error: true
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            junit-results.xml
            htmlcov/
            coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: spikeformer_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:test_password@localhost:5432/spikeformer_test
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/integration/ \
            --maxfail=3 \
            -v \
            --tb=short

  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run performance benchmarks
        run: |
          pytest tests/performance/ \
            --benchmark-only \
            --benchmark-json=benchmark-results.json \
            --benchmark-sort=mean
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark-results.json

  # Build and Package
  build:
    name: Build and Package
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      package-version: ${{ steps.version.outputs.version }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip build twine
          pip install -e ".[dev]"
      
      - name: Get version
        id: version
        run: |
          VERSION=$(python -c "import spikeformer; print(spikeformer.__version__)")
          echo "version=$VERSION" >> $GITHUB_OUTPUT
      
      - name: Build Python package
        run: |
          python -m build
          twine check dist/*
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        id: build
        run: |
          docker buildx build \
            --platform linux/amd64,linux/arm64 \
            --tag spikeformer:${{ github.sha }} \
            --tag spikeformer:latest \
            --output type=docker \
            .
          
          DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' spikeformer:latest)
          echo "digest=$DIGEST" >> $GITHUB_OUTPUT
      
      - name: Generate SBOM
        run: |
          python scripts/generate_sbom.py
          mkdir -p sbom-artifacts
          cp sbom/*.json sbom-artifacts/
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-artifacts
          path: |
            dist/
            sbom-artifacts/
      
      - name: Upload Docker image
        run: |
          docker save spikeformer:latest | gzip > spikeformer-docker.tar.gz
      
      - name: Upload Docker artifact
        uses: actions/upload-artifact@v3
        with:
          name: docker-image
          path: spikeformer-docker.tar.gz

  # Mutation Testing (Optional, runs on main branch)
  mutation-testing:
    name: Mutation Testing
    runs-on: ubuntu-latest
    needs: unit-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    continue-on-error: true
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install mutmut
          pip install -e ".[dev]"
      
      - name: Run mutation testing
        run: |
          mutmut run --paths-to-mutate spikeformer/ || true
          mutmut html || true
      
      - name: Upload mutation report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: mutation-report
          path: html/

  # Notification
  notify:
    name: Notify Results
    runs-on: ubuntu-latest
    needs: [security-scan, code-quality, unit-tests, integration-tests, build]
    if: always() && (github.event_name == 'push' && github.ref == 'refs/heads/main')
    
    steps:
      - name: Notify Slack
        uses: 8398a7/action-slack@v3
        if: always()
        with:
          status: ${{ job.status }}
          channel: '#ci-cd'
          webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
```

### 2. Security Workflow (.github/workflows/security.yml)

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'requirements.txt'
      - 'pyproject.toml'
      - 'Dockerfile'

jobs:
  comprehensive-security:
    name: Comprehensive Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      
      - name: Run comprehensive security scan
        run: |
          # Source code security
          bandit -r spikeformer/ -f json -o security-reports/bandit.json
          
          # Dependency security
          safety check --json --output security-reports/safety.json
          pip-audit --format=json --output=security-reports/pip-audit.json
          
          # Generate SBOM
          python scripts/generate_sbom.py
        continue-on-error: true
      
      - name: Build Docker image for scanning
        run: |
          docker build -t spikeformer:security-scan .
      
      - name: Run container security scan
        run: |
          # Install Trivy
          curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
          
          # Scan container
          trivy image --format json --output security-reports/trivy.json spikeformer:security-scan
          python scripts/container_security_scan.py --image spikeformer:security-scan
        continue-on-error: true
      
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: security-reports/
          retention-days: 30
      
      - name: Create security issue on failure
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Security scan failures detected',
              body: `Security scanning workflow failed on ${context.sha.substring(0, 8)}.
              
              Please review the security reports and address any critical issues.
              
              Workflow run: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
              labels: ['security', 'high-priority']
            })
```

### 3. Release Workflow (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., v1.0.0)'
        required: true
        type: string

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip build twine
          pip install -e ".[dev]"
      
      - name: Generate release notes
        run: |
          ./scripts/generate_release_notes.sh ${{ github.ref_name }} > release-notes.md
      
      - name: Build package
        run: |
          python -m build
          twine check dist/*
      
      - name: Build and sign Docker image
        run: |
          docker build -t spikeformer:${{ github.ref_name }} .
          docker tag spikeformer:${{ github.ref_name }} spikeformer:latest
          
          # Install cosign for signing
          curl -O -L "https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64"
          sudo mv cosign-linux-amd64 /usr/local/bin/cosign
          sudo chmod +x /usr/local/bin/cosign
          
          # Sign image (requires keyless signing setup)
          cosign sign --yes spikeformer:${{ github.ref_name }}
      
      - name: Generate SBOM
        run: |
          python scripts/generate_sbom.py
          
          # Generate container SBOM
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            anchore/syft:latest spikeformer:${{ github.ref_name }} -o spdx-json > sbom/container-sbom.spdx.json
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Push Docker images
        run: |
          docker push spikeformer:${{ github.ref_name }}
          docker push spikeformer:latest
      
      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref_name }}
          body_path: release-notes.md
          draft: false
          prerelease: false
      
      - name: Upload release assets
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: sbom/spikeformer-spdx.json
          asset_name: sbom.spdx.json
          asset_content_type: application/json
```

### 4. Deployment Workflows

#### Staging Deployment (.github/workflows/deploy-staging.yml)

```yaml
name: Deploy to Staging

on:
  push:
    branches: [develop]
  workflow_dispatch:

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'latest'
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/spikeformer-api \
            api=spikeformer:${{ github.sha }} \
            --namespace=spikeformer-staging
          
          kubectl rollout status deployment/spikeformer-api \
            --namespace=spikeformer-staging \
            --timeout=300s
      
      - name: Run health checks
        run: |
          ./scripts/health_check.sh https://staging.spikeformer.com
      
      - name: Run smoke tests
        run: |
          pytest tests/smoke/ --base-url=https://staging.spikeformer.com
```

#### Production Deployment (.github/workflows/deploy-production.yml)

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'latest'
      
      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
      
      - name: Blue-Green Deployment
        run: |
          # Deploy to green environment
          kubectl set image deployment/spikeformer-api \
            api=spikeformer:${{ github.sha }} \
            --namespace=spikeformer-green
          
          # Wait for rollout
          kubectl rollout status deployment/spikeformer-api \
            --namespace=spikeformer-green \
            --timeout=600s
          
          # Health check green environment
          ./scripts/health_check.sh https://green.spikeformer.com
          
          # Switch traffic to green
          kubectl patch service spikeformer-api \
            -p '{"spec":{"selector":{"environment":"green"}}}' \
            --namespace=spikeformer
          
          # Monitor for 5 minutes
          ./scripts/deployment_monitor.sh --duration=300
      
      - name: Cleanup on success
        if: success()
        run: |
          kubectl delete deployment spikeformer-api \
            --namespace=spikeformer-blue
      
      - name: Rollback on failure
        if: failure()
        run: |
          kubectl patch service spikeformer-api \
            -p '{"spec":{"selector":{"environment":"blue"}}}' \
            --namespace=spikeformer
          
          kubectl delete deployment spikeformer-api \
            --namespace=spikeformer-green
```

## Setup Instructions

### 1. Create Workflow Files
Manually create each workflow file in the `.github/workflows/` directory with the content provided above.

### 2. Configure Repository Secrets
Add the following secrets in your repository settings:

```bash
# GitHub Settings → Secrets and variables → Actions

# PyPI Publishing
PYPI_API_TOKEN="pypi-..."

# Docker Registry
DOCKER_USERNAME="your-username"
DOCKER_PASSWORD="your-password"

# Code Coverage
CODECOV_TOKEN="your-codecov-token"

# Notifications
SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# Kubernetes Deployment
KUBE_CONFIG_STAGING="base64-encoded-kubeconfig"
KUBE_CONFIG_PRODUCTION="base64-encoded-kubeconfig"
```

### 3. Set Up Branch Protection
Configure branch protection rules in repository settings to require status checks.

### 4. Test Workflows
1. Create a test branch
2. Make a small change
3. Create a pull request
4. Verify that all checks pass

## Monitoring and Maintenance

### 1. Regular Updates
- Update action versions monthly
- Review and update security configurations
- Monitor workflow execution times
- Update dependencies in requirements files

### 2. Performance Optimization
- Use caching for packages and dependencies
- Parallelize independent jobs
- Optimize Docker layer caching
- Monitor resource usage

### 3. Security Maintenance
- Regularly rotate secrets
- Update security scanning tools
- Review and update permissions
- Monitor for security advisories

---

*This setup provides a comprehensive CI/CD pipeline that follows security best practices and supports the neuromorphic computing domain's specific needs.*