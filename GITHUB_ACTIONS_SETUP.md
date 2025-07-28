# GitHub Actions Setup Instructions

⚠️ **Important**: The GitHub Actions workflow files in `.github/workflows/` are provided for reference only. You must manually create these workflows in your GitHub repository as I cannot modify GitHub Actions directly.

## Required Actions

### 1. Create CI/CD Workflows

Copy the following files to your GitHub repository:

1. **`.github/workflows/ci.yml`** - Continuous Integration workflow
2. **`.github/workflows/cd.yml`** - Continuous Deployment workflow

### 2. Configure Repository Secrets

Add these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets:
- `SNYK_TOKEN` - For dependency vulnerability scanning
- `SLACK_WEBHOOK_URL` - For deployment notifications (optional)

#### For Production Deployment:
- `PYPI_API_TOKEN` - For publishing to PyPI
- Container registry credentials (if using private registry)

### 3. Enable GitHub Pages (Optional)

If you want automatic documentation deployment:

1. Go to `Settings > Pages`
2. Select "GitHub Actions" as the source
3. The documentation will be automatically deployed on pushes to main

### 4. Configure Branch Protection

Recommended branch protection rules for `main`:

1. Go to `Settings > Branches`
2. Add rule for `main` branch:
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Required status checks:
     - `Code Quality`
     - `Security Scan`
     - `Unit Tests`
     - `Integration Tests`
     - `Build Verification`

### 5. Set Up Environments

Create the following environments in `Settings > Environments`:

#### Staging Environment
- Add required reviewers
- Set deployment branch rule to `main`

#### Production Environment  
- Add required reviewers
- Set deployment branch rule to `tags/v*`
- Add deployment protection rules

#### Release Environment
- Used for PyPI publishing
- Add required reviewers for release approval

### 6. Configure Issue and PR Templates

The following templates are provided in `.github/`:

- Issue templates for bugs, features, and hardware requests
- Pull request template with checklist

### 7. Additional Integrations

#### CodeCov (Recommended)
1. Sign up at https://codecov.io
2. Connect your GitHub repository
3. No additional configuration needed - workflows include codecov upload

#### Snyk (Recommended)
1. Sign up at https://snyk.io
2. Connect your GitHub repository
3. Add `SNYK_TOKEN` secret

## Workflow Overview

### CI Workflow (`ci.yml`)

Triggered on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Jobs:
1. **Code Quality** - Linting, formatting, type checking
2. **Security Scan** - Vulnerability scanning with Bandit and Safety
3. **Unit Tests** - Cross-platform testing (Linux, Windows, macOS)
4. **Integration Tests** - End-to-end integration testing
5. **Build Verification** - Package building and verification
6. **Docker Build** - Multi-stage Docker image building
7. **Documentation** - Sphinx documentation building
8. **Performance Benchmarks** - Automated performance testing
9. **Hardware Simulation** - Neuromorphic hardware simulation tests
10. **Dependency Scan** - Third-party vulnerability scanning

### CD Workflow (`cd.yml`)

Triggered on:
- Push to `main` branch (staging deployment)
- Tagged releases (production deployment)
- Published releases (PyPI publishing)

Jobs:
1. **Pre-deployment Tests** - Comprehensive testing before deployment
2. **Docker Publish** - Multi-variant container publishing
3. **PyPI Publish** - Python package publishing (releases only)
4. **Documentation Deploy** - GitHub Pages documentation
5. **GitHub Release** - Automated release creation
6. **Staging Deploy** - Staging environment deployment
7. **Production Deploy** - Production environment deployment (releases only)
8. **Post-deployment Verification** - Smoke tests and health checks
9. **Notifications** - Slack/email notifications

## Hardware-Specific Testing

The workflows include conditional hardware testing:

- Tests are skipped if hardware is not available
- Use environment variables to enable hardware tests:
  - `LOIHI2_AVAILABLE=true`
  - `SPINNAKER_AVAILABLE=true`
- Hardware tests run in separate jobs to avoid blocking

## Performance Monitoring

- Automated benchmarking on every PR
- Performance regression detection
- Benchmark results stored in GitHub Pages
- Memory usage monitoring during tests

## Security Features

- Automated dependency vulnerability scanning
- Code security analysis with Bandit
- Container image security scanning
- Secrets detection in commits
- Branch protection with required reviews

## Getting Started

1. Copy the workflow files to your repository
2. Configure the required secrets
3. Set up branch protection rules
4. Create your first pull request to test the CI pipeline
5. Tag a release to test the CD pipeline

## Troubleshooting

### Common Issues:

1. **Tests failing on Windows**: Check path separators and line endings
2. **Docker builds failing**: Verify Dockerfile syntax and dependencies
3. **PyPI publishing failing**: Ensure `PYPI_API_TOKEN` is correctly set
4. **Documentation deployment failing**: Check Sphinx configuration

### Getting Help:

- Check the Actions tab in GitHub for detailed logs
- Review the workflow files for job dependencies
- Ensure all required secrets are configured
- Verify branch protection rules are not too restrictive