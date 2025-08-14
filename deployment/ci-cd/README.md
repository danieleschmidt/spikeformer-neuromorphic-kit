# CI/CD Setup Instructions

## GitHub Actions Workflow

To set up the CI/CD pipeline, manually copy the `github-actions-template.yml` file to `.github/workflows/ci-cd.yml` in your repository.

```bash
mkdir -p .github/workflows/
cp deployment/ci-cd/github-actions-template.yml .github/workflows/ci-cd.yml
```

## Required GitHub Secrets

Configure these secrets in your GitHub repository settings:

- `KUBECONFIG_DEV`: Base64-encoded kubeconfig for development cluster
- `KUBECONFIG_STAGING`: Base64-encoded kubeconfig for staging cluster  
- `KUBECONFIG_PROD`: Base64-encoded kubeconfig for production cluster
- `SLACK_WEBHOOK_URL`: Slack webhook for deployment notifications

## Manual Setup Required

Due to GitHub App permissions, the workflow file must be added manually by a user with appropriate repository permissions.

## Workflow Features

- ✅ Multi-Python version testing (3.9, 3.10, 3.11)
- ✅ Multi-PyTorch version compatibility testing
- ✅ Security scanning with Trivy and CodeQL
- ✅ Docker image building and pushing
- ✅ Automated deployment to dev/staging/production
- ✅ Performance monitoring and SLA compliance
- ✅ Automatic rollback on deployment failure