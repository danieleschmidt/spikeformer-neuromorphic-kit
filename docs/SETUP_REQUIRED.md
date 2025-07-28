# Manual Setup Requirements

## Repository Administrator Tasks

The following items require repository admin permissions and must be set up manually:

### 1. GitHub Actions Workflows
* Create `.github/workflows/` directory
* Add CI/CD pipeline workflow file
* Configure hardware testing workflows
* Set up release automation
* **Reference**: [GitHub Actions Documentation](https://docs.github.com/en/actions)

### 2. Branch Protection Rules
* Protect `main` branch
* Require PR reviews (minimum 1)
* Require status checks to pass
* Enforce for administrators
* **Setup**: Repository Settings → Branches → Add rule

### 3. Repository Settings
* **Topics**: Add neuromorphic, spiking-neural-networks, transformer
* **Description**: Copy from package.json
* **Homepage**: Set project documentation URL
* **Features**: Enable Issues, Discussions as needed

### 4. Secrets and Variables
* Add required environment secrets
* Configure deployment credentials
* Set up external service tokens
* **Location**: Repository Settings → Secrets and variables

### 5. External Integrations
* **Code Quality**: Configure CodeClimate/SonarQube
* **Security**: Set up GitGuardian organization
* **Monitoring**: Connect to alerting systems
* **Notifications**: Add webhook URLs for CI status

### 6. Deployment Environments
* Create staging/production environments
* Configure environment protection rules
* Set up deployment approvals
* **Setup**: Repository Settings → Environments

## Security Considerations

* Review all external integrations
* Audit access permissions regularly
* Use least-privilege principle
* Enable security advisories

## Verification Steps

1. Test CI pipeline with sample PR
2. Verify branch protection enforcement
3. Check all integrations are working
4. Validate security scanning results

For detailed implementation, see [workflows documentation](workflows/README.md).