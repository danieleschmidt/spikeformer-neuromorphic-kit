# SLSA (Supply-chain Levels for Software Artifacts) Compliance

This document outlines the SLSA compliance strategy for the Spikeformer Neuromorphic Kit project.

## Current SLSA Level Assessment

**Target Level: SLSA Level 2**
**Current Status: In Progress**

### SLSA Level 1 Requirements ✅

- **Source**: All source code stored in version control (Git)
- **Build**: Automated build process defined (CI/CD pipelines)
- **Provenance**: Build provenance generation implemented

### SLSA Level 2 Requirements 🚧

- **Source**: ✅ Version controlled with authenticated commits
- **Build**: 🚧 Build service generates authenticated provenance  
- **Provenance**: 🚧 Tamper-resistant provenance with authenticated service identity

### SLSA Level 3 Requirements 🔄

- **Source**: 🔄 Source and build platform hardening
- **Build**: 🔄 Hardened build platform with isolation
- **Provenance**: 🔄 Non-forgeable provenance with stronger isolation

## Implementation Strategy

### 1. Source Requirements

#### Version Control Security
- ✅ Git repository with commit signing
- ✅ Branch protection rules enforced
- ✅ Two-person review requirement
- ✅ Signed commits verification

#### Access Control
- ✅ Repository access controls implemented
- ✅ Admin permissions limited
- ✅ Audit logging enabled
- 🚧 Multi-factor authentication required

### 2. Build Requirements

#### Build Environment
- 🚧 Ephemeral build environments (GitHub Actions)
- 🚧 Parameterless builds from VCS
- 🚧 Build service isolation
- 🚧 Dependency pinning and verification

#### Build Process
```yaml
# .github/workflows/slsa-build.yml (Reference - implement manually)
name: SLSA Compliant Build

on:
  push:
    tags: ['v*']
  release:
    types: [published]

permissions:
  contents: read
  packages: write
  id-token: write  # For OIDC token generation

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      sbom-digest: ${{ steps.sbom.outputs.digest }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up build environment
      # Isolated, ephemeral environment setup
      
    - name: Generate SBOM
      id: sbom
      # SBOM generation with integrity verification
      
    - name: Build artifacts
      id: build
      # Reproducible build process
      
    - name: Sign artifacts
      # Cryptographic signing of outputs
      
    - name: Generate provenance
      # SLSA provenance generation
```

### 3. Provenance Requirements

#### Provenance Generation
- 🚧 Automated provenance document creation
- 🚧 Service identity authentication
- 🚧 Build metadata inclusion
- 🚧 Cryptographic integrity protection

#### Provenance Format
```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "subject": [
    {
      "name": "spikeformer-neuromorphic-kit",
      "digest": {
        "sha256": "..."
      }
    }
  ],
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "predicate": {
    "builder": {
      "id": "https://github.com/actions/runner"
    },
    "buildType": "https://github.com/actions/workflow",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/your-org/spikeformer-neuromorphic-kit",
        "digest": {
          "sha1": "..."
        }
      }
    },
    "metadata": {
      "buildInvocationId": "...",
      "buildStartedOn": "2024-01-01T00:00:00Z",
      "buildFinishedOn": "2024-01-01T00:10:00Z",
      "completeness": {
        "parameters": true,
        "environment": false,
        "materials": true
      },
      "reproducible": false
    },
    "materials": [
      {
        "uri": "git+https://github.com/your-org/spikeformer-neuromorphic-kit",
        "digest": {
          "sha1": "..."
        }
      }
    ]
  }
}
```

## Verification Process

### Consumer Verification
1. **Artifact Integrity**: Verify checksums and signatures
2. **Provenance Verification**: Validate provenance signatures
3. **Source Verification**: Confirm source repository authenticity
4. **Build Verification**: Validate build process integrity

### Tools and Commands
```bash
# Verify artifact signatures
cosign verify --key cosign.pub spikeformer:latest

# Verify provenance
slsa-verifier verify-image spikeformer:latest \
  --source-uri github.com/your-org/spikeformer-neuromorphic-kit

# Verify SBOM
cosign verify-attestation --key cosign.pub \
  --type https://spdx.dev/Document spikeformer:latest
```

## Security Measures

### Cryptographic Controls
- **Signing Keys**: Ed25519 keys for artifact signing
- **Key Management**: Hardware security modules (HSM) for key storage
- **Rotation**: Regular key rotation schedule
- **Backup**: Secure key backup and recovery procedures

### Access Controls
- **Build Service**: Dedicated service accounts with minimal permissions
- **Signing**: Automated signing with service identity
- **Distribution**: Secure artifact distribution channels
- **Monitoring**: Build and deployment monitoring

## Compliance Monitoring

### Metrics and KPIs
- **Provenance Coverage**: % of releases with provenance
- **Signature Verification**: % of verified downloads
- **Build Reproducibility**: % of reproducible builds
- **Vulnerability Response**: Time to security patch deployment

### Audit Requirements
- **Quarterly Reviews**: SLSA compliance assessment
- **Annual Audits**: Third-party security audits
- **Incident Response**: Supply chain security incident procedures
- **Documentation**: Compliance evidence maintenance

## Tooling and Infrastructure

### Required Tools
- **GitHub Actions**: Build automation and provenance generation
- **Cosign**: Artifact signing and verification
- **SLSA Verifier**: Provenance verification
- **Sigstore**: Keyless signing infrastructure
- **SBOM Tools**: Software bill of materials generation

### Integration Points
- **CI/CD Pipeline**: Automated SLSA compliance checks
- **Package Registry**: Signed artifact distribution
- **Security Scanning**: Continuous vulnerability assessment
- **Monitoring**: Supply chain security monitoring

## Roadmap

### Phase 1: Foundation (Current)
- [x] Source control security hardening
- [x] Basic build automation
- [ ] Initial provenance generation
- [ ] SBOM integration

### Phase 2: Level 2 Compliance
- [ ] Authenticated build service
- [ ] Tamper-resistant provenance
- [ ] Automated verification
- [ ] Consumer tooling

### Phase 3: Level 3 Enhancement
- [ ] Hardened build platform
- [ ] Enhanced isolation
- [ ] Non-forgeable provenance
- [ ] Advanced verification

## Resources and References

### Standards and Specifications
- [SLSA Framework](https://slsa.dev/)
- [in-toto Attestation Framework](https://in-toto.io/)
- [Sigstore Project](https://www.sigstore.dev/)
- [Supply Chain Security Guide](https://cloud.google.com/software-supply-chain-security)

### Tools and Documentation
- [GitHub Actions SLSA](https://github.com/slsa-framework/slsa-github-generator)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [SLSA Verifier](https://github.com/slsa-framework/slsa-verifier)
- [SBOM Tools](https://github.com/anchore/syft)

---

*This SLSA compliance document is maintained by the security team and updated with each release cycle.*