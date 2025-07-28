# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial SpikeFormer Neuromorphic Kit implementation
- PyTorch to SNN conversion pipeline
- Support for Vision Transformer (ViT) architectures
- Intel Loihi 2 hardware backend (preliminary)
- SpiNNaker hardware backend (preliminary)
- Edge deployment capabilities
- Energy profiling and optimization tools
- Comprehensive testing framework
- Docker containerization with multi-stage builds
- Monitoring and observability with Prometheus/Grafana
- Security scanning and compliance tools
- Development environment with DevContainer support

### Features
- **Conversion Engine**: Automated transformer to SNN conversion
- **Hardware Backends**: Unified interface for neuromorphic hardware
- **Energy Monitoring**: Real-time power and energy measurement
- **Training Framework**: Hybrid ANN-SNN training capabilities
- **CLI Tools**: Command-line interface for common operations
- **Documentation**: Comprehensive guides and API documentation

### Infrastructure
- **CI/CD**: Automated testing, building, and deployment
- **Quality Gates**: Linting, formatting, type checking, security scanning
- **Monitoring**: Health checks, metrics collection, alerting
- **Security**: Vulnerability scanning, secret detection, compliance
- **Documentation**: Auto-generated API docs, user guides, runbooks

## [0.1.0] - 2025-01-28

### Added
- Initial project structure and SDLC automation
- Comprehensive development environment setup
- Testing infrastructure with unit, integration, and hardware tests
- Build and packaging system with Docker multi-stage builds
- Monitoring and observability stack
- Security scanning and compliance framework
- Documentation structure and development guides

### Infrastructure
- DevContainer configuration for consistent development environment
- Pre-commit hooks for code quality enforcement
- Docker Compose setup for development and production
- Prometheus and Grafana monitoring stack
- Automated security scanning with Bandit, Safety, and pip-audit
- GitHub Actions workflows for CI/CD (reference implementation)

### Documentation
- Project architecture and design documents
- Development setup and contribution guidelines
- Security policy and vulnerability reporting process
- Deployment runbooks and operational procedures

---

## Release Notes Template

### Version X.Y.Z - YYYY-MM-DD

#### 🚀 New Features
- Feature descriptions with impact

#### 🐛 Bug Fixes
- Bug fix descriptions

#### ⚡ Performance Improvements
- Performance optimization details

#### 🔧 Infrastructure Changes
- Infrastructure and tooling updates

#### 📚 Documentation
- Documentation updates and improvements

#### 🔒 Security
- Security improvements and fixes

#### ⚠️ Breaking Changes
- Breaking changes requiring user action

#### 🗑️ Deprecated
- Features marked for deprecation

---

## Contribution Guidelines

This changelog is automatically generated using semantic-release based on conventional commit messages. To contribute to the changelog:

1. Use conventional commit format: `type(scope): description`
2. Types that trigger releases:
   - `feat`: New features (minor version bump)
   - `fix`: Bug fixes (patch version bump)
   - `perf`: Performance improvements (patch version bump)
   - `refactor`: Code refactoring (patch version bump)
   - `revert`: Reverting changes (patch version bump)

3. Breaking changes:
   - Add `BREAKING CHANGE:` in commit body (major version bump)
   - Or use `!` after type: `feat!: breaking change`

4. Types that don't trigger releases:
   - `docs`: Documentation changes
   - `style`: Code style changes
   - `test`: Test additions/changes
   - `chore`: Maintenance tasks
   - `ci`: CI/CD changes
   - `build`: Build system changes

Example commit messages:
```
feat(conversion): add support for BERT architecture conversion
fix(hardware): resolve Loihi2 memory allocation issue
perf(inference): optimize spike tensor operations
docs(api): update conversion API documentation
BREAKING CHANGE: remove deprecated threshold_mode parameter
```