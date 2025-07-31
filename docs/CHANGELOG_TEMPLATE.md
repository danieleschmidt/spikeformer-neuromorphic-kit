# Changelog Template

This template provides guidelines for maintaining a comprehensive changelog for the Spikeformer Neuromorphic Kit.

## Format Guidelines

Follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles with neuromorphic-specific enhancements:

### Version Headers
```markdown
## [Unreleased]

## [1.2.0] - 2024-01-15
```

### Change Categories

#### Added
- New neuromorphic hardware support
- New model architectures
- New training algorithms
- New conversion tools
- New API endpoints
- New documentation sections

#### Changed
- Performance improvements
- API modifications (with migration notes)
- Model architecture updates
- Hardware abstraction changes
- Documentation restructuring

#### Deprecated
- Old API methods (with removal timeline)
- Legacy hardware support
- Outdated model formats
- Old configuration formats

#### Removed
- Discontinued features
- Removed hardware support
- Deleted deprecated APIs
- Removed dependencies

#### Fixed
- Bug fixes in neuromorphic operations
- Hardware-specific issues
- Performance regressions
- Memory leaks
- Training instabilities

#### Security
- Security vulnerabilities patched
- Dependency security updates
- Access control improvements
- Encryption enhancements

### Entry Format

Each entry should include:
1. **What changed** - Clear description
2. **Why it changed** - Context and motivation
3. **Impact** - Who is affected and how
4. **Migration** - How to adapt (for breaking changes)
5. **References** - Links to issues, PRs, or documentation

### Example Entries

```markdown
## [1.2.0] - 2024-01-15

### Added
- **Intel Loihi 2 Support**: Full compilation and deployment support for Intel's second-generation neuromorphic processor
  - Supports up to 1M neurons per chip
  - Automatic partitioning for multi-chip deployments
  - Real-time inference capabilities
  - See [Loihi 2 Documentation](docs/hardware/loihi2.md) for setup instructions
  - Resolves #123, #145

- **Spiking Transformer Architecture**: New energy-efficient transformer implementation
  - 50% reduction in power consumption vs standard transformers
  - Native support for temporal encoding
  - Hardware-agnostic design
  - Migration guide: [Spiking Transformers](docs/models/spiking-transformers.md)
  - Implements RFC-004

### Changed
- **BREAKING**: Model compilation API restructured for better hardware abstraction
  - Old: `model.compile_for_loihi()`
  - New: `compiler.compile(model, target="loihi2")`
  - Migration: See [Compilation API Migration](docs/migration/v1.2-compilation.md)
  - Affects all hardware compilation workflows
  - Resolves #234

- **Performance**: 3x faster inference on SpiNNaker hardware
  - Optimized memory management
  - Improved spike routing algorithms
  - Better core utilization
  - Benchmarks: [Performance Report](docs/benchmarks/v1.2-performance.md)

### Fixed
- **Memory Leak**: Fixed memory leak in continuous learning mode (#456)
  - Affected long-running training sessions
  - Memory usage now stable over extended periods
  - Performance impact: negligible

### Security
- **CVE-2024-12345**: Updated PyTorch dependency to address tensor deserialization vulnerability
  - Risk: Remote code execution via malicious model files
  - Impact: All users loading untrusted models
  - Action: Update to v1.2.0 immediately
```

## Release Notes Integration

Changelog entries should complement GitHub release notes:

- **Changelog**: Technical details, migration guides, comprehensive changes
- **Release Notes**: User-focused summary, highlights, download links

## Automation Integration

The changelog should integrate with:

1. **Conventional Commits**: Automatic entry generation from commit messages
2. **GitHub Releases**: Automatic release note creation
3. **Documentation**: Cross-references to updated docs
4. **Migration Guides**: Links to breaking change documentation

## Review Process

1. **Draft entries** for unreleased changes in `## [Unreleased]`
2. **Review entries** during PR process
3. **Finalize entries** before release
4. **Cross-reference** with documentation updates
5. **Validate links** and references

## Neuromorphic-Specific Considerations

- **Hardware compatibility** changes
- **Power consumption** improvements
- **Latency and throughput** changes
- **Model accuracy** impacts
- **Training efficiency** modifications
- **Edge deployment** considerations