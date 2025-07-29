# Pull Request

## Summary
<!-- Provide a clear and concise description of the changes -->


## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Hardware integration
- [ ] Security enhancement
- [ ] CI/CD improvement

## Related Issues
<!-- Link to related issues using #issue_number -->
Fixes #
Closes #
Related to #

## Changes Made
<!-- Detailed list of changes -->
- 
- 
- 

## Hardware Impact
<!-- If applicable, describe hardware-specific changes -->
- [ ] Loihi 2 compatibility verified
- [ ] SpiNNaker compatibility verified  
- [ ] Edge deployment tested
- [ ] GPU acceleration tested
- [ ] No hardware impact

## Testing Performed
<!-- Describe the testing you've performed -->
- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Hardware tests completed (specify hardware)
- [ ] Performance benchmarks run
- [ ] Manual testing completed
- [ ] Security scanning passed

### Test Details
<!-- Provide specific test results or commands used -->
```bash
# Example test commands run
pytest tests/unit/test_my_feature.py -v
pytest tests/integration/test_my_integration.py --hardware loihi2
```

## Performance Impact
<!-- If applicable, describe performance implications -->
- [ ] Performance improved
- [ ] Performance maintained  
- [ ] Performance degraded (justify below)
- [ ] Not applicable

**Benchmark Results:**
<!-- Include before/after performance metrics if relevant -->

## Security Considerations
<!-- Address any security implications -->
- [ ] No security impact
- [ ] Security review required
- [ ] Vulnerability addressed
- [ ] New security features added

**Security Analysis:**
<!-- Describe security implications and mitigations -->

## Breaking Changes
<!-- If applicable, describe breaking changes and migration path -->
- [ ] No breaking changes
- [ ] Breaking changes documented below

**Migration Guide:**
<!-- Provide clear migration instructions for breaking changes -->

## Documentation Updates
<!-- Check all that apply -->
- [ ] README updated
- [ ] API documentation updated
- [ ] Architecture documentation updated
- [ ] Runbook updated
- [ ] Examples updated
- [ ] No documentation changes needed

## Deployment Notes
<!-- Any special deployment considerations -->
- [ ] Standard deployment
- [ ] Requires configuration changes
- [ ] Requires database migration
- [ ] Requires environment updates

**Special Instructions:**
<!-- Provide any special deployment instructions -->

## Screenshots/Demos
<!-- If applicable, add screenshots or demo links -->

## Code Quality Checklist
<!-- Ensure code quality standards are met -->
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented appropriately
- [ ] Tests added/updated for changes
- [ ] All CI checks pass
- [ ] Performance impact assessed
- [ ] Security implications considered

## Hardware Testing Checklist
<!-- For hardware-related changes -->
- [ ] Tested on Loihi 2
- [ ] Tested on SpiNNaker
- [ ] Tested on GPU
- [ ] Tested on CPU
- [ ] Edge deployment verified
- [ ] Energy profiling completed
- [ ] Hardware compatibility documented

## Pre-merge Checklist
<!-- Final checks before merge -->
- [ ] All conversations resolved
- [ ] Required approvals obtained
- [ ] CI/CD pipeline passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Ready for merge

## Additional Notes
<!-- Any additional context or notes for reviewers -->

---

## For Reviewers

### Review Focus Areas
<!-- Highlight specific areas that need attention -->
- [ ] Algorithm correctness
- [ ] Hardware compatibility
- [ ] Performance optimization
- [ ] Security implications
- [ ] Code architecture
- [ ] Test coverage
- [ ] Documentation clarity

### Hardware Review Required
<!-- Check if hardware-specific review is needed -->
- [ ] Loihi 2 expert review needed
- [ ] SpiNNaker expert review needed
- [ ] Edge computing review needed
- [ ] GPU optimization review needed

### Review Checklist for Reviewers
- [ ] Code logic and implementation reviewed
- [ ] Test coverage adequate
- [ ] Documentation accurate and complete
- [ ] Security implications assessed
- [ ] Performance impact evaluated
- [ ] Hardware compatibility verified
- [ ] Breaking changes properly documented

---

**By submitting this pull request, I confirm that:**
- [ ] I have read and followed the [Contributing Guidelines](../CONTRIBUTING.md)
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my changes
- [ ] I have tested my changes thoroughly
- [ ] My changes do not introduce security vulnerabilities

<!-- 
Thank you for contributing to SpikeFormer! 🚀

For questions about this PR template or contribution process, 
please refer to our Contributing Guidelines or reach out to the maintainers.
-->