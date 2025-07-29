---
name: Hardware Issue
about: Report hardware-specific problems or compatibility issues
title: '[HARDWARE] '
labels: hardware
assignees: ''
---

## Hardware Platform
<!-- Select the affected hardware platform -->
- [ ] Intel Loihi 2
- [ ] SpiNNaker2
- [ ] NVIDIA GPU
- [ ] Edge devices (specify below)
- [ ] CPU-only simulation
- [ ] Multiple platforms

**Specific Hardware Details:**
- **Model/Version**: 
- **Driver Version**: 
- **SDK Version**: 
- **Board Configuration**: 

## Problem Description
<!-- Clearly describe the hardware-related issue -->

## Steps to Reproduce
1. 
2. 
3. 

## Expected Behavior
<!-- What should happen on this hardware? -->

## Actual Behavior
<!-- What actually happens? -->

## Error Messages
<!-- Include full error messages, stack traces, and logs -->
```
[Paste error messages here]
```

## Environment Information
### System Environment
- **OS**: 
- **Python Version**: 
- **SpikeFormer Version**: 
- **PyTorch Version**: 

### Hardware Environment
- **Available Memory**: 
- **Hardware Cores/Chips**: 
- **Power Constraints**: 
- **Temperature**: 

### Software Dependencies
```bash
# Output of: pip list | grep -E "(nxsdk|spynnaker|torch|spikeformer)"
[Paste relevant package versions]
```

## Hardware Configuration
<!-- Include relevant hardware configuration files -->
```yaml
# Hardware config (if applicable)
[Paste configuration here]
```

## Code Example
<!-- Minimal code that reproduces the issue -->
```python
# Minimal reproducible example
import spikeformer

# Your code here that demonstrates the issue
```

## Hardware Logs
<!-- Include hardware-specific logs -->
<details>
<summary>Hardware Logs</summary>

```
[Paste logs here - may be large]
```
</details>

## Performance Metrics
<!-- If performance-related issue -->
- **Expected Performance**: 
- **Actual Performance**: 
- **Energy Consumption**: 
- **Latency**: 

## Compatibility Testing
<!-- What compatibility testing was done? -->
- [ ] Tested on CPU simulation (works/doesn't work)
- [ ] Tested on different hardware version
- [ ] Tested with different SDK version
- [ ] Tested with different model sizes
- [ ] Tested with different configurations

## Hardware Expertise Required
<!-- What level of hardware expertise is needed to debug this? -->
- [ ] Basic hardware knowledge
- [ ] Advanced hardware knowledge
- [ ] Vendor-specific expertise
- [ ] Hardware access required

## Impact Assessment
<!-- How does this affect users? -->
- [ ] Completely blocks hardware usage
- [ ] Severely degrades performance
- [ ] Minor performance impact
- [ ] Cosmetic/logging issue
- [ ] Feature not available

## Workaround
<!-- Is there a temporary workaround? -->
- [ ] No workaround available
- [ ] Workaround available (describe below)
- [ ] Alternative hardware can be used

**Workaround Details:**

## Additional Context
<!-- Any other context about the hardware issue -->

## Related Hardware Issues
<!-- Link to related hardware issues -->
Related to: #
Duplicate of: #

---

## For Hardware Experts

### Hardware Analysis Checklist
- [ ] Hardware configuration validated
- [ ] Driver/SDK compatibility checked  
- [ ] Resource constraints identified
- [ ] Known hardware limitations considered
- [ ] Alternative hardware platforms tested

### Debug Information Needed
- [ ] Hardware diagnostic logs
- [ ] Resource utilization metrics
- [ ] Detailed error traces
- [ ] Hardware configuration dumps
- [ ] Performance profiling data

### Hardware Team Assignment
- [ ] @loihi2-experts (for Loihi 2 issues)
- [ ] @spinnaker-experts (for SpiNNaker issues)  
- [ ] @gpu-experts (for GPU issues)
- [ ] @edge-experts (for edge device issues)
- [ ] @hardware-team (general hardware issues)

### Priority for Hardware Team
- [ ] Critical (hardware completely unusable)
- [ ] High (major performance degradation) 
- [ ] Medium (minor issues, workaround available)
- [ ] Low (enhancement or optimization)

**Thank you for reporting this hardware issue! ⚡**

Hardware compatibility is crucial for neuromorphic computing. Our hardware experts will investigate and provide guidance.