# Getting Started with Spikeformer Neuromorphic Kit

## Overview

This guide will help you get started with the Spikeformer Neuromorphic Kit, a complete toolkit for training and deploying spiking transformer networks on neuromorphic hardware.

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Basic understanding of neural networks and transformers
- (Optional) Access to neuromorphic hardware (Loihi 2, SpiNNaker2)

## Installation

### Quick Start (PyPI)

```bash
pip install spikeformer-neuromorphic-kit
```

### Development Installation

```bash
git clone https://github.com/your-org/spikeformer-neuromorphic-kit
cd spikeformer-neuromorphic-kit
pip install -e ".[dev]"
```

### With Hardware Support

```bash
# For Intel Loihi 2
pip install -e ".[loihi2]"

# For SpiNNaker2
pip install -e ".[spinnaker]"

# All hardware backends
pip install -e ".[all]"
```

## First Example: Convert a Vision Transformer

```python
from spikeformer import SpikeformerConverter, EnergyProfiler
from transformers import ViTModel
import torch

# Load a pre-trained Vision Transformer
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Create converter with default settings
converter = SpikeformerConverter(
    timesteps=32,
    threshold=1.0,
    spike_encoding="rate"
)

# Convert to spiking neural network
spiking_vit = converter.convert(vit)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)

# Profile energy consumption
profiler = EnergyProfiler()
with profiler.measure():
    output = spiking_vit(sample_input)

print(f"Energy consumption: {profiler.energy_mJ:.2f} mJ")
print(f"GPU baseline reduction: {profiler.gpu_baseline_ratio:.1f}x")
```

## Key Concepts

### Spiking Neural Networks
- **Spikes**: Binary events that carry information through timing
- **Temporal Dynamics**: Information processing over time steps
- **Energy Efficiency**: Sparse computation reduces power consumption

### Conversion Process
1. **Calibration**: Analyze activation patterns in original model
2. **Threshold Setting**: Determine optimal firing thresholds
3. **Layer Mapping**: Convert each layer to spiking equivalent
4. **Fine-tuning**: Adjust parameters for accuracy retention

### Hardware Deployment
- **Loihi 2**: Intel's neuromorphic research chip
- **SpiNNaker2**: Massively parallel neuromorphic system
- **Edge Devices**: Commercial neuromorphic accelerators

## Next Steps

1. **[Training Guide](training.md)**: Learn to train spiking transformers from scratch
2. **[Hardware Guide](../hardware/loihi2-quickstart.md)**: Deploy to neuromorphic hardware
3. **[API Reference](../../api/index.md)**: Detailed API documentation
4. **[Examples](../../examples/)**: Complete example projects

## Troubleshooting

### Common Issues

**Q: Conversion accuracy is low**
A: Try increasing timesteps or adjusting threshold values. Use calibration data that matches your target domain.

**Q: Hardware deployment fails**
A: Ensure hardware drivers are installed and accessible. Check hardware-specific documentation.

**Q: High memory usage during conversion**
A: Use batch processing for large models or reduce calibration dataset size.

### Getting Help

- 📖 Check the [FAQ](faq.md)
- 💬 Join our [Discord community](https://discord.gg/your-org)
- 🐛 Report bugs on [GitHub Issues](https://github.com/your-org/spikeformer-neuromorphic-kit/issues)
- 📧 Email: neuromorphic@your-org.com