# SpikeFormer Neuromorphic Kit

A clean, minimal implementation of the **Spiking Transformer** architecture in pure PyTorch — built for neuromorphic hardware deployment and energy-efficient inference.

> **Core idea:** Replace floating-point activations with binary spike trains. On neuromorphic chips (Intel Loihi, IBM TrueNorth), spikes are processed as integer accumulations instead of multiply-accumulates — yielding 10–100× energy reductions at realistic spike rates.

---

## Architecture

```
Input (float)
  │
  ▼
Linear Embed ──► LIF Neuron (binary spikes)
  │
  ▼
┌─────────────────────────────────────┐
│  SpikeFormerBlock (×N)              │
│                                     │
│  x ──► LayerNorm ──► LIF₁ ─────────┤
│                       │ (spikes)    │
│                       ▼             │
│               SpikeAttention        │
│                Q·Kᵀ → LIF → ·V     │
│                       │             │
│                       ▼             │ + residual
│  x ──► LayerNorm ──► LIF₂ ─────────┤
│                       │ (spikes)    │
│                       ▼             │
│                  SpikeMLP           │
│              Linear → LIF → Linear  │
│                       │             │ + residual
└───────────────────────┼─────────────┘
                        ▼
              Mean pool (time + seq)
                        ▼
                  LayerNorm
                        ▼
                    Classifier
```

**Key components:**

| Component | What it does |
|-----------|-------------|
| `LIFNeuron` | Leaky Integrate-and-Fire with learnable decay α, subtract-reset, fast-sigmoid surrogate gradient for backprop |
| `SpikeAttention` | Multi-head attention where Q·Kᵀ scores pass through a LIF layer → binary attention weights. No softmax, no exp(). |
| `SpikeMLP` | Two-layer FFN with a LIF hidden layer — sparsifies internal activations |
| `SpikeRateTracker` | Context manager to measure firing rates per layer (energy efficiency metric) |

---

## Installation

```bash
pip install torch  # requires PyTorch ≥ 2.0
pip install -e .
```

---

## Quick Start

```python
from spikeformer import SpikeFormer, SpikeRateTracker
import torch

model = SpikeFormer(
    input_dim=64,    # input feature dimension
    dim=128,         # internal embedding dimension
    num_heads=4,
    num_layers=4,
    num_classes=10,
    timesteps=16,    # simulation timesteps (temporal resolution)
    tau_mem=20.0,    # LIF membrane time constant (ms)
    threshold=1.0,   # firing threshold
)

x = torch.randn(8, 64)         # (batch=8, input_dim=64)
logits, spike_rates = model(x)  # logits: (8, 10)

# Inspect energy efficiency
print(spike_rates)
# {'embed': 0.042, 'block0.lif1': 0.051, 'block0.attn': 0.003, ...}
```

### Spike Rate Monitoring

```python
with SpikeRateTracker(model) as tracker:
    logits, _ = model(x)

mean_rate = sum(tracker.rates.values()) / len(tracker.rates)
print(f"Mean spike rate: {mean_rate:.2%}")
print(f"Estimated energy reduction vs ANN: {1/mean_rate:.1f}×")
```

---

## Demo Results

Training a 2-layer SpikeFormer (103K params) on a 10-class synthetic dataset:

```
Epoch  Train Loss  Train Acc   Val Loss   Val Acc
    1      1.9300    69.19%     1.4768  100.00%
    5      0.0871   100.00%     0.0627  100.00%
   15      0.0154   100.00%     0.0154  100.00%

Best validation accuracy: 100.00%

Spike rates (fraction of neurons firing per timestep):
  block0.lif1              0.053  ██
  block0.lif2              0.053  ██
  block0.attn              0.000
  block0.ffn_hidden        0.000
  block1.lif1              0.053  ██
  block1.lif2              0.053  ██
  embed                    0.034  █
  mean                     0.027

Estimated energy reduction vs ANN: 36.5× fewer MACs
```

At 2.7% mean spike rate, the model uses ~36× fewer synaptic operations than an equivalent ANN — the defining advantage for neuromorphic deployment.

Run the demo:
```bash
python demo_basic.py
```

---

## Energy Efficiency Framing

**Why spike rates matter:**

- **ANN on GPU:** every neuron fires every forward pass → O(N·D) multiply-accumulates per layer
- **SNN on neuromorphic chip:** only firing neurons send events → O(spike\_rate · N · D) accumulations (no multiply — spike = ±1)
- At 5% spike rate: **20× fewer operations** → proportional energy reduction

**Typical results:**
| Spike Rate | Energy Reduction |
|-----------|-----------------|
| 10% | 10× |
| 5% | 20× |
| 2.7% (this model) | ~37× |
| 1% | 100× |

**Hardware targets:** Intel Loihi 2, IBM TrueNorth, BrainScaleS, SpiNNaker — all benefit directly from lower spike rates.

---

## Encoding Inputs

Three encoders convert continuous inputs to spike trains:

```python
from spikeformer import RateEncoder, LatencyEncoder, DirectEncoder

# Poisson rate coding (default for most tasks)
enc = RateEncoder(timesteps=16)
spikes = enc(x)        # (batch, 16, features) — binary

# Time-to-first-spike (higher value = fires sooner)
enc = LatencyEncoder(timesteps=16)
spikes = enc(x)        # exactly one spike per neuron per sequence

# Direct (repeat input T times, let LIF handle temporal dynamics)
enc = DirectEncoder(timesteps=16)
spikes = enc(x)        # (batch, 16, features) — float repeated
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

38 tests covering neurons, attention, full model end-to-end backprop, spike rate validity, and encoding.

---

## Project Structure

```
spikeformer/
  __init__.py      — public API
  neurons.py       — LIFNeuron, spike_fn, SpikeRateTracker
  models.py        — SpikeFormer, SpikeFormerBlock, SpikeAttention, SpikeMLP
  encoding.py      — RateEncoder, LatencyEncoder, DirectEncoder
tests/
  test_neurons.py
  test_models.py
  test_encoding.py
demo_basic.py      — training demo with spike rate report
```

---

## References

- **SpikeFormer** (Zhou et al., 2022): spike-based self-attention for vision
- **Surrogate Gradient Learning** (Neftci et al., 2019): making SNNs differentiable
- **LIF dynamics**: Gerstner & Kistler, *Spiking Neuron Models* (2002)
- **Neuromorphic efficiency**: Merolla et al., *Science* 2014 (TrueNorth)

---

## License

MIT
