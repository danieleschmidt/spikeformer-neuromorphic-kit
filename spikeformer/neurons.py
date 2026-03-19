"""
Spiking neuron models for neuromorphic computing.

Core building blocks:
- SpikingFunction: differentiable spike emission with surrogate gradients
- LIFNeuron: Leaky Integrate-and-Fire with membrane potential dynamics
- SpikeRateTracker: measures what % of neurons fire (energy efficiency metric)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Surrogate gradient
# ---------------------------------------------------------------------------

class _SpikeFunction(torch.autograd.Function):
    """
    Forward:  spike = 1 if membrane >= threshold else 0  (hard threshold)
    Backward: approximate with fast-sigmoid surrogate gradient so gradients
              can flow through the discontinuity.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, beta: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.beta = beta
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        membrane, = ctx.saved_tensors
        # Fast-sigmoid surrogate: d/dx [sigmoid(beta*(x-theta))] ≈ beta/(2*(1+|beta*(x-theta)|))^2
        # Simplified: 1/(1 + beta*|x - theta|)^2  (well-conditioned, widely used)
        x = membrane - ctx.threshold
        surrogate = 1.0 / (1.0 + ctx.beta * x.abs()) ** 2
        return grad_output * surrogate, None, None


def spike_fn(membrane: torch.Tensor, threshold: float = 1.0, beta: float = 4.0) -> torch.Tensor:
    """Differentiable spike emission."""
    return _SpikeFunction.apply(membrane, threshold, beta)


# ---------------------------------------------------------------------------
# LIF Neuron layer
# ---------------------------------------------------------------------------

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer.

    Dynamics (per time-step t):
        u[t] = decay * u[t-1] + x[t]   # leaky integration
        s[t] = spike_fn(u[t] - theta)  # fire if above threshold
        u[t] = u[t] - s[t] * theta     # subtract-reset

    Args:
        tau_mem:   membrane time constant in ms (controls decay α = exp(-1/τ))
        threshold: firing threshold θ
        beta:      surrogate gradient sharpness
        track_rates: if True, accumulate spike rate stats
    """

    def __init__(
        self,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        beta: float = 4.0,
        track_rates: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.track_rates = track_rates

        # decay factor — learnable so the network can tune time constants
        log_alpha = torch.log(torch.tensor(torch.exp(torch.tensor(-1.0 / tau_mem)).item()))
        self.log_alpha = nn.Parameter(log_alpha)

        # running stats (not parameters — reset manually)
        self.register_buffer("_spike_count", torch.tensor(0.0))
        self.register_buffer("_total_count", torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        """Membrane decay factor (clamped to (0,1) via sigmoid)."""
        return torch.sigmoid(self.log_alpha)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor of shape (batch, time, *features)

        Returns:
            spikes:   binary tensor (batch, time, *features)
            rates:    mean spike rate scalar for this forward pass
        """
        T = x.shape[1]
        alpha = self.alpha
        u = torch.zeros_like(x[:, 0])  # membrane potential, shape (batch, *features)
        spike_list = []

        for t in range(T):
            u = alpha * u + x[:, t]
            s = spike_fn(u, self.threshold, self.beta)
            u = u - s * self.threshold  # subtract-reset
            spike_list.append(s)

        spikes = torch.stack(spike_list, dim=1)  # (batch, time, *features)

        rate = spikes.mean()
        if self.track_rates:
            self._spike_count += spikes.sum()
            self._total_count += spikes.numel()

        return spikes, rate

    def reset_stats(self):
        self._spike_count.zero_()
        self._total_count.zero_()

    @property
    def spike_rate(self) -> float:
        """Overall spike rate since last reset_stats()."""
        if self._total_count == 0:
            return 0.0
        return (self._spike_count / self._total_count).item()


# ---------------------------------------------------------------------------
# Spike rate tracker (module-level)
# ---------------------------------------------------------------------------

class SpikeRateTracker:
    """
    Lightweight context manager that collects spike rates from all LIFNeuron
    modules inside a model during a forward pass.

    Usage::

        with SpikeRateTracker(model) as tracker:
            output = model(x)
        print(tracker.rates)   # {'encoder.lif': 0.12, ...}
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.rates: dict = {}
        self._handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LIFNeuron):
                module.reset_stats()
        return self

    def __exit__(self, *args):
        for name, module in self.model.named_modules():
            if isinstance(module, LIFNeuron):
                self.rates[name] = module.spike_rate
