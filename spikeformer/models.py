"""
SpikeFormer: Spiking Transformer architecture.

Architecture overview:
  Input → PatchEmbed/Linear → [SpikeFormerBlock × N] → Classifier

SpikeFormerBlock:
  x → LIF → SpikeAttention → LIF → SpikeMLP → x'  (residual added at float level)

Key property: activations *inside* the block are binary spike trains.
This makes them compatible with neuromorphic hardware (accumulate-and-fire
instead of multiply-accumulate — ~10–100× energy reduction).

Spike rates are tracked per-layer; lower is more energy-efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .neurons import LIFNeuron


# ---------------------------------------------------------------------------
# Spiking self-attention
# ---------------------------------------------------------------------------

class SpikeAttention(nn.Module):
    """
    Spike-based multi-head self-attention.

    Replaces softmax attention with spike-train accumulation:
      - Q, K, V projections receive binary spikes → produce float activations
      - Attention scores: S_q @ S_k^T  (inner product of spike trains)
      - Scale by 1/sqrt(d_head) then apply LIF to get spike attention weights
      - Output: spike_weights @ V

    This is an integer accumulation at the hardware level (no exp, no division
    in the spike domain) — key to neuromorphic efficiency.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        tau_mem: float = 20.0,
        threshold: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # LIF in attention pathway — produces spike attention "scores"
        self.attn_lif = LIFNeuron(tau_mem=tau_mem, threshold=threshold)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, seq_len, dim)  — spike trains from prev LIF

        Returns:
            out: (batch, time, seq_len, dim)
            attn_rate: mean spike rate in attention LIF
        """
        B, T, N, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Project — x is sparse (binary), so this is efficient accumulation
        q = self.q_proj(x).reshape(B, T, N, H, Dh)  # (B,T,N,H,Dh)
        k = self.k_proj(x).reshape(B, T, N, H, Dh)
        v = self.v_proj(x).reshape(B, T, N, H, Dh)

        # Attention scores across sequence dim: (B,T,H,N,N)
        q = q.permute(0, 1, 3, 2, 4)  # (B,T,H,N,Dh)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,T,H,N,N)

        # Reshape for LIF: treat (B,H,N,N) as the feature dims, T is time
        # attn_scores: (B,T,H,N,N) → reshape to (B*H*N, T, N) to run LIF over time
        BHN = B * H * N
        attn_t = attn_scores.permute(0, 2, 3, 1, 4).reshape(BHN, T, N)
        spike_attn, attn_rate = self.attn_lif(attn_t)            # (BHN, T, N)
        spike_attn = spike_attn.reshape(B, H, N, T, N).permute(0, 3, 1, 2, 4)  # (B,T,H,N,N)

        # Aggregate values with spike attention weights
        out = torch.matmul(spike_attn, v)  # (B,T,H,N,Dh)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N, D)
        out = self.out_proj(out)

        return out, attn_rate


# ---------------------------------------------------------------------------
# Spike FFN (Feed-Forward Network)
# ---------------------------------------------------------------------------

class SpikeMLP(nn.Module):
    """
    Two-layer feed-forward network in spike domain.
    Linear → LIF (spikes) → Linear.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, tau_mem: float = 20.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.lif = LIFNeuron(tau_mem=tau_mem)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, time, seq_len, dim)

        Returns:
            out: (batch, time, seq_len, dim)
            rate: spike rate in hidden LIF
        """
        h = self.fc1(x)           # float
        h, rate = self.lif(h)     # sparse spikes
        out = self.fc2(h)         # accumulate back to float
        return out, rate


# ---------------------------------------------------------------------------
# SpikeFormer Block
# ---------------------------------------------------------------------------

class SpikeFormerBlock(nn.Module):
    """
    Single SpikeFormer block:

        x_float → LIF₁ (encode to spikes)
                → SpikeAttention
                + residual(x_float)
                → LIF₂ (encode to spikes)
                → SpikeMLP
                + residual

    LayerNorm operates at float level (pre-norm), on x before spiking.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lif1 = LIFNeuron(tau_mem=tau_mem, threshold=threshold)
        self.attn = SpikeAttention(dim, num_heads, tau_mem=tau_mem, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.lif2 = LIFNeuron(tau_mem=tau_mem, threshold=threshold)
        self.ffn = SpikeMLP(dim, mlp_ratio, tau_mem=tau_mem)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, time, seq_len, dim)

        Returns:
            x: (batch, time, seq_len, dim)
            rates: dict of spike rates for this block
        """
        # --- Attention branch ---
        h = self.norm1(x)
        spikes1, r_lif1 = self.lif1(h)
        attn_out, r_attn = self.attn(spikes1)
        x = x + attn_out

        # --- FFN branch ---
        h = self.norm2(x)
        spikes2, r_lif2 = self.lif2(h)
        ffn_out, r_ffn = self.ffn(spikes2)
        x = x + ffn_out

        rates = {
            "lif1": r_lif1,
            "attn": r_attn,
            "lif2": r_lif2,
            "ffn_hidden": r_ffn,
        }
        return x, rates


# ---------------------------------------------------------------------------
# Full SpikeFormer model
# ---------------------------------------------------------------------------

class SpikeFormer(nn.Module):
    """
    SpikeFormer: Spiking Transformer for classification.

    Inputs are projected to `dim`-dimensional space, then processed by
    N SpikeFormerBlocks.  At each block, activations are converted to
    binary spike trains — enabling neuromorphic hardware mapping.

    Args:
        input_dim:  feature dimension of each input token
        dim:        internal embedding dimension
        num_heads:  number of attention heads
        num_layers: number of SpikeFormerBlocks
        num_classes: output classes
        seq_len:    number of input tokens (sequence length)
        timesteps:  number of simulation timesteps (controls temporal coding)
        mlp_ratio:  FFN hidden-dim ratio
        tau_mem:    LIF membrane time constant (ms)
        threshold:  LIF firing threshold
        dropout:    dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        seq_len: int = 1,
        timesteps: int = 16,
        mlp_ratio: float = 4.0,
        tau_mem: float = 20.0,
        threshold: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.timesteps = timesteps

        # Patch / token embedding
        self.embed = nn.Linear(input_dim, dim, bias=False)
        self.embed_lif = LIFNeuron(tau_mem=tau_mem, threshold=threshold)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikeFormerBlock(dim, num_heads, mlp_ratio, tau_mem, threshold, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: float tensor (batch, seq_len, input_dim)
               or (batch, input_dim) — will be unsqueezed to seq_len=1

        Returns:
            logits: (batch, num_classes)
            spike_rates: dict mapping layer-name → mean spike rate
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        B, N, D = x.shape
        T = self.timesteps

        # Project and expand over time axis: (batch, time, seq, dim)
        x_proj = self.embed(x)                         # (B, N, dim)
        x_time = x_proj.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, N, dim)

        # Initial embedding LIF
        x_time, r_embed = self.embed_lif(x_time.reshape(B * N, T, -1))
        x_time = x_time.reshape(B, T, N, -1)

        spike_rates = {"embed": r_embed.item() if torch.is_tensor(r_embed) else r_embed}

        # SpikeFormer blocks
        for i, block in enumerate(self.blocks):
            x_time, block_rates = block(x_time)
            for k, v in block_rates.items():
                key = f"block{i}.{k}"
                spike_rates[key] = v.item() if torch.is_tensor(v) else v

        # Pool over time and sequence → classify
        # Mean over time and sequence dims
        out = x_time.mean(dim=(1, 2))          # (B, dim)
        out = self.norm(out)
        logits = self.head(out)                # (B, num_classes)

        return logits, spike_rates
