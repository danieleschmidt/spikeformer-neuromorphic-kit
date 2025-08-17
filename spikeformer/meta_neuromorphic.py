"""Meta-Neuromorphic Computing: Next-Generation Adaptive Spiking Systems.

This module implements cutting-edge meta-learning algorithms for neuromorphic systems
that can adapt their architecture and learning rules dynamically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MetaNeuromorphicConfig:
    """Configuration for meta-neuromorphic systems."""
    meta_learning_rate: float = 1e-4
    adaptation_steps: int = 5
    num_meta_layers: int = 3
    meta_hidden_dim: int = 512
    plasticity_types: List[str] = None
    architecture_search_space: List[str] = None
    
    def __post_init__(self):
        if self.plasticity_types is None:
            self.plasticity_types = ["hebbian", "anti_hebbian", "oja", "bcm"]
        if self.architecture_search_space is None:
            self.architecture_search_space = ["attention", "convolution", "recurrent", "skip"]


class DynamicPlasticityRule(nn.Module):
    """Dynamic plasticity rule that adapts based on context."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Meta-network to generate plasticity parameters
        self.meta_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # [hebbian, anti_hebbian, oja, bcm] weights
        )
        
        # Learnable base plasticity rates
        self.base_rates = nn.Parameter(torch.tensor([0.01, 0.01, 0.01, 0.01]))
        
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                weights: torch.Tensor) -> torch.Tensor:
        """Compute dynamic weight updates based on spike patterns."""
        batch_size, seq_len, input_dim = pre_spikes.shape
        
        # Compute spike correlations
        pre_activity = pre_spikes.mean(dim=1)  # (batch, input_dim)
        post_activity = post_spikes.mean(dim=1)  # (batch, output_dim)
        
        # Create context vector
        context = torch.cat([pre_activity.mean(dim=-1, keepdim=True).expand(-1, input_dim),
                           post_activity.mean(dim=-1, keepdim=True).expand(-1, input_dim)], dim=-1)
        
        # Generate dynamic plasticity weights
        plasticity_weights = torch.softmax(self.meta_net(context), dim=-1)  # (batch, 4)
        
        # Compute different plasticity rules
        hebbian = torch.einsum('bti,bto->bio', pre_spikes, post_spikes)
        anti_hebbian = -hebbian
        
        # Oja's rule (normalized Hebbian)
        post_norm = post_spikes.norm(dim=1, keepdim=True) + 1e-6
        oja = hebbian / post_norm
        
        # BCM rule (bidirectional plasticity)
        post_squared = (post_spikes ** 2).mean(dim=1, keepdim=True)
        bcm = hebbian * (post_spikes.mean(dim=1, keepdim=True) - post_squared)
        
        # Combine plasticity rules
        plasticity_rules = torch.stack([hebbian, anti_hebbian, oja, bcm], dim=-1)  # (batch, input, output, 4)
        
        # Apply dynamic weights
        weight_updates = torch.einsum('bior,br->bio', plasticity_rules, 
                                    plasticity_weights * self.base_rates)
        
        return weight_updates


class NeuralArchitectureEvolution(nn.Module):
    """Neural architecture that evolves its structure during training."""
    
    def __init__(self, config: MetaNeuromorphicConfig):
        super().__init__()
        self.config = config
        
        # Architecture genes (learnable)
        self.architecture_genes = nn.Parameter(torch.randn(len(config.architecture_search_space)))
        
        # Module bank
        self.module_bank = nn.ModuleDict({
            "attention": self._create_attention_module(),
            "convolution": self._create_conv_module(),
            "recurrent": self._create_rnn_module(),
            "skip": nn.Identity()
        })
        
        # Evolution controller
        self.evolution_controller = nn.Sequential(
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.meta_hidden_dim, len(config.architecture_search_space))
        )
        
    def _create_attention_module(self) -> nn.Module:
        """Create a spiking attention module."""
        return SpikingMultiHeadAttention(
            embed_dim=self.config.meta_hidden_dim,
            num_heads=8,
            dropout=0.1
        )
    
    def _create_conv_module(self) -> nn.Module:
        """Create a spiking convolution module."""
        return SpikingConv1d(
            in_channels=self.config.meta_hidden_dim,
            out_channels=self.config.meta_hidden_dim,
            kernel_size=3,
            padding=1
        )
    
    def _create_rnn_module(self) -> nn.Module:
        """Create a spiking RNN module."""
        return SpikingLSTM(
            input_size=self.config.meta_hidden_dim,
            hidden_size=self.config.meta_hidden_dim
        )
    
    def forward(self, x: torch.Tensor, performance_feedback: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with dynamic architecture selection."""
        # Evolve architecture based on performance
        if performance_feedback is not None:
            evolution_signal = self.evolution_controller(performance_feedback)
            self.architecture_genes.data += 0.01 * evolution_signal
        
        # Softmax over architecture choices
        arch_probs = torch.softmax(self.architecture_genes, dim=0)
        
        # Apply modules with learned weights
        output = torch.zeros_like(x)
        for i, (name, module) in enumerate(self.module_bank.items()):
            if name in self.config.architecture_search_space:
                idx = self.config.architecture_search_space.index(name)
                weight = arch_probs[idx]
                
                if name == "attention":
                    module_output = module(x, x, x)[0]  # Take values only
                else:
                    module_output = module(x)
                
                output += weight * module_output
        
        return output


class MetaNeuromorphicLearner(nn.Module):
    """Meta-learning system for neuromorphic networks."""
    
    def __init__(self, config: MetaNeuromorphicConfig):
        super().__init__()
        self.config = config
        
        # Meta-learner network
        self.meta_learner = nn.Sequential(
            nn.Linear(config.meta_hidden_dim * 2, config.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.meta_hidden_dim, config.meta_hidden_dim)
        )
        
        # Dynamic plasticity system
        self.plasticity_rule = DynamicPlasticityRule(config.meta_hidden_dim)
        
        # Evolving architecture
        self.evolving_arch = NeuralArchitectureEvolution(config)
        
        # Memory for meta-learning
        self.task_memory = {}
        self.performance_history = []
        
    def meta_forward(self, x: torch.Tensor, task_id: str, 
                    adaptation_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Meta-learning forward pass."""
        batch_size = x.size(0)
        
        # Retrieve or initialize task-specific parameters
        if task_id not in self.task_memory:
            self.task_memory[task_id] = torch.zeros(batch_size, self.config.meta_hidden_dim)
        
        task_memory = self.task_memory[task_id]
        
        # Combine input with task memory
        meta_input = torch.cat([x, task_memory], dim=-1)
        
        # Meta-learning update
        meta_output = self.meta_learner(meta_input)
        
        # Update task memory
        self.task_memory[task_id] = meta_output.detach()
        
        # Apply evolving architecture
        performance_feedback = self._compute_performance_feedback()
        output = self.evolving_arch(meta_output, performance_feedback)
        
        return output
    
    def _compute_performance_feedback(self) -> torch.Tensor:
        """Compute performance feedback for architecture evolution."""
        if len(self.performance_history) < 2:
            return torch.zeros(self.config.meta_hidden_dim)
        
        # Simple performance trend
        recent_performance = np.mean(self.performance_history[-5:])
        older_performance = np.mean(self.performance_history[-10:-5])
        trend = recent_performance - older_performance
        
        # Convert to feedback signal
        feedback = torch.randn(self.config.meta_hidden_dim) * trend
        return feedback
    
    def update_performance(self, performance: float):
        """Update performance history for meta-learning."""
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)


class SpikingMultiHeadAttention(nn.Module):
    """Spiking version of multi-head attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.spike_fn = SpikeFunction.apply
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spiking multi-head attention forward pass."""
        batch_size, seq_len, embed_dim = query.shape
        
        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Spiking attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.spike_fn(attn_weights)  # Convert to spikes
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class SpikingConv1d(nn.Module):
    """Spiking 1D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 padding: int = 0, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, stride=stride)
        self.spike_fn = SpikeFunction.apply
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking convolution."""
        # Transpose for conv1d (batch, channels, seq_len)
        x = x.transpose(1, 2)
        conv_out = self.conv(x)
        spikes = self.spike_fn(conv_out)
        # Transpose back
        return spikes.transpose(1, 2)


class SpikingLSTM(nn.Module):
    """Spiking LSTM implementation."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.spike_fn = SpikeFunction.apply
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking LSTM."""
        batch_size, seq_len, input_size = x.shape
        
        # Initialize hidden states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            # Concatenate input and hidden state
            combined = torch.cat([xt, h], dim=1)
            
            # Compute gates
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            g = torch.tanh(self.cell_gate(combined))
            
            # Update cell state
            c = f * c + i * g
            
            # Update hidden state with spikes
            h = o * torch.tanh(c)
            h_spikes = self.spike_fn(h)
            
            outputs.append(h_spikes)
        
        return torch.stack(outputs, dim=1)


class SpikeFunction(torch.autograd.Function):
    """Surrogate gradient function for spike generation."""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Fast sigmoid surrogate gradient
        grad_input = grad_input * torch.exp(-torch.abs(input)) / (1 + torch.exp(-torch.abs(input)))**2
        return grad_input


class MetaNeuromorphicFramework:
    """Complete framework for meta-neuromorphic computing."""
    
    def __init__(self, config: MetaNeuromorphicConfig):
        self.config = config
        self.learner = MetaNeuromorphicLearner(config)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), 
                                        lr=config.meta_learning_rate)
        
        # Performance tracking
        self.task_performances = {}
        self.adaptation_history = []
        
    def meta_train(self, support_data: torch.Tensor, query_data: torch.Tensor,
                  task_id: str, num_steps: int = None) -> Dict[str, float]:
        """Meta-training on a task."""
        if num_steps is None:
            num_steps = self.config.adaptation_steps
        
        self.learner.train()
        
        # Support phase (adaptation)
        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # Forward pass on support data
            support_output = self.learner.meta_forward(support_data, task_id)
            
            # Compute adaptation loss (task-specific)
            adaptation_loss = self._compute_adaptation_loss(support_output, support_data)
            
            # Backward pass
            adaptation_loss.backward()
            self.optimizer.step()
        
        # Query phase (evaluation)
        with torch.no_grad():
            query_output = self.learner.meta_forward(query_data, task_id)
            query_loss = self._compute_adaptation_loss(query_output, query_data)
            
            # Update performance tracking
            performance = 1.0 / (1.0 + query_loss.item())
            self.task_performances[task_id] = performance
            self.learner.update_performance(performance)
        
        return {
            "adaptation_loss": adaptation_loss.item(),
            "query_loss": query_loss.item(),
            "performance": performance
        }
    
    def _compute_adaptation_loss(self, output: torch.Tensor, 
                               target: torch.Tensor) -> torch.Tensor:
        """Compute adaptation loss for meta-learning."""
        # Reconstruction loss
        recon_loss = F.mse_loss(output, target)
        
        # Sparsity loss (encourage spiking)
        sparsity_loss = torch.mean(torch.abs(output))
        
        # Plasticity regularization
        plasticity_reg = 0.0
        for name, param in self.learner.named_parameters():
            if "plasticity" in name:
                plasticity_reg += torch.norm(param, p=2)
        
        total_loss = recon_loss + 0.01 * sparsity_loss + 0.001 * plasticity_reg
        return total_loss
    
    def adapt_to_task(self, task_data: torch.Tensor, task_id: str) -> torch.Tensor:
        """Rapidly adapt to a new task."""
        self.learner.eval()
        
        with torch.no_grad():
            # Few-shot adaptation
            adapted_output = self.learner.meta_forward(task_data, task_id)
            
        return adapted_output
    
    def get_architecture_genes(self) -> torch.Tensor:
        """Get current architecture genes."""
        return self.learner.evolving_arch.architecture_genes.detach()
    
    def get_task_memory(self, task_id: str) -> Optional[torch.Tensor]:
        """Get task-specific memory."""
        return self.learner.task_memory.get(task_id)


# Factory function
def create_meta_neuromorphic_system(config: Optional[MetaNeuromorphicConfig] = None) -> MetaNeuromorphicFramework:
    """Create a complete meta-neuromorphic computing system."""
    if config is None:
        config = MetaNeuromorphicConfig()
    
    logger.info(f"Creating meta-neuromorphic system with config: {config}")
    
    framework = MetaNeuromorphicFramework(config)
    
    logger.info("Meta-neuromorphic system created successfully")
    logger.info(f"Architecture search space: {config.architecture_search_space}")
    logger.info(f"Plasticity types: {config.plasticity_types}")
    
    return framework