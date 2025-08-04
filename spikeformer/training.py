"""Advanced training frameworks for spiking neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from collections import defaultdict, deque
import matplotlib.pyplot as plt

from .optimization import (
    OptimizationConfig, SpikeRegularizer, AdaptiveThresholdOptimizer,
    STDPOptimizer, NeuromorphicScheduler, GradientOptimizer,
    create_optimizer, create_scheduler
)
from .profiling import EnergyProfiler
from .models import SpikingTransformer, SpikingViT
from .neurons import SpikingNeuron


@dataclass
class TrainingConfig:
    """Configuration for spiking neural network training."""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Spiking-specific parameters
    surrogate_gradient: str = "fast_sigmoid"
    spike_regularization_weight: float = 0.01
    temporal_loss_weight: float = 0.1
    energy_loss_weight: float = 0.05
    
    # Training strategies
    use_knowledge_distillation: bool = False
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    
    use_progressive_training: bool = False
    progressive_timesteps: List[int] = field(default_factory=lambda: [8, 16, 32])
    
    use_adaptive_thresholds: bool = True
    use_stdp: bool = False
    use_energy_aware_training: bool = False
    
    # Regularization
    weight_decay: float = 1e-5
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Validation and early stopping
    validation_freq: int = 1
    early_stopping_patience: int = 15
    save_best_model: bool = True
    
    # Hardware-aware training
    hardware_constraints: Optional[Dict[str, Any]] = None
    target_energy_budget: Optional[float] = None  # in mJ
    target_latency_budget: Optional[float] = None  # in ms


@dataclass
class TrainingState:
    """Training state tracking."""
    epoch: int = 0
    step: int = 0
    best_val_acc: float = 0.0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    early_stopped: bool = False
    total_training_time: float = 0.0


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for ANN-to-SNN training."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Standard cross-entropy loss
        ce_loss = loss_fn(student_logits, targets)
        
        # Distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * distill_loss
        
        return total_loss, ce_loss, distill_loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for spiking networks."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, spike_outputs: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss."""
        if spike_outputs.dim() < 3:  # Need time dimension
            return torch.tensor(0.0, device=spike_outputs.device)
        
        # Calculate temporal variance
        temporal_mean = spike_outputs.mean(dim=1, keepdim=True)
        temporal_var = ((spike_outputs - temporal_mean) ** 2).mean()
        
        return self.weight * temporal_var


class EnergyAwareLoss(nn.Module):
    """Energy-aware loss function for neuromorphic optimization."""
    
    def __init__(self, energy_weight: float = 0.05, target_energy: float = 1.0):
        super().__init__()
        self.energy_weight = energy_weight
        self.target_energy = target_energy  # Target energy in mJ
        
    def forward(self, spike_outputs: torch.Tensor, 
                estimated_energy: Optional[float] = None) -> torch.Tensor:
        """Compute energy-aware loss."""
        # Spike rate penalty (more spikes = more energy)
        spike_rate = spike_outputs.mean()
        energy_penalty = self.energy_weight * (spike_rate ** 2)
        
        # If we have actual energy measurements, use them
        if estimated_energy is not None:
            energy_deviation = abs(estimated_energy - self.target_energy) / self.target_energy
            energy_penalty += self.energy_weight * energy_deviation
        
        return energy_penalty


class SpikingTrainer:
    """Advanced trainer for spiking neural networks."""
    
    def __init__(self, model: nn.Module, config: TrainingConfig, 
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training state
        self.state = TrainingState()
        self.logger = logging.getLogger(__name__)
        
        # Optimization components
        self.opt_config = OptimizationConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            spike_regularization=config.spike_regularization_weight
        )
        
        self.optimizer = create_optimizer(model, "spiking_adam", self.opt_config)
        self.scheduler = create_scheduler(self.optimizer, "neuromorphic", self.opt_config)
        
        # Regularization and optimization components
        self.spike_regularizer = SpikeRegularizer(self.opt_config)
        self.gradient_optimizer = GradientOptimizer(model, self.opt_config)
        
        if config.use_adaptive_thresholds:
            self.threshold_optimizer = AdaptiveThresholdOptimizer(model, self.opt_config)
        else:
            self.threshold_optimizer = None
            
        if config.use_stdp:
            self.stdp_optimizer = STDPOptimizer(model, self.opt_config)
        else:
            self.stdp_optimizer = None
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        
        if config.use_knowledge_distillation:
            self.distillation_loss = KnowledgeDistillationLoss(
                temperature=config.distillation_temperature,
                alpha=config.distillation_alpha
            )
        else:
            self.distillation_loss = None
            
        self.temporal_loss = TemporalConsistencyLoss(config.temporal_loss_weight)
        
        if config.use_energy_aware_training:
            self.energy_loss = EnergyAwareLoss(
                energy_weight=config.energy_loss_weight,
                target_energy=config.target_energy_budget or 1.0
            )
            self.energy_profiler = EnergyProfiler()
        else:
            self.energy_loss = None
            self.energy_profiler = None
        
        # Progressive training
        if config.use_progressive_training:
            self.progressive_timesteps = config.progressive_timesteps
            self.current_timestep_idx = 0
        else:
            self.progressive_timesteps = None
    
    def train_epoch(self, train_loader: DataLoader, 
                   teacher_model: Optional[nn.Module] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if teacher_model:
            teacher_model.eval()
        
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'spike_rate': 0.0,
            'energy_consumption': 0.0
        }
        
        num_batches = len(train_loader)
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            batch_size = data.size(0)
            
            # Progressive training - adjust timesteps
            if self.progressive_timesteps:
                self._update_progressive_timesteps()
            
            # Reset neuron states
            self._reset_model_states()
            
            # Energy profiling
            if self.energy_profiler:
                self.energy_profiler.start_profiling()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs, spike_outputs = outputs[0], outputs[1]
            else:
                spike_outputs = outputs
            
            # Compute losses
            loss_components = self._compute_losses(
                outputs, spike_outputs, targets, data, teacher_model
            )
            
            total_loss = loss_components['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient optimization
            grad_norm = self.gradient_optimizer.clip_gradients(max_norm=1.0)
            
            # Check for gradient anomalies
            anomalies = self.gradient_optimizer.detect_gradient_anomalies()
            if any(anomalies.values()):
                self.logger.warning(f"Gradient anomalies detected: {anomalies}")
            
            # Optimizer step
            self.optimizer.step()
            
            # STDP updates (if enabled)
            if self.stdp_optimizer:
                self.stdp_optimizer.apply_stdp_updates()
            
            # Adaptive threshold updates (if enabled)
            if self.threshold_optimizer:
                self._update_adaptive_thresholds(spike_outputs)
            
            # Energy profiling
            energy_consumed = 0.0
            if self.energy_profiler:
                measurements = self.energy_profiler.stop_profiling()
                if measurements:
                    energy_consumed = sum(m.energy_joules for m in measurements)
            
            # Compute metrics
            batch_metrics = self._compute_batch_metrics(
                outputs, targets, spike_outputs, loss_components, energy_consumed
            )
            
            # Update epoch metrics
            for key in epoch_metrics:
                if key in batch_metrics:
                    epoch_metrics[key] += batch_metrics[key] * batch_size
            
            total_samples += batch_size
            self.state.step += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.state.epoch}, Batch {batch_idx}/{num_batches}: "
                    f"Loss: {batch_metrics['loss']:.4f}, "
                    f"Acc: {batch_metrics['accuracy']:.3f}, "
                    f"Spike Rate: {batch_metrics['spike_rate']:.3f}"
                )
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= total_samples
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'spike_rate': 0.0
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                batch_size = data.size(0)
                
                # Reset states
                self._reset_model_states()
                
                # Forward pass
                outputs = self.model(data)
                
                if isinstance(outputs, tuple):
                    outputs, spike_outputs = outputs[0], outputs[1]
                else:
                    spike_outputs = outputs
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Compute metrics
                batch_metrics = self._compute_batch_metrics(
                    outputs, targets, spike_outputs, {'total_loss': loss}, 0.0
                )
                
                # Update validation metrics
                for key in val_metrics:
                    if key in batch_metrics:
                        val_metrics[key] += batch_metrics[key] * batch_size
                
                total_samples += batch_size
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= total_samples
        
        return val_metrics
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            teacher_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """Complete training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        training_start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, teacher_model)
            
            # Validation phase
            val_metrics = {}
            if val_loader and epoch % self.config.validation_freq == 0:
                val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, NeuromorphicScheduler):
                val_loss = val_metrics.get('loss', train_metrics['loss'])
                spike_rate = train_metrics.get('spike_rate', 0.1)
                self.scheduler.step(val_loss, spike_rate)
            else:
                self.scheduler.step()
            
            # Adaptive threshold updates
            if self.threshold_optimizer:
                self.threshold_optimizer.adapt_thresholds()
            
            epoch_time = time.time() - epoch_start_time
            
            # Record history
            epoch_data = {
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            
            self.state.training_history.append(epoch_data)
            
            # Early stopping check
            if val_metrics:
                val_acc = val_metrics.get('accuracy', 0.0)
                val_loss = val_metrics.get('loss', float('inf'))
                
                if val_acc > self.state.best_val_acc:
                    self.state.best_val_acc = val_acc
                    self.state.best_val_loss = val_loss
                    self.state.patience_counter = 0
                    
                    if self.config.save_best_model:
                        self._save_checkpoint('best_model.pth')
                else:
                    self.state.patience_counter += 1
                
                if self.state.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    self.state.early_stopped = True
                    break
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.3f}"
            )
            
            if val_metrics:
                self.logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.3f}"
                )
        
        self.state.total_training_time = time.time() - training_start_time
        
        return {
            'training_history': self.state.training_history,
            'best_val_accuracy': self.state.best_val_acc,
            'best_val_loss': self.state.best_val_loss,
            'total_training_time': self.state.total_training_time,
            'early_stopped': self.state.early_stopped
        }
    
    def _compute_losses(self, outputs: torch.Tensor, spike_outputs: torch.Tensor,
                       targets: torch.Tensor, inputs: torch.Tensor,
                       teacher_model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        loss_components = {}
        
        # Primary classification loss
        if self.distillation_loss and teacher_model:
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            total_loss, ce_loss, distill_loss = self.distillation_loss(
                outputs, teacher_outputs, targets, self.criterion
            )
            
            loss_components['classification_loss'] = ce_loss
            loss_components['distillation_loss'] = distill_loss
        else:
            ce_loss = self.criterion(outputs, targets)
            loss_components['classification_loss'] = ce_loss
            total_loss = ce_loss
        
        # Spike regularization
        reg_loss = self.spike_regularizer(self.model, spike_outputs)
        loss_components['regularization_loss'] = reg_loss
        total_loss += reg_loss
        
        # Temporal consistency loss
        temporal_loss = self.temporal_loss(spike_outputs)
        loss_components['temporal_loss'] = temporal_loss
        total_loss += temporal_loss
        
        # Energy-aware loss
        if self.energy_loss:
            energy_loss = self.energy_loss(spike_outputs)
            loss_components['energy_loss'] = energy_loss
            total_loss += energy_loss
        
        loss_components['total_loss'] = total_loss
        
        return loss_components
    
    def _compute_batch_metrics(self, outputs: torch.Tensor, targets: torch.Tensor,
                              spike_outputs: torch.Tensor, loss_components: Dict[str, torch.Tensor],
                              energy_consumed: float) -> Dict[str, float]:
        """Compute metrics for a single batch."""
        # Accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean().item()
        
        # Spike rate
        spike_rate = spike_outputs.mean().item()
        
        # Loss
        total_loss = loss_components['total_loss'].item()
        
        metrics = {
            'loss': total_loss,
            'accuracy': accuracy,
            'spike_rate': spike_rate,
            'energy_consumption': energy_consumed
        }
        
        # Add individual loss components
        for key, value in loss_components.items():
            if key != 'total_loss':
                metrics[f'{key}_value'] = value.item()
        
        return metrics
    
    def _reset_model_states(self):
        """Reset all neuron states in the model."""
        for module in self.model.modules():
            if hasattr(module, 'reset_state'):
                module.reset_state()
    
    def _update_progressive_timesteps(self):
        """Update timesteps for progressive training."""
        if not self.progressive_timesteps:
            return
        
        # Update timesteps based on training progress
        progress = self.state.epoch / self.config.num_epochs
        target_idx = min(int(progress * len(self.progressive_timesteps)), 
                        len(self.progressive_timesteps) - 1)
        
        if target_idx != self.current_timestep_idx:
            self.current_timestep_idx = target_idx
            target_timesteps = self.progressive_timesteps[target_idx]
            
            # Update model timesteps
            for module in self.model.modules():
                if hasattr(module, 'timesteps'):
                    module.timesteps = target_timesteps
            
            self.logger.info(f"Updated timesteps to {target_timesteps}")
    
    def _update_adaptive_thresholds(self, spike_outputs: torch.Tensor):
        """Update adaptive thresholds based on spike outputs."""
        if not self.threshold_optimizer:
            return
        
        # Collect statistics for threshold adaptation
        for name, module in self.model.named_modules():
            if isinstance(module, SpikingNeuron):
                # Mock membrane potential for statistics
                # In practice, this would come from the neuron's state
                mock_membrane = torch.randn_like(spike_outputs.mean(dim=1))
                self.threshold_optimizer.collect_statistics(
                    name, spike_outputs.mean(dim=1), mock_membrane
                )
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'training_state': self.state,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.state = checkpoint['training_state']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.state.training_history:
            return
        
        history = self.state.training_history
        epochs = [h['epoch'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        train_losses = [h['train_metrics']['loss'] for h in history]
        val_losses = [h['val_metrics'].get('loss', np.nan) for h in history]
        
        axes[0, 0].plot(epochs, train_losses, label='Train Loss')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Accuracy plot
        train_accs = [h['train_metrics']['accuracy'] for h in history]
        val_accs = [h['val_metrics'].get('accuracy', np.nan) for h in history]
        
        axes[0, 1].plot(epochs, train_accs, label='Train Acc')
        axes[0, 1].plot(epochs, val_accs, label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Spike rate plot
        spike_rates = [h['train_metrics'].get('spike_rate', 0) for h in history]
        axes[1, 0].plot(epochs, spike_rates)
        axes[1, 0].set_title('Spike Rate')
        axes[1, 0].set_xlabel('Epoch')
        
        # Learning rate plot
        lrs = [h['learning_rate'] for h in history]
        axes[1, 1].plot(epochs, lrs)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.state.training_history:
            return {}
        
        final_metrics = self.state.training_history[-1]
        
        summary = {
            'total_epochs': len(self.state.training_history),
            'best_validation_accuracy': self.state.best_val_acc,
            'best_validation_loss': self.state.best_val_loss,
            'final_train_accuracy': final_metrics['train_metrics']['accuracy'],
            'final_train_loss': final_metrics['train_metrics']['loss'],
            'total_training_time': self.state.total_training_time,
            'early_stopped': self.state.early_stopped,
            'avg_spike_rate': np.mean([h['train_metrics'].get('spike_rate', 0) 
                                     for h in self.state.training_history]),
            'total_energy_consumed': sum([h['train_metrics'].get('energy_consumption', 0) 
                                        for h in self.state.training_history])
        }
        
        # Add threshold adaptation summary if used
        if self.threshold_optimizer:
            summary['threshold_report'] = self.threshold_optimizer.get_threshold_report()
        
        return summary


# Convenience functions for common training scenarios
def train_spiking_classifier(model: nn.Module, train_loader: DataLoader,
                            val_loader: Optional[DataLoader] = None,
                            config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
    """Train a spiking classifier with default settings."""
    if config is None:
        config = TrainingConfig()
    
    trainer = SpikingTrainer(model, config)
    return trainer.fit(train_loader, val_loader)


def train_with_knowledge_distillation(student_model: nn.Module, teacher_model: nn.Module,
                                     train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                                     temperature: float = 4.0, alpha: float = 0.7) -> Dict[str, Any]:
    """Train spiking student with knowledge distillation from teacher."""
    config = TrainingConfig(
        use_knowledge_distillation=True,
        distillation_temperature=temperature,
        distillation_alpha=alpha
    )
    
    trainer = SpikingTrainer(student_model, config)
    return trainer.fit(train_loader, val_loader, teacher_model)


def energy_aware_training(model: nn.Module, train_loader: DataLoader,
                         energy_budget: float, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
    """Train with energy awareness."""
    config = TrainingConfig(
        use_energy_aware_training=True,
        target_energy_budget=energy_budget,
        energy_loss_weight=0.1
    )
    
    trainer = SpikingTrainer(model, config)
    return trainer.fit(train_loader, val_loader)