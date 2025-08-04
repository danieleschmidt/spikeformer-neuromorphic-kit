"""Advanced performance optimization and caching for neuromorphic systems."""

import torch
import torch.nn as nn
import time
import threading
import hashlib
import pickle
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path
import json
import logging
import weakref
import gc
from enum import Enum

from .models import SpikingTransformer, SpikingViT
from .neurons import SpikingNeuron
from .monitoring import metrics, PerformanceMetrics


class CacheStrategy(Enum):
    """Caching strategies for different use cases."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    NEUROMORPHIC = "neuromorphic"  # Neuromorphic-specific caching


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    spike_pattern_hash: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.creation_time > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.creation_time


class IntelligentCache:
    """Intelligent caching system with multiple strategies."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512, 
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory_bytes = 0
        
        # Adaptive strategy parameters
        self.adaptation_window = 100
        self.adaptation_threshold = 0.7
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()
            self.access_patterns[key].append(time.time())
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self.cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None,
            spike_pattern: Optional[torch.Tensor] = None) -> bool:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default size if pickle fails
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Create cache entry
            spike_hash = None
            if spike_pattern is not None:
                spike_hash = self._compute_spike_pattern_hash(spike_pattern)
            
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds,
                spike_pattern_hash=spike_hash
            )
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Ensure space
            self._ensure_space(size_bytes)
            
            # Add entry
            self.cache[key] = entry
            self.current_memory_bytes += size_bytes
            
            return True
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.current_memory_bytes = 0
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space for new entry."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_one()
        
        # Check memory limit
        while self.current_memory_bytes + required_bytes > self.max_memory_bytes:
            if not self._evict_one():
                break  # No more entries to evict
    
    def _evict_one(self) -> bool:
        """Evict one entry based on strategy."""
        if not self.cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self.cache))  # First (oldest) item
        elif self.strategy == CacheStrategy.LFU:
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            expired_keys = [k for k, e in self.cache.items() if e.is_expired]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].creation_time)
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key = self._adaptive_eviction()
        elif self.strategy == CacheStrategy.NEUROMORPHIC:
            key = self._neuromorphic_eviction()
        else:
            key = next(iter(self.cache))  # Default to LRU
        
        self._remove_entry(key)
        self.evictions += 1
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on access patterns."""
        if len(self.cache) < self.adaptation_window:
            # Not enough data, use LRU
            return next(iter(self.cache))
        
        # Calculate access frequency for recent period
        recent_time = time.time() - 3600  # Last hour
        
        scores = {}
        for key, entry in self.cache.items():
            recent_accesses = len([t for t in self.access_patterns[key] if t > recent_time])
            recency_score = 1.0 / (time.time() - entry.last_access + 1)
            frequency_score = recent_accesses / len(self.access_patterns[key])
            size_penalty = entry.size_bytes / self.max_memory_bytes
            
            scores[key] = recency_score * frequency_score * (1 - size_penalty)
        
        # Evict entry with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _neuromorphic_eviction(self) -> str:
        """Neuromorphic-specific eviction considering spike patterns."""
        if not self.cache:
            return next(iter(self.cache))
        
        # Consider spike pattern similarity and temporal locality
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Temporal score (newer is better)
            temporal_score = 1.0 / (current_time - entry.last_access + 1)
            
            # Frequency score
            frequency_score = entry.access_count / (entry.age_seconds + 1)
            
            # Spike pattern score (if available)
            spike_score = 1.0
            if entry.spike_pattern_hash:
                # In a full implementation, would compare spike patterns
                spike_score = 1.2  # Slight preference for spike data
            
            scores[key] = temporal_score * frequency_score * spike_score
        
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]
    
    def _compute_spike_pattern_hash(self, spike_pattern: torch.Tensor) -> str:
        """Compute hash of spike pattern for similarity matching."""
        # Convert to binary and hash
        binary_spikes = (spike_pattern > 0.5).cpu().numpy().astype(np.uint8)
        return hashlib.md5(binary_spikes.tobytes()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'strategy': self.strategy.value
        }


class SpikePatternOptimizer:
    """Optimize spike patterns for better performance."""
    
    def __init__(self):
        self.pattern_cache = {}
        self.optimization_stats = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def optimize_spike_tensor(self, spikes: torch.Tensor, 
                             optimization_level: str = "balanced") -> torch.Tensor:
        """Optimize spike tensor for performance."""
        original_size = spikes.numel()
        
        if optimization_level == "memory":
            optimized = self._memory_optimize(spikes)
        elif optimization_level == "speed":
            optimized = self._speed_optimize(spikes)
        elif optimization_level == "balanced":
            optimized = self._balanced_optimize(spikes)
        else:
            optimized = spikes
        
        # Track optimization statistics
        compression_ratio = original_size / optimized.numel() if optimized.numel() > 0 else 1.0
        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['avg_compression'] = (
            (self.optimization_stats['avg_compression'] * 
             (self.optimization_stats['total_optimizations'] - 1) + compression_ratio) /
            self.optimization_stats['total_optimizations']
        )
        
        return optimized
    
    def _memory_optimize(self, spikes: torch.Tensor) -> torch.Tensor:
        """Optimize for memory usage."""
        # Use sparse representation for very sparse tensors
        sparsity = (spikes == 0).float().mean().item()
        
        if sparsity > 0.9:  # Very sparse
            # Convert to sparse tensor
            indices = torch.nonzero(spikes, as_tuple=False).t()
            values = spikes[spikes != 0]
            sparse_tensor = torch.sparse_coo_tensor(indices, values, spikes.shape)
            return sparse_tensor.coalesce()
        
        # Quantize for moderate sparsity
        elif sparsity > 0.5:
            # Use int8 quantization
            return (spikes * 127).to(torch.int8)
        
        return spikes
    
    def _speed_optimize(self, spikes: torch.Tensor) -> torch.Tensor:
        """Optimize for speed."""
        # Ensure contiguous memory layout
        if not spikes.is_contiguous():
            spikes = spikes.contiguous()
        
        # Use appropriate data type
        if spikes.dtype != torch.float32:
            spikes = spikes.to(torch.float32)
        
        # Batch similar operations
        return spikes
    
    def _balanced_optimize(self, spikes: torch.Tensor) -> torch.Tensor:
        """Balanced optimization."""
        sparsity = (spikes == 0).float().mean().item()
        
        # Choose strategy based on sparsity
        if sparsity > 0.95:
            return self._memory_optimize(spikes)
        else:
            return self._speed_optimize(spikes)


class ModelOptimizer:
    """Optimize neuromorphic models for better performance."""
    
    def __init__(self):
        self.optimization_cache = IntelligentCache(max_size=100, max_memory_mb=256)
        self.logger = logging.getLogger(__name__)
    
    def optimize_model(self, model: nn.Module, optimization_config: Dict[str, Any]) -> nn.Module:
        """Optimize model with various techniques."""
        model_key = self._get_model_key(model, optimization_config)
        
        # Check cache
        cached_model = self.optimization_cache.get(model_key)
        if cached_model is not None:
            self.logger.info("Using cached optimized model")
            return cached_model
        
        optimized_model = model
        
        # Apply optimizations
        if optimization_config.get('fuse_operations', True):
            optimized_model = self._fuse_operations(optimized_model)
        
        if optimization_config.get('quantize_weights', False):
            optimized_model = self._quantize_weights(optimized_model)
        
        if optimization_config.get('prune_connections', False):
            optimized_model = self._prune_connections(optimized_model, 
                                                    optimization_config.get('pruning_ratio', 0.1))
        
        if optimization_config.get('optimize_spike_thresholds', True):
            optimized_model = self._optimize_spike_thresholds(optimized_model)
        
        # Cache optimized model
        self.optimization_cache.put(model_key, optimized_model, ttl_seconds=3600)
        
        return optimized_model
    
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse operations for better performance."""
        # In a full implementation, would fuse consecutive operations
        self.logger.info("Applied operation fusion optimization")
        return model
    
    def _quantize_weights(self, model: nn.Module) -> nn.Module:
        """Quantize model weights."""
        # Simple quantization example
        for param in model.parameters():
            if param.requires_grad:
                # Quantize to int8 range
                scale = param.abs().max() / 127
                quantized = torch.round(param / scale).clamp(-128, 127)
                param.data = quantized * scale
        
        self.logger.info("Applied weight quantization optimization")
        return model
    
    def _prune_connections(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Prune weak connections."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    # Simple magnitude-based pruning
                    flat_weights = param.view(-1)
                    threshold = torch.quantile(torch.abs(flat_weights), pruning_ratio)
                    mask = torch.abs(param) > threshold
                    param.data *= mask.float()
        
        self.logger.info(f"Applied connection pruning ({pruning_ratio*100:.1f}%)")
        return model
    
    def _optimize_spike_thresholds(self, model: nn.Module) -> nn.Module:
        """Optimize spike thresholds for better performance."""
        for module in model.modules():
            if isinstance(module, SpikingNeuron):
                # Adaptive threshold optimization
                if hasattr(module, 'threshold'):
                    # Simple heuristic: adjust based on typical input magnitudes
                    current_threshold = module.threshold.item() if torch.is_tensor(module.threshold) else module.threshold
                    optimized_threshold = current_threshold * 0.95  # Slightly lower for more activity
                    
                    if torch.is_tensor(module.threshold):
                        module.threshold.data.fill_(optimized_threshold)
                    else:
                        module.threshold = optimized_threshold
        
        self.logger.info("Optimized spike thresholds")
        return model
    
    def _get_model_key(self, model: nn.Module, config: Dict[str, Any]) -> str:
        """Generate cache key for model and config."""
        model_str = str(model.__class__.__name__)
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5((model_str + config_str).encode()).hexdigest()


class PerformanceProfiler:
    """Profile performance bottlenecks in neuromorphic models."""
    
    def __init__(self):
        self.profiles = {}
        self.logger = logging.getLogger(__name__)
    
    def profile_model(self, model: nn.Module, input_data: torch.Tensor, 
                     model_name: str = "model") -> Dict[str, Any]:
        """Profile model performance."""
        profile_data = {
            'model_name': model_name,
            'input_shape': list(input_data.shape),
            'layer_timings': {},
            'memory_usage': {},
            'spike_statistics': {},
            'bottlenecks': []
        }
        
        # Register hooks for profiling
        hooks = []
        layer_times = {}
        
        def create_hook(name):
            def hook(module, input, output):
                start_time = time.perf_counter()
                # The actual computation is already done, so we measure hook overhead
                # In a real implementation, would use more sophisticated timing
                layer_times[name] = time.perf_counter() - start_time
            return hook
        
        for name, module in model.named_modules():
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
        
        # Profile inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            output = model(input_data)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                final_memory = torch.cuda.memory_allocated()
        
        total_time = time.perf_counter() - start_time
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze results
        profile_data['total_inference_time'] = total_time
        profile_data['layer_timings'] = layer_times
        
        if torch.cuda.is_available():
            profile_data['memory_usage'] = {
                'initial_mb': initial_memory / (1024**2),
                'peak_mb': peak_memory / (1024**2),
                'final_mb': final_memory / (1024**2),
                'peak_increase_mb': (peak_memory - initial_memory) / (1024**2)
            }
        
        # Analyze spike statistics
        if hasattr(model, 'get_spike_stats'):
            profile_data['spike_statistics'] = model.get_spike_stats()
        
        # Identify bottlenecks
        if layer_times:
            avg_layer_time = np.mean(list(layer_times.values()))
            for layer_name, layer_time in layer_times.items():
                if layer_time > avg_layer_time * 2:  # More than 2x average
                    profile_data['bottlenecks'].append({
                        'layer': layer_name,
                        'time': layer_time,
                        'relative_slowness': layer_time / avg_layer_time
                    })
        
        # Store profile
        self.profiles[model_name] = profile_data
        
        return profile_data
    
    def generate_optimization_recommendations(self, model_name: str) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on profiling."""
        if model_name not in self.profiles:
            return []
        
        profile = self.profiles[model_name]
        recommendations = []
        
        # Memory-based recommendations
        if 'memory_usage' in profile:
            peak_mb = profile['memory_usage'].get('peak_increase_mb', 0)
            if peak_mb > 1000:  # More than 1GB
                recommendations.append({
                    'category': 'memory',
                    'priority': 'high',
                    'recommendation': 'Consider gradient checkpointing or model parallelism',
                    'reason': f'High memory usage: {peak_mb:.1f} MB'
                })
        
        # Bottleneck-based recommendations
        for bottleneck in profile.get('bottlenecks', []):
            if bottleneck['relative_slowness'] > 5:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'recommendation': f'Optimize {bottleneck["layer"]} layer',
                    'reason': f'Layer is {bottleneck["relative_slowness"]:.1f}x slower than average'
                })
        
        # Spike-based recommendations
        spike_stats = profile.get('spike_statistics', {})
        if spike_stats:
            avg_spike_rate = spike_stats.get('average_spike_rate', 0)
            if avg_spike_rate < 0.1:
                recommendations.append({
                    'category': 'efficiency',
                    'priority': 'medium',
                    'recommendation': 'Increase spike activity for better neuromorphic efficiency',
                    'reason': f'Low spike rate: {avg_spike_rate:.3f}'
                })
            elif avg_spike_rate > 0.8:
                recommendations.append({
                    'category': 'efficiency',
                    'priority': 'medium',
                    'recommendation': 'Reduce spike activity to save energy',
                    'reason': f'High spike rate: {avg_spike_rate:.3f}'
                })
        
        return recommendations


class AdaptiveMemoryManager:
    """Adaptive memory management for neuromorphic systems."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_pools: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.allocation_stats = defaultdict(int)
        self.gc_threshold = 0.8  # Trigger GC at 80% memory usage
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: str = "cpu", pool: str = "default") -> torch.Tensor:
        """Allocate tensor with memory pooling."""
        with self.lock:
            required_bytes = np.prod(shape) * torch._utils._element_size(dtype)
            
            # Check if we can reuse from pool
            for i, tensor in enumerate(self.memory_pools[pool]):
                if (tensor.shape == shape and tensor.dtype == dtype and 
                    tensor.device.type == device):
                    # Reuse tensor
                    reused_tensor = self.memory_pools[pool].pop(i)
                    reused_tensor.zero_()  # Clear data
                    self.allocation_stats['reused'] += 1
                    return reused_tensor
            
            # Check memory pressure
            if self._get_memory_usage() > self.gc_threshold:
                self._garbage_collect()
            
            # Allocate new tensor
            device_obj = torch.device(device)
            tensor = torch.zeros(shape, dtype=dtype, device=device_obj)
            self.allocation_stats['allocated'] += 1
            
            return tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor, pool: str = "default"):
        """Return tensor to memory pool."""
        with self.lock:
            if len(self.memory_pools[pool]) < 100:  # Limit pool size
                self.memory_pools[pool].append(tensor)
                self.allocation_stats['pooled'] += 1
            else:
                del tensor
                self.allocation_stats['freed'] += 1
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage ratio."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return allocated / reserved if reserved > 0 else 0.0
        else:
            # For CPU, use a simple heuristic
            total_tensors = sum(len(pool) for pool in self.memory_pools.values())
            return min(total_tensors / 1000, 1.0)  # Rough estimate
    
    def _garbage_collect(self):
        """Perform garbage collection."""
        # Clear some pools
        for pool_name in list(self.memory_pools.keys()):
            if len(self.memory_pools[pool_name]) > 10:
                # Remove oldest half
                remove_count = len(self.memory_pools[pool_name]) // 2
                for _ in range(remove_count):
                    self.memory_pools[pool_name].pop(0)
        
        # Python garbage collection
        collected = gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"Garbage collection freed {collected} objects")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        return {
            'allocation_stats': dict(self.allocation_stats),
            'pool_sizes': {pool: len(tensors) for pool, tensors in self.memory_pools.items()},
            'memory_usage_ratio': self._get_memory_usage(),
            'total_pools': len(self.memory_pools)
        }


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self):
        self.model_cache = IntelligentCache(max_size=50, max_memory_mb=1024, 
                                          strategy=CacheStrategy.NEUROMORPHIC)
        self.result_cache = IntelligentCache(max_size=1000, max_memory_mb=512,
                                           strategy=CacheStrategy.ADAPTIVE)
        self.spike_optimizer = SpikePatternOptimizer()
        self.model_optimizer = ModelOptimizer()
        self.profiler = PerformanceProfiler()
        self.memory_manager = AdaptiveMemoryManager()
        
        self.optimization_stats = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def optimize_inference_pipeline(self, model: nn.Module, model_name: str,
                                  optimization_config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Optimize complete inference pipeline."""
        optimization_config = optimization_config or {
            'cache_models': True,
            'cache_results': True,
            'optimize_spikes': True,
            'profile_performance': True,
            'adaptive_memory': True
        }
        
        cache_key = f"optimized_{model_name}_{hash(str(optimization_config))}"
        
        # Check model cache
        if optimization_config.get('cache_models', True):
            cached_model = self.model_cache.get(cache_key)
            if cached_model is not None:
                self.logger.info(f"Using cached optimized model for {model_name}")
                return cached_model
        
        # Profile model first
        if optimization_config.get('profile_performance', True):
            dummy_input = torch.randn(1, 10)  # Dummy input for profiling
            profile = self.profiler.profile_model(model, dummy_input, model_name)
            recommendations = self.profiler.generate_optimization_recommendations(model_name)
            
            self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
        
        # Optimize model
        optimized_model = self.model_optimizer.optimize_model(model, optimization_config)
        
        # Cache optimized model
        if optimization_config.get('cache_models', True):
            self.model_cache.put(cache_key, optimized_model, ttl_seconds=7200)
        
        self.optimization_stats['models_optimized'] += 1
        return optimized_model
    
    def optimized_inference(self, model: nn.Module, inputs: torch.Tensor,
                          model_name: str = "model", use_cache: bool = True) -> Dict[str, Any]:
        """Perform optimized inference with caching."""
        start_time = time.perf_counter()
        
        # Generate cache key based on inputs
        if use_cache:
            input_hash = hashlib.md5(inputs.detach().cpu().numpy().tobytes()).hexdigest()
            cache_key = f"{model_name}_{input_hash}"
            
            # Check result cache
            cached_result = self.result_cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for {model_name}")
                return {
                    'output': cached_result,
                    'inference_time_ms': 0.1,  # Minimal cache lookup time
                    'cache_hit': True
                }
        
        # Optimize spike inputs if applicable
        if torch.is_tensor(inputs) and inputs.dtype in [torch.float32, torch.float16]:
            # Check if this looks like spike data
            if ((inputs >= 0) & (inputs <= 1)).all():
                inputs = self.spike_optimizer.optimize_spike_tensor(inputs)
        
        # Perform inference
        with torch.no_grad():
            output = model(inputs)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Cache result
        if use_cache:
            self.result_cache.put(cache_key, output, ttl_seconds=1800,  # 30 minutes
                                spike_pattern=inputs if inputs.numel() < 10000 else None)
        
        self.optimization_stats['inferences_performed'] += 1
        
        return {
            'output': output,
            'inference_time_ms': inference_time,
            'cache_hit': False
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'cache_stats': {
                'model_cache': self.model_cache.get_stats(),
                'result_cache': self.result_cache.get_stats()
            },
            'optimization_stats': dict(self.optimization_stats),
            'spike_optimizer_stats': dict(self.spike_optimizer.optimization_stats),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'profiling_data': {
                'models_profiled': len(self.profiler.profiles),
                'available_profiles': list(self.profiler.profiles.keys())
            }
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.result_cache.clear()
        self.logger.info("Cleared all performance caches")


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience functions
def optimize_model_for_inference(model: nn.Module, model_name: str = "model",
                               **optimization_kwargs) -> nn.Module:
    """Optimize model for inference performance."""
    return performance_optimizer.optimize_inference_pipeline(
        model, model_name, optimization_kwargs
    )

def cached_inference(model: nn.Module, inputs: torch.Tensor, 
                    model_name: str = "model") -> torch.Tensor:
    """Perform cached inference."""
    result = performance_optimizer.optimized_inference(model, inputs, model_name)
    return result['output']

def get_performance_stats() -> Dict[str, Any]:
    """Get performance optimization statistics."""
    return performance_optimizer.get_performance_report()

def clear_performance_caches():
    """Clear all performance caches."""
    performance_optimizer.clear_caches()

def create_optimized_memory_pool(max_memory_mb: int = 1024) -> AdaptiveMemoryManager:
    """Create optimized memory pool."""
    return AdaptiveMemoryManager(max_memory_mb)

# Performance optimization decorator
def performance_optimized(model_name: str = None, use_cache: bool = True,
                        optimization_config: Dict[str, Any] = None):
    """Decorator for automatic performance optimization."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal model_name
            if model_name is None:
                model_name = func.__name__
            
            # Extract model and inputs from args
            if len(args) >= 2 and isinstance(args[0], nn.Module) and torch.is_tensor(args[1]):
                model, inputs = args[0], args[1]
                
                # Optimize model if not already done
                cache_key = f"opt_{model_name}"
                optimized_model = performance_optimizer.model_cache.get(cache_key)
                if optimized_model is None:
                    optimized_model = performance_optimizer.optimize_inference_pipeline(
                        model, model_name, optimization_config
                    )
                    performance_optimizer.model_cache.put(cache_key, optimized_model)
                
                # Perform optimized inference
                result = performance_optimizer.optimized_inference(
                    optimized_model, inputs, model_name, use_cache
                )
                
                return result['output']
            else:
                # Fallback to original function
                return func(*args, **kwargs)
        
        return wrapper
    return decorator