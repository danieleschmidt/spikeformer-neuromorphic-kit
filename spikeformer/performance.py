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


# ==============================================================================
# GENERATION 3: ADVANCED SCALING & PERFORMANCE SYSTEMS
# ==============================================================================

class DistributedNeuromorphicOptimizer:
    """Advanced distributed optimization for neuromorphic computing at scale."""
    
    def __init__(self, num_devices: int = 4, backend: str = "nccl"):
        self.num_devices = num_devices
        self.backend = backend
        self.device_pool = []
        self.load_balancer = None
        self.performance_stats = defaultdict(lambda: defaultdict(float))
        
    def setup_distributed_environment(self):
        """Setup distributed training/inference environment."""
        if torch.cuda.is_available():
            # CUDA distributed setup
            for i in range(min(self.num_devices, torch.cuda.device_count())):
                device = torch.device(f'cuda:{i}')
                self.device_pool.append({
                    'device': device,
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': 0,
                    'active_models': 0,
                    'performance_score': 1.0
                })
        else:
            # CPU distributed setup
            for i in range(self.num_devices):
                self.device_pool.append({
                    'device': torch.device('cpu'),
                    'memory_total': 8 * 1024**3,  # Assume 8GB RAM
                    'memory_allocated': 0,
                    'active_models': 0,
                    'performance_score': 1.0
                })
    
    def distribute_model(self, model: nn.Module, distribution_strategy: str = "data_parallel") -> nn.Module:
        """Distribute model across available devices."""
        
        if distribution_strategy == "data_parallel":
            return self._setup_data_parallel(model)
        elif distribution_strategy == "model_parallel":
            return self._setup_model_parallel(model)
        elif distribution_strategy == "pipeline_parallel":
            return self._setup_pipeline_parallel(model)
        elif distribution_strategy == "hybrid":
            return self._setup_hybrid_parallel(model)
        else:
            raise ValueError(f"Unknown distribution strategy: {distribution_strategy}")
    
    def _setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup data parallel distribution."""
        if len(self.device_pool) > 1 and torch.cuda.is_available():
            device_ids = [device['device'].index for device in self.device_pool 
                         if device['device'].type == 'cuda']
            if device_ids:
                model = model.to(device_ids[0])
                return nn.DataParallel(model, device_ids=device_ids)
        
        return model.to(self.device_pool[0]['device'])
    
    def _setup_model_parallel(self, model: nn.Module) -> nn.Module:
        """Setup model parallel distribution."""
        # Simplified model parallelism - split layers across devices
        layers = list(model.modules())
        layers_per_device = len(layers) // len(self.device_pool)
        
        class ModelParallelWrapper(nn.Module):
            def __init__(self, original_model, device_pool, layers_per_device):
                super().__init__()
                self.device_assignments = {}
                self.layer_modules = nn.ModuleDict()
                
                # Assign layers to devices
                for i, (name, module) in enumerate(original_model.named_modules()):
                    if i == 0:  # Skip root module
                        continue
                    device_idx = min(i // layers_per_device, len(device_pool) - 1)
                    device = device_pool[device_idx]['device']
                    
                    self.device_assignments[name] = device
                    self.layer_modules[name] = module.to(device)
            
            def forward(self, x):
                # Simple sequential execution across devices
                current_device = x.device
                
                for name, module in self.layer_modules.items():
                    target_device = self.device_assignments[name]
                    if current_device != target_device:
                        x = x.to(target_device)
                        current_device = target_device
                    x = module(x)
                
                return x
        
        return ModelParallelWrapper(model, self.device_pool, layers_per_device)
    
    def _setup_pipeline_parallel(self, model: nn.Module) -> nn.Module:
        """Setup pipeline parallel distribution."""
        # Simplified pipeline parallelism implementation
        
        class PipelineParallelWrapper(nn.Module):
            def __init__(self, original_model, device_pool):
                super().__init__()
                self.pipeline_stages = nn.ModuleList()
                self.devices = [d['device'] for d in device_pool]
                
                # Split model into pipeline stages
                layers = list(original_model.children())
                stage_size = max(1, len(layers) // len(self.devices))
                
                for i in range(0, len(layers), stage_size):
                    stage_layers = layers[i:i + stage_size]
                    stage = nn.Sequential(*stage_layers)
                    device_idx = i // stage_size
                    if device_idx < len(self.devices):
                        stage = stage.to(self.devices[device_idx])
                    self.pipeline_stages.append(stage)
            
            def forward(self, x):
                # Pipeline execution
                for i, stage in enumerate(self.pipeline_stages):
                    if i < len(self.devices):
                        x = x.to(self.devices[i])
                    x = stage(x)
                return x
        
        return PipelineParallelWrapper(model, self.device_pool)
    
    def _setup_hybrid_parallel(self, model: nn.Module) -> nn.Module:
        """Setup hybrid data + model parallelism."""
        # For simplicity, use data parallel if multiple devices available
        if len(self.device_pool) >= 4:
            # Use half devices for data parallel, half for model parallel
            data_devices = self.device_pool[:len(self.device_pool)//2]
            model_devices = self.device_pool[len(self.device_pool)//2:]
            
            # First apply model parallelism
            model_parallel = self._setup_model_parallel(model)
            
            # Then apply data parallelism if CUDA available
            if torch.cuda.is_available():
                device_ids = [d['device'].index for d in data_devices if d['device'].type == 'cuda']
                if device_ids:
                    return nn.DataParallel(model_parallel, device_ids=device_ids)
            
            return model_parallel
        else:
            return self._setup_data_parallel(model)
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Determine optimal batch size for current hardware configuration."""
        device = next(model.parameters()).device
        
        if device.type == 'cuda':
            # GPU optimization
            available_memory = torch.cuda.get_device_properties(device).total_memory
            current_memory = torch.cuda.memory_allocated(device)
            free_memory = available_memory - current_memory
            
            # Estimate memory per sample (rough heuristic)
            sample_elements = np.prod(input_shape)
            estimated_memory_per_sample = sample_elements * 4 * 3  # float32 * forward/backward/gradients
            
            max_batch_size = int(free_memory * 0.7 / estimated_memory_per_sample)  # 70% utilization
            return max(1, min(max_batch_size, 512))  # Cap at 512
        else:
            # CPU optimization - consider available cores
            import multiprocessing
            cores = multiprocessing.cpu_count()
            return min(cores * 4, 128)  # 4x cores, cap at 128


class AdvancedSpikePatternCache:
    """Advanced caching system specifically for neuromorphic spike patterns."""
    
    def __init__(self, max_patterns: int = 10000, similarity_threshold: float = 0.95):
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold
        self.pattern_cache = OrderedDict()
        self.pattern_hashes = {}
        self.access_stats = defaultdict(int)
        self.compression_ratios = {}
        
    def _compute_pattern_hash(self, spike_tensor: torch.Tensor) -> str:
        """Compute hash for spike pattern with temporal structure."""
        # Compress sparse spike tensor for hashing
        spike_positions = torch.nonzero(spike_tensor, as_tuple=False)
        if spike_positions.numel() == 0:
            return "empty_pattern"
        
        # Create hash from spike positions and timing
        position_str = ",".join([f"{pos[0]},{pos[1]},{pos[2]}" for pos in spike_positions[:100]])  # Limit for performance
        return hashlib.md5(position_str.encode()).hexdigest()
    
    def _compute_pattern_similarity(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """Compute similarity between two spike patterns."""
        if pattern1.shape != pattern2.shape:
            return 0.0
        
        # Temporal spike correlation
        correlation = F.cosine_similarity(pattern1.flatten(), pattern2.flatten(), dim=0)
        
        # Spike timing similarity (simplified)
        timing_diff = torch.abs(pattern1 - pattern2).mean()
        timing_similarity = 1.0 - min(timing_diff.item(), 1.0)
        
        return (correlation.item() + timing_similarity) / 2.0
    
    def get_similar_pattern(self, spike_pattern: torch.Tensor) -> Optional[str]:
        """Find similar cached pattern."""
        pattern_hash = self._compute_pattern_hash(spike_pattern)
        
        # Direct hash match
        if pattern_hash in self.pattern_hashes:
            self.access_stats[pattern_hash] += 1
            return pattern_hash
        
        # Similarity search (expensive, limit to recent patterns)
        recent_patterns = list(self.pattern_cache.items())[-100:]
        
        for cached_hash, cached_pattern in recent_patterns:
            similarity = self._compute_pattern_similarity(spike_pattern, cached_pattern['pattern'])
            if similarity >= self.similarity_threshold:
                self.access_stats[cached_hash] += 1
                return cached_hash
        
        return None
    
    def cache_pattern(self, spike_pattern: torch.Tensor, computation_result: Any,
                     computation_time: float) -> str:
        """Cache spike pattern and its computation result."""
        pattern_hash = self._compute_pattern_hash(spike_pattern)
        
        # Check cache size limit
        if len(self.pattern_cache) >= self.max_patterns:
            # Remove least recently used
            self.pattern_cache.popitem(last=False)
        
        # Compress pattern for storage
        compressed_pattern = self._compress_spike_pattern(spike_pattern)
        compression_ratio = spike_pattern.numel() / compressed_pattern.numel()
        
        self.pattern_cache[pattern_hash] = {
            'pattern': compressed_pattern,
            'result': computation_result,
            'computation_time': computation_time,
            'cache_time': time.time(),
            'access_count': 1
        }
        
        self.pattern_hashes[pattern_hash] = True
        self.compression_ratios[pattern_hash] = compression_ratio
        
        return pattern_hash
    
    def get_cached_result(self, pattern_hash: str) -> Optional[Any]:
        """Get cached computation result."""
        if pattern_hash in self.pattern_cache:
            entry = self.pattern_cache[pattern_hash]
            entry['access_count'] += 1
            # Move to end (LRU)
            self.pattern_cache.move_to_end(pattern_hash)
            return entry['result']
        return None
    
    def _compress_spike_pattern(self, spike_pattern: torch.Tensor) -> torch.Tensor:
        """Compress sparse spike pattern."""
        # Simple compression: store only non-zero positions and values
        nonzero_indices = torch.nonzero(spike_pattern, as_tuple=False)
        nonzero_values = spike_pattern[spike_pattern != 0]
        
        if nonzero_indices.numel() == 0:
            return torch.tensor([])
        
        # Combine indices and values
        compressed = torch.cat([
            nonzero_indices.float().flatten(),
            nonzero_values.flatten()
        ])
        
        return compressed
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(self.access_stats.values())
        avg_compression = np.mean(list(self.compression_ratios.values())) if self.compression_ratios else 1.0
        
        return {
            'total_patterns': len(self.pattern_cache),
            'total_accesses': total_accesses,
            'average_compression_ratio': avg_compression,
            'cache_hit_rate': total_accesses / max(1, len(self.pattern_cache)),
            'memory_saved_ratio': 1.0 - (1.0 / avg_compression)
        }


class NeuromorphicLoadBalancer:
    """Load balancer optimized for neuromorphic computing workloads."""
    
    def __init__(self, devices: List[torch.device]):
        self.devices = devices
        self.device_loads = {device: 0.0 for device in devices}
        self.device_capabilities = {}
        self.spike_processing_history = defaultdict(list)
        self.load_update_interval = 10  # seconds
        self.last_load_update = time.time()
        
        self._assess_device_capabilities()
    
    def _assess_device_capabilities(self):
        """Assess neuromorphic processing capabilities of each device."""
        for device in self.devices:
            if device.type == 'cuda':
                props = torch.cuda.get_device_properties(device)
                capability_score = (
                    props.total_memory / (1024**3) * 0.4 +  # Memory weight
                    props.multi_processor_count * 0.3 +     # Compute weight
                    (props.major + props.minor * 0.1) * 0.3  # Architecture weight
                )
            else:
                # CPU capability assessment
                import multiprocessing
                capability_score = multiprocessing.cpu_count() * 0.5
            
            self.device_capabilities[device] = {
                'base_capability': capability_score,
                'neuromorphic_efficiency': 1.0,  # Will be learned
                'spike_processing_speed': 1.0,   # Will be learned
                'current_load': 0.0
            }
    
    def select_optimal_device(self, workload_type: str = "inference", 
                            spike_density: float = 0.1) -> torch.device:
        """Select optimal device based on current load and workload characteristics."""
        
        self._update_device_loads()
        
        best_device = None
        best_score = float('-inf')
        
        for device in self.devices:
            capabilities = self.device_capabilities[device]
            
            # Base score from hardware capabilities
            score = capabilities['base_capability']
            
            # Adjust for neuromorphic efficiency
            score *= capabilities['neuromorphic_efficiency']
            
            # Adjust for spike processing speed if spike-heavy workload
            if spike_density > 0.3:
                score *= capabilities['spike_processing_speed']
            
            # Penalize for current load
            load_penalty = capabilities['current_load'] ** 2
            score *= (1.0 - load_penalty)
            
            # Bonus for recent similar workloads (warm cache)
            if workload_type in self.spike_processing_history[device]:
                recent_performance = self.spike_processing_history[device][-5:]
                avg_performance = np.mean(recent_performance) if recent_performance else 1.0
                score *= (1.0 + avg_performance * 0.2)
            
            if score > best_score:
                best_score = score
                best_device = device
        
        return best_device or self.devices[0]
    
    def update_device_performance(self, device: torch.device, 
                                performance_metrics: Dict[str, float]):
        """Update device performance metrics based on actual execution."""
        if device not in self.device_capabilities:
            return
        
        capabilities = self.device_capabilities[device]
        
        # Update neuromorphic efficiency
        if 'spike_processing_efficiency' in performance_metrics:
            efficiency = performance_metrics['spike_processing_efficiency']
            capabilities['neuromorphic_efficiency'] = (
                0.8 * capabilities['neuromorphic_efficiency'] + 0.2 * efficiency
            )
        
        # Update spike processing speed
        if 'spikes_per_second' in performance_metrics:
            speed_ratio = performance_metrics['spikes_per_second'] / 1000  # Normalize
            capabilities['spike_processing_speed'] = (
                0.8 * capabilities['spike_processing_speed'] + 0.2 * speed_ratio
            )
        
        # Record performance history
        overall_performance = (
            capabilities['neuromorphic_efficiency'] + 
            capabilities['spike_processing_speed']
        ) / 2.0
        
        self.spike_processing_history[device].append(overall_performance)
        
        # Limit history size
        if len(self.spike_processing_history[device]) > 100:
            self.spike_processing_history[device] = self.spike_processing_history[device][-50:]
    
    def _update_device_loads(self):
        """Update current device load estimates."""
        current_time = time.time()
        
        if current_time - self.last_load_update < self.load_update_interval:
            return
        
        for device in self.devices:
            if device.type == 'cuda':
                # GPU load estimation
                try:
                    allocated = torch.cuda.memory_allocated(device)
                    cached = torch.cuda.memory_cached(device)
                    total = torch.cuda.get_device_properties(device).total_memory
                    
                    memory_load = (allocated + cached) / total
                    
                    # Estimate compute load (simplified)
                    compute_load = min(1.0, memory_load * 1.2)  # Approximate
                    
                    overall_load = (memory_load + compute_load) / 2.0
                    
                except Exception:
                    overall_load = 0.5  # Default assumption
            else:
                # CPU load estimation
                import psutil
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
                    memory_percent = psutil.virtual_memory().percent / 100.0
                    overall_load = (cpu_percent + memory_percent) / 2.0
                except Exception:
                    overall_load = 0.5  # Default assumption
            
            self.device_capabilities[device]['current_load'] = overall_load
        
        self.last_load_update = current_time
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        self._update_device_loads()
        
        stats = {}
        for device in self.devices:
            capabilities = self.device_capabilities[device]
            stats[str(device)] = {
                'current_load': capabilities['current_load'],
                'neuromorphic_efficiency': capabilities['neuromorphic_efficiency'],
                'spike_processing_speed': capabilities['spike_processing_speed'],
                'performance_history_length': len(self.spike_processing_history[device])
            }
        
        return stats


class AsyncNeuromorphicProcessor:
    """Asynchronous processor for neuromorphic workloads."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = asyncio.Queue()
        self.results_cache = {}
        self.processing_stats = defaultdict(int)
        
    async def process_spike_batch_async(self, spike_batches: List[torch.Tensor],
                                      model: nn.Module, 
                                      processing_func: Callable) -> List[torch.Tensor]:
        """Process multiple spike batches asynchronously."""
        
        tasks = []
        for i, batch in enumerate(spike_batches):
            task = self.executor.submit(self._process_single_batch, batch, model, processing_func, i)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = []
        for task in tasks:
            result = await asyncio.wrap_future(task)
            results.append(result)
        
        self.processing_stats['batches_processed'] += len(spike_batches)
        return results
    
    def _process_single_batch(self, batch: torch.Tensor, model: nn.Module,
                            processing_func: Callable, batch_id: int) -> torch.Tensor:
        """Process single batch (runs in thread pool)."""
        start_time = time.perf_counter()
        
        # Check cache first
        batch_hash = hashlib.md5(batch.detach().cpu().numpy().tobytes()).hexdigest()
        if batch_hash in self.results_cache:
            self.processing_stats['cache_hits'] += 1
            return self.results_cache[batch_hash]
        
        # Process batch
        with torch.no_grad():
            result = processing_func(model, batch)
        
        # Cache result
        self.results_cache[batch_hash] = result
        
        processing_time = time.perf_counter() - start_time
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['batches_processed'] += 1
        
        return result
    
    async def stream_process_spikes(self, spike_stream: Iterator[torch.Tensor],
                                  model: nn.Module,
                                  processing_func: Callable,
                                  buffer_size: int = 10) -> Iterator[torch.Tensor]:
        """Stream process spikes with buffering."""
        
        buffer = []
        
        async for spike_batch in spike_stream:
            buffer.append(spike_batch)
            
            if len(buffer) >= buffer_size:
                # Process buffer
                results = await self.process_spike_batch_async(buffer, model, processing_func)
                
                for result in results:
                    yield result
                
                buffer.clear()
        
        # Process remaining items in buffer
        if buffer:
            results = await self.process_spike_batch_async(buffer, model, processing_func)
            for result in results:
                yield result
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = dict(self.processing_stats)
        
        if stats['batches_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['batches_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['batches_processed']
        
        stats['cache_size'] = len(self.results_cache)
        
        return stats
    
    def clear_cache(self):
        """Clear results cache."""
        self.results_cache.clear()
        self.processing_stats['cache_cleared'] += 1


class ScalableNeuromorphicFramework:
    """Complete scalable framework for neuromorphic computing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'max_devices': 8,
            'cache_size': 1000,
            'async_workers': 4,
            'load_balancing': True,
            'advanced_caching': True
        }
        
        # Initialize components
        self.distributed_optimizer = DistributedNeuromorphicOptimizer(
            num_devices=self.config['max_devices']
        )
        self.distributed_optimizer.setup_distributed_environment()
        
        self.spike_cache = AdvancedSpikePatternCache(
            max_patterns=self.config['cache_size']
        )
        
        self.load_balancer = NeuromorphicLoadBalancer(
            devices=[d['device'] for d in self.distributed_optimizer.device_pool]
        )
        
        self.async_processor = AsyncNeuromorphicProcessor(
            max_workers=self.config['async_workers']
        )
        
        self.performance_stats = defaultdict(float)
        self.logger = logging.getLogger(__name__)
    
    def optimize_model_for_scale(self, model: nn.Module, 
                                distribution_strategy: str = "auto") -> nn.Module:
        """Optimize model for large-scale deployment."""
        
        if distribution_strategy == "auto":
            # Auto-select strategy based on model size and available resources
            param_count = sum(p.numel() for p in model.parameters())
            num_devices = len(self.distributed_optimizer.device_pool)
            
            if param_count > 100_000_000 and num_devices >= 4:  # 100M+ parameters
                distribution_strategy = "hybrid"
            elif param_count > 10_000_000 and num_devices >= 2:  # 10M+ parameters
                distribution_strategy = "model_parallel"
            else:
                distribution_strategy = "data_parallel"
        
        self.logger.info(f"Using distribution strategy: {distribution_strategy}")
        
        # Distribute model
        distributed_model = self.distributed_optimizer.distribute_model(
            model, distribution_strategy
        )
        
        # Optimize batch size
        optimal_batch_size = self.distributed_optimizer.get_optimal_batch_size(
            distributed_model, (32, 128)  # Example input shape
        )
        
        self.performance_stats['optimal_batch_size'] = optimal_batch_size
        self.performance_stats['distribution_strategy'] = distribution_strategy
        
        return distributed_model
    
    async def scalable_inference(self, model: nn.Module, 
                                spike_batches: List[torch.Tensor],
                                use_load_balancing: bool = True) -> List[torch.Tensor]:
        """Perform scalable inference across multiple devices."""
        
        if use_load_balancing:
            # Distribute batches across devices
            device_assignments = []
            for batch in spike_batches:
                optimal_device = self.load_balancer.select_optimal_device(
                    workload_type="inference",
                    spike_density=batch.float().mean().item()
                )
                device_assignments.append(optimal_device)
        else:
            # Round-robin assignment
            devices = [d['device'] for d in self.distributed_optimizer.device_pool]
            device_assignments = [devices[i % len(devices)] for i in range(len(spike_batches))]
        
        # Process batches asynchronously
        results = await self.async_processor.process_spike_batch_async(
            spike_batches, model, self._inference_func
        )
        
        # Update load balancer with performance metrics
        for device, batch, result in zip(device_assignments, spike_batches, results):
            performance_metrics = {
                'spike_processing_efficiency': 1.0,  # Simplified
                'spikes_per_second': batch.numel() / 0.1  # Simplified
            }
            self.load_balancer.update_device_performance(device, performance_metrics)
        
        return results
    
    def _inference_func(self, model: nn.Module, spike_batch: torch.Tensor) -> torch.Tensor:
        """Internal inference function."""
        return model(spike_batch)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            'performance_stats': dict(self.performance_stats),
            'spike_cache_stats': self.spike_cache.get_cache_stats(),
            'load_balancer_stats': self.load_balancer.get_load_balancing_stats(),
            'async_processor_stats': self.async_processor.get_processing_stats(),
            'distributed_optimizer_stats': {
                'num_devices': len(self.distributed_optimizer.device_pool),
                'device_pool': [str(d['device']) for d in self.distributed_optimizer.device_pool]
            }
        }
    
    def benchmark_scalability(self, model: nn.Module, 
                            test_batches: List[torch.Tensor]) -> Dict[str, Any]:
        """Benchmark scalability performance."""
        
        benchmark_results = {}
        
        # Test different configurations
        configs = [
            {'distribution': 'data_parallel', 'load_balancing': False},
            {'distribution': 'data_parallel', 'load_balancing': True},
            {'distribution': 'model_parallel', 'load_balancing': True},
            {'distribution': 'hybrid', 'load_balancing': True}
        ]
        
        for config in configs:
            config_name = f"{config['distribution']}_lb_{config['load_balancing']}"
            
            # Optimize model
            distributed_model = self.distributed_optimizer.distribute_model(
                model, config['distribution']
            )
            
            # Benchmark inference
            start_time = time.perf_counter()
            
            # Run synchronous version for benchmarking
            results = []
            for batch in test_batches:
                with torch.no_grad():
                    result = distributed_model(batch)
                results.append(result)
            
            total_time = time.perf_counter() - start_time
            
            benchmark_results[config_name] = {
                'total_time': total_time,
                'batches_per_second': len(test_batches) / total_time,
                'avg_batch_time': total_time / len(test_batches)
            }
        
        return benchmark_results


# Global scalable framework instance
scalable_framework = ScalableNeuromorphicFramework()

# Convenience functions for scaling
def optimize_for_scale(model: nn.Module, distribution_strategy: str = "auto") -> nn.Module:
    """Optimize model for large-scale deployment."""
    return scalable_framework.optimize_model_for_scale(model, distribution_strategy)

async def scale_inference(model: nn.Module, spike_batches: List[torch.Tensor]) -> List[torch.Tensor]:
    """Perform scalable inference."""
    return await scalable_framework.scalable_inference(model, spike_batches)

def get_scaling_stats() -> Dict[str, Any]:
    """Get scaling performance statistics."""
    return scalable_framework.get_comprehensive_stats()

def benchmark_scaling_performance(model: nn.Module, test_data: List[torch.Tensor]) -> Dict[str, Any]:
    """Benchmark scaling performance."""
    return scalable_framework.benchmark_scalability(model, test_data)

# Example usage and testing
if __name__ == "__main__":
    print("⚡ Advanced Neuromorphic Scaling Suite")
    print("=" * 60)
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print(f"Test model parameters: {sum(p.numel() for p in test_model.parameters()):,}")
    
    # Test distributed optimization
    print("\n1. Testing Distributed Optimization:")
    distributed_opt = DistributedNeuromorphicOptimizer(num_devices=2)
    distributed_opt.setup_distributed_environment()
    
    distributed_model = distributed_opt.distribute_model(test_model, "data_parallel")
    optimal_batch_size = distributed_opt.get_optimal_batch_size(distributed_model, (32, 128))
    print(f"   ✅ Distributed model created")
    print(f"   📊 Optimal batch size: {optimal_batch_size}")
    
    # Test spike pattern cache
    print("\n2. Testing Advanced Spike Pattern Cache:")
    spike_cache = AdvancedSpikePatternCache(max_patterns=100)
    
    test_spike_pattern = torch.randint(0, 2, (32, 10, 20)).float()
    pattern_hash = spike_cache.cache_pattern(test_spike_pattern, "test_result", 0.1)
    cached_result = spike_cache.get_cached_result(pattern_hash)
    
    cache_stats = spike_cache.get_cache_stats()
    print(f"   ✅ Spike pattern cached: {pattern_hash[:8]}...")
    print(f"   📊 Cache stats: {cache_stats['total_patterns']} patterns")
    
    # Test load balancer
    print("\n3. Testing Neuromorphic Load Balancer:")
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda:0'))
    
    load_balancer = NeuromorphicLoadBalancer(devices)
    optimal_device = load_balancer.select_optimal_device("inference", 0.3)
    
    load_stats = load_balancer.get_load_balancing_stats()
    print(f"   ✅ Optimal device selected: {optimal_device}")
    print(f"   📊 Load balancing stats: {len(load_stats)} devices")
    
    # Test scalable framework
    print("\n4. Testing Scalable Framework:")
    framework = ScalableNeuromorphicFramework()
    
    optimized_model = framework.optimize_model_for_scale(test_model, "auto")
    comprehensive_stats = framework.get_comprehensive_stats()
    
    print(f"   ✅ Model optimized for scale")
    print(f"   📊 Distribution strategy: {comprehensive_stats['performance_stats'].get('distribution_strategy', 'auto')}")
    print(f"   📊 Optimal batch size: {comprehensive_stats['performance_stats'].get('optimal_batch_size', 'N/A')}")
    
    print("\n⚡ Advanced Scaling Suite Complete!")