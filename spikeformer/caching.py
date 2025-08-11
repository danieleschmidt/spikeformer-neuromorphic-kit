"""Advanced caching mechanisms for neuromorphic computing systems."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass
import hashlib
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from functools import wraps, lru_cache
import weakref
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

from .error_handling import safe_execute, ErrorSeverity


@dataclass
class CacheConfig:
    """Configuration for caching systems."""
    max_memory_mb: int = 1024  # 1GB default
    ttl_seconds: int = 3600  # 1 hour default
    max_entries: int = 1000
    enable_compression: bool = True
    enable_async_loading: bool = True
    prefetch_size: int = 10
    cache_hit_threshold: float = 0.1  # Minimum hit rate to maintain entry


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, size_bytes: int = 0, ttl: int = 3600):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.ttl = ttl
        self.compressed = False
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at
    
    def get_value(self) -> Any:
        """Get value with access tracking."""
        self.touch()
        if self.compressed:
            return self._decompress(self.value)
        return self.value
    
    def _compress(self, value: Any) -> bytes:
        """Compress value for storage."""
        import gzip
        serialized = pickle.dumps(value)
        return gzip.compress(serialized)
    
    def _decompress(self, compressed_value: bytes) -> Any:
        """Decompress value from storage."""
        import gzip
        decompressed = gzip.decompress(compressed_value)
        return pickle.loads(decompressed)
    
    def compress(self):
        """Compress the stored value."""
        if not self.compressed and self.value is not None:
            compressed_value = self._compress(self.value)
            self.value = compressed_value
            self.compressed = True
            # Update size estimate
            self.size_bytes = len(compressed_value)


class AdaptiveCache:
    """Adaptive cache with intelligent eviction and prefetching."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.access_patterns = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
        # Background thread for maintenance
        self._maintenance_thread = None
        self._stop_maintenance = False
        self._start_maintenance()
    
    def _start_maintenance(self):
        """Start background maintenance thread."""
        def maintenance_loop():
            while not self._stop_maintenance:
                try:
                    self._cleanup_expired()
                    self._adapt_cache_size()
                    self._compress_old_entries()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    self.logger.error(f"Cache maintenance error: {e}")
        
        self._maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self._maintenance_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None or entry.is_expired():
                self.miss_count += 1
                if entry is not None:
                    self._remove_entry(key)
                return default
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            
            # Track access pattern
            self.access_patterns[key].append(time.time())
            
            return entry.get_value()
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, 
            size_hint: Optional[int] = None) -> bool:
        """Put value into cache."""
        if ttl is None:
            ttl = self.config.ttl_seconds
        
        # Estimate size
        size_bytes = size_hint or self._estimate_size(value)
        
        # Check if value is too large
        if size_bytes > self.config.max_memory_mb * 1024 * 1024:
            self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
            return False
        
        with self.lock:
            # Remove existing entry
            if key in self.cache:
                self._remove_entry(key)
            
            # Evict if necessary
            while (len(self.cache) >= self.config.max_entries or
                   self.total_size + size_bytes > self.config.max_memory_mb * 1024 * 1024):
                if not self._evict_entry():
                    break
            
            # Add new entry
            entry = CacheEntry(key, value, size_bytes, ttl)
            if self.config.enable_compression and size_bytes > 1024:  # Compress large entries
                entry.compress()
            
            self.cache[key] = entry
            self.total_size += entry.size_bytes
            
            return True
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if torch.is_tensor(value):
                return value.numel() * value.element_size()
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        entry = self.cache.pop(key, None)
        if entry:
            self.total_size -= entry.size_bytes
    
    def _evict_entry(self) -> bool:
        """Evict least useful entry."""
        if not self.cache:
            return False
        
        # Score entries for eviction
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Factors: recency, frequency, size, age
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = entry.access_count
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            age_penalty = entry.get_age() / 3600  # Hours
            
            # Lower score = more likely to evict
            scores[key] = recency_score * frequency_score / (size_penalty + age_penalty + 1)
        
        # Evict lowest scoring entry
        victim_key = min(scores.keys(), key=lambda k: scores[k])
        self._remove_entry(victim_key)
        
        return True
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
    
    def _adapt_cache_size(self):
        """Adapt cache size based on hit rate."""
        total_accesses = self.hit_count + self.miss_count
        if total_accesses > 100:  # Need sufficient data
            hit_rate = self.hit_count / total_accesses
            
            # Adjust max entries based on hit rate
            if hit_rate > 0.8:  # High hit rate, can grow
                self.config.max_entries = min(self.config.max_entries * 1.1, 10000)
            elif hit_rate < 0.3:  # Low hit rate, should shrink
                self.config.max_entries = max(self.config.max_entries * 0.9, 100)
    
    def _compress_old_entries(self):
        """Compress old, infrequently accessed entries."""
        if not self.config.enable_compression:
            return
        
        with self.lock:
            current_time = time.time()
            for entry in self.cache.values():
                # Compress entries older than 10 minutes with low access
                if (not entry.compressed and 
                    current_time - entry.last_accessed > 600 and
                    entry.access_count < 5):
                    try:
                        entry.compress()
                    except Exception as e:
                        self.logger.warning(f"Failed to compress cache entry: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self.hit_count + self.miss_count
        return {
            'entries': len(self.cache),
            'total_size_mb': self.total_size / (1024 * 1024),
            'hit_rate': self.hit_count / total_accesses if total_accesses > 0 else 0,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'max_entries': self.config.max_entries,
            'max_memory_mb': self.config.max_memory_mb
        }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.total_size = 0
            self.access_patterns.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        self._stop_maintenance = True
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=1)


class TensorCache(AdaptiveCache):
    """Specialized cache for PyTorch tensors."""
    
    def __init__(self, config: CacheConfig, device: str = 'cpu'):
        super().__init__(config)
        self.device = device
    
    def put_tensor(self, key: str, tensor: torch.Tensor, ttl: Optional[int] = None) -> bool:
        """Put tensor into cache with device handling."""
        # Move to CPU for caching if needed
        if tensor.device.type != 'cpu' and self.device == 'cpu':
            cached_tensor = tensor.cpu()
        else:
            cached_tensor = tensor.clone()
        
        return self.put(key, cached_tensor, ttl, tensor.numel() * tensor.element_size())
    
    def get_tensor(self, key: str, target_device: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get tensor from cache with device handling."""
        tensor = self.get(key)
        if tensor is None:
            return None
        
        # Move to target device
        if target_device and tensor.device.type != target_device:
            return tensor.to(target_device)
        
        return tensor


class ModelStateCache:
    """Cache for model states and intermediate computations."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.state_cache = AdaptiveCache(config)
        self.gradient_cache = AdaptiveCache(config)
        self.activation_cache = TensorCache(config)
    
    def cache_model_state(self, model: nn.Module, epoch: int, step: int) -> str:
        """Cache model state dict."""
        key = f"model_state_{epoch}_{step}"
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.state_cache.put(key, state_dict)
        return key
    
    def restore_model_state(self, model: nn.Module, key: str) -> bool:
        """Restore model state from cache."""
        state_dict = self.state_cache.get(key)
        if state_dict is None:
            return False
        
        # Move to model's device
        device = next(model.parameters()).device
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return True
    
    def cache_gradients(self, model: nn.Module, key: str):
        """Cache model gradients."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        
        self.gradient_cache.put(key, gradients)
    
    def restore_gradients(self, model: nn.Module, key: str) -> bool:
        """Restore gradients from cache."""
        gradients = self.gradient_cache.get(key)
        if gradients is None:
            return False
        
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            if name in gradients:
                param.grad = gradients[name].to(device)
        
        return True
    
    def cache_activations(self, layer_name: str, activations: torch.Tensor) -> str:
        """Cache layer activations."""
        key = f"activations_{layer_name}_{hash(activations.data_ptr())}"
        self.activation_cache.put_tensor(key, activations)
        return key
    
    def get_activations(self, key: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """Get cached activations."""
        return self.activation_cache.get_tensor(key, device)


class ComputationCache:
    """Cache for expensive computations in neuromorphic models."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.computation_cache = AdaptiveCache(config)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _compute_hash(self, *args, **kwargs) -> str:
        """Compute hash for cache key."""
        hasher = hashlib.sha256()
        
        for arg in args:
            if torch.is_tensor(arg):
                hasher.update(arg.cpu().numpy().tobytes())
            else:
                hasher.update(str(arg).encode())
        
        for key, value in sorted(kwargs.items()):
            hasher.update(key.encode())
            if torch.is_tensor(value):
                hasher.update(value.cpu().numpy().tobytes())
            else:
                hasher.update(str(value).encode())
        
        return hasher.hexdigest()
    
    def cached_computation(self, func: Callable, ttl: int = 3600):
        """Decorator for caching expensive computations."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            cache_key = f"{func.__name__}_{self._compute_hash(*args, **kwargs)}"
            
            # Try to get from cache
            result = self.computation_cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            self.computation_cache.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    def async_compute_and_cache(self, func: Callable, cache_key: str, 
                               *args, **kwargs):
        """Asynchronously compute and cache result."""
        def compute_and_store():
            try:
                result = func(*args, **kwargs)
                self.computation_cache.put(cache_key, result)
            except Exception as e:
                logging.error(f"Async computation failed: {e}")
        
        self.executor.submit(compute_and_store)
    
    def warmup_cache(self, computation_configs: List[Dict[str, Any]]):
        """Warm up cache with common computations."""
        futures = []
        
        for config in computation_configs:
            func = config['function']
            args = config.get('args', [])
            kwargs = config.get('kwargs', {})
            cache_key = f"{func.__name__}_{self._compute_hash(*args, **kwargs)}"
            
            # Skip if already cached
            if self.computation_cache.get(cache_key) is not None:
                continue
            
            future = self.executor.submit(func, *args, **kwargs)
            futures.append((future, cache_key))
        
        # Wait for completion and cache results
        for future, cache_key in futures:
            try:
                result = future.result(timeout=60)
                self.computation_cache.put(cache_key, result)
            except Exception as e:
                logging.error(f"Warmup computation failed for {cache_key}: {e}")


class MemoryAwareCache:
    """Memory-aware cache that monitors system resources."""
    
    def __init__(self, config: CacheConfig, memory_limit_ratio: float = 0.8):
        self.config = config
        self.memory_limit_ratio = memory_limit_ratio
        self.cache = AdaptiveCache(config)
        self._monitor_thread = None
        self._stop_monitor = False
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start memory monitoring thread."""
        def monitor_loop():
            while not self._stop_monitor:
                try:
                    self._check_memory_pressure()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logging.error(f"Memory monitoring error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _check_memory_pressure(self):
        """Check memory pressure and adjust cache."""
        import psutil
        
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        if memory_usage > self.memory_limit_ratio:
            # Aggressive cache cleanup
            self._emergency_cleanup()
        elif memory_usage > 0.7:
            # Moderate cleanup
            self._reduce_cache_size(0.8)
    
    def _emergency_cleanup(self):
        """Emergency cache cleanup to free memory."""
        with self.cache.lock:
            # Remove 50% of entries, starting with least recently used
            target_size = len(self.cache.cache) // 2
            
            # Sort by last access time
            sorted_entries = sorted(
                self.cache.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            for key, _ in sorted_entries[:target_size]:
                self.cache._remove_entry(key)
    
    def _reduce_cache_size(self, factor: float):
        """Reduce cache size by given factor."""
        with self.cache.lock:
            target_size = int(len(self.cache.cache) * factor)
            
            while len(self.cache.cache) > target_size:
                if not self.cache._evict_entry():
                    break
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get from cache with memory awareness."""
        return self.cache.get(key, default)
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put into cache with memory awareness."""
        # Check memory before adding
        import psutil
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        if memory_usage > self.memory_limit_ratio:
            return False  # Skip caching under memory pressure
        
        return self.cache.put(key, value, ttl)
    
    def __del__(self):
        """Cleanup monitoring."""
        self._stop_monitor = True
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1)


# Global cache instances
_global_tensor_cache = None
_global_computation_cache = None
_global_model_cache = None


def get_tensor_cache() -> TensorCache:
    """Get global tensor cache instance."""
    global _global_tensor_cache
    if _global_tensor_cache is None:
        config = CacheConfig(max_memory_mb=512, max_entries=500)
        _global_tensor_cache = TensorCache(config)
    return _global_tensor_cache


def get_computation_cache() -> ComputationCache:
    """Get global computation cache instance."""
    global _global_computation_cache
    if _global_computation_cache is None:
        config = CacheConfig(max_memory_mb=256, max_entries=1000)
        _global_computation_cache = ComputationCache(config)
    return _global_computation_cache


def get_model_cache() -> ModelStateCache:
    """Get global model state cache instance."""
    global _global_model_cache
    if _global_model_cache is None:
        config = CacheConfig(max_memory_mb=1024, max_entries=100)
        _global_model_cache = ModelStateCache(config)
    return _global_model_cache


def cached(ttl: int = 3600, cache_type: str = 'computation'):
    """Decorator for caching function results."""
    def decorator(func):
        if cache_type == 'computation':
            cache = get_computation_cache()
            return cache.cached_computation(func, ttl)
        else:
            # Default to simple LRU cache
            return lru_cache(maxsize=128)(func)
    
    return decorator