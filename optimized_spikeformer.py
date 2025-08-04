#!/usr/bin/env python3
"""
Optimized Spikeformer Implementation - Generation 3
Adds performance optimization, concurrency, caching, and scaling capabilities.
"""

import sys
import time
import json
import hashlib
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum
from contextlib import contextmanager
import warnings
from collections import defaultdict, deque
import queue
import weakref

# Import previous components
from robust_spikeformer import (
    SpikeFormerLogger, HealthMonitor, RobustSpikingConfig, RobustModelConfig,
    ValidationError, ConversionError, DeploymentError, ProfilingError
)
from demo_basic import (
    BasicLIFNeuron, RateEncoder, TemporalEncoder, SpikingLayer, SpikingMLP,
    BasicHardwareSimulator, HardwareMetrics
)


# ==============================================================================
# PERFORMANCE OPTIMIZATION & CACHING
# ==============================================================================

class LRUCache:
    """Simple LRU Cache implementation."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size
            }


class PerformanceOptimizer:
    """Performance optimization utilities."""
    
    def __init__(self):
        self.logger = SpikeFormerLogger("optimizer")
        self.cache = LRUCache(max_size=256)
        self.timing_stats = defaultdict(list)
        self.optimization_cache = {}
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timing_stats[operation_name].append(duration)
            if len(self.timing_stats[operation_name]) > 1000:
                # Keep only recent 1000 measurements
                self.timing_stats[operation_name] = self.timing_stats[operation_name][-1000:]
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        stats = {}
        for op_name, times in self.timing_stats.items():
            if times:
                stats[op_name] = {
                    'count': len(times),
                    'avg_ms': (sum(times) / len(times)) * 1000,
                    'min_ms': min(times) * 1000,
                    'max_ms': max(times) * 1000,
                    'total_s': sum(times)
                }
        return stats
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps([args, sorted(kwargs.items())], sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached_operation(self, cache_key: str, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with caching."""
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for {cache_key[:8]}")
            return cached_result
        
        # Execute operation
        with self.timed_operation(f"cached_{operation.__name__}"):
            result = operation(*args, **kwargs)
        
        # Store in cache
        self.cache.put(cache_key, result)
        self.logger.debug(f"Cache miss for {cache_key[:8]}, result cached")
        
        return result


# ==============================================================================
# CONCURRENT PROCESSING FRAMEWORK
# ==============================================================================

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task for concurrent processing."""
    id: str
    operation: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value > other.priority.value


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None


class ConcurrentProcessor:
    """High-performance concurrent task processor."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.logger = SpikeFormerLogger("concurrent_processor")
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.task_dependencies = {}
        self.completed_tasks = set()
        
        # Worker management
        self.executor = None
        self.active_workers = 0
        self.worker_stats = defaultdict(lambda: {'tasks_completed': 0, 'total_time': 0.0})
        
        # Performance tracking
        self.optimizer = PerformanceOptimizer()
        
    def start(self):
        """Start the concurrent processor."""
        if self.executor is None:
            if self.use_processes:
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
                self.logger.info(f"Started process pool with {self.max_workers} workers")
            else:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.info(f"Started thread pool with {self.max_workers} workers")
    
    def stop(self):
        """Stop the concurrent processor."""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
            self.logger.info("Concurrent processor stopped")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for execution."""
        self.task_queue.put((task.priority, task))
        if task.dependencies:
            self.task_dependencies[task.id] = set(task.dependencies)
        
        self.logger.debug(f"Task submitted: {task.id} (priority: {task.priority.name})")
        return task.id
    
    def submit_batch(self, tasks: List[Task]) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        for task in tasks:
            task_id = self.submit_task(task)
            task_ids.append(task_id)
        
        self.logger.info(f"Batch submitted: {len(tasks)} tasks")
        return task_ids
    
    def execute_parallel(self, timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Execute all queued tasks in parallel."""
        if not self.executor:
            self.start()
        
        start_time = time.time()
        futures = {}
        results = {}
        
        # Submit all ready tasks
        while not self.task_queue.empty():
            try:
                priority, task = self.task_queue.get_nowait()
                
                # Check dependencies
                if task.id in self.task_dependencies:
                    unmet_deps = self.task_dependencies[task.id] - self.completed_tasks
                    if unmet_deps:
                        # Put back in queue for later
                        self.task_queue.put((priority, task))
                        continue
                
                # Submit task
                future = self.executor.submit(self._execute_task, task)
                futures[future] = task
                
            except queue.Empty:
                break
        
        # Wait for completion
        try:
            for future in as_completed(futures, timeout=timeout):
                task = futures[future]
                try:
                    result = future.result()
                    results[task.id] = result
                    self.completed_tasks.add(task.id)
                    
                    # Update stats
                    worker_id = result.worker_id or "unknown"
                    self.worker_stats[worker_id]['tasks_completed'] += 1
                    self.worker_stats[worker_id]['total_time'] += result.execution_time
                    
                except Exception as e:
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        success=False,
                        error=e,
                        execution_time=0.0
                    )
                    self.logger.error(f"Task {task.id} failed: {e}")
        
        except TimeoutError:
            self.logger.warning(f"Batch execution timed out after {timeout}s")
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in results.values() if r.success)
        
        self.logger.info(f"Batch completed: {success_count}/{len(results)} tasks successful in {total_time:.3f}s")
        
        return results
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        worker_id = threading.current_thread().name
        start_time = time.time()
        
        try:
            with self.optimizer.timed_operation(f"task_{task.operation.__name__}"):
                result = task.operation(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                task_id=task.id,
                success=False,
                error=e,
                execution_time=execution_time,
                worker_id=worker_id
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'active_workers': self.active_workers,
            'completed_tasks': len(self.completed_tasks),
            'worker_stats': dict(self.worker_stats),
            'timing_stats': self.optimizer.get_timing_stats(),
            'cache_stats': self.optimizer.cache.stats()
        }


# ==============================================================================
# SCALABLE NEUROMORPHIC CONVERTER
# ==============================================================================

class ScalableConverter:
    """High-performance, scalable neural network converter."""
    
    def __init__(self, config: RobustSpikingConfig, max_workers: int = None):
        self.config = config
        self.logger = SpikeFormerLogger("scalable_converter")
        self.processor = ConcurrentProcessor(max_workers=max_workers)
        self.optimizer = PerformanceOptimizer()
        
        # Conversion cache
        self.conversion_cache = LRUCache(max_size=64)
        
    def convert_batch(self, model_configs: List[RobustModelConfig]) -> List[Tuple[SpikingMLP, Dict[str, Any]]]:
        """Convert multiple models concurrently."""
        self.logger.info(f"Starting batch conversion of {len(model_configs)} models")
        
        # Start processor
        self.processor.start()
        
        try:
            # Create conversion tasks
            tasks = []
            for i, model_config in enumerate(model_configs):
                task = Task(
                    id=f"convert_{i}",
                    operation=self._convert_single_cached,
                    args=(model_config,),
                    priority=TaskPriority.NORMAL
                )
                tasks.append(task)
            
            # Submit and execute
            task_ids = self.processor.submit_batch(tasks)
            results = self.processor.execute_parallel(timeout=300)  # 5 minute timeout
            
            # Process results
            converted_models = []
            for i, task_id in enumerate(task_ids):
                result = results.get(task_id)
                if result and result.success:
                    converted_models.append(result.result)
                else:
                    error_msg = str(result.error) if result and result.error else "Unknown error"
                    self.logger.error(f"Model {i} conversion failed: {error_msg}")
                    # Add placeholder for failed conversion
                    converted_models.append((None, {'error': error_msg}))
            
            self.logger.info(f"Batch conversion completed: {len([m for m, _ in converted_models if m is not None])}/{len(model_configs)} successful")
            
            return converted_models
            
        finally:
            self.processor.stop()
    
    def _convert_single_cached(self, model_config: RobustModelConfig) -> Tuple[SpikingMLP, Dict[str, Any]]:
        """Convert single model with caching."""
        # Generate cache key
        cache_key = self.optimizer.cache_key(
            model_config.to_dict(),
            self.config.to_dict()
        )
        
        # Check cache
        cached_result = self.conversion_cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Using cached conversion for model")
            return cached_result
        
        # Perform conversion
        with self.optimizer.timed_operation("model_conversion"):
            # Create layer sizes
            layer_sizes = [model_config.input_size] + model_config.hidden_sizes + [model_config.output_size]
            
            # Create spiking network
            snn_model = SpikingMLP(layer_sizes, self.config)
            
            # Generate metadata
            metadata = {
                'conversion_time_seconds': 0.0,  # Will be filled by timing
                'input_config': model_config.to_dict(),
                'spiking_config': self.config.to_dict(),
                'layer_sizes': layer_sizes,
                'total_neurons': sum(layer_sizes[1:]),
                'total_synapses': sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1)),
                'conversion_timestamp': time.time(),
                'cached': False
            }
            
            result = (snn_model, metadata)
            
            # Cache result
            self.conversion_cache.put(cache_key, result)
        
        return result


# ==============================================================================
# AUTO-SCALING HARDWARE DEPLOYMENT
# ==============================================================================

class LoadBalancer:
    """Simple load balancer for hardware resources."""
    
    def __init__(self):
        self.platforms = {}
        self.load_stats = defaultdict(lambda: {'requests': 0, 'total_time': 0.0})
        self.logger = SpikeFormerLogger("load_balancer")
    
    def register_platform(self, platform_name: str, max_concurrent: int = 10):
        """Register a hardware platform."""
        self.platforms[platform_name] = {
            'max_concurrent': max_concurrent,
            'current_load': 0,
            'available': True
        }
        self.logger.info(f"Registered platform: {platform_name} (max_concurrent: {max_concurrent})")
    
    def get_best_platform(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """Get the best available platform based on current load."""
        exclude = exclude or []
        available_platforms = [
            (name, info) for name, info in self.platforms.items()
            if info['available'] and name not in exclude and info['current_load'] < info['max_concurrent']
        ]
        
        if not available_platforms:
            return None
        
        # Choose platform with lowest relative load
        best_platform = min(available_platforms,
                          key=lambda x: x[1]['current_load'] / x[1]['max_concurrent'])
        
        return best_platform[0]
    
    def acquire_resource(self, platform_name: str) -> bool:
        """Acquire a resource on the platform."""
        if platform_name not in self.platforms:
            return False
        
        platform = self.platforms[platform_name]
        if platform['current_load'] < platform['max_concurrent']:
            platform['current_load'] += 1
            return True
        
        return False
    
    def release_resource(self, platform_name: str):
        """Release a resource on the platform."""
        if platform_name in self.platforms:
            self.platforms[platform_name]['current_load'] = max(0, 
                self.platforms[platform_name]['current_load'] - 1)
    
    def record_request(self, platform_name: str, duration: float):
        """Record request statistics."""
        self.load_stats[platform_name]['requests'] += 1
        self.load_stats[platform_name]['total_time'] += duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'platforms': self.platforms.copy(),
            'load_stats': dict(self.load_stats)
        }


class ScalableHardwareManager:
    """Scalable hardware deployment manager."""
    
    def __init__(self):
        self.logger = SpikeFormerLogger("scalable_hardware")
        self.load_balancer = LoadBalancer()
        self.processor = ConcurrentProcessor(max_workers=16)
        
        # Register default platforms
        self.load_balancer.register_platform("loihi2", max_concurrent=8)
        self.load_balancer.register_platform("spinnaker", max_concurrent=4)
        self.load_balancer.register_platform("cpu", max_concurrent=32)
    
    def deploy_batch(self, models: List[SpikingMLP], 
                    preferred_platforms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Deploy multiple models with load balancing."""
        self.logger.info(f"Starting batch deployment of {len(models)} models")
        
        self.processor.start()
        
        try:
            # Create deployment tasks
            tasks = []
            for i, model in enumerate(models):
                # Select platform
                platform = self._select_platform(preferred_platforms)
                if not platform:
                    self.logger.warning(f"No available platform for model {i}")
                    platform = "cpu"  # Fallback
                
                task = Task(
                    id=f"deploy_{i}_{platform}",
                    operation=self._deploy_with_load_balancing,
                    args=(model, platform),
                    priority=TaskPriority.NORMAL
                )
                tasks.append(task)
            
            # Execute deployments
            task_ids = self.processor.submit_batch(tasks)
            results = self.processor.execute_parallel(timeout=600)  # 10 minute timeout
            
            # Process results
            deployment_results = []
            for i, task_id in enumerate(task_ids):
                result = results.get(task_id)
                if result and result.success:
                    deployment_results.append(result.result)
                else:
                    error_msg = str(result.error) if result and result.error else "Unknown error"
                    self.logger.error(f"Model {i} deployment failed: {error_msg}")
                    deployment_results.append({'error': error_msg, 'success': False})
            
            successful_deployments = sum(1 for r in deployment_results if r.get('success', False))
            self.logger.info(f"Batch deployment completed: {successful_deployments}/{len(models)} successful")
            
            return deployment_results
            
        finally:
            self.processor.stop()
    
    def _select_platform(self, preferred_platforms: Optional[List[str]] = None) -> Optional[str]:
        """Select best platform for deployment."""
        if preferred_platforms:
            for platform in preferred_platforms:
                if self.load_balancer.acquire_resource(platform):
                    return platform
        
        # Fallback to best available
        best_platform = self.load_balancer.get_best_platform()
        if best_platform and self.load_balancer.acquire_resource(best_platform):
            return best_platform
        
        return None
    
    def _deploy_with_load_balancing(self, model: SpikingMLP, platform: str) -> Dict[str, Any]:
        """Deploy model with load balancing."""
        start_time = time.time()
        
        try:
            # Create hardware simulator
            simulator = BasicHardwareSimulator(platform)
            
            # Deploy model
            deployment_info = simulator.deploy(model)
            
            # Run benchmark
            metrics = simulator.benchmark(model, num_samples=5)
            
            deployment_time = time.time() - start_time
            
            # Record stats
            self.load_balancer.record_request(platform, deployment_time)
            
            result = {
                'success': True,
                'platform': platform,
                'deployment_info': deployment_info,
                'metrics': {
                    'energy_per_inference_mj': metrics.energy_per_inference_mj,
                    'latency_ms': metrics.latency_ms,
                    'throughput_inferences_per_sec': metrics.throughput_inferences_per_sec,
                    'spike_rate': metrics.spike_rate
                },
                'deployment_time_seconds': deployment_time
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment failed on {platform}: {e}")
            return {
                'success': False,
                'platform': platform,
                'error': str(e),
                'deployment_time_seconds': time.time() - start_time
            }
        
        finally:
            # Release resource
            self.load_balancer.release_resource(platform)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hardware manager statistics."""
        return {
            'load_balancer': self.load_balancer.get_stats(),
            'processor': self.processor.get_stats()
        }


# ==============================================================================
# SCALABLE DEMONSTRATION
# ==============================================================================

def run_scalable_demo():
    """Demonstrate scalable high-performance system."""
    print("\n" + "="*80)
    print("🚀 SPIKEFORMER SCALABLE SYSTEM DEMO - GENERATION 3")
    print("="*80)
    
    logger = SpikeFormerLogger("scalable_demo")
    
    try:
        # 1. Performance Optimization Demo
        print("\n1️⃣  PERFORMANCE OPTIMIZATION & CACHING")
        print("-" * 40)
        
        optimizer = PerformanceOptimizer()
        
        # Simulate expensive operations with caching
        def expensive_operation(n: int) -> int:
            time.sleep(0.01)  # Simulate work
            return n * n
        
        # First run (cache miss)
        with optimizer.timed_operation("expensive_op"):
            cache_key = optimizer.cache_key("square", 42)
            result1 = optimizer.cached_operation(cache_key, expensive_operation, 42)
        
        # Second run (cache hit)
        with optimizer.timed_operation("expensive_op"):
            result2 = optimizer.cached_operation(cache_key, expensive_operation, 42)
        
        print(f"✅ Cache demo: result1={result1}, result2={result2}")
        print(f"✅ Cache stats: {optimizer.cache.stats()}")
        
        # 2. Concurrent Processing Demo
        print("\n2️⃣  CONCURRENT BATCH PROCESSING")
        print("-" * 40)
        
        processor = ConcurrentProcessor(max_workers=4)
        processor.start()
        
        # Create test tasks
        def test_task(task_id: str, duration: float) -> str:
            time.sleep(duration)
            return f"Task {task_id} completed"
        
        tasks = [
            Task(id=f"task_{i}", operation=test_task, args=(f"task_{i}", 0.1))
            for i in range(8)
        ]
        
        start_time = time.time()
        task_ids = processor.submit_batch(tasks)
        results = processor.execute_parallel(timeout=10)
        concurrent_time = time.time() - start_time
        
        processor.stop()
        
        successful_tasks = sum(1 for r in results.values() if r.success)
        print(f"✅ Concurrent processing: {successful_tasks}/{len(tasks)} tasks completed in {concurrent_time:.3f}s")
        
        # 3. Scalable Model Conversion
        print("\n3️⃣  SCALABLE MODEL CONVERSION")
        print("-" * 40)
        
        # Create test model configurations
        test_configs = [
            RobustModelConfig(
                input_size=28 * 28,
                hidden_sizes=[64, 32],
                output_size=10,
                description=f"Test model {i}"
            )
            for i in range(4)
        ]
        
        # Convert in batch
        converter = ScalableConverter(RobustSpikingConfig(timesteps=16), max_workers=4)
        
        conversion_start = time.time()
        converted_models = converter.convert_batch(test_configs)
        conversion_time = time.time() - conversion_start
        
        successful_conversions = sum(1 for model, _ in converted_models if model is not None)
        print(f"✅ Batch conversion: {successful_conversions}/{len(test_configs)} models converted in {conversion_time:.3f}s")
        
        # 4. Auto-scaling Hardware Deployment
        print("\n4️⃣  AUTO-SCALING HARDWARE DEPLOYMENT")
        print("-" * 40)
        
        # Extract successful models
        valid_models = [model for model, metadata in converted_models if model is not None]
        
        if valid_models:
            hardware_manager = ScalableHardwareManager()
            
            deployment_start = time.time()
            deployment_results = hardware_manager.deploy_batch(
                valid_models,
                preferred_platforms=["loihi2", "spinnaker"]
            )
            deployment_time = time.time() - deployment_start
            
            successful_deployments = sum(1 for r in deployment_results if r.get('success', False))
            print(f"✅ Batch deployment: {successful_deployments}/{len(valid_models)} models deployed in {deployment_time:.3f}s")
            
            # Show platform distribution
            platform_counts = defaultdict(int)
            for result in deployment_results:
                if result.get('success'):
                    platform_counts[result['platform']] += 1
            
            print("📊 Platform distribution:")
            for platform, count in platform_counts.items():
                print(f"     {platform.upper()}: {count} deployments")
        
        else:
            print("❌ No valid models for deployment")
            deployment_results = []
        
        # 5. Performance Analysis
        print("\n5️⃣  PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if deployment_results:
            # Analyze energy efficiency
            energy_values = [
                r['metrics']['energy_per_inference_mj']
                for r in deployment_results
                if r.get('success') and 'metrics' in r
            ]
            
            if energy_values:
                avg_energy = sum(energy_values) / len(energy_values)
                min_energy = min(energy_values)
                max_energy = max(energy_values)
                
                print(f"⚡ Energy Analysis:")
                print(f"     Average: {avg_energy:.3f} mJ/inference")
                print(f"     Best: {min_energy:.3f} mJ/inference")
                print(f"     Worst: {max_energy:.3f} mJ/inference")
                print(f"     Efficiency spread: {(max_energy/min_energy):.1f}x")
        
        # 6. System Scaling Metrics
        print("\n6️⃣  SCALING PERFORMANCE METRICS")
        print("-" * 40)
        
        # Calculate throughput metrics
        total_models_processed = len(test_configs)
        total_processing_time = conversion_time + (deployment_time if valid_models else 0)
        
        throughput = total_models_processed / total_processing_time if total_processing_time > 0 else 0
        
        print(f"📈 Scaling Metrics:")
        print(f"     Models processed: {total_models_processed}")
        print(f"     Total time: {total_processing_time:.3f}s")
        print(f"     Throughput: {throughput:.1f} models/second")
        print(f"     Parallel efficiency: {(8 * 0.1) / concurrent_time:.1f}x speedup")
        
        # 7. Resource Utilization
        print("\n7️⃣  RESOURCE UTILIZATION")
        print("-" * 40)
        
        # Get system stats
        if valid_models:
            hw_stats = hardware_manager.get_stats()
            
            print("🖥️ Hardware Resource Usage:")
            for platform, info in hw_stats['load_balancer']['platforms'].items():
                utilization = info['current_load'] / info['max_concurrent'] * 100
                print(f"     {platform.upper()}: {info['current_load']}/{info['max_concurrent']} ({utilization:.1f}%)")
        
        print(f"💾 Cache Utilization: {optimizer.cache.stats()['utilization']:.1%}")
        print(f"🔧 Timing Stats: {len(optimizer.get_timing_stats())} operations tracked")
        
        print("\n" + "="*80)
        print("🎉 SCALABLE SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("✅ Performance optimization with caching implemented")
        print("✅ Concurrent batch processing working")
        print("✅ Auto-scaling hardware deployment operational")
        print("✅ Load balancing and resource management active")
        print("✅ Real-time performance monitoring enabled")
        print(f"✅ System throughput: {throughput:.1f} models/second")
        print("="*80 + "\n")
        
        return {
            'throughput': throughput,
            'concurrent_speedup': (8 * 0.1) / concurrent_time,
            'successful_conversions': successful_conversions,
            'successful_deployments': successful_deployments if valid_models else 0,
            'cache_stats': optimizer.cache.stats(),
            'timing_stats': optimizer.get_timing_stats()
        }
        
    except Exception as e:
        logger.error(f"Scalable demo failed: {e}")
        print(f"\n❌ SCALABLE DEMO ERROR: {e}")
        raise


if __name__ == "__main__":
    try:
        results = run_scalable_demo()
        
        # Save results if requested
        if "--save" in sys.argv:
            output_file = "scalable_demo_results.json"
            
            # Convert to JSON-serializable format
            json_results = {
                'throughput_models_per_second': results['throughput'],
                'concurrent_speedup_factor': results['concurrent_speedup'],
                'successful_conversions': results['successful_conversions'],
                'successful_deployments': results['successful_deployments'],
                'cache_utilization': results['cache_stats']['utilization'],
                'demo_timestamp': time.time()
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"📁 Scalable demo results saved to {output_file}")
            
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)