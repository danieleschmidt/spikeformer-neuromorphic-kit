"""Advanced concurrency and parallelization for neuromorphic computing."""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass
import threading
import asyncio
import concurrent.futures
import queue
import time
import logging
import os
import signal
from abc import ABC, abstractmethod
from contextlib import contextmanager
import functools

from .caching import get_tensor_cache, get_computation_cache
from .error_handling import safe_execute, ErrorSeverity


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency settings."""
    max_workers: int = 4
    batch_size_per_worker: int = 32
    enable_data_parallelism: bool = True
    enable_model_parallelism: bool = False
    enable_pipeline_parallelism: bool = False
    async_data_loading: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


class ParallelDataLoader:
    """Advanced parallel data loader with prefetching."""
    
    def __init__(self, dataset, config: ConcurrencyConfig, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.config = config
        self.transform = transform
        self.data_queue = queue.Queue(maxsize=config.prefetch_factor * config.max_workers)
        self.workers = []
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        
    def _worker_fn(self, worker_id: int, indices: List[int]):
        """Worker function for data loading."""
        try:
            for idx in indices:
                if self.stop_event.is_set():
                    break
                
                # Load data
                data = self.dataset[idx]
                
                # Apply transform if provided
                if self.transform:
                    data = self.transform(data)
                
                # Put in queue
                self.data_queue.put((worker_id, idx, data), timeout=10)
                
        except Exception as e:
            self.logger.error(f"Worker {worker_id} error: {e}")
            self.data_queue.put((worker_id, -1, None))  # Error signal
    
    def start_workers(self, indices: List[int]):
        """Start worker threads for data loading."""
        # Split indices among workers
        chunk_size = len(indices) // self.config.max_workers
        worker_indices = [
            indices[i*chunk_size:(i+1)*chunk_size] 
            for i in range(self.config.max_workers)
        ]
        
        # Handle remainder
        remainder = len(indices) % self.config.max_workers
        if remainder > 0:
            worker_indices[-1].extend(indices[-remainder:])
        
        # Start workers
        for worker_id, worker_idx in enumerate(worker_indices):
            if worker_idx:  # Only start if there are indices
                worker = threading.Thread(
                    target=self._worker_fn,
                    args=(worker_id, worker_idx)
                )
                worker.daemon = True
                worker.start()
                self.workers.append(worker)
    
    def __iter__(self):
        """Iterate over data with parallel loading."""
        indices = list(range(len(self.dataset)))
        self.start_workers(indices)
        
        loaded_count = 0
        total_items = len(indices)
        
        while loaded_count < total_items:
            try:
                worker_id, idx, data = self.data_queue.get(timeout=30)
                
                if idx == -1:  # Error signal
                    self.logger.warning(f"Worker {worker_id} encountered error")
                    continue
                
                yield data
                loaded_count += 1
                
            except queue.Empty:
                self.logger.warning("Data queue timeout")
                break
        
        self.stop()
    
    def stop(self):
        """Stop all workers."""
        self.stop_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        # Clear queue
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass
        
        self.workers.clear()


class AsyncSpikingInference:
    """Asynchronous inference for spiking neural networks."""
    
    def __init__(self, model: nn.Module, config: ConcurrencyConfig):
        self.model = model
        self.config = config
        self.inference_queue = asyncio.Queue(maxsize=100)
        self.result_cache = get_computation_cache()
        self.device_pool = self._create_device_pool()
        self.logger = logging.getLogger(__name__)
        
    def _create_device_pool(self) -> List[str]:
        """Create pool of available devices."""
        devices = ['cpu']
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        
        return devices
    
    async def inference_worker(self, device: str):
        """Async worker for inference."""
        model_copy = self.model.to(device) if device != 'cpu' else self.model
        model_copy.# eval() removed for security)
        
        while True:
            try:
                # Get inference request
                request_id, inputs, callback = await self.inference_queue.get()
                
                if request_id is None:  # Shutdown signal
                    break
                
                # Move inputs to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, (list, tuple)):
                    inputs = [x.to(device) if torch.is_tensor(x) else x for x in inputs]
                
                # Perform inference
                with torch.no_grad():
                    outputs = model_copy(inputs)
                    
                    # Move results back to CPU for callback
                    if torch.is_tensor(outputs):
                        outputs = outputs.cpu()
                    elif isinstance(outputs, (list, tuple)):
                        outputs = [x.cpu() if torch.is_tensor(x) else x for x in outputs]
                
                # Call callback with results
                if callback:
                    await callback(request_id, outputs)
                
                self.inference_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Inference worker error on {device}: {e}")
                if callback:
                    await callback(request_id, None)
    
    async def submit_inference(self, inputs: torch.Tensor, 
                             callback: Optional[Callable] = None) -> str:
        """Submit inference request."""
        request_id = f"req_{int(time.time() * 1000000)}"
        await self.inference_queue.put((request_id, inputs, callback))
        return request_id
    
    async def start_workers(self):
        """Start async inference workers."""
        tasks = []
        for device in self.device_pool:
            task = asyncio.create_task(self.inference_worker(device))
            tasks.append(task)
        
        return tasks
    
    async def shutdown_workers(self, tasks: List[asyncio.Task]):
        """Shutdown inference workers."""
        # Send shutdown signals
        for _ in self.device_pool:
            await self.inference_queue.put((None, None, None))
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)


class ParallelSpikingTrainer:
    """Parallel training system for spiking neural networks."""
    
    def __init__(self, model: nn.Module, config: ConcurrencyConfig, world_size: int = 1):
        self.model = model
        self.config = config
        self.world_size = world_size
        self.rank = 0
        self.local_rank = 0
        self.device = 'cpu'
        self.ddp_model = None
        self.logger = logging.getLogger(__name__)
    
    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training."""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Set device
        if torch.cuda.is_available():
            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = 'cpu'
        
        # Initialize process group
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        # Wrap model with DDP
        self.model = self.model.to(self.device)
        self.ddp_model = DDP(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None)
    
    def train_step_parallel(self, batch_data: torch.Tensor, labels: torch.Tensor,
                          optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """Parallel training step."""
        batch_data = batch_data.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.ddp_model(batch_data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient synchronization happens automatically in DDP
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            accuracy = (pred == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def cleanup_distributed(self):
        """Cleanup distributed training."""
        if self.world_size > 1:
            destroy_process_group()


class PipelineParallelModel(nn.Module):
    """Pipeline parallel implementation for large spiking models."""
    
    def __init__(self, layers: List[nn.Module], devices: List[str], chunk_size: int = 32):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.devices = devices
        self.chunk_size = chunk_size
        self.num_stages = len(layers)
        
        # Distribute layers across devices
        for i, (layer, device) in enumerate(zip(self.layers, self.devices)):
            layer.to(device)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with pipeline parallelism."""
        # Split input into chunks for pipelining
        chunks = torch.split(inputs, self.chunk_size, dim=0)
        
        # Process chunks in pipeline
        outputs = []
        pipeline_cache = {}
        
        for chunk_idx, chunk in enumerate(chunks):
            current_input = chunk
            
            for stage_idx, (layer, device) in enumerate(zip(self.layers, self.devices)):
                current_input = current_input.to(device)
                
                # Forward through current stage
                with torch.cuda.device(device) if 'cuda' in device else torch.no_grad():
                    current_input = layer(current_input)
                
                # Cache intermediate results for next chunk overlap
                cache_key = f"stage_{stage_idx}_chunk_{chunk_idx}"
                pipeline_cache[cache_key] = current_input.detach()
            
            outputs.append(current_input)
        
        # Concatenate outputs
        return torch.cat(outputs, dim=0)


class WorkerPool:
    """Advanced worker pool for neuromorphic computations."""
    
    def __init__(self, config: ConcurrencyConfig, worker_initializer: Optional[Callable] = None):
        self.config = config
        self.worker_initializer = worker_initializer
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="spikeformer_worker"
        )
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.worker_stats = {i: {'tasks_completed': 0, 'total_time': 0.0} 
                           for i in range(config.max_workers)}
        self.logger = logging.getLogger(__name__)
    
    def submit_task(self, func: Callable, args: Tuple, kwargs: Dict, 
                   priority: int = 1, task_id: str = None) -> str:
        """Submit task to worker pool."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        future = self.executor.submit(self._execute_task, func, args, kwargs, task_id)
        self.results[task_id] = future
        
        return task_id
    
    def _execute_task(self, func: Callable, args: Tuple, kwargs: Dict, task_id: str) -> Any:
        """Execute task with error handling and timing."""
        worker_id = threading.current_thread().ident
        start_time = time.time()
        
        try:
            # Initialize worker if needed
            if self.worker_initializer:
                self.worker_initializer()
            
            # Execute task
            result = func(*args, **kwargs)
            
            # Update statistics
            execution_time = time.time() - start_time
            if worker_id in self.worker_stats:
                self.worker_stats[worker_id]['tasks_completed'] += 1
                self.worker_stats[worker_id]['total_time'] += execution_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}")
            raise
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result."""
        future = self.results.get(task_id)
        if future is None:
            raise ValueError(f"Task {task_id} not found")
        
        return future.result(timeout=timeout)
    
    def map_async(self, func: Callable, args_list: List[Tuple], 
                 callback: Optional[Callable] = None) -> List[str]:
        """Map function over arguments asynchronously."""
        task_ids = []
        
        for i, args in enumerate(args_list):
            task_id = f"map_task_{i}_{int(time.time() * 1000000)}"
            self.submit_task(func, args, {}, task_id=task_id)
            task_ids.append(task_id)
        
        # Handle callback if provided
        if callback:
            def collect_results():
                results = []
                for task_id in task_ids:
                    try:
                        result = self.get_result(task_id)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task {task_id} failed: {e}")
                        results.append(None)
                callback(results)
            
            threading.Thread(target=collect_results, daemon=True).start()
        
        return task_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        total_tasks = sum(stats['tasks_completed'] for stats in self.worker_stats.values())
        total_time = sum(stats['total_time'] for stats in self.worker_stats.values())
        
        return {
            'total_tasks': total_tasks,
            'total_time': total_time,
            'average_task_time': total_time / total_tasks if total_tasks > 0 else 0,
            'worker_stats': self.worker_stats,
            'active_tasks': len([f for f in self.results.values() if not f.done()])
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)


class AsyncBatchProcessor:
    """Asynchronous batch processor for inference."""
    
    def __init__(self, model: nn.Module, batch_size: int = 32, max_latency: float = 0.1):
        self.model = model
        self.batch_size = batch_size
        self.max_latency = max_latency
        self.pending_requests = []
        self.request_futures = {}
        self.processing_lock = asyncio.Lock()
        self.batch_timer = None
        self.logger = logging.getLogger(__name__)
    
    async def process_request(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process single request, batching with others."""
        request_id = f"req_{int(time.time() * 1000000)}"
        future = asyncio.Future()
        
        async with self.processing_lock:
            self.pending_requests.append((request_id, inputs))
            self.request_futures[request_id] = future
            
            # Start batch timer if not already running
            if self.batch_timer is None:
                self.batch_timer = asyncio.create_task(self._batch_timer())
            
            # Process batch if full
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch()
        
        return await future
    
    async def _batch_timer(self):
        """Timer to process partial batches."""
        try:
            await asyncio.sleep(self.max_latency)
            
            async with self.processing_lock:
                if self.pending_requests:
                    await self._process_batch()
                self.batch_timer = None
                
        except asyncio.CancelledError:
            pass
    
    async def _process_batch(self):
        """Process current batch of requests."""
        if not self.pending_requests:
            return
        
        # Extract batch
        batch_requests = self.pending_requests[:]
        self.pending_requests.clear()
        
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        try:
            # Prepare batch inputs
            request_ids, inputs_list = zip(*batch_requests)
            
            # Stack inputs into batch
            batch_inputs = torch.stack(inputs_list)
            
            # Process batch
            self.model.eval()
            with torch.no_grad():
                batch_outputs = self.model(batch_inputs)
            
            # Distribute results
            for i, request_id in enumerate(request_ids):
                future = self.request_futures.pop(request_id)
                if not future.cancelled():
                    future.set_result(batch_outputs[i])
        
        except Exception as e:
            # Handle errors
            self.logger.error(f"Batch processing error: {e}")
            
            for request_id in request_ids:
                future = self.request_futures.pop(request_id, None)
                if future and not future.cancelled():
                    future.set_exception(e)


def parallel_spike_encoding(inputs: torch.Tensor, encoding_type: str = 'rate',
                          timesteps: int = 32, num_workers: int = 4) -> torch.Tensor:
    """Parallel spike encoding for large inputs."""
    
    def encode_chunk(chunk: torch.Tensor, enc_type: str, t_steps: int) -> torch.Tensor:
        """Encode a chunk of inputs to spikes."""
        if enc_type == 'rate':
            # Rate coding
            rates = torch.sigmoid(chunk)  # Normalize to [0,1]
            spikes = torch.rand(t_steps, *chunk.shape) < rates.unsqueeze(0)
            return spikes.float()
        elif enc_type == 'temporal':
            # Temporal coding
            spike_times = (chunk.sigmoid() * t_steps).long().clamp(0, t_steps - 1)
            spikes = torch.zeros(t_steps, *chunk.shape)
            for t in range(t_steps):
                mask = (spike_times == t)
                spikes[t][mask] = 1.0
            return spikes
        else:
            raise ValueError(f"Unknown encoding type: {enc_type}")
    
    # Split inputs into chunks
    chunk_size = max(1, inputs.size(0) // num_workers)
    chunks = torch.split(inputs, chunk_size, dim=0)
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(encode_chunk, chunk, encoding_type, timesteps)
            for chunk in chunks
        ]
        
        encoded_chunks = [future.result() for future in futures]
    
    # Concatenate results
    return torch.cat(encoded_chunks, dim=1)  # Concatenate along batch dimension


@contextmanager
def distributed_context(rank: int, world_size: int):
    """Context manager for distributed training."""
    try:
        # Setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        init_process_group(
            backend='gloo',  # Use gloo for CPU, nccl for GPU
            rank=rank,
            world_size=world_size
        )
        
        yield rank, world_size
        
    finally:
        # Cleanup
        destroy_process_group()


def run_distributed_training(train_fn: Callable, world_size: int, *args, **kwargs):
    """Run distributed training across multiple processes."""
    def train_worker(rank: int):
        with distributed_context(rank, world_size):
            train_fn(rank, world_size, *args, **kwargs)
    
    # Start processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_worker, args=(rank,))
        p.start()
        processes.append(p)
    
    # Wait for completion
    for p in processes:
        p.join()


# Global worker pool
_global_worker_pool = None


def get_worker_pool() -> WorkerPool:
    """Get global worker pool instance."""
    global _global_worker_pool
    if _global_worker_pool is None:
        config = ConcurrencyConfig(max_workers=4)
        _global_worker_pool = WorkerPool(config)
    return _global_worker_pool