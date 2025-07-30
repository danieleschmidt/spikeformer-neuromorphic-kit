"""Monitoring and metrics collection for SpikeFormer with OpenTelemetry integration."""

import time
import os
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry, 
    generate_latest, CONTENT_TYPE_LATEST
)
import torch

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    otel_metrics = None


@dataclass
class MetricValue:
    """Container for metric values with metadata."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    

class SpikeFormerMetrics:
    """Centralized metrics collection for SpikeFormer with OpenTelemetry support."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, enable_otel: bool = True):
        self.registry = registry or CollectorRegistry()
        self.enable_otel = enable_otel and OTEL_AVAILABLE
        self._setup_metrics()
        self._setup_opentelemetry()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def _setup_opentelemetry(self):
        """Initialize OpenTelemetry instrumentation."""
        if not self.enable_otel:
            self.tracer = None
            self.meter = None
            self.otel_metrics = {}
            return
            
        try:
            # Set up resource with neuromorphic-specific attributes
            resource = Resource.create({
                "service.name": "spikeformer-neuromorphic-kit",
                "service.version": "0.1.0",
                "service.namespace": "neuromorphic-ai",
                "neuromorphic.framework": "spikeformer",
                "ml.framework": "pytorch",
                "neuromorphic.hardware.available": os.getenv("NEUROMORPHIC_HARDWARE", "simulation"),
                "neuromorphic.loihi2.enabled": os.getenv("LOIHI2_AVAILABLE", "false"),
                "neuromorphic.spinnaker.enabled": os.getenv("SPINNAKER_AVAILABLE", "false"),
            })
            
            # Set up tracing
            trace.set_tracer_provider(TracerProvider(resource=resource))
            otlp_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                insecure=True
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            self.tracer = trace.get_tracer(__name__)
            
            # Set up metrics
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                    insecure=True
                ),
                export_interval_millis=5000
            )
            otel_metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))
            self.meter = otel_metrics.get_meter(__name__)
            
            # Create neuromorphic-specific OpenTelemetry metrics
            self.otel_metrics = {
                "inference_counter": self.meter.create_counter(
                    "spikeformer_inferences_total",
                    description="Total number of neuromorphic model inferences"
                ),
                "energy_histogram": self.meter.create_histogram(
                    "spikeformer_energy_consumption_mj", 
                    description="Energy consumption per inference in millijoules"
                ),
                "spike_sparsity_gauge": self.meter.create_up_down_counter(
                    "spikeformer_spike_sparsity_ratio",
                    description="Neural spike sparsity ratio"
                ),
                "hardware_utilization_gauge": self.meter.create_up_down_counter(
                    "spikeformer_hardware_utilization_percent",
                    description="Neuromorphic hardware utilization percentage"
                ),
                "training_loss_histogram": self.meter.create_histogram(
                    "spikeformer_training_loss",
                    description="Training loss values"
                ),
                "conversion_accuracy_gauge": self.meter.create_up_down_counter(
                    "spikeformer_conversion_accuracy_percent",
                    description="ANN-to-SNN conversion accuracy percentage"
                )
            }
            
        except Exception as e:
            print(f"Warning: Failed to initialize OpenTelemetry: {e}")
            self.tracer = None
            self.meter = None
            self.otel_metrics = {}
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        
        # Application metrics
        self.requests_total = Counter(
            'spikeformer_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'spikeformer_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'spikeformer_model_accuracy',
            'Current model accuracy',
            ['model_name', 'dataset'],
            registry=self.registry
        )
        
        self.inference_duration = Histogram(
            'spikeformer_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_name', 'batch_size'],
            registry=self.registry
        )
        
        self.spike_rate = Gauge(
            'spikeformer_spike_rate',
            'Current spike rate',
            ['layer', 'model_name'],
            registry=self.registry
        )
        
        self.spike_rate_baseline = Gauge(
            'spikeformer_spike_rate_baseline',
            'Baseline spike rate for comparison',
            ['layer', 'model_name'],
            registry=self.registry
        )
        
        # Energy efficiency metrics
        self.energy_consumption = Gauge(
            'spikeformer_energy_consumption_joules',
            'Energy consumption in joules',
            ['component', 'backend'],
            registry=self.registry
        )
        
        self.power_consumption = Gauge(
            'spikeformer_power_consumption_watts',
            'Power consumption in watts',
            ['component', 'backend'],
            registry=self.registry
        )
        
        self.energy_efficiency_ratio = Gauge(
            'spikeformer_energy_efficiency_ratio',
            'Energy efficiency ratio vs baseline',
            ['model_name', 'backend'],
            registry=self.registry
        )
        
        # Hardware-specific metrics
        self.loihi2_chip_utilization = Gauge(
            'spikeformer_loihi2_chip_utilization',
            'Loihi2 chip utilization',
            ['chip_id'],
            registry=self.registry
        )
        
        self.spinnaker_communication_errors = Counter(
            'spikeformer_spinnaker_communication_errors_total',
            'SpiNNaker communication errors',
            ['board_id', 'error_type'],
            registry=self.registry
        )
        
        # System resource metrics
        self.memory_usage = Gauge(
            'spikeformer_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.memory_limit = Gauge(
            'spikeformer_memory_limit_bytes',
            'Memory limit in bytes',
            ['component'],
            registry=self.registry
        )
        
        self.cpu_utilization = Gauge(
            'spikeformer_cpu_utilization_percent',
            'CPU utilization percentage',
            ['core'],
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'spikeformer_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'spikeformer_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Training metrics
        self.training_loss = Gauge(
            'spikeformer_training_loss',
            'Current training loss',
            ['model_name', 'loss_type'],
            registry=self.registry
        )
        
        self.training_steps = Counter(
            'spikeformer_training_steps_total',
            'Total training steps',
            ['model_name'],
            registry=self.registry
        )
        
        self.training_active = Gauge(
            'spikeformer_training_active',
            'Whether training is currently active',
            ['model_name'],
            registry=self.registry
        )
        
        # Queue and throughput metrics
        self.inference_queue_size = Gauge(
            'spikeformer_inference_queue_size',
            'Current inference queue size',
            ['backend'],
            registry=self.registry
        )
        
        self.throughput = Gauge(
            'spikeformer_throughput_requests_per_second',
            'Current throughput in requests per second',
            ['backend'],
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_free = Gauge(
            'spikeformer_disk_free_bytes',
            'Free disk space in bytes',
            ['path'],
            registry=self.registry
        )
        
        self.disk_total = Gauge(
            'spikeformer_disk_total_bytes',
            'Total disk space in bytes',
            ['path'],
            registry=self.registry
        )
        
        # Application info
        self.build_info = Info(
            'spikeformer_build_info',
            'Build information',
            registry=self.registry
        )
        
    def start_monitoring(self, interval: float = 15.0):
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
            
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(interval):
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                self._collect_disk_metrics()
            except Exception as e:
                # Log error but continue monitoring
                print(f"Monitoring error: {e}")
                
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.labels(component="system").set(memory.used)
            self.memory_limit.labels(component="system").set(memory.total)
            
            # CPU metrics
            cpu_percents = psutil.cpu_percent(percpu=True)
            for i, cpu_percent in enumerate(cpu_percents):
                self.cpu_utilization.labels(core=str(i)).set(cpu_percent)
                
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # GPU utilization (simplified - would need nvidia-ml-py for real metrics)
                    # For now, just track memory usage
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    self.gpu_memory_usage.labels(gpu_id=str(i)).set(memory_allocated)
                    
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            
    def _collect_disk_metrics(self):
        """Collect disk usage metrics."""
        try:
            disk_usage = psutil.disk_usage('/')
            self.disk_free.labels(path="/").set(disk_usage.free)
            self.disk_total.labels(path="/").set(disk_usage.total)
            
        except Exception as e:
            print(f"Error collecting disk metrics: {e}")
    
    # High-level metric recording methods
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
    def record_inference(self, model_name: str, batch_size: int, duration: float):
        """Record model inference metrics."""
        self.inference_duration.labels(
            model_name=model_name,
            batch_size=str(batch_size)
        ).observe(duration)
        
    def record_energy_consumption(self, component: str, backend: str, 
                                 energy_joules: float, power_watts: float):
        """Record energy consumption metrics."""
        self.energy_consumption.labels(
            component=component,
            backend=backend
        ).set(energy_joules)
        
        self.power_consumption.labels(
            component=component,
            backend=backend
        ).set(power_watts)
        
    def record_spike_rate(self, layer: str, model_name: str, spike_rate: float):
        """Record spike rate metrics."""
        self.spike_rate.labels(
            layer=layer,
            model_name=model_name
        ).set(spike_rate)
        
    def record_accuracy(self, model_name: str, dataset: str, accuracy: float):
        """Record model accuracy metrics."""
        self.model_accuracy.labels(
            model_name=model_name,
            dataset=dataset
        ).set(accuracy)
        
    def record_training_step(self, model_name: str, loss: float, step: int):
        """Record training progress."""
        self.training_steps.labels(model_name=model_name).inc()
        self.training_loss.labels(
            model_name=model_name,
            loss_type="total"
        ).set(loss)
        
    def set_build_info(self, version: str, commit: str, build_date: str):
        """Set build information."""
        self.build_info.info({
            'version': version,
            'commit': commit,
            'build_date': build_date
        })
        
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class EnergyProfiler:
    """Energy profiling and monitoring."""
    
    def __init__(self, metrics: Optional[SpikeFormerMetrics] = None):
        self.metrics = metrics
        self.measurements = deque(maxlen=1000)  # Keep last 1000 measurements
        self._start_time = None
        self._start_energy = None
        
    def __enter__(self):
        """Start energy measurement context."""
        self._start_time = time.time()
        self._start_energy = self._get_current_energy()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End energy measurement and record results."""
        end_time = time.time()
        end_energy = self._get_current_energy()
        
        duration = end_time - self._start_time
        energy_consumed = end_energy - self._start_energy
        power_avg = energy_consumed / duration if duration > 0 else 0
        
        measurement = {
            'duration_seconds': duration,
            'energy_joules': energy_consumed,
            'power_watts': power_avg,
            'timestamp': end_time
        }
        
        self.measurements.append(measurement)
        
        if self.metrics:
            self.metrics.record_energy_consumption(
                component="inference",
                backend="cpu",  # Would be determined dynamically
                energy_joules=energy_consumed,
                power_watts=power_avg
            )
            
    def _get_current_energy(self) -> float:
        """Get current energy reading (mock implementation)."""
        # In a real implementation, this would interface with hardware
        # energy monitoring APIs or estimate based on utilization
        return time.time() * 0.1  # Mock energy measurement
        
    def get_energy_statistics(self) -> Dict[str, float]:
        """Get energy consumption statistics."""
        if not self.measurements:
            return {}
            
        energies = [m['energy_joules'] for m in self.measurements]
        powers = [m['power_watts'] for m in self.measurements]
        
        return {
            'avg_energy_joules': sum(energies) / len(energies),
            'max_energy_joules': max(energies),
            'min_energy_joules': min(energies),
            'avg_power_watts': sum(powers) / len(powers),
            'max_power_watts': max(powers),
            'min_power_watts': min(powers),
            'total_measurements': len(self.measurements)
        }


# Global metrics instance
metrics = SpikeFormerMetrics()

# Convenience functions
def start_monitoring(interval: float = 15.0):
    """Start global monitoring."""
    metrics.start_monitoring(interval)

def stop_monitoring():
    """Stop global monitoring."""
    metrics.stop_monitoring()

def get_metrics() -> str:
    """Get global metrics."""
    return metrics.get_metrics()

def create_energy_profiler() -> EnergyProfiler:
    """Create energy profiler with global metrics."""
    return EnergyProfiler(metrics)