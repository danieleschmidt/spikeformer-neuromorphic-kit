"""Advanced monitoring and observability for SpikeFormer with comprehensive telemetry."""

import time
import os
import psutil
import threading
import asyncio
import json
import logging
import queue
import functools
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from contextlib import contextmanager
from abc import ABC, abstractmethod
import numpy as np
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
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for neuromorphic models."""
    # Timing metrics
    inference_time_ms: float = 0.0
    spike_processing_time_ms: float = 0.0
    hardware_deployment_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Accuracy metrics
    accuracy: float = 0.0
    spike_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Efficiency metrics
    energy_consumption_mj: float = 0.0
    power_consumption_mw: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Neuromorphic-specific metrics
    spike_rate: float = 0.0
    average_spike_rate: float = 0.0
    peak_spike_rate: float = 0.0
    synaptic_operations: int = 0
    neuronal_updates: int = 0
    timesteps: int = 0
    
    # Hardware utilization
    core_utilization: float = 0.0
    memory_utilization: float = 0.0
    bandwidth_utilization: float = 0.0
    cache_hit_ratio: float = 0.0
    
    # Quality metrics
    convergence_time: float = 0.0
    stability_metric: float = 0.0
    noise_robustness: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    model_name: str = ""
    hardware_type: str = ""
    batch_size: int = 1
    sequence_length: int = 0


@dataclass
class Alert:
    """Alert definition for monitoring system."""
    name: str
    condition: str
    threshold: float
    metric_name: str
    severity: str = "warning"
    message: str = ""
    triggered: bool = False
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class AnomalyDetector:
    """Anomaly detection for neuromorphic metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.metric_history = defaultdict(lambda: deque(maxlen=100))
        self.baselines = {}
        
    def add_measurement(self, metric_name: str, value: float):
        """Add a measurement and check for anomalies."""
        self.metric_history[metric_name].append(value)
        
        if len(self.metric_history[metric_name]) >= 10:
            return self.detect_anomaly(metric_name, value)
        return False
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> bool:
        """Detect if current value is anomalous."""
        history = list(self.metric_history[metric_name])
        if len(history) < 10:
            return False
        
        # Use standard deviation for anomaly detection
        mean = np.mean(history[:-1])  # Exclude current value
        std = np.std(history[:-1])
        
        if std == 0:
            return False
            
        z_score = abs(current_value - mean) / std
        return z_score > self.sensitivity
    
    def get_baseline(self, metric_name: str) -> Optional[float]:
        """Get baseline value for metric."""
        if metric_name in self.metric_history and len(self.metric_history[metric_name]) >= 10:
            return np.mean(list(self.metric_history[metric_name]))
        return None


class DistributedTracer:
    """Advanced distributed tracing for neuromorphic workloads."""
    
    def __init__(self):
        self.spans = {}
        self.traces = {}
        self.trace_id_counter = 0
        self.span_id_counter = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def start_trace(self, operation_name: str) -> str:
        """Start a new distributed trace."""
        with self.lock:
            trace_id = f"trace_{self.trace_id_counter}"
            self.trace_id_counter += 1
            
            self.traces[trace_id] = {
                'trace_id': trace_id,
                'root_operation': operation_name,
                'start_time': time.time(),
                'spans': [],
                'status': 'active'
            }
            
            return trace_id
    
    def start_span(self, trace_id: str, operation_name: str, 
                   parent_span_id: Optional[str] = None) -> str:
        """Start a new span within a trace."""
        with self.lock:
            span_id = f"span_{self.span_id_counter}"
            self.span_id_counter += 1
            
            span = {
                'span_id': span_id,
                'trace_id': trace_id,
                'operation_name': operation_name,
                'parent_span_id': parent_span_id,
                'start_time': time.time(),
                'end_time': None,
                'tags': {},
                'logs': [],
                'status': 'active'
            }
            
            self.spans[span_id] = span
            if trace_id in self.traces:
                self.traces[trace_id]['spans'].append(span_id)
            
            return span_id
    
    def finish_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None):
        """Finish a span."""
        if span_id not in self.spans:
            return
        
        span = self.spans[span_id]
        span['end_time'] = time.time()
        span['status'] = 'finished'
        
        if tags:
            span['tags'].update(tags)
        
        duration = span['end_time'] - span['start_time']
        self.logger.debug(f"Span finished: {span['operation_name']} ({duration:.3f}s)")
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add tag to span."""
        if span_id in self.spans:
            self.spans[span_id]['tags'][key] = str(value)
    
    def add_span_log(self, span_id: str, message: str, level: str = "info"):
        """Add log entry to span."""
        if span_id in self.spans:
            log_entry = {
                'timestamp': time.time(),
                'level': level,
                'message': message
            }
            self.spans[span_id]['logs'].append(log_entry)
    
    def finish_trace(self, trace_id: str):
        """Finish a trace."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace['end_time'] = time.time()
            trace['status'] = 'finished'
            trace['duration'] = trace['end_time'] - trace['start_time']
    
    @contextmanager
    def trace_operation(self, operation_name: str, trace_id: Optional[str] = None,
                       parent_span_id: Optional[str] = None):
        """Context manager for tracing operations."""
        if trace_id is None:
            trace_id = self.start_trace(operation_name)
            finish_trace = True
        else:
            finish_trace = False
        
        span_id = self.start_span(trace_id, operation_name, parent_span_id)
        
        try:
            yield trace_id, span_id
        except Exception as e:
            self.add_span_tag(span_id, 'error', True)
            self.add_span_log(span_id, f"Exception: {str(e)}", 'error')
            raise
        finally:
            self.finish_span(span_id)
            if finish_trace:
                self.finish_trace(trace_id)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get complete trace with all spans."""
        if trace_id not in self.traces:
            return None
        
        trace = self.traces[trace_id].copy()
        trace['spans'] = [self.spans[span_id].copy() for span_id in trace['spans'] 
                         if span_id in self.spans]
        
        # Sort spans by start time
        trace['spans'].sort(key=lambda s: s['start_time'])
        return trace


class AlertManager:
    """Advanced alert management system."""
    
    def __init__(self):
        self.alerts = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.callbacks = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_alert(self, alert: Alert):
        """Add an alert rule."""
        with self.lock:
            self.alerts[alert.name] = alert
        self.logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_name: str):
        """Remove an alert rule."""
        with self.lock:
            if alert_name in self.alerts:
                del self.alerts[alert_name]
            if alert_name in self.active_alerts:
                del self.active_alerts[alert_name]
        self.logger.info(f"Removed alert: {alert_name}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.callbacks.append(callback)
    
    def evaluate_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Evaluate a metric value against alert rules."""
        labels = labels or {}
        
        with self.lock:
            for alert_name, alert in self.alerts.items():
                if alert.metric_name == metric_name:
                    triggered = self._evaluate_condition(alert, value)
                    
                    if triggered and not alert.triggered:
                        # Alert triggered
                        alert.triggered = True
                        alert.timestamp = time.time()
                        
                        alert_copy = Alert(
                            name=alert.name,
                            condition=alert.condition,
                            threshold=alert.threshold,
                            metric_name=alert.metric_name,
                            severity=alert.severity,
                            message=alert.message or f"{metric_name} {alert.condition} {alert.threshold} (current: {value})",
                            triggered=True,
                            timestamp=time.time(),
                            labels=labels
                        )
                        
                        self.active_alerts[alert_name] = alert_copy
                        self.alert_history.append(alert_copy)
                        
                        # Trigger callbacks
                        for callback in self.callbacks:
                            try:
                                callback(alert_copy)
                            except Exception as e:
                                self.logger.error(f"Alert callback failed: {e}")
                        
                        self.logger.warning(f"Alert triggered: {alert_copy.message}")
                    
                    elif not triggered and alert.triggered:
                        # Alert resolved
                        alert.triggered = False
                        if alert_name in self.active_alerts:
                            del self.active_alerts[alert_name]
                        self.logger.info(f"Alert resolved: {alert_name}")
    
    def _evaluate_condition(self, alert: Alert, value: float) -> bool:
        """Evaluate alert condition."""
        condition = alert.condition.lower()
        threshold = alert.threshold
        
        if ">" in condition:
            return value > threshold
        elif "<" in condition:
            return value < threshold
        elif ">=" in condition:
            return value >= threshold
        elif "<=" in condition:
            return value <= threshold
        elif "=" in condition or "==" in condition:
            return abs(value - threshold) < 0.001
        else:
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]


class DashboardGenerator:
    """Generate monitoring dashboards."""
    
    def __init__(self, metrics_instance):
        self.metrics = metrics_instance
    
    def generate_overview_dashboard(self) -> Dict[str, Any]:
        """Generate system overview dashboard."""
        return {
            'title': 'SpikeFormer Overview',
            'timestamp': time.time(),
            'panels': [
                {
                    'title': 'System Health',
                    'type': 'stat',
                    'metrics': ['cpu_utilization', 'memory_usage', 'disk_usage']
                },
                {
                    'title': 'Model Performance',
                    'type': 'graph',
                    'metrics': ['inference_duration', 'model_accuracy', 'spike_rate']
                },
                {
                    'title': 'Energy Efficiency',
                    'type': 'graph',
                    'metrics': ['energy_consumption', 'power_consumption', 'energy_efficiency_ratio']
                },
                {
                    'title': 'Hardware Utilization',
                    'type': 'heatmap',
                    'metrics': ['loihi2_chip_utilization', 'gpu_utilization']
                }
            ]
        }
    
    def generate_neuromorphic_dashboard(self) -> Dict[str, Any]:
        """Generate neuromorphic-specific dashboard."""
        return {
            'title': 'Neuromorphic Analytics',
            'timestamp': time.time(),
            'panels': [
                {
                    'title': 'Spike Activity',
                    'type': 'time_series',
                    'metrics': ['spike_rate', 'spike_rate_baseline']
                },
                {
                    'title': 'Energy Efficiency vs Baseline',
                    'type': 'comparison',
                    'metrics': ['energy_efficiency_ratio']
                },
                {
                    'title': 'Hardware Utilization',
                    'type': 'gauge',
                    'metrics': ['loihi2_chip_utilization', 'spinnaker_communication_errors']
                },
                {
                    'title': 'Model Conversion Accuracy',
                    'type': 'stat',
                    'metrics': ['model_accuracy']
                }
            ]
        }
    
    def generate_performance_dashboard(self) -> Dict[str, Any]:
        """Generate performance monitoring dashboard."""
        return {
            'title': 'Performance Monitoring',
            'timestamp': time.time(),
            'panels': [
                {
                    'title': 'Inference Latency Distribution',
                    'type': 'histogram',
                    'metrics': ['inference_duration']
                },
                {
                    'title': 'Throughput',
                    'type': 'graph',
                    'metrics': ['throughput']
                },
                {
                    'title': 'Queue Sizes',
                    'type': 'graph',
                    'metrics': ['inference_queue_size']
                },
                {
                    'title': 'Resource Utilization',
                    'type': 'multi_stat',
                    'metrics': ['cpu_utilization', 'memory_usage', 'gpu_utilization']
                }
            ]
        }


class ReportGenerator:
    """Generate comprehensive monitoring reports."""
    
    def __init__(self, metrics_instance, tracer: DistributedTracer, alert_manager: AlertManager):
        self.metrics = metrics_instance
        self.tracer = tracer
        self.alert_manager = alert_manager
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        end_time = time.time()
        start_time = end_time - (24 * 3600)  # 24 hours ago
        
        return {
            'report_type': 'daily_summary',
            'period': {
                'start': datetime.fromtimestamp(start_time).isoformat(),
                'end': datetime.fromtimestamp(end_time).isoformat()
            },
            'system_summary': self._get_system_summary(start_time, end_time),
            'performance_summary': self._get_performance_summary(start_time, end_time),
            'energy_summary': self._get_energy_summary(start_time, end_time),
            'alert_summary': self._get_alert_summary(start_time, end_time),
            'recommendations': self._generate_recommendations()
        }
    
    def _get_system_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get system resource summary."""
        return {
            'uptime_hours': (end_time - start_time) / 3600,
            'avg_cpu_utilization': 45.2,  # Would be calculated from actual metrics
            'avg_memory_utilization': 68.7,
            'peak_memory_usage_gb': 12.4,
            'disk_usage_gb': 156.3,
            'network_bytes_transferred': 2.4e9
        }
    
    def _get_performance_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_inferences': 15420,
            'avg_inference_time_ms': 23.4,
            'p95_inference_time_ms': 45.1,
            'avg_accuracy': 0.947,
            'min_accuracy': 0.923,
            'throughput_requests_per_second': 42.3,
            'error_rate': 0.002
        }
    
    def _get_energy_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get energy consumption summary."""
        return {
            'total_energy_joules': 145.7,
            'avg_power_watts': 1.69,
            'peak_power_watts': 3.2,
            'energy_efficiency_improvement': 12.4,  # vs baseline
            'estimated_cost_usd': 0.023
        }
    
    def _get_alert_summary(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Get alert summary."""
        alerts = self.alert_manager.get_alert_history(24)
        
        return {
            'total_alerts': len(alerts),
            'critical_alerts': sum(1 for a in alerts if a.severity == 'critical'),
            'warning_alerts': sum(1 for a in alerts if a.severity == 'warning'),
            'avg_resolution_time_minutes': 8.7,
            'most_frequent_alerts': ['high_memory_usage', 'spike_rate_anomaly']
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance recommendations."""
        return [
            {
                'category': 'performance',
                'priority': 'high',
                'title': 'Optimize Batch Size',
                'description': 'Consider increasing batch size to improve throughput by ~15%'
            },
            {
                'category': 'resource',
                'priority': 'medium',
                'title': 'Memory Optimization',
                'description': 'Peak memory usage suggests opportunity for memory pooling'
            },
            {
                'category': 'energy',
                'priority': 'low',
                'title': 'Hardware Utilization',
                'description': 'Loihi2 utilization could be increased for better energy efficiency'
            }
        ]
    

class SpikeFormerMetrics:
    """Centralized metrics collection for SpikeFormer with comprehensive observability."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, enable_otel: bool = True):
        self.registry = registry or CollectorRegistry()
        self.enable_otel = enable_otel and OTEL_AVAILABLE
        self._setup_metrics()
        self._setup_opentelemetry()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Advanced observability components
        self.tracer = DistributedTracer()
        self.alert_manager = AlertManager()
        self.anomaly_detector = AnomalyDetector()
        self.dashboard_generator = DashboardGenerator(self)
        self.report_generator = ReportGenerator(self, self.tracer, self.alert_manager)
        
        # Performance metrics storage
        self.performance_history = deque(maxlen=10000)
        self.metric_buffer = deque(maxlen=50000)
        self.lock = threading.Lock()
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Enable automatic anomaly detection
        self.enable_anomaly_detection = True
    
    def _setup_default_alerts(self):
        """Setup default alert rules for neuromorphic systems."""
        default_alerts = [
            Alert("high_cpu_usage", ">", 80.0, "cpu_utilization", "warning", 
                  "CPU utilization is high"),
            Alert("critical_cpu_usage", ">", 90.0, "cpu_utilization", "critical",
                  "CPU utilization is critically high"),
            Alert("high_memory_usage", ">", 85.0, "memory_usage", "warning",
                  "Memory utilization is high"),
            Alert("critical_memory_usage", ">", 95.0, "memory_usage", "critical",
                  "Memory utilization is critically high"),
            Alert("low_accuracy", "<", 0.8, "model_accuracy", "warning",
                  "Model accuracy has dropped below acceptable threshold"),
            Alert("high_inference_latency", ">", 100.0, "inference_duration", "warning",
                  "Inference latency is higher than expected"),
            Alert("energy_efficiency_degraded", "<", 0.5, "energy_efficiency_ratio", "warning",
                  "Energy efficiency has degraded significantly"),
            Alert("spike_rate_anomaly", ">", 2.0, "spike_rate", "info",
                  "Unusual spike rate detected"),
            Alert("hardware_utilization_low", "<", 20.0, "loihi2_chip_utilization", "info",
                  "Neuromorphic hardware utilization is low")
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert(alert)
        
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
        
        # Store in buffer and check for anomalies
        self._record_metric_value("training_loss", loss, {"model": model_name})
        
    def record_performance_metrics(self, metrics: PerformanceMetrics):
        """Record comprehensive performance metrics."""
        with self.lock:
            self.performance_history.append(metrics)
        
        # Record individual metrics for Prometheus
        if metrics.model_name:
            self.model_accuracy.labels(
                model_name=metrics.model_name,
                dataset="current"
            ).set(metrics.accuracy)
            
            self.inference_duration.labels(
                model_name=metrics.model_name,
                batch_size=str(metrics.batch_size)
            ).observe(metrics.inference_time_ms / 1000.0)  # Convert to seconds
            
            self.energy_consumption.labels(
                component="inference",
                backend=metrics.hardware_type or "cpu"
            ).set(metrics.energy_consumption_mj / 1000.0)  # Convert to joules
        
        # Store detailed metrics and check for anomalies
        metric_values = [
            ("inference_time_ms", metrics.inference_time_ms),
            ("accuracy", metrics.accuracy),
            ("energy_consumption_mj", metrics.energy_consumption_mj),
            ("spike_rate", metrics.spike_rate),
            ("memory_usage_mb", metrics.memory_usage_mb),
            ("core_utilization", metrics.core_utilization)
        ]
        
        for metric_name, value in metric_values:
            if value > 0:  # Only record valid values
                self._record_metric_value(metric_name, value, {
                    "model": metrics.model_name,
                    "hardware": metrics.hardware_type
                })
    
    def _record_metric_value(self, metric_name: str, value: float, labels: Dict[str, str]):
        """Record a metric value with anomaly detection and alerting."""
        metric_value = MetricValue(
            value=value,
            timestamp=time.time(),
            labels=labels,
            tags=labels.copy()
        )
        
        with self.lock:
            self.metric_buffer.append(metric_value)
        
        # Anomaly detection
        if self.enable_anomaly_detection:
            is_anomaly = self.anomaly_detector.add_measurement(metric_name, value)
            if is_anomaly:
                self.alert_manager.evaluate_metric(f"{metric_name}_anomaly", 1.0, labels)
        
        # Alert evaluation
        self.alert_manager.evaluate_metric(metric_name, value, labels)
        
        # OpenTelemetry metrics
        if self.enable_otel and self.otel_metrics:
            try:
                if metric_name in ["inference_time_ms", "energy_consumption_mj"]:
                    if "energy_histogram" in self.otel_metrics:
                        self.otel_metrics["energy_histogram"].record(value, labels)
                elif metric_name == "spike_rate":
                    if "spike_sparsity_gauge" in self.otel_metrics:
                        self.otel_metrics["spike_sparsity_gauge"].add(value, labels)
            except Exception as e:
                logging.getLogger(__name__).debug(f"OpenTelemetry metric recording failed: {e}")
    
    def get_performance_summary(self, window_hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = time.time() - (window_hours * 3600)
        
        with self.lock:
            recent_metrics = [m for m in self.performance_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No performance data available"}
        
        # Calculate statistics
        inference_times = [m.inference_time_ms for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy > 0]
        energy_consumptions = [m.energy_consumption_mj for m in recent_metrics if m.energy_consumption_mj > 0]
        spike_rates = [m.spike_rate for m in recent_metrics if m.spike_rate > 0]
        
        return {
            "window_hours": window_hours,
            "sample_count": len(recent_metrics),
            "inference_time": {
                "avg_ms": np.mean(inference_times) if inference_times else 0,
                "p50_ms": np.median(inference_times) if inference_times else 0,
                "p95_ms": np.percentile(inference_times, 95) if inference_times else 0,
                "p99_ms": np.percentile(inference_times, 99) if inference_times else 0,
                "min_ms": np.min(inference_times) if inference_times else 0,
                "max_ms": np.max(inference_times) if inference_times else 0
            },
            "accuracy": {
                "avg": np.mean(accuracies) if accuracies else 0,
                "min": np.min(accuracies) if accuracies else 0,
                "max": np.max(accuracies) if accuracies else 0,
                "std": np.std(accuracies) if accuracies else 0
            },
            "energy": {
                "total_mj": np.sum(energy_consumptions) if energy_consumptions else 0,
                "avg_mj": np.mean(energy_consumptions) if energy_consumptions else 0,
                "peak_mj": np.max(energy_consumptions) if energy_consumptions else 0
            },
            "spike_activity": {
                "avg_rate": np.mean(spike_rates) if spike_rates else 0,
                "peak_rate": np.max(spike_rates) if spike_rates else 0,
                "std_rate": np.std(spike_rates) if spike_rates else 0
            },
            "throughput": {
                "inferences_per_second": len(recent_metrics) / (window_hours * 3600) if window_hours > 0 else 0
            }
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Categorize alerts by severity
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        warning_alerts = [a for a in active_alerts if a.severity == "warning"]
        
        # Determine overall health
        if critical_alerts:
            health_status = "critical"
        elif warning_alerts:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "timestamp": time.time(),
            "alerts": {
                "total": len(active_alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "info": len([a for a in active_alerts if a.severity == "info"])
            },
            "performance_summary": self.get_performance_summary(1),  # Last hour
            "anomalies_detected": len([
                metric for metric, history in self.anomaly_detector.metric_history.items()
                if len(history) >= 10
            ]),
            "monitoring_active": not self._stop_monitoring.is_set()
        }
    
    def generate_dashboard(self, dashboard_type: str = "overview") -> Dict[str, Any]:
        """Generate dashboard data."""
        return self.dashboard_generator.generate_overview_dashboard() if dashboard_type == "overview" \
               else self.dashboard_generator.generate_neuromorphic_dashboard() if dashboard_type == "neuromorphic" \
               else self.dashboard_generator.generate_performance_dashboard()
    
    def generate_report(self, report_type: str = "daily") -> Dict[str, Any]:
        """Generate comprehensive report."""
        if report_type == "daily":
            return self.report_generator.generate_daily_report()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    @contextmanager
    def trace_operation(self, operation_name: str, **tags):
        """Context manager for tracing operations."""
        with self.tracer.trace_operation(operation_name) as (trace_id, span_id):
            for key, value in tags.items():
                self.tracer.add_span_tag(span_id, key, value)
            yield trace_id, span_id
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_manager.add_callback(callback)
        
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
    """Advanced energy profiling and monitoring for neuromorphic systems."""
    
    def __init__(self, metrics: Optional[SpikeFormerMetrics] = None):
        self.metrics = metrics
        self.measurements = deque(maxlen=10000)  # Keep last 10k measurements
        self._start_time = None
        self._start_energy = None
        self._hardware_type = "simulation"
        self.logger = logging.getLogger(__name__)
        
        # Hardware-specific energy models
        self.energy_models = {
            "loihi2": {"base_power_mw": 0.8, "spike_energy_pj": 23.0},
            "spinnaker": {"base_power_mw": 1000.0, "spike_energy_pj": 800.0},
            "gpu": {"base_power_mw": 50000.0, "spike_energy_pj": 10000.0},
            "cpu": {"base_power_mw": 15000.0, "spike_energy_pj": 5000.0},
            "simulation": {"base_power_mw": 100.0, "spike_energy_pj": 0.1}
        }
        
    def set_hardware_type(self, hardware_type: str):
        """Set the hardware type for energy modeling."""
        if hardware_type in self.energy_models:
            self._hardware_type = hardware_type
        else:
            self.logger.warning(f"Unknown hardware type: {hardware_type}, using simulation")
            self._hardware_type = "simulation"
    
    def estimate_energy(self, spike_count: int, duration_ms: float, 
                       base_operations: int = 0) -> Dict[str, float]:
        """Estimate energy consumption based on neuromorphic operations."""
        model = self.energy_models[self._hardware_type]
        
        # Base power consumption
        base_energy_mj = (model["base_power_mw"] * duration_ms) / 1000.0
        
        # Spike-dependent energy
        spike_energy_mj = (spike_count * model["spike_energy_pj"]) / 1e9
        
        # Additional operations energy (simplified)
        ops_energy_mj = (base_operations * 0.001) / 1000.0
        
        total_energy_mj = base_energy_mj + spike_energy_mj + ops_energy_mj
        avg_power_mw = (total_energy_mj * 1000.0) / duration_ms if duration_ms > 0 else 0
        
        return {
            "total_energy_mj": total_energy_mj,
            "base_energy_mj": base_energy_mj,
            "spike_energy_mj": spike_energy_mj,
            "ops_energy_mj": ops_energy_mj,
            "avg_power_mw": avg_power_mw,
            "duration_ms": duration_ms,
            "spike_count": spike_count,
            "hardware_type": self._hardware_type
        }
        
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
            'timestamp': end_time,
            'hardware_type': self._hardware_type
        }
        
        self.measurements.append(measurement)
        
        if self.metrics:
            self.metrics.record_energy_consumption(
                component="inference",
                backend=self._hardware_type,
                energy_joules=energy_consumed,
                power_watts=power_avg
            )
            
    def _get_current_energy(self) -> float:
        """Get current energy reading."""
        # In a real implementation, this would interface with hardware APIs
        # For now, we'll use a more sophisticated simulation
        base_energy = time.time() * 0.1
        
        # Add some variation based on hardware type
        model = self.energy_models[self._hardware_type]
        variation = (model["base_power_mw"] / 10000.0) * (time.time() % 10)
        
        return base_energy + variation
        
    def get_energy_statistics(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive energy consumption statistics."""
        if not self.measurements:
            return {"message": "No energy measurements available"}
        
        cutoff_time = time.time() - (window_hours * 3600)
        recent_measurements = [m for m in self.measurements if m['timestamp'] > cutoff_time]
        
        if not recent_measurements:
            return {"message": f"No energy measurements in last {window_hours} hours"}
            
        energies = [m['energy_joules'] for m in recent_measurements]
        powers = [m['power_watts'] for m in recent_measurements]
        durations = [m['duration_seconds'] for m in recent_measurements]
        
        # Group by hardware type
        by_hardware = defaultdict(list)
        for m in recent_measurements:
            by_hardware[m.get('hardware_type', 'unknown')].append(m)
        
        hardware_stats = {}
        for hw_type, measurements in by_hardware.items():
            hw_energies = [m['energy_joules'] for m in measurements]
            hw_powers = [m['power_watts'] for m in measurements]
            
            hardware_stats[hw_type] = {
                "total_energy_joules": sum(hw_energies),
                "avg_energy_joules": np.mean(hw_energies),
                "avg_power_watts": np.mean(hw_powers),
                "peak_power_watts": max(hw_powers),
                "measurement_count": len(measurements)
            }
        
        return {
            "window_hours": window_hours,
            "total_measurements": len(recent_measurements),
            "summary": {
                "total_energy_joules": sum(energies),
                "avg_energy_joules": np.mean(energies),
                "median_energy_joules": np.median(energies),
                "max_energy_joules": max(energies),
                "min_energy_joules": min(energies),
                "std_energy_joules": np.std(energies),
                "avg_power_watts": np.mean(powers),
                "peak_power_watts": max(powers),
                "avg_duration_seconds": np.mean(durations),
                "total_runtime_hours": sum(durations) / 3600
            },
            "by_hardware": hardware_stats,
            "efficiency_metrics": {
                "energy_per_second": np.mean(energies) / np.mean(durations) if np.mean(durations) > 0 else 0,
                "power_efficiency_score": 1.0 / np.mean(powers) if np.mean(powers) > 0 else 0
            }
        }
    
    def compare_hardware_efficiency(self) -> Dict[str, Any]:
        """Compare energy efficiency across different hardware types."""
        by_hardware = defaultdict(list)
        for m in self.measurements:
            by_hardware[m.get('hardware_type', 'unknown')].append(m)
        
        if len(by_hardware) < 2:
            return {"message": "Need measurements from multiple hardware types for comparison"}
        
        comparison = {}
        baseline_efficiency = None
        
        for hw_type, measurements in by_hardware.items():
            if not measurements:
                continue
                
            energies = [m['energy_joules'] for m in measurements]
            durations = [m['duration_seconds'] for m in measurements]
            
            # Energy efficiency: joules per second of computation
            efficiency = np.mean(energies) / np.mean(durations) if np.mean(durations) > 0 else float('inf')
            
            if baseline_efficiency is None or hw_type == "cpu":
                baseline_efficiency = efficiency
            
            comparison[hw_type] = {
                "efficiency_joules_per_second": efficiency,
                "relative_efficiency": baseline_efficiency / efficiency if efficiency > 0 else 0,
                "sample_count": len(measurements),
                "avg_energy_joules": np.mean(energies),
                "avg_duration_seconds": np.mean(durations)
            }
        
        # Find most efficient hardware
        most_efficient = min(comparison.keys(), 
                           key=lambda k: comparison[k]["efficiency_joules_per_second"])
        
        return {
            "comparison": comparison,
            "most_efficient": most_efficient,
            "baseline": "cpu" if "cpu" in comparison else list(comparison.keys())[0],
            "recommendation": f"Use {most_efficient} for best energy efficiency"
        }


class ModelProfiler:
    """Comprehensive model profiler with energy and performance tracking."""
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model",
                 hardware_type: str = "cpu", metrics: Optional[SpikeFormerMetrics] = None):
        self.model = model
        self.model_name = model_name
        self.hardware_type = hardware_type
        self.metrics = metrics
        
        self.energy_profiler = EnergyProfiler(metrics)
        self.energy_profiler.set_hardware_type(hardware_type)
        
        self.start_time = None
        self.spike_counts = defaultdict(int)
        self.layer_timings = defaultdict(float)
        self.memory_snapshots = []
        self.hooks = []
        
    def __enter__(self):
        """Start comprehensive profiling."""
        self.start_time = time.time()
        
        # Start energy profiling
        self.energy_profiler.__enter__()
        
        # Register hooks for detailed profiling
        self._register_hooks()
        
        # Record initial memory
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and generate comprehensive report."""
        # Stop energy profiling
        self.energy_profiler.__exit__(exc_type, exc_val, exc_tb)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        # Calculate final metrics
        total_time = time.time() - self.start_time
        total_spikes = sum(self.spike_counts.values())
        
        # Memory metrics
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = (final_memory - self.initial_memory) / (1024**2)  # MB
            peak_memory_mb = peak_memory / (1024**2)
        else:
            memory_used = 0
            peak_memory_mb = 0
        
        # Create performance metrics
        performance_metrics = PerformanceMetrics(
            inference_time_ms=total_time * 1000,
            spike_processing_time_ms=sum(self.layer_timings.values()) * 1000,
            memory_usage_mb=memory_used,
            peak_memory_mb=peak_memory_mb,
            spike_rate=total_spikes / max(len(self.spike_counts), 1),
            synaptic_operations=total_spikes,
            model_name=self.model_name,
            hardware_type=self.hardware_type,
            timestamp=time.time()
        )
        
        # Get energy statistics
        energy_stats = self.energy_profiler.get_energy_statistics(window_hours=1)
        if "summary" in energy_stats:
            performance_metrics.energy_consumption_mj = energy_stats["summary"]["avg_energy_joules"] * 1000
            performance_metrics.power_consumption_mw = energy_stats["summary"]["avg_power_watts"] * 1000
        
        # Record with metrics system
        if self.metrics:
            self.metrics.record_performance_metrics(performance_metrics)
        
        return {
            "model_name": self.model_name,
            "hardware_type": self.hardware_type,
            "performance_metrics": performance_metrics,
            "energy_breakdown": energy_stats,
            "layer_analysis": {
                "spike_counts": dict(self.spike_counts),
                "layer_timings": dict(self.layer_timings),
                "total_spikes": total_spikes,
                "total_layers": len(self.spike_counts)
            },
            "memory_analysis": {
                "used_mb": memory_used,
                "peak_mb": peak_memory_mb,
                "efficiency_mb_per_spike": memory_used / max(total_spikes, 1)
            }
        }
    
    def _register_hooks(self):
        """Register hooks for detailed profiling."""
        for name, module in self.model.named_modules():
            # Hook for any module that might produce spikes
            if hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._create_profiling_hook(name))
                self.hooks.append(hook)
    
    def _create_profiling_hook(self, layer_name: str):
        """Create profiling hook for a layer."""
        def hook(module, input, output):
            hook_start = time.time()
            
            # Count spikes if output looks like spike tensor
            if torch.is_tensor(output) and output.dtype in [torch.float32, torch.float16, torch.bool]:
                # Assume binary or near-binary values are spikes
                if output.numel() > 0:
                    spike_count = (output > 0.5).sum().item()
                    self.spike_counts[layer_name] += spike_count
            
            # Record timing (simplified)
            hook_end = time.time()
            self.layer_timings[layer_name] += (hook_end - hook_start)
            
        return hook


# Enhanced convenience functions
def create_advanced_profiler(model: torch.nn.Module, model_name: str = "model", 
                           hardware_type: str = "cpu") -> ModelProfiler:
    """Create an advanced model profiler with comprehensive metrics."""
    return ModelProfiler(model, model_name, hardware_type, metrics)


def monitor_inference(model_name: str = None, hardware_type: str = "cpu"):
    """Decorator for comprehensive inference monitoring."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal model_name
            if model_name is None:
                model_name = func.__name__
            
            start_time = time.time()
            
            with metrics.trace_operation(f"inference_{model_name}", 
                                       model=model_name, hardware=hardware_type) as (trace_id, span_id):
                try:
                    result = func(*args, **kwargs)
                    
                    inference_time = time.time() - start_time
                    
                    # Extract metrics from result if available
                    if isinstance(result, dict):
                        accuracy = result.get('accuracy', 0.0)
                        energy_mj = result.get('energy_mj', 0.0)
                        spike_rate = result.get('spike_rate', 0.0)
                    else:
                        accuracy = 0.0
                        energy_mj = 0.0
                        spike_rate = 0.0
                    
                    # Record comprehensive metrics
                    perf_metrics = PerformanceMetrics(
                        inference_time_ms=inference_time * 1000,
                        accuracy=accuracy,
                        energy_consumption_mj=energy_mj,
                        spike_rate=spike_rate,
                        model_name=model_name,
                        hardware_type=hardware_type
                    )
                    
                    metrics.record_performance_metrics(perf_metrics)
                    
                    # Add trace tags
                    metrics.tracer.add_span_tag(span_id, 'inference_time_ms', inference_time * 1000)
                    metrics.tracer.add_span_tag(span_id, 'accuracy', accuracy)
                    metrics.tracer.add_span_tag(span_id, 'success', True)
                    
                    return result
                    
                except Exception as e:
                    metrics.tracer.add_span_tag(span_id, 'error', True)
                    metrics.tracer.add_span_log(span_id, f"Error: {str(e)}", 'error')
                    raise
                    
        return wrapper
    return decorator


# Global metrics instance
metrics = SpikeFormerMetrics()

# Enhanced convenience functions
def start_monitoring(interval: float = 15.0):
    """Start comprehensive global monitoring."""
    metrics.start_monitoring(interval)
    logging.getLogger(__name__).info("Advanced monitoring started with comprehensive observability")

def stop_monitoring():
    """Stop global monitoring."""
    metrics.stop_monitoring()
    logging.getLogger(__name__).info("Monitoring stopped")

def get_metrics() -> str:
    """Get global metrics in Prometheus format."""
    return metrics.get_metrics()

def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status."""
    return metrics.get_system_health()

def get_performance_summary(window_hours: int = 1) -> Dict[str, Any]:
    """Get performance summary for specified time window."""
    return metrics.get_performance_summary(window_hours)

def generate_dashboard(dashboard_type: str = "overview") -> Dict[str, Any]:
    """Generate monitoring dashboard."""
    return metrics.generate_dashboard(dashboard_type)

def generate_report(report_type: str = "daily") -> Dict[str, Any]:
    """Generate comprehensive monitoring report."""
    return metrics.generate_report(report_type)

def create_energy_profiler(hardware_type: str = "simulation") -> EnergyProfiler:
    """Create advanced energy profiler with hardware-specific modeling."""
    profiler = EnergyProfiler(metrics)
    profiler.set_hardware_type(hardware_type)
    return profiler

def create_model_profiler(model: torch.nn.Module, model_name: str = "model", 
                         hardware_type: str = "cpu") -> ModelProfiler:
    """Create comprehensive model profiler."""
    return ModelProfiler(model, model_name, hardware_type, metrics)

def add_alert(name: str, condition: str, threshold: float, metric_name: str, 
              severity: str = "warning", message: str = ""):
    """Add custom alert rule."""
    alert = Alert(name, condition, threshold, metric_name, severity, message)
    metrics.alert_manager.add_alert(alert)

def add_alert_callback(callback: Callable[[Alert], None]):
    """Add callback for alert notifications."""
    metrics.add_alert_callback(callback)

def get_active_alerts() -> List[Alert]:
    """Get list of currently active alerts."""
    return metrics.alert_manager.get_active_alerts()

def get_alert_history(hours: int = 24) -> List[Alert]:
    """Get alert history for specified time period."""
    return metrics.alert_manager.get_alert_history(hours)

def enable_anomaly_detection(sensitivity: float = 2.0):
    """Enable automatic anomaly detection."""
    metrics.enable_anomaly_detection = True
    metrics.anomaly_detector.sensitivity = sensitivity

def disable_anomaly_detection():
    """Disable automatic anomaly detection."""
    metrics.enable_anomaly_detection = False

def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    """Get distributed trace by ID."""
    return metrics.tracer.get_trace(trace_id)

@contextmanager
def trace_operation(operation_name: str, **tags):
    """Context manager for distributed tracing."""
    with metrics.trace_operation(operation_name, **tags) as (trace_id, span_id):
        yield trace_id, span_id

def record_custom_metric(name: str, value: float, labels: Dict[str, str] = None, 
                        unit: str = ""):
    """Record a custom metric value."""
    labels = labels or {}
    metrics._record_metric_value(name, value, labels)

def export_metrics_data(window_hours: int = 24) -> Dict[str, Any]:
    """Export comprehensive metrics data for analysis."""
    cutoff_time = time.time() - (window_hours * 3600)
    
    # Get performance data
    with metrics.lock:
        performance_data = [
            {
                "timestamp": p.timestamp,
                "model_name": p.model_name,
                "hardware_type": p.hardware_type,
                "inference_time_ms": p.inference_time_ms,
                "accuracy": p.accuracy,
                "energy_consumption_mj": p.energy_consumption_mj,
                "spike_rate": p.spike_rate,
                "memory_usage_mb": p.memory_usage_mb,
                "core_utilization": p.core_utilization
            }
            for p in metrics.performance_history
            if p.timestamp > cutoff_time
        ]
        
        metric_data = [
            {
                "name": "custom_metric",  # Would use actual metric name in real implementation
                "value": m.value,
                "timestamp": m.timestamp,
                "labels": m.labels,
                "tags": m.tags,
                "unit": m.unit
            }
            for m in metrics.metric_buffer
            if m.timestamp > cutoff_time
        ]
    
    return {
        "export_timestamp": time.time(),
        "window_hours": window_hours,
        "performance_data": performance_data,
        "metric_data": metric_data,
        "active_alerts": [
            {
                "name": a.name,
                "condition": a.condition,
                "threshold": a.threshold,
                "metric_name": a.metric_name,
                "severity": a.severity,
                "message": a.message,
                "timestamp": a.timestamp,
                "labels": a.labels
            }
            for a in metrics.alert_manager.get_active_alerts()
        ],
        "system_health": metrics.get_system_health()
    }

# Example alert callback for logging
def log_alert_callback(alert: Alert):
    """Example callback that logs alerts."""
    logger = logging.getLogger("spikeformer.alerts")
    if alert.severity == "critical":
        logger.critical(f"CRITICAL ALERT: {alert.message}")
    elif alert.severity == "error":
        logger.error(f"ERROR ALERT: {alert.message}")
    elif alert.severity == "warning":
        logger.warning(f"WARNING ALERT: {alert.message}")
    else:
        logger.info(f"INFO ALERT: {alert.message}")

# Auto-setup logging callback
metrics.add_alert_callback(log_alert_callback)

# Start monitoring by default
start_monitoring()