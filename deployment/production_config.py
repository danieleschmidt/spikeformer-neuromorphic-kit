"""Production deployment configuration and orchestration for neuromorphic systems."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"
    RESEARCH = "research"


class SecurityLevel(Enum):
    """Security levels for deployment."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class HardwareSpec:
    """Hardware specifications for deployment."""
    chip_type: str
    chip_count: int
    cores_per_chip: int
    memory_per_chip_mb: int
    total_memory_gb: float
    network_bandwidth_gbps: float
    power_budget_watts: Optional[float] = None
    thermal_limit_celsius: Optional[float] = None
    
    def __post_init__(self):
        if self.total_memory_gb is None:
            self.total_memory_gb = (self.chip_count * self.memory_per_chip_mb) / 1024


@dataclass
class ResourceLimits:
    """Resource limits and quotas."""
    max_cpu_cores: int = 16
    max_memory_gb: float = 32.0
    max_gpu_memory_gb: float = 16.0
    max_storage_gb: float = 100.0
    max_network_bandwidth_mbps: int = 1000
    max_concurrent_requests: int = 1000
    max_batch_size: int = 32
    timeout_seconds: int = 30
    max_model_size_gb: float = 5.0


@dataclass
class SecurityConfig:
    """Security configuration for deployment."""
    level: SecurityLevel = SecurityLevel.INTERNAL
    enable_encryption: bool = True
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_audit_logging: bool = True
    enable_rate_limiting: bool = True
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    api_key_required: bool = True
    jwt_secret_key: Optional[str] = None
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    def __post_init__(self):
        if self.level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
            self.enable_encryption = True
            self.enable_authentication = True
            self.enable_authorization = True
            self.api_key_required = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = True
    enable_health_checks: bool = True
    metrics_interval_seconds: int = 60
    log_level: str = "INFO"
    retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage_percent": 80.0,
        "memory_usage_percent": 85.0,
        "disk_usage_percent": 90.0,
        "error_rate_percent": 5.0,
        "latency_p99_ms": 1000.0,
        "temperature_celsius": 75.0
    })
    webhook_urls: List[str] = field(default_factory=list)
    email_alerts: List[str] = field(default_factory=list)
    slack_webhook: Optional[str] = None


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    enable_auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    cooldown_seconds: int = 300
    enable_predictive_scaling: bool = False
    enable_cost_optimization: bool = True


@dataclass
class NetworkConfig:
    """Network configuration for deployment."""
    port: int = 8080
    host: str = "0.0.0.0"
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_gzip: bool = True
    max_request_size_mb: float = 100.0
    connection_timeout_seconds: int = 30
    read_timeout_seconds: int = 60
    write_timeout_seconds: int = 60
    keep_alive_timeout_seconds: int = 5
    max_connections: int = 1000
    enable_http2: bool = True
    enable_websockets: bool = False


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "postgresql"  # postgresql, mysql, mongodb, redis
    host: str = "localhost"
    port: int = 5432
    database: str = "neuromorphic"
    username: str = "admin"
    password: Optional[str] = None
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    max_overflow: int = 20
    connection_timeout_seconds: int = 30
    enable_encryption: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 30


@dataclass
class ProductionConfig:
    """Comprehensive production deployment configuration."""
    
    # Basic configuration
    service_name: str = "neuromorphic-inference"
    version: str = "1.0.0"
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    deployment_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Infrastructure
    hardware_spec: HardwareSpec = field(default_factory=lambda: HardwareSpec(
        chip_type="loihi2",
        chip_count=4,
        cores_per_chip=128,
        memory_per_chip_mb=2048,
        total_memory_gb=8.0,
        network_bandwidth_gbps=10.0,
        power_budget_watts=500.0,
        thermal_limit_celsius=80.0
    ))
    
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Model configuration
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {
        "enable_caching": True,
        "enable_compression": True,
        "enable_batch_optimization": True,
        "enable_model_quantization": True,
        "enable_energy_optimization": True,
        "enable_distributed_inference": True,
        "enable_experiment_tracking": False,
        "enable_data_validation": True,
        "enable_model_versioning": True,
        "enable_canary_deployment": False
    })
    
    # Environment variables
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Deployment metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate hardware specs
        if self.hardware_spec.chip_count <= 0:
            issues.append("Hardware spec: chip_count must be positive")
        
        if self.hardware_spec.cores_per_chip <= 0:
            issues.append("Hardware spec: cores_per_chip must be positive")
        
        if self.hardware_spec.total_memory_gb <= 0:
            issues.append("Hardware spec: total_memory_gb must be positive")
        
        # Validate resource limits
        if self.resource_limits.max_memory_gb < 1.0:
            issues.append("Resource limits: max_memory_gb too low")
        
        if self.resource_limits.max_concurrent_requests <= 0:
            issues.append("Resource limits: max_concurrent_requests must be positive")
        
        # Validate security
        if self.security.level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]:
            if not self.security.api_key_required:
                issues.append("Security: API key required for confidential/restricted deployments")
            
            if not self.security.ssl_cert_path:
                issues.append("Security: SSL certificate required for confidential/restricted deployments")
        
        # Validate scaling
        if self.scaling.min_replicas <= 0:
            issues.append("Scaling: min_replicas must be positive")
        
        if self.scaling.max_replicas < self.scaling.min_replicas:
            issues.append("Scaling: max_replicas must be >= min_replicas")
        
        # Validate network
        if not (1 <= self.network.port <= 65535):
            issues.append("Network: port must be between 1 and 65535")
        
        # Environment-specific validations
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if not self.security.enable_authentication:
                issues.append("Production: authentication must be enabled")
            
            if not self.monitoring.enable_health_checks:
                issues.append("Production: health checks must be enabled")
            
            if self.monitoring.log_level == "DEBUG":
                issues.append("Production: DEBUG log level not recommended")
        
        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def save_to_file(self, filepath: Path, format: str = "yaml"):
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                f.write(self.to_json())
        elif format.lower() == "yaml":
            with open(filepath, 'w') as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclasses
        if 'hardware_spec' in data and isinstance(data['hardware_spec'], dict):
            data['hardware_spec'] = HardwareSpec(**data['hardware_spec'])
        
        if 'resource_limits' in data and isinstance(data['resource_limits'], dict):
            data['resource_limits'] = ResourceLimits(**data['resource_limits'])
        
        if 'security' in data and isinstance(data['security'], dict):
            security_data = data['security']
            if 'level' in security_data:
                security_data['level'] = SecurityLevel(security_data['level'])
            data['security'] = SecurityConfig(**security_data)
        
        if 'monitoring' in data and isinstance(data['monitoring'], dict):
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        if 'scaling' in data and isinstance(data['scaling'], dict):
            data['scaling'] = ScalingConfig(**data['scaling'])
        
        if 'network' in data and isinstance(data['network'], dict):
            data['network'] = NetworkConfig(**data['network'])
        
        if 'database' in data and isinstance(data['database'], dict):
            data['database'] = DatabaseConfig(**data['database'])
        
        if 'environment' in data:
            data['environment'] = DeploymentEnvironment(data['environment'])
        
        return cls(**data)

    @classmethod
    def from_file(cls, filepath: Path) -> 'ProductionConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() == '.json':
                data = json.load(f)
            elif filepath.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(data)

    @classmethod
    def from_environment(cls, env_prefix: str = "NEUROMORPHIC_") -> 'ProductionConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Basic configuration
        if f"{env_prefix}SERVICE_NAME" in os.environ:
            config.service_name = os.environ[f"{env_prefix}SERVICE_NAME"]
        
        if f"{env_prefix}VERSION" in os.environ:
            config.version = os.environ[f"{env_prefix}VERSION"]
        
        if f"{env_prefix}ENVIRONMENT" in os.environ:
            config.environment = DeploymentEnvironment(os.environ[f"{env_prefix}ENVIRONMENT"])
        
        # Network configuration
        if f"{env_prefix}PORT" in os.environ:
            config.network.port = int(os.environ[f"{env_prefix}PORT"])
        
        if f"{env_prefix}HOST" in os.environ:
            config.network.host = os.environ[f"{env_prefix}HOST"]
        
        # Security configuration
        if f"{env_prefix}API_KEY_REQUIRED" in os.environ:
            config.security.api_key_required = os.environ[f"{env_prefix}API_KEY_REQUIRED"].lower() == "true"
        
        if f"{env_prefix}JWT_SECRET_KEY" in os.environ:
            config.security.jwt_secret_key = os.environ[f"{env_prefix}JWT_SECRET_KEY"]
        
        if f"{env_prefix}SSL_CERT_PATH" in os.environ:
            config.security.ssl_cert_path = os.environ[f"{env_prefix}SSL_CERT_PATH"]
        
        if f"{env_prefix}SSL_KEY_PATH" in os.environ:
            config.security.ssl_key_path = os.environ[f"{env_prefix}SSL_KEY_PATH"]
        
        # Database configuration
        if f"{env_prefix}DB_HOST" in os.environ:
            config.database.host = os.environ[f"{env_prefix}DB_HOST"]
        
        if f"{env_prefix}DB_PORT" in os.environ:
            config.database.port = int(os.environ[f"{env_prefix}DB_PORT"])
        
        if f"{env_prefix}DB_NAME" in os.environ:
            config.database.database = os.environ[f"{env_prefix}DB_NAME"]
        
        if f"{env_prefix}DB_USER" in os.environ:
            config.database.username = os.environ[f"{env_prefix}DB_USER"]
        
        if f"{env_prefix}DB_PASSWORD" in os.environ:
            config.database.password = os.environ[f"{env_prefix}DB_PASSWORD"]
        
        # Collect all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config.environment_variables[key] = value
        
        return config


def create_development_config() -> ProductionConfig:
    """Create development environment configuration."""
    config = ProductionConfig()
    config.environment = DeploymentEnvironment.DEVELOPMENT
    config.service_name = "neuromorphic-dev"
    
    # Relaxed security for development
    config.security.level = SecurityLevel.INTERNAL
    config.security.api_key_required = False
    config.security.enable_authentication = False
    
    # Debug logging
    config.monitoring.log_level = "DEBUG"
    config.monitoring.enable_tracing = True
    
    # Smaller resource limits
    config.resource_limits.max_memory_gb = 8.0
    config.resource_limits.max_concurrent_requests = 100
    
    # Single chip for development
    config.hardware_spec.chip_count = 1
    config.hardware_spec.cores_per_chip = 64
    
    # No auto-scaling in development
    config.scaling.enable_auto_scaling = False
    config.scaling.min_replicas = 1
    config.scaling.max_replicas = 1
    
    return config


def create_staging_config() -> ProductionConfig:
    """Create staging environment configuration."""
    config = ProductionConfig()
    config.environment = DeploymentEnvironment.STAGING
    config.service_name = "neuromorphic-staging"
    
    # Production-like security but less restrictive
    config.security.level = SecurityLevel.INTERNAL
    config.security.api_key_required = True
    config.security.enable_authentication = True
    
    # Medium resource allocation
    config.resource_limits.max_memory_gb = 16.0
    config.resource_limits.max_concurrent_requests = 500
    
    # Fewer chips than production
    config.hardware_spec.chip_count = 2
    
    # Limited auto-scaling
    config.scaling.min_replicas = 1
    config.scaling.max_replicas = 5
    
    return config


def create_production_config() -> ProductionConfig:
    """Create production environment configuration."""
    config = ProductionConfig()
    config.environment = DeploymentEnvironment.PRODUCTION
    config.service_name = "neuromorphic-production"
    
    # Full security enabled
    config.security.level = SecurityLevel.CONFIDENTIAL
    config.security.enable_encryption = True
    config.security.enable_authentication = True
    config.security.enable_authorization = True
    config.security.enable_audit_logging = True
    config.security.api_key_required = True
    
    # Production monitoring
    config.monitoring.enable_metrics = True
    config.monitoring.enable_logging = True
    config.monitoring.enable_tracing = True
    config.monitoring.enable_health_checks = True
    config.monitoring.log_level = "INFO"
    
    # Full resource allocation
    config.resource_limits.max_memory_gb = 32.0
    config.resource_limits.max_concurrent_requests = 1000
    
    # Multiple chips for redundancy and performance
    config.hardware_spec.chip_count = 4
    
    # Full auto-scaling capability
    config.scaling.enable_auto_scaling = True
    config.scaling.enable_predictive_scaling = True
    config.scaling.min_replicas = 2
    config.scaling.max_replicas = 10
    
    # Enable all production features
    config.feature_flags.update({
        "enable_caching": True,
        "enable_compression": True,
        "enable_batch_optimization": True,
        "enable_model_quantization": True,
        "enable_energy_optimization": True,
        "enable_distributed_inference": True,
        "enable_data_validation": True,
        "enable_model_versioning": True,
    })
    
    return config


def create_edge_config() -> ProductionConfig:
    """Create edge deployment configuration."""
    config = ProductionConfig()
    config.environment = DeploymentEnvironment.EDGE
    config.service_name = "neuromorphic-edge"
    
    # Minimal security for edge devices
    config.security.level = SecurityLevel.PUBLIC
    config.security.api_key_required = False
    config.security.enable_authentication = False
    
    # Minimal monitoring to save resources
    config.monitoring.enable_tracing = False
    config.monitoring.log_level = "WARNING"
    config.monitoring.retention_days = 7
    
    # Very limited resources
    config.resource_limits.max_memory_gb = 4.0
    config.resource_limits.max_concurrent_requests = 50
    config.resource_limits.max_batch_size = 8
    
    # Single edge chip
    config.hardware_spec.chip_type = "edge"
    config.hardware_spec.chip_count = 1
    config.hardware_spec.cores_per_chip = 32
    config.hardware_spec.memory_per_chip_mb = 512
    config.hardware_spec.power_budget_watts = 10.0
    
    # No scaling for edge devices
    config.scaling.enable_auto_scaling = False
    config.scaling.min_replicas = 1
    config.scaling.max_replicas = 1
    
    # Optimize for energy efficiency
    config.feature_flags.update({
        "enable_energy_optimization": True,
        "enable_model_quantization": True,
        "enable_compression": True,
        "enable_batch_optimization": False,  # Limited batching on edge
        "enable_distributed_inference": False,
        "enable_caching": True,  # Local caching important for edge
    })
    
    return config


def create_research_config() -> ProductionConfig:
    """Create research environment configuration."""
    config = ProductionConfig()
    config.environment = DeploymentEnvironment.RESEARCH
    config.service_name = "neuromorphic-research"
    
    # Minimal security for research
    config.security.level = SecurityLevel.INTERNAL
    config.security.api_key_required = False
    
    # Extensive monitoring for research insights
    config.monitoring.enable_tracing = True
    config.monitoring.log_level = "DEBUG"
    config.monitoring.metrics_interval_seconds = 10  # High frequency
    
    # Flexible resource allocation
    config.resource_limits.max_memory_gb = 64.0
    config.resource_limits.max_concurrent_requests = 200
    
    # Multiple chip types for comparison
    config.hardware_spec.chip_count = 8
    
    # Manual scaling for research control
    config.scaling.enable_auto_scaling = False
    
    # Enable experimental features
    config.feature_flags.update({
        "enable_experiment_tracking": True,
        "enable_model_versioning": True,
        "enable_canary_deployment": True,
        "enable_data_validation": True,
    })
    
    return config


# Configuration factory
def create_config_for_environment(environment: str) -> ProductionConfig:
    """Create configuration for specified environment."""
    environment = environment.lower()
    
    if environment in ["dev", "development"]:
        return create_development_config()
    elif environment in ["stage", "staging"]:
        return create_staging_config()
    elif environment in ["prod", "production"]:
        return create_production_config()
    elif environment == "edge":
        return create_edge_config()
    elif environment in ["research", "lab"]:
        return create_research_config()
    else:
        raise ValueError(f"Unknown environment: {environment}")


if __name__ == "__main__":
    # Generate sample configurations
    environments = ["development", "staging", "production", "edge", "research"]
    
    output_dir = Path("config_samples")
    output_dir.mkdir(exist_ok=True)
    
    for env in environments:
        config = create_config_for_environment(env)
        
        # Validate configuration
        issues = config.validate()
        if issues:
            print(f"Configuration issues for {env}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"Configuration for {env}: ✓ Valid")
        
        # Save to files
        config.save_to_file(output_dir / f"{env}_config.yaml", format="yaml")
        config.save_to_file(output_dir / f"{env}_config.json", format="json")
    
    print(f"Sample configurations saved to {output_dir}")