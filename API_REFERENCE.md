# 📚 SpikeFormer API Reference

## Core Classes and Functions

### Models (`spikeformer.models`)

#### `SpikingTransformer`
Main spiking transformer architecture with temporal dynamics.

```python
class SpikingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        timesteps: int = 32,
        threshold: float = 1.0
    )
```

**Parameters:**
- `vocab_size`: Vocabulary size for token embeddings
- `d_model`: Model dimension
- `nhead`: Number of attention heads  
- `num_layers`: Number of transformer layers
- `timesteps`: Number of simulation timesteps
- `threshold`: Spiking threshold

**Example:**
```python
from spikeformer import SpikingTransformer

model = SpikingTransformer(
    vocab_size=10000,
    d_model=768,
    nhead=12,
    num_layers=6,
    timesteps=32
)

# Forward pass
outputs = model(input_ids)  # Shape: (batch, timesteps, seq_len, d_model)
```

#### `SpikingAttention`
Event-driven spiking attention mechanism.

```python
class SpikingAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        timesteps: int = 32,
        dropout: float = 0.1,
        sparsity_target: float = 0.1
    )
```

---

### Neurons (`spikeformer.neurons`)

#### `LifNeuron`
Leaky Integrate-and-Fire neuron with surrogate gradients.

```python
class LifNeuron(SpikingNeuron):
    def __init__(
        self,
        threshold: float = 1.0,
        tau_mem: float = 20.0,
        reset: str = "subtract",
        surrogate_type: str = "fast_sigmoid"
    )
```

**Parameters:**
- `threshold`: Spiking threshold
- `tau_mem`: Membrane time constant
- `reset`: Reset mechanism ("subtract" or "zero")
- `surrogate_type`: Surrogate gradient function

**Example:**
```python
from spikeformer.neurons import LifNeuron

neuron = LifNeuron(threshold=1.0, tau_mem=20.0)
spikes = neuron(membrane_potential)  # Shape: (batch, timesteps, features)
```

#### `AdLifNeuron`
Adaptive Leaky Integrate-and-Fire with adaptation current.

```python
class AdLifNeuron(LifNeuron):
    def __init__(
        self,
        threshold: float = 1.0,
        tau_mem: float = 20.0,
        tau_adp: float = 200.0,
        beta: float = 0.1,
        **kwargs
    )
```

---

### Encoding (`spikeformer.encoding`)

#### `RateCoding`
Rate-based spike encoding for continuous inputs.

```python
class RateCoding(SpikeEncoder):
    def __init__(self, timesteps: int = 32)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Returns: (batch, timesteps, *input_shape)
```

#### `PoissonCoding`
Poisson-distributed spike encoding.

```python
class PoissonCoding(SpikeEncoder):
    def __init__(self, timesteps: int = 32, max_rate: float = 100.0)
```

**Example:**
```python
from spikeformer.encoding import RateCoding, PoissonCoding

# Rate coding
rate_encoder = RateCoding(timesteps=32)
rate_spikes = rate_encoder.encode(continuous_data)

# Poisson coding  
poisson_encoder = PoissonCoding(timesteps=32, max_rate=50.0)
poisson_spikes = poisson_encoder.encode(continuous_data)
```

---

### Conversion (`spikeformer.conversion`)

#### `SpikeformerConverter`
Convert pre-trained ANNs to SNNs.

```python
class SpikeformerConverter:
    def __init__(
        self,
        timesteps: int = 32,
        threshold: float = 1.0,
        spike_encoding: str = "rate",
        neuron_model: str = "LIF"
    )
    
    def convert(self, ann_model: nn.Module) -> nn.Module:
        # Returns converted spiking model
```

**Example:**
```python
from spikeformer import SpikeformerConverter
from transformers import ViTModel

# Load pre-trained model
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

# Convert to spiking
converter = SpikeformerConverter(timesteps=32, threshold=1.0)
spiking_vit = converter.convert(vit)
```

#### `ConversionPipeline`
Complete conversion pipeline with calibration.

```python
class ConversionPipeline:
    def __init__(
        self,
        calibration_data: DataLoader,
        conversion_mode: str = "layer_wise",
        optimization_target: str = "accuracy"
    )
    
    def convert(
        self,
        ann_model: nn.Module,
        calibration_samples: int = 1000,
        fine_tune_epochs: int = 5
    ) -> nn.Module
```

---

### Multi-Modal Fusion (`spikeformer.fusion`)

#### `MultiModalSpikeTransformer`
Complete multi-modal spiking transformer.

```python
class MultiModalSpikeTransformer(nn.Module):
    def __init__(
        self,
        fusion_config: FusionConfig,
        modality_dims: Dict[str, int],
        num_classes: int,
        num_layers: int = 6
    )
```

#### `CrossModalAttentionFusion`
Cross-modal attention fusion mechanism.

```python
class CrossModalAttentionFusion(nn.Module):
    def forward(
        self, 
        modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor
```

**Example:**
```python
from spikeformer.fusion import MultiModalSpikeTransformer, FusionConfig

config = FusionConfig(
    fusion_type="cross_attention",
    shared_dim=256,
    num_heads=8
)

model = MultiModalSpikeTransformer(
    fusion_config=config,
    modality_dims={"vision": 256, "audio": 256, "text": 256},
    num_classes=10
)

outputs = model({
    "vision": vision_spikes,
    "audio": audio_spikes, 
    "text": text_spikes
})
```

---

### Research (`spikeformer.research`)

#### `AdaptiveSpikeThreshold`
Adaptive threshold mechanism based on activity patterns.

```python
class AdaptiveSpikeThreshold(nn.Module):
    def __init__(
        self,
        initial_threshold: float = 1.0,
        adaptation_rate: float = 0.1,
        window_size: int = 100
    )
    
    def forward(
        self, 
        membrane_potential: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns: (spikes, current_threshold)
```

#### `MetaplasticSynapses`
Metaplastic synapses with activity-dependent plasticity.

```python
class MetaplasticSynapses(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        metaplastic_rate: float = 0.01
    )
```

#### `LifetimeLearningSpikingTransformer`
Spiking transformer with continual learning via EWC.

```python
class LifetimeLearningSpikingTransformer(SpikingTransformer):
    def begin_new_task(self) -> None
    def estimate_parameter_importance(self, dataloader, criterion) -> None
    def ewc_loss(self) -> torch.Tensor
```

---

### Hardware (`spikeformer.hardware`)

#### `NeuromorphicDeployer`
Abstract base for hardware deployment.

```python
class NeuromorphicDeployer(ABC):
    @abstractmethod
    def compile(self, model: nn.Module) -> HardwareModel
    
    @abstractmethod
    def deploy(self, compiled_model: HardwareModel) -> DeployedModel
```

#### Hardware-Specific Deployers

```python
# Intel Loihi 2
from spikeformer.hardware import Loihi2Deployer
deployer = Loihi2Deployer()
loihi_model = deployer.compile(spiking_model)

# SpiNNaker
from spikeformer.hardware import SpiNNakerDeployer
deployer = SpiNNakerDeployer(board_config="spin2-48chip")
spinn_model = deployer.deploy(spiking_model)
```

---

### Profiling (`spikeformer.profiling`)

#### `EnergyProfiler`
Energy consumption profiling and analysis.

```python
class EnergyProfiler:
    def __init__(self, backend: str = "simulation")
    
    def measure(self) -> EnergyRecording:
        # Context manager for energy measurement
        
    def compare(self, models: List[nn.Module]) -> ComparisonReport
```

**Example:**
```python
from spikeformer.profiling import EnergyProfiler

profiler = EnergyProfiler(backend="simulation")

with profiler.measure() as recording:
    for batch in dataloader:
        output = model(batch)

metrics = recording.get_metrics()
print(f"Energy per sample: {metrics['energy_per_sample_uj']:.2f} μJ")
```

#### `PowerMonitor`
Real-time power monitoring during inference.

```python
class PowerMonitor:
    def __init__(
        self,
        backend: str = "loihi2",
        sampling_rate_hz: int = 1000
    )
    
    def record(self) -> PowerRecording
```

---

### Training (`spikeformer.training`)

#### `SpikingTrainer`
Complete training system for spiking neural networks.

```python
class SpikingTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "AdamW",
        loss_fn: str = "spike_count_loss",
        learning_rate: float = 1e-4
    )
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        callbacks: List[str] = None
    ) -> TrainingHistory
```

#### `HybridTrainer`
Hybrid ANN-SNN training with knowledge distillation.

```python
class HybridTrainer:
    def __init__(
        self,
        model: nn.Module,
        ann_epochs: int = 50,
        snn_epochs: int = 20,
        surrogate_gradient: str = "fast_sigmoid"
    )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_model: nn.Module = None,
        distillation_weight: float = 0.5
    )
```

---

### Validation (`spikeformer.validation`)

#### `ComprehensiveValidator`
Complete model validation suite.

```python
class ComprehensiveValidator:
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD)
    
    def validate_model(self, model: nn.Module) -> List[ValidationResult]
    def validate_input_data(self, data: torch.Tensor) -> List[ValidationResult]
    def validate_hardware_compatibility(self, hardware_type: str) -> List[ValidationResult]
```

**Example:**
```python
from spikeformer.validation import ComprehensiveValidator

validator = ComprehensiveValidator()
results = validator.validate_model(spiking_model)

for result in results:
    if result.level == ValidationLevel.ERROR:
        print(f"Error: {result.message}")
```

---

### Caching (`spikeformer.caching`)

#### Global Cache Functions

```python
from spikeformer.caching import (
    get_tensor_cache,
    get_computation_cache,
    cached
)

# Tensor caching
tensor_cache = get_tensor_cache()
tensor_cache.put_tensor("key", tensor)
cached_tensor = tensor_cache.get_tensor("key")

# Computation caching decorator
@cached(ttl=3600)
def expensive_computation(data):
    return complex_operation(data)
```

#### `ModelStateCache`
Caching for model states and checkpoints.

```python
class ModelStateCache:
    def cache_model_state(self, model: nn.Module, epoch: int, step: int) -> str
    def restore_model_state(self, model: nn.Module, key: str) -> bool
```

---

### Concurrency (`spikeformer.concurrency`)

#### `AsyncSpikingInference`
Asynchronous inference with batching.

```python
class AsyncSpikingInference:
    def __init__(self, model: nn.Module, config: ConcurrencyConfig)
    
    async def submit_inference(
        self,
        inputs: torch.Tensor,
        callback: Optional[Callable] = None
    ) -> str
```

#### `ParallelSpikingTrainer`
Distributed training with DDP.

```python
class ParallelSpikingTrainer:
    def setup_distributed(self, rank: int, world_size: int)
    def train_step_parallel(
        self,
        batch_data: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]
```

---

### Internationalization (`spikeformer.i18n`)

#### Global Translation Functions

```python
from spikeformer.i18n import t, set_locale, format_energy, format_time

# Set locale
set_locale("es")  # Spanish

# Translate messages
error_msg = t("error.model_not_found", model_name="MyModel")

# Format units
energy_str = format_energy(1500.0)  # "1.50 mJ"
time_str = format_time(0.025)  # "25.00 ms"
```

#### `TranslationManager`
Comprehensive translation management.

```python
class TranslationManager:
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str
    def add_custom_translations(self, locale: str, translations: Dict[str, str])
    def get_available_locales(self) -> List[str]
```

---

### Compliance (`spikeformer.compliance`)

#### `PrivacyManager`
GDPR/CCPA compliant privacy management.

```python
class PrivacyManager:
    def __init__(self, organization_name: str, dpo_contact: str)
    
    def register_data_subject(self, subject: DataSubject) -> bool
    def record_consent(self, subject_id: str, purposes: List[ProcessingPurpose]) -> bool
    def handle_data_subject_request(self, request_type: str, subject_id: str) -> Dict[str, Any]
```

#### `ComplianceAuditor`
Automated compliance assessment.

```python
class ComplianceAuditor:
    def assess_gdpr_compliance(self) -> ComplianceReport
    def generate_compliance_report(self, framework: ComplianceFramework) -> ComplianceReport
```

---

### Cross-Platform (`spikeformer.cross_platform`)

#### Global Platform Functions

```python
from spikeformer.cross_platform import (
    get_platform_info,
    optimize_config_for_platform,
    check_platform_compatibility
)

# Get platform information
platform_info = get_platform_info()
print(f"OS: {platform_info.os.value}, Arch: {platform_info.architecture.value}")

# Optimize configuration
optimized_config = optimize_config_for_platform(base_config)

# Check compatibility
compatible, issues = check_platform_compatibility(requirements)
```

#### `CrossPlatformManager`
Complete platform compatibility management.

```python
class CrossPlatformManager:
    def get_platform_info(self) -> PlatformInfo
    def get_optimal_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]
    def check_hardware_compatibility(self, requirements: Dict[str, Any]) -> Tuple[bool, List[str]]
```

---

## Configuration Classes

### `SpikingConfig`
Core spiking model configuration.

```python
@dataclass
class SpikingConfig:
    timesteps: int = 32
    threshold: float = 1.0
    neuron_model: str = "LIF"
    spike_encoding: str = "rate"
    surrogate_gradient: str = "fast_sigmoid"
    tau_mem: float = 20.0
    tau_adp: float = 200.0
    beta: float = 0.1
    dropout: float = 0.1
    layer_norm: bool = True
```

### `FusionConfig`
Multi-modal fusion configuration.

```python
@dataclass
class FusionConfig:
    fusion_type: str = "cross_attention"
    shared_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    temperature: float = 1.0
    timesteps: int = 32
```

### `ConcurrencyConfig`
Parallel processing configuration.

```python
@dataclass
class ConcurrencyConfig:
    max_workers: int = 4
    batch_size_per_worker: int = 32
    enable_data_parallelism: bool = True
    enable_model_parallelism: bool = False
    async_data_loading: bool = True
    prefetch_factor: int = 2
```

---

## Utility Functions

### Energy and Time Formatting
```python
from spikeformer.i18n import format_energy, format_time, format_percentage

energy_str = format_energy(1500.0)  # "1.50 mJ"
time_str = format_time(0.025)  # "25.00 ms" 
percent_str = format_percentage(0.851, decimals=1)  # "85.1%"
```

### Parallel Processing
```python
from spikeformer.concurrency import parallel_spike_encoding

spikes = parallel_spike_encoding(
    inputs=continuous_data,
    encoding_type="rate",
    timesteps=32,
    num_workers=4
)
```

### Platform Detection
```python
from spikeformer.cross_platform import PlatformDetector

detector = PlatformDetector()
platform_info = detector.detect_platform()

if PlatformCapability.CUDA_GPU in platform_info.capabilities:
    print("CUDA GPU available")
```

---

## Error Handling

### Custom Exceptions
```python
from spikeformer.error_handling import (
    NeuromorphicError,
    HardwareCompatibilityError,
    ConversionError,
    SpikingDynamicsError
)

from spikeformer.i18n import LocalizedError

# Localized error with automatic translation
raise LocalizedError("error.model_not_found", model_name="MyModel")
```

### Error Context Manager
```python
from spikeformer.error_handling import error_context

with error_context("spike_encoding", "encoding_operation"):
    encoded_spikes = encoder.encode(data)
```

---

## Complete Example

```python
import torch
from spikeformer import *
from spikeformer.fusion import MultiModalSpikeTransformer, FusionConfig
from spikeformer.i18n import set_locale, t

# Set Spanish locale
set_locale("es")

# Create multi-modal model
fusion_config = FusionConfig(
    fusion_type="cross_attention",
    shared_dim=256,
    num_heads=8,
    timesteps=32
)

model = MultiModalSpikeTransformer(
    fusion_config=fusion_config,
    modality_dims={"vision": 256, "audio": 256, "text": 256},
    num_classes=10
)

# Create encoders
from spikeformer.multimodal import VisionSpikeEncoder, AudioSpikeEncoder, TextSpikeEncoder

vision_encoder = VisionSpikeEncoder(embed_dim=256, timesteps=32)
audio_encoder = AudioSpikeEncoder(embed_dim=256, timesteps=32)
text_encoder = TextSpikeEncoder(vocab_size=10000, embed_dim=256, timesteps=32)

# Energy profiling
profiler = EnergyProfiler(backend="simulation")

with profiler.measure() as recording:
    # Encode inputs
    vision_spikes = vision_encoder.encode(vision_data)
    audio_spikes = audio_encoder.encode(audio_data)
    text_spikes = text_encoder.encode(text_data)
    
    # Multi-modal inference
    outputs = model({
        "vision": vision_spikes,
        "audio": audio_spikes,
        "text": text_spikes
    })

# Get localized results
metrics = recording.get_metrics()
print(t("success.benchmark_complete", metrics=metrics))
print(f"Energía: {format_energy(metrics['energy_per_sample_uj'])}")
```

This completes the comprehensive API reference for the SpikeFormer neuromorphic toolkit.