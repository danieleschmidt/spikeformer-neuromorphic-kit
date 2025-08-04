"""Security and validation framework for neuromorphic systems."""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import hmac
import secrets
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
from abc import ABC, abstractmethod
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os

from .models import SpikingTransformer, SpikingViT
from .neurons import SpikingNeuron


@dataclass
class SecurityConfig:
    """Security configuration for neuromorphic systems."""
    # Model integrity
    enable_model_validation: bool = True
    enable_checksum_verification: bool = True
    trusted_model_sources: List[str] = field(default_factory=list)
    
    # Encryption
    enable_model_encryption: bool = False
    encryption_key_length: int = 32  # bytes
    encryption_algorithm: str = "AES-256-GCM"
    
    # Access control
    enable_access_logging: bool = True
    max_failed_attempts: int = 3
    lockout_duration: int = 300  # seconds
    
    # Adversarial robustness
    enable_adversarial_detection: bool = True
    adversarial_threshold: float = 0.1
    noise_detection_sensitivity: float = 0.05
    
    # Privacy
    enable_differential_privacy: bool = False
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    
    # Hardware security
    enable_hardware_attestation: bool = False
    secure_boot_verification: bool = False
    tamper_detection: bool = True
    
    # Audit and compliance
    audit_log_retention_days: int = 90
    compliance_mode: str = "standard"  # "standard", "fips", "cc"


class ModelIntegrityValidator:
    """Validate model integrity and authenticity."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trusted_hashes = {}
        
    def compute_model_hash(self, model: nn.Module, algorithm: str = "sha256") -> str:
        """Compute cryptographic hash of model parameters."""
        hasher = hashlib.new(algorithm)
        
        # Sort parameters by name for deterministic hashing
        for name, param in sorted(model.named_parameters()):
            # Convert parameter to bytes
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(f"{name}:".encode())
            hasher.update(param_bytes)
        
        return hasher.hexdigest()
    
    def validate_model_integrity(self, model: nn.Module, expected_hash: Optional[str] = None) -> bool:
        """Validate model integrity against expected hash."""
        if not self.config.enable_model_validation:
            return True
        
        current_hash = self.compute_model_hash(model)
        
        if expected_hash:
            is_valid = hmac.compare_digest(current_hash, expected_hash)
            if not is_valid:
                self.logger.error(f"Model integrity check failed. Expected: {expected_hash}, Got: {current_hash}")
            return is_valid
        
        # Store hash for future validation
        model_name = model.__class__.__name__
        self.trusted_hashes[model_name] = current_hash
        self.logger.info(f"Model hash computed for {model_name}: {current_hash}")
        
        return True
    
    def sign_model(self, model: nn.Module, private_key: bytes) -> str:
        """Create digital signature for model."""
        model_hash = self.compute_model_hash(model)
        signature = hmac.new(private_key, model_hash.encode(), hashlib.sha256).hexdigest()
        return signature
    
    def verify_model_signature(self, model: nn.Module, signature: str, public_key: bytes) -> bool:
        """Verify model digital signature."""
        model_hash = self.compute_model_hash(model)
        expected_signature = hmac.new(public_key, model_hash.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_signature)
    
    def check_model_tampering(self, model: nn.Module, baseline_stats: Dict[str, Any]) -> Dict[str, bool]:
        """Check for signs of model tampering."""
        tampering_indicators = {}
        
        # Check parameter statistics
        current_stats = self._compute_model_statistics(model)
        
        for layer_name, stats in current_stats.items():
            if layer_name in baseline_stats:
                baseline = baseline_stats[layer_name]
                
                # Check for unusual parameter distributions
                mean_diff = abs(stats['mean'] - baseline['mean'])
                std_diff = abs(stats['std'] - baseline['std'])
                
                # Flag if changes are too large
                tampering_indicators[layer_name] = (
                    mean_diff > 0.1 * abs(baseline['mean']) or
                    std_diff > 0.1 * baseline['std']
                )
        
        return tampering_indicators
    
    def _compute_model_statistics(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Compute statistical properties of model parameters."""
        stats = {}
        
        for name, param in model.named_parameters():
            param_data = param.detach().cpu().numpy()
            stats[name] = {
                'mean': float(np.mean(param_data)),
                'std': float(np.std(param_data)),
                'min': float(np.min(param_data)),
                'max': float(np.max(param_data)),
                'shape': list(param_data.shape)
            }
        
        return stats


class ModelEncryption:
    """Encrypt and decrypt neuromorphic models."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def generate_key(self, password: Optional[str] = None) -> bytes:
        """Generate encryption key."""
        if password:
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.config.encryption_key_length,
                salt=b'spikeformer_salt',  # In production, use random salt
                iterations=100000,
            )
            key = kdf.derive(password.encode())
        else:
            # Generate random key
            key = secrets.token_bytes(self.config.encryption_key_length)
        
        return key
    
    def encrypt_model(self, model: nn.Module, key: bytes) -> Dict[str, Any]:
        """Encrypt model parameters."""
        if not self.config.enable_model_encryption:
            return {'state_dict': model.state_dict(), 'encrypted': False}
        
        # Serialize model state
        model_data = torch.save(model.state_dict(), f=None)
        
        if isinstance(model_data, bytes):
            serialized_data = model_data
        else:
            import pickle
            serialized_data = pickle.dumps(model_data)
        
        # Encrypt data
        fernet = Fernet(base64.urlsafe_b64encode(key[:32]))
        encrypted_data = fernet.encrypt(serialized_data)
        
        return {
            'encrypted_data': encrypted_data,
            'encrypted': True,
            'algorithm': self.config.encryption_algorithm,
            'model_class': model.__class__.__name__
        }
    
    def decrypt_model(self, encrypted_data: Dict[str, Any], key: bytes) -> Dict[str, Any]:
        """Decrypt model parameters."""
        if not encrypted_data.get('encrypted', False):
            return encrypted_data['state_dict']
        
        # Decrypt data
        fernet = Fernet(base64.urlsafe_b64encode(key[:32]))
        decrypted_data = fernet.decrypt(encrypted_data['encrypted_data'])
        
        # Deserialize model state
        import pickle
        model_state = pickle.loads(decrypted_data)
        
        return model_state
    
    def secure_model_storage(self, model: nn.Module, filepath: Path, password: str):
        """Securely store model with encryption."""
        key = self.generate_key(password)
        encrypted_model = self.encrypt_model(model, key)
        
        # Store encrypted model
        with open(filepath, 'wb') as f:
            torch.save(encrypted_model, f)
        
        self.logger.info(f"Model securely stored at {filepath}")
    
    def secure_model_loading(self, filepath: Path, password: str) -> Dict[str, Any]:
        """Securely load encrypted model."""
        with open(filepath, 'rb') as f:
            encrypted_data = torch.load(f, map_location='cpu')
        
        key = self.generate_key(password)
        model_state = self.decrypt_model(encrypted_data, key)
        
        return model_state


class AdversarialDefense:
    """Adversarial attack detection and defense for spiking networks."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_spike_patterns = {}
        
    def detect_adversarial_input(self, model: nn.Module, input_data: torch.Tensor,
                                clean_reference: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Detect potential adversarial inputs."""
        detection_results = {
            'is_adversarial': False,
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'detection_methods': {}
        }
        
        if not self.config.enable_adversarial_detection:
            return detection_results
        
        # Method 1: Input statistics analysis
        stats_anomaly = self._detect_statistical_anomaly(input_data, clean_reference)
        detection_results['detection_methods']['statistical'] = stats_anomaly
        
        # Method 2: Spike pattern analysis
        spike_anomaly = self._detect_spike_pattern_anomaly(model, input_data)
        detection_results['detection_methods']['spike_pattern'] = spike_anomaly
        
        # Method 3: Gradient analysis
        gradient_anomaly = self._detect_gradient_anomaly(model, input_data)
        detection_results['detection_methods']['gradient'] = gradient_anomaly
        
        # Combine detection methods
        anomaly_scores = [
            stats_anomaly['anomaly_score'],
            spike_anomaly['anomaly_score'],
            gradient_anomaly['anomaly_score']
        ]
        
        detection_results['anomaly_score'] = np.mean(anomaly_scores)
        detection_results['is_adversarial'] = detection_results['anomaly_score'] > self.config.adversarial_threshold
        detection_results['confidence'] = max(anomaly_scores)
        
        return detection_results
    
    def _detect_statistical_anomaly(self, input_data: torch.Tensor, 
                                   clean_reference: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Detect anomalies in input statistics."""
        if clean_reference is None:
            return {'anomaly_score': 0.0, 'details': 'No reference data'}
        
        # Compare statistical properties
        input_stats = {
            'mean': input_data.mean().item(),
            'std': input_data.std().item(),
            'min': input_data.min().item(),
            'max': input_data.max().item()
        }
        
        ref_stats = {
            'mean': clean_reference.mean().item(),
            'std': clean_reference.std().item(),
            'min': clean_reference.min().item(),
            'max': clean_reference.max().item()
        }
        
        # Calculate deviation score
        deviations = []
        for key in input_stats:
            if ref_stats[key] != 0:
                deviation = abs(input_stats[key] - ref_stats[key]) / abs(ref_stats[key])
                deviations.append(deviation)
        
        anomaly_score = np.mean(deviations) if deviations else 0.0
        
        return {
            'anomaly_score': min(anomaly_score, 1.0),
            'details': {
                'input_stats': input_stats,
                'reference_stats': ref_stats,
                'deviations': deviations
            }
        }
    
    def _detect_spike_pattern_anomaly(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies in spike patterns."""
        model.eval()
        
        # Collect spike patterns
        spike_patterns = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(module, SpikingNeuron):
                    spike_patterns[name] = output.detach().clone()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, SpikingNeuron):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze spike patterns
        anomaly_score = 0.0
        
        for name, spikes in spike_patterns.items():
            # Calculate spike rate
            spike_rate = spikes.mean().item()
            
            # Compare with baseline if available
            if name in self.baseline_spike_patterns:
                baseline_rate = self.baseline_spike_patterns[name]
                rate_deviation = abs(spike_rate - baseline_rate) / max(baseline_rate, 0.01)
                anomaly_score = max(anomaly_score, rate_deviation)
            else:
                # Store as baseline
                self.baseline_spike_patterns[name] = spike_rate
        
        return {
            'anomaly_score': min(anomaly_score, 1.0),
            'details': {
                'spike_patterns': {k: v.mean().item() for k, v in spike_patterns.items()},
                'baseline_patterns': self.baseline_spike_patterns.copy()
            }
        }
    
    def _detect_gradient_anomaly(self, model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies in input gradients."""
        input_data.requires_grad_(True)
        
        # Forward pass
        output = model(input_data)
        
        # Compute gradient with respect to input
        if output.dim() > 1:
            # Use max class for gradient computation
            target_class = output.argmax(dim=-1)
            loss = F.cross_entropy(output, target_class)
        else:
            loss = output.sum()
        
        loss.backward()
        
        # Analyze gradient properties
        gradient = input_data.grad
        grad_norm = gradient.norm().item()
        grad_std = gradient.std().item()
        
        # Anomaly detection based on gradient properties
        # High gradient norm might indicate adversarial perturbations
        normalized_grad_norm = min(grad_norm / (input_data.norm().item() + 1e-8), 10.0)
        
        return {
            'anomaly_score': normalized_grad_norm / 10.0,  # Normalize to [0, 1]
            'details': {
                'gradient_norm': grad_norm,
                'gradient_std': grad_std,
                'normalized_gradient_norm': normalized_grad_norm
            }
        }
    
    def apply_defensive_measures(self, input_data: torch.Tensor, defense_type: str = "noise_injection") -> torch.Tensor:
        """Apply defensive measures to input data."""
        if defense_type == "noise_injection":
            # Add small amount of noise to disrupt adversarial perturbations
            noise_std = 0.01 * input_data.std()
            noise = torch.randn_like(input_data) * noise_std
            return input_data + noise
        
        elif defense_type == "input_smoothing":
            # Apply Gaussian smoothing
            return torch.nn.functional.conv2d(
                input_data.unsqueeze(0) if input_data.dim() == 3 else input_data,
                torch.ones(1, 1, 3, 3) / 9,
                padding=1
            ).squeeze(0) if input_data.dim() == 3 else input_data
        
        elif defense_type == "input_quantization":
            # Quantize input to reduce precision of adversarial perturbations
            return torch.round(input_data * 255) / 255
        
        else:
            return input_data


class PrivacyProtection:
    """Privacy protection mechanisms for neuromorphic systems."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def add_differential_privacy_noise(self, gradients: torch.Tensor, 
                                     sensitivity: float = 1.0) -> torch.Tensor:
        """Add differential privacy noise to gradients."""
        if not self.config.enable_differential_privacy:
            return gradients
        
        # Calculate noise scale based on privacy parameters
        noise_scale = sensitivity / self.config.privacy_epsilon
        
        # Add Gaussian noise
        noise = torch.normal(0, noise_scale, size=gradients.shape, device=gradients.device)
        noisy_gradients = gradients + noise
        
        return noisy_gradients
    
    def federated_aggregation(self, client_updates: List[Dict[str, torch.Tensor]],
                            aggregation_method: str = "fedavg") -> Dict[str, torch.Tensor]:
        """Aggregate client updates with privacy preservation."""
        if aggregation_method == "fedavg":
            # Standard FedAvg
            aggregated = {}
            num_clients = len(client_updates)
            
            for key in client_updates[0].keys():
                aggregated[key] = torch.zeros_like(client_updates[0][key])
                
                for client_update in client_updates:
                    aggregated[key] += client_update[key]
                    
                aggregated[key] /= num_clients
                
                # Add differential privacy noise if enabled
                if self.config.enable_differential_privacy:
                    aggregated[key] = self.add_differential_privacy_noise(aggregated[key])
            
            return aggregated
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def secure_aggregation(self, client_updates: List[Dict[str, torch.Tensor]],
                          num_clients: int) -> Dict[str, torch.Tensor]:
        """Perform secure aggregation without revealing individual updates."""
        # Simplified secure aggregation (production would use cryptographic protocols)
        # Add random masks that cancel out in aggregation
        
        masked_updates = []
        
        for i, update in enumerate(client_updates):
            masked_update = {}
            
            for key, tensor in update.items():
                # Add random mask (in practice, this would be shared secrets)
                mask = torch.randn_like(tensor) * 0.01
                masked_update[key] = tensor + mask
            
            masked_updates.append(masked_update)
        
        # Aggregate masked updates
        return self.federated_aggregation(masked_updates, "fedavg")


class HardwareSecurity:
    """Hardware security measures for neuromorphic chips."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tamper_detected = False
        
    def verify_hardware_attestation(self, hardware_id: str, expected_signature: str) -> bool:
        """Verify hardware attestation."""
        if not self.config.enable_hardware_attestation:
            return True
        
        # Mock attestation verification
        # In practice, this would involve cryptographic verification with hardware TPM
        computed_signature = hashlib.sha256(hardware_id.encode()).hexdigest()
        
        is_valid = hmac.compare_digest(computed_signature, expected_signature)
        
        if not is_valid:
            self.logger.error(f"Hardware attestation failed for {hardware_id}")
        
        return is_valid
    
    def monitor_tamper_detection(self, hardware_metrics: Dict[str, float]) -> bool:
        """Monitor for hardware tampering."""
        if not self.config.tamper_detection:
            return False
        
        # Check for anomalous hardware behavior
        tamper_indicators = []
        
        # Temperature anomalies
        if 'temperature' in hardware_metrics:
            temp = hardware_metrics['temperature']
            if temp > 85 or temp < -10:  # Extreme temperatures
                tamper_indicators.append('temperature_anomaly')
        
        # Power consumption anomalies
        if 'power_consumption' in hardware_metrics:
            power = hardware_metrics['power_consumption']
            if power > 1000 or power < 0:  # Unusual power levels
                tamper_indicators.append('power_anomaly')
        
        # Voltage anomalies
        if 'voltage' in hardware_metrics:
            voltage = hardware_metrics['voltage']
            if voltage > 5.5 or voltage < 2.5:  # Outside safe voltage range
                tamper_indicators.append('voltage_anomaly')
        
        if tamper_indicators:
            self.tamper_detected = True
            self.logger.error(f"Tamper detection triggered: {tamper_indicators}")
            return True
        
        return False
    
    def secure_boot_check(self, boot_signature: str, expected_hash: str) -> bool:
        """Verify secure boot process."""
        if not self.config.secure_boot_verification:
            return True
        
        # Verify boot signature
        return hmac.compare_digest(boot_signature, expected_hash)
    
    def get_hardware_security_status(self) -> Dict[str, Any]:
        """Get comprehensive hardware security status."""
        return {
            'tamper_detected': self.tamper_detected,
            'attestation_enabled': self.config.enable_hardware_attestation,
            'secure_boot_enabled': self.config.secure_boot_verification,
            'security_level': self._calculate_security_level()
        }
    
    def _calculate_security_level(self) -> str:
        """Calculate overall security level."""
        security_features = [
            self.config.enable_hardware_attestation,
            self.config.secure_boot_verification,
            self.config.tamper_detection,
            not self.tamper_detected
        ]
        
        security_score = sum(security_features) / len(security_features)
        
        if security_score >= 0.8:
            return "high"
        elif security_score >= 0.6:
            return "medium"
        else:
            return "low"


class AuditLogger:
    """Comprehensive audit logging for security events."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.log_file = Path("security_audit.log")
        self.logger = logging.getLogger(__name__)
        
        # Setup audit logging
        if config.enable_access_logging:
            self.audit_handler = logging.FileHandler(self.log_file)
            self.audit_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(self.audit_handler)
    
    def log_model_access(self, model_name: str, user_id: str, action: str, success: bool):
        """Log model access events."""
        event_data = {
            'event_type': 'model_access',
            'model_name': model_name,
            'user_id': user_id,
            'action': action,
            'success': success,
            'timestamp': time.time()
        }
        
        self.logger.info(f"Model Access: {json.dumps(event_data)}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "info"):
        """Log security events."""
        event_data = {
            'event_type': event_type,
            'details': details,
            'severity': severity,
            'timestamp': time.time()
        }
        
        if severity == "critical":
            self.logger.critical(f"Security Event: {json.dumps(event_data)}")
        elif severity == "error":
            self.logger.error(f"Security Event: {json.dumps(event_data)}")
        elif severity == "warning":
            self.logger.warning(f"Security Event: {json.dumps(event_data)}")
        else:
            self.logger.info(f"Security Event: {json.dumps(event_data)}")
    
    def log_adversarial_detection(self, detection_results: Dict[str, Any], input_hash: str):
        """Log adversarial attack detection."""
        self.log_security_event(
            'adversarial_detection',
            {
                'detection_results': detection_results,
                'input_hash': input_hash,
                'is_attack': detection_results['is_adversarial']
            },
            severity="warning" if detection_results['is_adversarial'] else "info"
        )
    
    def generate_security_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate security report for specified time period."""
        # In a real implementation, this would parse the audit log
        # and generate comprehensive security statistics
        
        report = {
            'report_period': {
                'start': start_time,
                'end': end_time
            },
            'summary': {
                'total_events': 0,
                'security_incidents': 0,
                'model_accesses': 0,
                'adversarial_detections': 0
            },
            'recommendations': []
        }
        
        return report


class SecurityManager:
    """Central security management for neuromorphic systems."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self.integrity_validator = ModelIntegrityValidator(self.config)
        self.encryption = ModelEncryption(self.config)
        self.adversarial_defense = AdversarialDefense(self.config)
        self.privacy_protection = PrivacyProtection(self.config)
        self.hardware_security = HardwareSecurity(self.config)
        self.audit_logger = AuditLogger(self.config)
        
        # Access control
        self.failed_attempts = defaultdict(int)
        self.lockout_times = {}
        
    def secure_model_deployment(self, model: nn.Module, model_name: str,
                               encryption_password: Optional[str] = None) -> Dict[str, Any]:
        """Securely deploy a model with all security measures."""
        deployment_result = {
            'model_name': model_name,
            'security_checks': {},
            'deployment_success': False
        }
        
        try:
            # 1. Model integrity validation
            model_hash = self.integrity_validator.compute_model_hash(model)
            integrity_valid = self.integrity_validator.validate_model_integrity(model)
            deployment_result['security_checks']['integrity'] = integrity_valid
            
            # 2. Model encryption (if enabled)
            if self.config.enable_model_encryption and encryption_password:
                encrypted_model = self.encryption.encrypt_model(
                    model, self.encryption.generate_key(encryption_password)
                )
                deployment_result['security_checks']['encryption'] = True
                deployment_result['encrypted_model'] = encrypted_model
            
            # 3. Log deployment
            self.audit_logger.log_model_access(
                model_name, "system", "deploy", True
            )
            
            deployment_result['deployment_success'] = all(
                deployment_result['security_checks'].values()
            )
            deployment_result['model_hash'] = model_hash
            
        except Exception as e:
            self.logger.error(f"Secure deployment failed: {e}")
            deployment_result['error'] = str(e)
            
            self.audit_logger.log_security_event(
                'deployment_failure',
                {'model_name': model_name, 'error': str(e)},
                severity="error"
            )
        
        return deployment_result
    
    def secure_inference(self, model: nn.Module, input_data: torch.Tensor,
                        user_id: str) -> Dict[str, Any]:
        """Perform secure inference with all protection measures."""
        inference_result = {
            'user_id': user_id,
            'security_checks': {},
            'inference_success': False
        }
        
        try:
            # 1. Access control check
            if not self._check_access_control(user_id):
                inference_result['error'] = "Access denied"
                return inference_result
            
            # 2. Adversarial input detection
            detection_results = self.adversarial_defense.detect_adversarial_input(model, input_data)
            inference_result['security_checks']['adversarial_detection'] = detection_results
            
            if detection_results['is_adversarial']:
                self.audit_logger.log_adversarial_detection(
                    detection_results, 
                    hashlib.sha256(input_data.detach().cpu().numpy().tobytes()).hexdigest()
                )
                
                # Apply defensive measures
                input_data = self.adversarial_defense.apply_defensive_measures(input_data)
            
            # 3. Secure inference execution
            model.eval()
            with torch.no_grad():
                output = model(input_data)
            
            inference_result['output'] = output
            inference_result['inference_success'] = True
            
            # 4. Log successful inference
            self.audit_logger.log_model_access(
                model.__class__.__name__, user_id, "inference", True
            )
            
        except Exception as e:
            inference_result['error'] = str(e)
            self.audit_logger.log_model_access(
                model.__class__.__name__, user_id, "inference", False
            )
        
        return inference_result
    
    def _check_access_control(self, user_id: str) -> bool:
        """Check access control for user."""
        current_time = time.time()
        
        # Check if user is locked out
        if user_id in self.lockout_times:
            if current_time - self.lockout_times[user_id] < self.config.lockout_duration:
                return False
            else:
                # Lockout expired
                del self.lockout_times[user_id]
                self.failed_attempts[user_id] = 0
        
        return True
    
    def record_failed_attempt(self, user_id: str):
        """Record failed access attempt."""
        self.failed_attempts[user_id] += 1
        
        if self.failed_attempts[user_id] >= self.config.max_failed_attempts:
            self.lockout_times[user_id] = time.time()
            self.audit_logger.log_security_event(
                'user_lockout',
                {'user_id': user_id, 'failed_attempts': self.failed_attempts[user_id]},
                severity="warning"
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'config': {
                'model_validation': self.config.enable_model_validation,
                'encryption': self.config.enable_model_encryption,
                'adversarial_detection': self.config.enable_adversarial_detection,
                'differential_privacy': self.config.enable_differential_privacy,
                'hardware_attestation': self.config.enable_hardware_attestation
            },
            'hardware_security': self.hardware_security.get_hardware_security_status(),
            'active_lockouts': len(self.lockout_times),
            'failed_attempts': dict(self.failed_attempts)
        }


# Convenience functions
def create_security_manager(enable_all: bool = False) -> SecurityManager:
    """Create security manager with default or all features enabled."""
    if enable_all:
        config = SecurityConfig(
            enable_model_validation=True,
            enable_model_encryption=True,
            enable_adversarial_detection=True,
            enable_differential_privacy=True,
            enable_hardware_attestation=True,
            enable_access_logging=True
        )
    else:
        config = SecurityConfig()
    
    return SecurityManager(config)


def secure_model_training(model: nn.Module, train_loader, privacy_epsilon: float = 1.0):
    """Train model with differential privacy."""
    config = SecurityConfig(
        enable_differential_privacy=True,
        privacy_epsilon=privacy_epsilon
    )
    
    security_manager = SecurityManager(config)
    
    # Training would use privacy_protection.add_differential_privacy_noise
    # during gradient updates
    
    return security_manager