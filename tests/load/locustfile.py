"""Load testing configuration using Locust for API endpoints."""

from locust import HttpUser, task, between
import json
import base64
import io
import numpy as np
from PIL import Image


class SpikeformerAPIUser(HttpUser):
    """Simulated user for load testing Spikeformer API endpoints."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize test data when user starts."""
        # Create sample image data
        self.sample_image = self._create_sample_image()
        self.sample_text = "This is a sample text for transformer processing."
    
    def _create_sample_image(self):
        """Create a sample image for testing."""
        # Create a random 224x224 RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to base64 for JSON transmission
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_b64
    
    @task(3)
    def test_health_check(self):
        """Test health check endpoint (most frequent)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def test_model_info(self):
        """Test model information endpoint."""
        with self.client.get("/api/v1/models", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    models = response.json()
                    if isinstance(models, list) and len(models) > 0:
                        response.success()
                    else:
                        response.failure("No models returned")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Model info failed: {response.status_code}")
    
    @task(1)
    def test_image_inference(self):
        """Test image inference endpoint."""
        payload = {
            "model_type": "vision",
            "model_name": "spikeformer-vit-base",
            "image": self.sample_image,
            "config": {
                "timesteps": 16,
                "threshold": 1.0,
                "return_energy_metrics": True
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post(
            "/api/v1/inference/image",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" in result and "energy_metrics" in result:
                        response.success()
                    else:
                        response.failure("Missing required fields in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 503:
                response.failure("Service temporarily unavailable")
            else:
                response.failure(f"Image inference failed: {response.status_code}")
    
    @task(1)
    def test_text_inference(self):
        """Test text inference endpoint."""
        payload = {
            "model_type": "language",
            "model_name": "spikeformer-bert-base",
            "text": self.sample_text,
            "config": {
                "timesteps": 20,
                "max_length": 128,
                "return_energy_metrics": True
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post(
            "/api/v1/inference/text",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" in result:
                        response.success()
                    else:
                        response.failure("Missing predictions in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Text inference failed: {response.status_code}")
    
    @task(1)
    def test_batch_inference(self):
        """Test batch inference endpoint."""
        payload = {
            "model_type": "vision",
            "model_name": "spikeformer-vit-small",
            "images": [self.sample_image] * 4,  # Batch of 4 images
            "config": {
                "timesteps": 8,
                "batch_size": 4,
                "optimize_for": "throughput"
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post(
            "/api/v1/inference/batch",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "predictions" in result and len(result["predictions"]) == 4:
                        response.success()
                    else:
                        response.failure("Incorrect batch predictions count")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Batch inference failed: {response.status_code}")


class HardwareAPIUser(HttpUser):
    """Specialized user for testing hardware-specific endpoints."""
    
    wait_time = between(2, 5)  # Longer wait times for hardware operations
    
    @task(1)
    def test_hardware_status(self):
        """Test hardware availability status."""
        with self.client.get("/api/v1/hardware/status", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    status = response.json()
                    if "loihi2" in status or "spinnaker" in status:
                        response.success()
                    else:
                        response.failure("No hardware status returned")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Hardware status failed: {response.status_code}")
    
    @task(1)
    def test_deployment_request(self):
        """Test model deployment to hardware."""
        payload = {
            "model_name": "spikeformer-vit-tiny",
            "hardware_target": "loihi2",
            "config": {
                "num_chips": 1,
                "optimization_level": 2,
                "power_budget_mw": 500
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        with self.client.post(
            "/api/v1/hardware/deploy",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 202]:  # Accept both sync and async responses
                response.success()
            elif response.status_code == 503:
                response.failure("Hardware unavailable")
            else:
                response.failure(f"Deployment failed: {response.status_code}")


class ModelManagementUser(HttpUser):
    """User for testing model management endpoints."""
    
    wait_time = between(3, 8)  # Longer wait for management operations
    
    @task(2)
    def test_list_models(self):
        """Test model listing endpoint."""
        with self.client.get("/api/v1/models?include_metrics=true", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model listing failed: {response.status_code}")
    
    @task(1)
    def test_model_metrics(self):
        """Test model performance metrics endpoint."""
        model_name = "spikeformer-vit-base"
        
        with self.client.get(f"/api/v1/models/{model_name}/metrics", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    metrics = response.json()
                    if "accuracy" in metrics and "energy_efficiency" in metrics:
                        response.success()
                    else:
                        response.failure("Missing metrics data")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Model metrics failed: {response.status_code}")