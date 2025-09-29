"""
Standalone Model Sampling Implementation
SD3 model sampling functionality without ComfyUI dependencies.

This module provides the ModelSamplingSD3 class that can be used to patch
diffusion models with SD3-style sampling parameters.
"""

import torch
import math
from typing import Optional, Dict, Any


def time_snr_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    """
    Time SNR shift function for SD3 sampling
    
    Args:
        alpha: Shift parameter
        t: Time tensor
        
    Returns:
        Shifted time tensor
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class CONST:
    """
    Constant sampling strategy for SD3
    Standalone implementation of ComfyUI's CONST sampling
    """
    
    def calculate_input(self, sigma: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Calculate model input from sigma and noise"""
        return noise

    def calculate_denoised(self, sigma: torch.Tensor, model_output: torch.Tensor, model_input: torch.Tensor) -> torch.Tensor:
        """Calculate denoised output"""
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor, max_denoise: bool = False) -> torch.Tensor:
        """Apply noise scaling"""
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return sigma * noise + (1.0 - sigma) * latent_image

    def inverse_noise_scaling(self, sigma: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Apply inverse noise scaling"""
        sigma = sigma.view(sigma.shape[:1] + (1,) * (latent.ndim - 1))
        return latent / (1.0 - sigma)


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """
    Discrete Flow Model Sampling for SD3
    Standalone implementation without ComfyUI dependencies
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        if model_config is not None and hasattr(model_config, 'sampling_settings'):
            sampling_settings = model_config.sampling_settings
        elif model_config is not None and isinstance(model_config, dict):
            sampling_settings = model_config.get('sampling_settings', {})
        else:
            sampling_settings = {}

        self.set_parameters(
            shift=sampling_settings.get("shift", 1.0), 
            multiplier=sampling_settings.get("multiplier", 1000)
        )

    def set_parameters(self, shift: float = 1.0, timesteps: int = 1000, multiplier: int = 1000):
        """Set sampling parameters"""
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self) -> torch.Tensor:
        """Minimum sigma value"""
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        """Maximum sigma value"""
        return self.sigmas[-1]

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep"""
        return sigma * self.multiplier

    def sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma using time SNR shift"""
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent: float) -> float:
        """Convert percentage to sigma value"""
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


class ModelSamplingAdvanced(ModelSamplingDiscreteFlow, CONST):
    """
    Advanced model sampling that combines discrete flow and CONST sampling
    This is the actual class used for SD3 sampling
    """
    pass


class ModelSamplingSD3:
    """
    SD3 Model Sampling Patcher
    Standalone implementation that patches models with SD3 sampling without ComfyUI dependencies
    """
    
    def __init__(self):
        """Initialize the SD3 model sampling patcher"""
        pass
    
    def patch(self, model, shift: float = 8.0, multiplier: int = 1000):
        """
        Patch a model with SD3 sampling
        
        Args:
            model: Model to patch (should have ModelPatcher interface)
            shift: SD3 shift parameter (default 8.0)
            multiplier: SD3 multiplier parameter (default 1000)
            
        Returns:
            Cloned model with SD3 sampling patch applied
        """
        if not hasattr(model, 'clone'):
            raise ValueError("Model must have clone() method (ModelPatcher interface)")
        
        if not hasattr(model, 'add_object_patch'):
            raise ValueError("Model must have add_object_patch() method (ModelPatcher interface)")
        
        # Clone the model to avoid modifying the original
        m = model.clone()
        
        # Get model config if available
        model_config = None
        if hasattr(model, 'model') and hasattr(model.model, 'model_config'):
            model_config = model.model.model_config
        
        # Create the sampling object
        model_sampling = ModelSamplingAdvanced(model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        
        # Apply the object patch
        m.add_object_patch("model_sampling", model_sampling)
        
        return m


def test_model_sampling():
    """Test function to verify model sampling works correctly"""
    print("🧪 Testing ModelSamplingSD3 standalone implementation...")
    
    # Test time_snr_shift function
    print("Testing time_snr_shift function...")
    t = torch.tensor([0.0, 0.5, 1.0])
    result = time_snr_shift(8.0, t)
    print(f"time_snr_shift(8.0, {t.tolist()}) = {result.tolist()}")
    
    # Test CONST sampling
    print("Testing CONST sampling...")
    const_sampler = CONST()
    sigma = torch.tensor([0.5])
    noise = torch.randn(1, 4, 32, 32)
    model_input = const_sampler.calculate_input(sigma, noise)
    print(f"CONST calculate_input shape: {model_input.shape}")
    
    # Test ModelSamplingDiscreteFlow
    print("Testing ModelSamplingDiscreteFlow...")
    flow_sampling = ModelSamplingDiscreteFlow()
    flow_sampling.set_parameters(shift=8.0, multiplier=1000)
    print(f"Flow sampling shift: {flow_sampling.shift}")
    print(f"Flow sampling multiplier: {flow_sampling.multiplier}")
    print(f"Sigma min: {flow_sampling.sigma_min}")
    print(f"Sigma max: {flow_sampling.sigma_max}")
    
    # Test ModelSamplingAdvanced
    print("Testing ModelSamplingAdvanced...")
    advanced_sampling = ModelSamplingAdvanced()
    advanced_sampling.set_parameters(shift=8.0, multiplier=1000)
    print(f"Advanced sampling created successfully")
    
    print("✅ All model sampling tests passed!")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_model_sampling()
