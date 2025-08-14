"""
Sampler component for the standalone pipeline
Adapted from ComfyUI KSampler but simplified for direct use
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

import comfy.samplers
import comfy.sample

class KSampler:
    """K-Sampler for denoising latents"""
    
    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, 
               sampler_name, scheduler, denoise=1.0):
        """Sample using the provided parameters"""
        
        # Get device from model - handle ModelPatcher objects
        if hasattr(model, 'parameters'):
            device = next(model.parameters()).device
        elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            device = next(model.model.parameters()).device
        elif hasattr(model, 'device'):
            device = model.device
        else:
            # Default to CUDA if we can't determine device
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Get sampler and scheduler
        sampler = comfy.samplers.KSampler(model, steps, device, sampler_name, scheduler, denoise)
        
        # Generate noise tensor with the same shape as latent_image
        # The noise should have the same spatial dimensions but with 4 channels (latent space)
        noise_shape = latent_image.shape
        print(f"6a. Generating noise tensor with shape: {noise_shape}")
        
        # Set seed for reproducible noise generation
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate noise on the same device as the model
        noise = torch.randn(noise_shape, device=device, dtype=latent_image.dtype)
        print(f"6a. Generated noise tensor: {noise.shape} on {device}")
        
        # Sample the latent
        samples = sampler.sample(noise, positive, negative, cfg, latent_image=latent_image, seed=seed)
        
        return samples 