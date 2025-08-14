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
        
        # Get device from model
        device = next(model.parameters()).device
        
        # Get sampler and scheduler
        sampler = comfy.samplers.KSampler(model, steps, device, sampler_name, scheduler, denoise)
        
        # Sample the latent
        samples = sampler.sample(None, positive, negative, cfg, latent_image=latent_image, seed=seed)
        
        return samples 