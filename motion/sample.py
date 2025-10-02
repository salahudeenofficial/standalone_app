"""
Standalone sample.py module
Replicates ComfyUI's sample.py functionality for standalone use.
Provides core sampling functions without external ComfyUI dependencies.
"""

import torch
import numpy as np
import logging
import math
from typing import Optional, Union, Callable, Any

# Import motion utilities
from wan_vae_components.model_management import get_torch_device, unet_offload_device

# Configure logging
logger = logging.getLogger(__name__)


def repeat_to_batch_size(tensor, batch_size, dim=0):
    """
    Repeat tensor to match batch size
    
    Args:
        tensor: Input tensor
        batch_size: Target batch size
        dim: Dimension to repeat along
        
    Returns:
        Tensor repeated to match batch size
    """
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


def prepare_noise(latent_image, seed, noise_inds=None):
    """
    Creates random noise given a latent image and a seed.
    Optional arg skip can be used to skip and discard x number of noise generations for a given seed
    
    Args:
        latent_image: Template latent tensor for shape/dtype
        seed: Random seed for reproducibility
        noise_inds: Optional noise indices for advanced noise generation
        
    Returns:
        Random noise tensor
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def fix_empty_latent_channels(model, latent_image):
    """
    Resize the empty latent image so it has the right number of channels
    
    Args:
        model: Model object containing latent format information
        latent_image: Latent tensor to fix
        
    Returns:
        Fixed latent tensor
    """
    try:
        # Try to get latent format from model
        latent_format = None
        
        if hasattr(model, 'get_model_object'):
            try:
                latent_format = model.get_model_object("latent_format")
            except:
                pass
        elif hasattr(model, 'model') and hasattr(model.model, 'latent_format'):
            latent_format = model.model.latent_format
        elif hasattr(model, 'latent_format'):
            latent_format = model.latent_format
        elif hasattr(model, '_latent_format'):
            latent_format = model._latent_format
        
        if latent_format is None:
            # Special handling for PureVaceWanModel and similar models
            if hasattr(model, '__class__') and 'Vace' in model.__class__.__name__:
                # Assume WAN format: 16 channels, 3D latent
                latent_format = type('LatentFormat', (), {
                    'latent_channels': 16,
                    'latent_dimensions': 3
                })()
            else:
                # Fallback: assume standard latent format
                logger.warning("Could not get latent_format from model, using fallback")
                return latent_image
            
        # Fix channel count if needed
        if latent_format.latent_channels != latent_image.shape[1] and torch.count_nonzero(latent_image) == 0:
            latent_image = repeat_to_batch_size(latent_image, latent_format.latent_channels, dim=1)
            
        # Fix dimensions if needed
        if latent_format.latent_dimensions == 3 and latent_image.ndim == 4:
            latent_image = latent_image.unsqueeze(2)
            
        return latent_image
        
    except Exception as e:
        logger.warning(f"Failed to fix latent channels: {e}")
        return latent_image


def prepare_sampling(model, noise_shape, positive, negative, noise_mask):
    """
    Legacy function - kept for compatibility but not used anymore
    
    Args:
        model: Model object
        noise_shape: Shape of noise tensor
        positive: Positive conditioning
        negative: Negative conditioning  
        noise_mask: Optional noise mask
        
    Returns:
        Tuple of (model, positive, negative, noise_mask, [])
    """
    logger.warning("Warning: prepare_sampling isn't used anymore and can be removed")
    return model, positive, negative, noise_mask, []


def cleanup_additional_models(models):
    """
    Legacy function - kept for compatibility but not used anymore
    
    Args:
        models: List of models to cleanup
    """
    logger.warning("Warning: cleanup_additional_models isn't used anymore and can be removed")


def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, 
           denoise=1.0, disable_noise=False, start_step=None, last_step=None, 
           force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, 
           disable_pbar=False, seed=None):
    """
    Main sampling function - replicates ComfyUI's sample function
    
    Args:
        model: ModelPatcher containing the diffusion model
        noise: Initial noise tensor
        steps: Number of sampling steps
        cfg: CFG scale for guidance
        sampler_name: Name of the sampler algorithm
        scheduler: Name of the noise scheduler
        positive: Positive conditioning
        negative: Negative conditioning
        latent_image: Optional latent image for img2img
        denoise: Denoising strength (0.0 to 1.0)
        disable_noise: Whether to disable noise generation
        start_step: Start step for partial sampling
        last_step: End step for partial sampling
        force_full_denoise: Force complete denoising
        noise_mask: Optional mask for selective denoising
        sigmas: Custom sigma schedule
        callback: Progress callback function
        disable_pbar: Whether to disable progress bar
        seed: Random seed
        
    Returns:
        Denoised samples tensor
    """
    try:
        # Import the standalone ksampler
        from standalone_ksampler import StandaloneKSampler
        
        # Create KSampler instance
        sampler = StandaloneKSampler(
            model=model, 
            steps=steps, 
            device=model.load_device if hasattr(model, 'load_device') else get_torch_device(),
            sampler=sampler_name, 
            scheduler=scheduler, 
            denoise=denoise, 
            model_options=getattr(model, 'model_options', {})
        )

        # Perform sampling
        samples = sampler.sample(
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=cfg,
            latent_image=latent_image,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            denoise_mask=noise_mask,
            sigmas=sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed
        )
        
        # Move to intermediate device (usually CPU for memory management)
        samples = samples.to(unet_offload_device())
        return samples
        
    except ImportError:
        logger.error("Could not import StandaloneKSampler. Make sure standalone_ksampler.py is available.")
        raise RuntimeError("StandaloneKSampler not available")
    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        raise


def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, 
                  noise_mask=None, callback=None, disable_pbar=False, seed=None):
    """
    Custom sampling function with pre-calculated sigmas
    
    Args:
        model: ModelPatcher containing the diffusion model
        noise: Initial noise tensor
        cfg: CFG scale for guidance
        sampler: Sampler object
        sigmas: Pre-calculated sigma schedule
        positive: Positive conditioning
        negative: Negative conditioning
        latent_image: Optional latent image for img2img
        noise_mask: Optional mask for selective denoising
        callback: Progress callback function
        disable_pbar: Whether to disable progress bar
        seed: Random seed
        
    Returns:
        Denoised samples tensor
    """
    try:
        # Import the standalone ksampler
        from standalone_ksampler import StandaloneKSampler
        
        # Create KSampler instance with custom sigmas
        sampler_instance = StandaloneKSampler(
            model=model,
            steps=len(sigmas) - 1,
            device=model.load_device if hasattr(model, 'load_device') else get_torch_device(),
            sampler="euler",  # Default sampler for custom sampling
            scheduler="simple",  # Default scheduler for custom sampling
            denoise=1.0,
            model_options=getattr(model, 'model_options', {})
        )

        # Perform sampling with custom sigmas
        samples = sampler_instance.sample(
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=cfg,
            latent_image=latent_image,
            denoise_mask=noise_mask,
            sigmas=sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed
        )
        
        # Move to intermediate device
        samples = samples.to(unet_offload_device())
        return samples
        
    except ImportError:
        logger.error("Could not import StandaloneKSampler. Make sure standalone_ksampler.py is available.")
        raise RuntimeError("StandaloneKSampler not available")
    except Exception as e:
        logger.error(f"Custom sampling failed: {e}")
        raise


# Convenience functions for common operations

def create_noise_for_latent(latent_shape, seed=None, device=None):
    """
    Create random noise for a given latent shape
    
    Args:
        latent_shape: Shape tuple for the latent tensor
        seed: Random seed (optional)
        device: Target device (auto-detected if None)
        
    Returns:
        Random noise tensor
    """
    if device is None:
        device = get_torch_device()
        
    if seed is not None:
        generator = torch.manual_seed(seed)
        noise = torch.randn(latent_shape, device='cpu', generator=generator)
    else:
        noise = torch.randn(latent_shape, device='cpu')
        
    return noise.to(device)


def prepare_latent_for_sampling(latent_dict, model=None):
    """
    Prepare latent dictionary for sampling
    
    Args:
        latent_dict: Dictionary containing 'samples' key
        model: Optional model for fixing latent channels
        
    Returns:
        Prepared latent tensor
    """
    if 'samples' not in latent_dict:
        raise ValueError("Latent dictionary must contain 'samples' key")
        
    latent_image = latent_dict['samples']
    
    # Fix empty latent channels if model is provided
    if model is not None:
        latent_image = fix_empty_latent_channels(model, latent_image)
        
    return latent_image


def create_sampling_callback(progress_callback=None):
    """
    Create a sampling progress callback
    
    Args:
        progress_callback: Optional custom progress callback
        
    Returns:
        Callback function for sampling progress
    """
    def default_callback(step, total_steps):
        if step % max(1, total_steps // 10) == 0:  # Report every 10%
            progress = (step + 1) / total_steps * 100
            logger.info(f"Sampling progress: {progress:.1f}% ({step + 1}/{total_steps})")
    
    return progress_callback if progress_callback is not None else default_callback


# Test functions

def test_sample_functions():
    """Test the sample.py functions"""
    print("🧪 Testing sample.py functions")
    print("=" * 40)
    
    try:
        # Test noise generation
        print("1. Testing noise generation...")
        dummy_latent = torch.zeros(1, 4, 32, 32)
        noise = prepare_noise(dummy_latent, seed=42)
        print(f"   ✅ Noise shape: {noise.shape}, dtype: {noise.dtype}")
        
        # Test noise with indices
        noise_inds = np.array([0, 1, 0, 1])
        noise_with_inds = prepare_noise(dummy_latent, seed=42, noise_inds=noise_inds)
        print(f"   ✅ Noise with indices shape: {noise_with_inds.shape}")
        
        # Test latent channel fixing
        print("\n2. Testing latent channel fixing...")
        
        class MockLatentFormat:
            def __init__(self):
                self.latent_channels = 4
                self.latent_dimensions = 2
                
        class MockModel:
            def get_model_object(self, name):
                if name == "latent_format":
                    return MockLatentFormat()
                return None
        
        mock_model = MockModel()
        fixed_latent = fix_empty_latent_channels(mock_model, dummy_latent)
        print(f"   ✅ Fixed latent shape: {fixed_latent.shape}")
        
        # Test convenience functions
        print("\n3. Testing convenience functions...")
        noise2 = create_noise_for_latent((1, 4, 32, 32), seed=123)
        print(f"   ✅ Created noise shape: {noise2.shape}")
        
        latent_dict = {'samples': dummy_latent}
        prepared_latent = prepare_latent_for_sampling(latent_dict, mock_model)
        print(f"   ✅ Prepared latent shape: {prepared_latent.shape}")
        
        callback = create_sampling_callback()
        print(f"   ✅ Created callback: {type(callback).__name__}")
        
        print("\n🎉 All sample.py tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_sample_functions()
