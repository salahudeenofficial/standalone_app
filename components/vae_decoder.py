"""
VAE decoder component for the standalone pipeline
Adapted from ComfyUI VAEDecode but simplified for direct use
Following ComfyUI VAEDecode implementation exactly
"""

import torch

class VAEDecode:
    """Decode latent images back to pixel space - ComfyUI compatible"""
    
    def decode(self, vae, samples):
        """
        Decode latent samples using VAE
        Following ComfyUI VAEDecode.decode() implementation exactly
        """
        images = vae.decode(samples["samples"])
        
        # Combine batches if needed (ComfyUI logic)
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
        return (images,)  # Return tuple like ComfyUI 