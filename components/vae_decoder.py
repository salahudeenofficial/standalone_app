"""
VAE decoder component for the standalone pipeline
Adapted from ComfyUI VAEDecode but simplified for direct use
"""

import torch

class VAEDecode:
    """Decode latent images back to pixel space"""
    
    def decode(self, vae, samples):
        """Decode latent samples using VAE"""
        images = vae.decode(samples["samples"])
        
        # Combine batches if needed
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
        return images 