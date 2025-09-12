"""
WAN Latent Format for motion pipeline
Standalone implementation of WAN21 latent format processing
"""

import torch

class Wan21_LatentFormat:
    """WAN 2.1 Latent Format - standalone implementation"""
    
    def __init__(self):
        self.latent_channels = 16
        self.latent_dimensions = 3
        self.scale_factor = 1.0
        
        # WAN21 specific normalization parameters
        self.latents_mean = torch.tensor([
            -0.4134, 0.0715, -0.5517, 0.3632, 0.1922, 0.9497, -0.2503, 0.2921,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, self.latent_channels, 1, 1, 1)
        
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        """Process latent for VAE input (normalize)"""
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        """Process latent for VAE output (denormalize)"""
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
