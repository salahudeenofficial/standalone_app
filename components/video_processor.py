"""
Video processing component for the standalone pipeline
Adapted from ComfyUI TrimVideoLatent but simplified for direct use
"""

import torch

class TrimVideoLatent:
    """Trim video latents by removing specified number of frames"""
    
    def op(self, samples, trim_amount):
        """Trim the video latent samples"""
        samples_out = samples.copy()
        
        s1 = samples["samples"]
        samples_out["samples"] = s1[:, :, trim_amount:]
        return samples_out 