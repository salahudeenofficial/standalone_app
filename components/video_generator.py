"""
Video generation component for the standalone pipeline
Adapted from ComfyUI WanVaceToVideo but simplified for direct use
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

import comfy.model_management
import comfy.utils
import comfy.latent_formats

class WanVaceToVideo:
    """Generate initial video latents from reference image and control video"""
    
    def encode(self, positive, negative, vae, width, height, length, batch_size, 
               strength, control_video=None, control_masks=None, reference_image=None):
        """Encode reference image and control video to initial latents"""
        
        latent_length = ((length - 1) // 4) + 1
        
        # Process control video
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5
            
        # Process reference image
        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([
                reference_image, 
                comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))
            ], dim=1)
            
        # Process masks
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
                )
                
        # Process control video latents
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5
        
        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)
            
        # Process mask for latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact'
        ).squeeze(0)
        
        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]
            
        mask = mask.unsqueeze(0)
        
        # Set conditioning values
        positive = self._set_conditioning_values(
            positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}
        )
        negative = self._set_conditioning_values(
            negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}
        )
        
        # Create initial latent
        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8], 
            device=comfy.model_management.intermediate_device()
        )
        out_latent = {"samples": latent}
        
        return (positive, negative, out_latent, trim_latent)
    
    def _set_conditioning_values(self, conditioning, values, append=True):
        """Helper to set conditioning values"""
        # Simplified version - in full implementation this would use node_helpers
        return conditioning 