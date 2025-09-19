"""
VAE decoder component for the standalone pipeline
Adapted from ComfyUI VAEDecode with tiled decoding fallback
Following ComfyUI VAEDecode implementation exactly
"""

import torch
import logging

class VAEDecode:
    """Decode latent images back to pixel space - ComfyUI compatible with tiled fallback"""
    
    def decode(self, vae, samples):
        """
        Decode latent samples using VAE with ComfyUI-style tiled fallback
        Following ComfyUI VAEDecode.decode() implementation exactly
        """
        try:
            # Try regular decode first (ComfyUI approach)
            images = vae.decode(samples["samples"])
            
        except Exception as e:
            # ComfyUI-style fallback to tiled decoding
            logging.warning("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            
            # Check if VAE supports tiled decoding
            if hasattr(vae, 'decode_tiled'):
                print("   🔧 Using VAE tiled decoding fallback...")
                
                # Extract tensor from dict for tiled decoding
                latent_tensor = samples["samples"]
                print(f"   📊 Latent tensor shape: {latent_tensor.shape}")
                
                # Determine dimensions for tiled decoding
                dims = latent_tensor.ndim - 2  # Subtract batch and channel dimensions
                
                if dims == 1:
                    # 1D tiled decoding
                    images = vae.decode_tiled(latent_tensor, tile_x=128, overlap=32)
                elif dims == 2:
                    # 2D tiled decoding
                    images = vae.decode_tiled(latent_tensor, tile_x=64, tile_y=64, overlap=16)
                elif dims == 3:
                    # 3D tiled decoding (video) - use conservative tile sizes
                    tile_size = 32  # Conservative for video
                    overlap = 8
                    images = vae.decode_tiled(
                        latent_tensor, 
                        tile_x=tile_size, 
                        tile_y=tile_size, 
                        tile_t=2,  # Small temporal tiles
                        overlap=overlap,
                        overlap_t=1
                    )
                else:
                    raise ValueError(f"Unsupported tensor dimensions: {latent_tensor.shape}")
                
                print("   ✅ Tiled decoding successful")
            else:
                # Fallback to CPU if no tiled decoding support
                print("   ⚠️  VAE doesn't support tiled decoding, falling back to CPU...")
                vae_cpu = vae.to('cpu')
                latent_cpu = {"samples": samples["samples"].cpu()}
                images = vae_cpu.decode(latent_cpu["samples"])
                images = images.to(samples["samples"].device)
                print("   ✅ CPU fallback successful")
        
        # Combine batches if needed (ComfyUI logic)
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            
        return (images,)  # Return tuple like ComfyUI 