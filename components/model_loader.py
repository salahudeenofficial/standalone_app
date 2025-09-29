"""
Model loading components for the standalone pipeline
Adapted from ComfyUI nodes but simplified for direct use
"""

import torch
import os
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

import comfy.sd
import comfy.utils
import comfy.model_management

class UNETLoader:
    """Load UNET diffusion models"""
    
    def __init__(self):
        self.models_dir = os.environ.get("COMFY_MODEL_PATH", "models")
        
    def load_unet(self, unet_name, weight_dtype="default"):
        """Load a UNET model"""
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
            
        unet_path = os.path.join(self.models_dir, "diffusion_models", unet_name)
        if not os.path.exists(unet_path):
            raise FileNotFoundError(f"UNET model not found: {unet_path}")
            
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return model

class CLIPLoader:
    """Load CLIP text encoder models"""
    
    def __init__(self):
        self.models_dir = os.environ.get("COMFY_MODEL_PATH", "models")
        
    def load_clip(self, clip_name, type="wan", device="default"):
        """Load a CLIP model"""
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
            
        clip_path = os.path.join(self.models_dir, "text_encoders", clip_name)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"CLIP model not found: {clip_path}")
            
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path], 
            embedding_directory=os.path.join(self.models_dir, "embeddings"),
            clip_type=clip_type, 
            model_options=model_options
        )
        return clip

class VAELoader:
    """Load VAE models"""
    
    def __init__(self):
        self.models_dir = os.environ.get("COMFY_MODEL_PATH", "models")
        
    def load_vae(self, vae_name):
        """Load a VAE model"""
        vae_path = os.path.join(self.models_dir, "vaes", vae_name)
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found: {vae_path}")
            
        # Load the VAE state dict using comfy.utils.load_torch_file
        vae_state_dict = comfy.utils.load_torch_file(vae_path)
        
        # Create VAE instance with the loaded state dict
        vae = comfy.sd.VAE(sd=vae_state_dict)
        vae.throw_exception_if_invalid()
        
        return vae 