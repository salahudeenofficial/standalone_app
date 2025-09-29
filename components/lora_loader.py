"""
LoRA loading component for the standalone pipeline
Adapted from ComfyUI LoraLoader but simplified for direct use
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

class LoraLoader:
    """Load and apply LoRA models to diffusion and CLIP models"""
    
    def __init__(self):
        self.models_dir = os.environ.get("COMFY_MODEL_PATH", "models")
        self.loaded_lora = None
        
    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        """Load and apply a LoRA to the model and clip"""
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
            
        lora_path = os.path.join(self.models_dir, "loras", lora_name)
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA not found: {lora_path}")
            
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
                
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)
            
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora) 