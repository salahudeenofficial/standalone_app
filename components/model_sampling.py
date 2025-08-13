"""
Model sampling component for the standalone pipeline
Adapted from ComfyUI ModelSamplingSD3 but simplified for direct use
"""

import torch
import sys
from pathlib import Path

# Add the comfy modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy"))
sys.path.insert(0, str(Path(__file__).parent.parent / "comfy_extras"))

import comfy.model_sampling

class ModelSamplingSD3:
    """Apply SD3 model sampling with shift parameter"""
    
    def patch(self, model, shift, multiplier=1000):
        """Patch the model with SD3 sampling"""
        m = model.clone()
        
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST
        
        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
            
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        return m 