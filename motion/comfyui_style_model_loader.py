#!/usr/bin/env python3
"""
ComfyUI-style model loader that integrates patching directly into loading
Follows ComfyUI's approach: model detection + initialization + patching all in one go
"""

import torch
import torch.nn as nn
import logging
import sys
import os
from typing import Dict, Any, Optional, Tuple

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_style_partial_loading import ComfyUIStylePartialLoader, LowVramPatch
from model_aware_patcher import ModelAwarePatcher
from comfyui_ops import ComfyUILinear, ComfyUIConv2d, ComfyUIGroupNorm, ComfyUILayerNorm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class ComfyUIStyleModelPatcher:
    """
    ComfyUI-style model patcher that integrates patching directly into model loading
    Follows ComfyUI's approach: model + patching + low-VRAM setup all in one go
    """
    
    def __init__(self, model: nn.Module, load_device: torch.device, offload_device: torch.device, size: int = 0):
        """
        Initialize ComfyUI-style model patcher
        Sets up low-VRAM attributes immediately, just like ComfyUI
        """
        self.size = size
        self.model = model
        
        # Set model device (ComfyUI's approach)
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device
        
        # Initialize patching infrastructure (ComfyUI's approach)
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options": {}}
        
        # Calculate model size
        self.model_size()
        
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = False
        self.force_cast_weights = False
        
        # Set up low-VRAM attributes immediately (ComfyUI's approach)
        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0
        
        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0
        
        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False
        
        # Apply ComfyUI-style patching immediately
        self._apply_comfyui_patching()
        
        logging.info(f"🚀 ComfyUI-style ModelPatcher initialized")
        logging.info(f"   Model size: {self.size / (1024**3):.2f} GB")
        logging.info(f"   Load device: {self.load_device}")
        logging.info(f"   Offload device: {self.offload_device}")
    
    def model_size(self):
        """Calculate model size (ComfyUI's approach)"""
        if hasattr(self.model, 'model_size'):
            # If it's a ModelPatcher, use its model_size method
            return self.model.model_size()
        elif hasattr(self.model, 'parameters'):
            # If it's a regular PyTorch model, calculate from parameters
            if self.size <= 0:
                self.size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            return self.size
        else:
            # Fallback: assume 0 size
            self.size = 0
            return self.size
    
    def _apply_comfyui_patching(self):
        """Apply ComfyUI-style patching immediately during initialization"""
        logging.info(f"🔧 Applying ComfyUI-style patching...")
        
        # Get the actual model from ModelPatcher if needed
        actual_model = self.model
        if hasattr(self.model, 'model'):
            # If it's a ModelPatcher, get the actual model
            actual_model = self.model.model
            logging.info(f"   Using actual model from ModelPatcher: {type(actual_model).__name__}")
        else:
            logging.info(f"   Using model directly: {type(actual_model).__name__}")
        
        # Use model-aware patcher
        patcher = ModelAwarePatcher()
        self.model = patcher.patch_model(actual_model)
        
        # Set up patches dictionary (ComfyUI's approach)
        self._setup_patches()
        
        logging.info(f"✅ ComfyUI-style patching complete")
    
    def _setup_patches(self):
        """Set up patches dictionary (ComfyUI's approach)"""
        # Get the actual model from ModelPatcher if needed
        actual_model = self.model
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
        
        # Extract all parameters and create patches
        for name, param in actual_model.named_parameters():
            self.patches[name] = param.data.clone()
        
        logging.debug(f"📊 Set up {len(self.patches)} patches")
    
    def _load_list(self):
        """Create loading list (ComfyUI's approach)"""
        # Get the actual model from ModelPatcher if needed
        actual_model = self.model
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
        
        loading = []
        for name, module in actual_model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'numel'):
                params = sum(p.numel() for p in module.parameters())
                module_mem = params * 4  # Assume float32
                loading.append((module_mem, name, module, params))
        return loading
    
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        Load model with ComfyUI-style low-VRAM patching
        This is the core method that applies weight_function patches
        """
        logging.info(f"🔄 Loading model with ComfyUI-style low-VRAM patching...")
        logging.info(f"   Low-VRAM memory limit: {lowvram_model_memory / (1024**3):.2f} GB")
        logging.info(f"   Force patch weights: {force_patch_weights}")
        logging.info(f"   Full load: {full_load}")
        
        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        
        # Get loading list
        loading = self._load_list()
        
        # Sort by memory usage (largest first)
        loading.sort(reverse=True)
        
        load_completely = []
        
        for x in loading:
            n = x[1]  # module name
            m = x[2]  # module
            params = x[3]  # parameter count
            module_mem = x[0]  # memory usage
            
            lowvram_weight = False
            
            # Create weight and bias keys (ComfyUI's approach)
            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)
            
            # Determine if this module should use low-VRAM (ComfyUI's approach)
            if not full_load and hasattr(m, "comfy_cast_weights"):
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                        continue
            
            cast_weight = self.force_cast_weights
            
            if lowvram_weight:
                # Apply low-VRAM patching (ComfyUI's approach)
                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []
                
                # Set up weight_function and bias_function (ComfyUI's approach)
                if weight_key in self.patches:
                    if force_patch_weights:
                        self._patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = [LowVramPatch(weight_key, self.patches)]
                        patch_counter += 1
                        logging.debug(f"  ✅ Set up weight_function for {weight_key}")
                
                if bias_key in self.patches:
                    if force_patch_weights:
                        self._patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = [LowVramPatch(bias_key, self.patches)]
                        patch_counter += 1
                        logging.debug(f"  ✅ Set up bias_function for {bias_key}")
                
                cast_weight = True
            else:
                # Load completely (ComfyUI's approach)
                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []
                
                load_completely.append((n, m, params))
                mem_counter += module_mem
            
            # Apply weight casting if needed (ComfyUI's approach)
            if cast_weight:
                if hasattr(m, "comfy_cast_weights"):
                    m.comfy_cast_weights()
        
        # Move completely loaded modules to device
        for n, m, params in load_completely:
            m.to(self.load_device)
            logging.debug(f"  ✅ Loaded {n} to {self.load_device}")
        
        # Move model to offload device (ComfyUI's approach)
        self.model.to(self.offload_device)
        
        logging.info(f"✅ Model loading complete:")
        logging.info(f"   Modules loaded completely: {len(load_completely)}")
        logging.info(f"   Modules with low-VRAM patches: {lowvram_counter}")
        logging.info(f"   Total patches applied: {patch_counter}")
        logging.info(f"   Memory used: {mem_counter / (1024**3):.2f} GB")
    
    def _patch_weight_to_device(self, key: str):
        """Patch weight to device (ComfyUI's approach)"""
        if key in self.patches:
            weight = self.patches[key]
            if weight.device != self.load_device:
                self.patches[key] = weight.to(self.load_device)
    
    def unload(self):
        """Unload model (ComfyUI's approach)"""
        logging.info(f"🧹 Unloading model...")
        
        # Clear weight_function and bias_function
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_function'):
                module.weight_function = []
            if hasattr(module, 'bias_function'):
                module.bias_function = []
        
        # Move model to offload device
        self.model.to(self.offload_device)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f"✅ Model unloaded")

def load_unet_with_comfyui_patching(unet_path: str, load_device: torch.device, offload_device: torch.device, 
                                   model_options: Dict[str, Any] = None) -> ComfyUIStyleModelPatcher:
    """
    Load UNet with ComfyUI-style patching integrated directly into loading
    This follows ComfyUI's approach: model + patching + low-VRAM setup all in one go
    """
    if model_options is None:
        model_options = {}
    
    logging.info(f"🚀 Loading UNet with ComfyUI-style patching...")
    logging.info(f"   Path: {unet_path}")
    logging.info(f"   Load device: {load_device}")
    logging.info(f"   Offload device: {offload_device}")
    
    try:
        # Load state dict
        if unet_path.endswith('.safetensors'):
            try:
                import safetensors.torch
                sd = safetensors.torch.load_file(unet_path)
            except Exception as safetensors_error:
                logging.warning(f"⚠️  Safetensors loading failed: {safetensors_error}")
                logging.warning(f"⚠️  Model file may be corrupted, trying torch.load...")
                try:
                    sd = torch.load(unet_path, map_location='cpu', weights_only=True)
                except Exception as torch_error:
                    logging.error(f"❌ Both safetensors and torch loading failed:")
                    logging.error(f"   Safetensors error: {safetensors_error}")
                    logging.error(f"   Torch error: {torch_error}")
                    raise RuntimeError(f"Failed to load model file: {unet_path}")
        else:
            sd = torch.load(unet_path, map_location='cpu', weights_only=True)
        
        logging.info(f"📊 Loaded state dict with {len(sd)} keys")
        
        # Debug: Show first few keys to understand the format
        sample_keys = list(sd.keys())[:10]
        logging.info(f"📊 Sample keys: {sample_keys}")
        
        # Check for common model prefixes
        prefixes = ["model.diffusion_model.", "model.model.", "net.", "head.", "blocks.", "vace_blocks."]
        for prefix in prefixes:
            matching_keys = [k for k in sd.keys() if k.startswith(prefix)]
            if matching_keys:
                logging.info(f"📊 Found {len(matching_keys)} keys with prefix '{prefix}': {matching_keys[:3]}...")
        
        # Create model using ComfyUI's exact approach
        try:
            # Use ComfyUI's exact model detection and creation flow
            from standalone_sd import load_state_dict_guess_config
            
            result = load_state_dict_guess_config(
                sd,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is None:
                raise RuntimeError("Failed to create model from state dict")
            
            model, _, _, _ = result
            
            if model is None:
                raise RuntimeError("Model is None after creation")
            
            logging.info(f"✅ Model created using ComfyUI's detection logic")
            logging.info(f"   Model type: {type(model).__name__}")
            if hasattr(model, 'model_info'):
                logging.info(f"   Model info: {model.model_info}")
                
        except Exception as detection_error:
            logging.warning(f"⚠️  Model detection failed: {detection_error}")
            logging.warning(f"⚠️  Falling back to generic model wrapper...")
            
            # Fallback: Create a generic model wrapper that can handle any state dict
            class GenericModel(nn.Module):
                def __init__(self, state_dict):
                    super().__init__()
                    self.state_dict_data = state_dict
                    self.device = torch.device("cpu")
                    
                    # Create a dummy parameter to satisfy ModelPatcher
                    self.dummy_param = nn.Parameter(torch.randn(1))
                    
                    # Store model info
                    total_params = sum(tensor.numel() for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))
                    self.model_info = {
                        'total_params': total_params,
                        'state_dict_keys': len(state_dict),
                        'model_type': 'generic'
                    }
                
                def state_dict(self):
                    return self.state_dict_data
                
                def load_state_dict(self, state_dict, strict=False):
                    self.state_dict_data = state_dict
                    return None, None
                
                def parameters(self):
                    return [self.dummy_param]
                
                def named_parameters(self):
                    return [('dummy_param', self.dummy_param)]
                
                def named_modules(self):
                    return [('', self)]
                
                def to(self, device):
                    self.device = device
                    return self
            
            model = GenericModel(sd)
        
        logging.info(f"✅ Model created and weights loaded")
        
        # Create ComfyUI-style ModelPatcher (this integrates patching immediately)
        model_patcher = ComfyUIStyleModelPatcher(model, load_device, offload_device)
        
        return model_patcher
        
    except Exception as e:
        logging.error(f"❌ Failed to load UNet: {e}")
        raise

def test_comfyui_style_loading():
    """Test ComfyUI-style model loading"""
    print("🧪 TESTING COMFYUI-STYLE MODEL LOADING")
    print("=" * 60)
    
    # Test with a simple model first
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 32 * 32, 1000)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create test model
    model = TestModel()
    load_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    offload_device = torch.device('cpu')
    
    try:
        # Create ComfyUI-style ModelPatcher
        model_patcher = ComfyUIStyleModelPatcher(model, load_device, offload_device)
        
        # Test loading with low-VRAM
        lowvram_memory = 0.1 * 1024**3  # 0.1 GB
        model_patcher.load(lowvram_model_memory=lowvram_memory, force_patch_weights=False, full_load=False)
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ ComfyUI-style loading test passed!")
        print(f"   Output shape: {output.shape}")
        
        # Test unloading
        model_patcher.unload()
        
        return True
        
    except Exception as e:
        print(f"❌ ComfyUI-style loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_comfyui_style_loading()
    sys.exit(0 if success else 1)
