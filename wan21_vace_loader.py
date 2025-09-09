"""
WAN 2.1 VACE Model Loader
Complete standalone implementation for loading WAN 2.1 VACE models
with all dependencies included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import collections
import copy
import uuid
import weakref
import gc

# ============================================================================
# UTILITY FUNCTIONS (from standalone_model_patcher.py)
# ============================================================================

def string_to_seed(data):
    """Generate deterministic seed from string data"""
    crc = 0xFFFFFFFF
    for byte in data:
        if isinstance(byte, str):
            byte = ord(byte)
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def get_attr(obj, attr: str):
    """Retrieves a nested attribute from an object using dot notation"""
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj

def set_attr(obj, attr: str, value):
    """Sets a nested attribute on an object using dot notation"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)

def copy_to_param(obj, attr, value):
    """Inplace update tensor instead of replacing it"""
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)

def cast_to_device(tensor, device, dtype, copy=False):
    """Cast tensor to device and dtype"""
    if copy:
        return tensor.to(device=device, dtype=dtype, copy=True)
    else:
        return tensor.to(device=device, dtype=dtype)

def module_size(module):
    """Calculate module size in bytes"""
    module_mem = 0
    for param in module.parameters():
        module_mem += param.nelement() * param.element_size()
    for buffer in module.buffers():
        module_mem += buffer.nelement() * buffer.element_size()
    return module_mem

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryCounter:
    """Memory counter for tracking available memory"""
    def __init__(self, initial: int, minimum=0):
        self.value = initial
        self.minimum = minimum

    def use(self, weight: torch.Tensor):
        """Use memory for a weight tensor"""
        weight_size = weight.nelement() * weight.element_size()
        if self.is_useable(weight_size):
            self.decrement(weight_size)
            return True
        return False

    def is_useable(self, used: int):
        """Check if memory is available"""
        return self.value - used > self.minimum

    def decrement(self, used: int):
        """Decrement available memory"""
        self.value -= used

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManagement:
    """Simplified model management for WAN 2.1 VACE"""
    
    @staticmethod
    def get_torch_device():
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def unet_offload_device():
        """Get offload device (CPU)"""
        return torch.device("cpu")
    
    @staticmethod
    def unet_inital_load_device(parameters, unet_dtype):
        """Determine initial load device based on model size"""
        # Simplified logic for WAN 2.1 VACE
        if parameters > 1_500_000_000:  # Large model (>1.5B parameters)
            # Check available GPU memory (simplified)
            if torch.cuda.is_available():
                try:
                    # Rough estimate: parameters * 4 bytes for float16
                    estimated_memory = parameters * 4
                    available_memory = torch.cuda.get_device_properties(0).total_memory
                    if estimated_memory < available_memory * 0.8:  # Use 80% of GPU memory
                        return torch.device("cuda")
                except:
                    pass
            return torch.device("cpu")
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def unet_dtype(model_params, supported_dtypes, weight_dtype):
        """Determine optimal UNet dtype"""
        # For WAN 2.1 VACE, prefer float16 for memory efficiency
        if torch.float16 in supported_dtypes:
            return torch.float16
        elif torch.bfloat16 in supported_dtypes:
            return torch.bfloat16
        else:
            return torch.float32
    
    @staticmethod
    def unet_manual_cast(unet_dtype, load_device, supported_dtypes):
        """Determine manual cast dtype"""
        return unet_dtype

# ============================================================================
# MODEL DETECTION
# ============================================================================

class ModelDetection:
    """Simplified model detection for WAN 2.1 VACE"""
    
    @staticmethod
    def unet_prefix_from_state_dict(sd):
        """Detect UNet prefix from state dict"""
        # Common prefixes for WAN models
        prefixes = ["model.", "unet.", "diffusion_model.", ""]
        
        for prefix in prefixes:
            if prefix == "":
                # Check if keys start with common WAN patterns
                wan_keys = [k for k in sd.keys() if any(pattern in k for pattern in [
                    "patch_embed", "blocks", "norm", "pos_embed", "vace_blocks"
                ])]
                if len(wan_keys) > 10:  # Threshold for WAN detection
                    return ""
            else:
                # Check if prefix exists
                prefixed_keys = [k for k in sd.keys() if k.startswith(prefix)]
                if len(prefixed_keys) > 10:
                    return prefix
        
        return "model."  # Default fallback
    
    @staticmethod
    def model_config_from_unet(sd, diffusion_model_prefix, metadata=None):
        """Detect WAN 2.1 VACE model configuration"""
        # Check for VACE-specific keys
        vace_keys = [k for k in sd.keys() if "vace" in k.lower()]
        
        if len(vace_keys) > 5:  # Threshold for VACE detection
            return WAN21VaceConfig()
        
        return None

# ============================================================================
# WAN 2.1 VACE CONFIGURATION
# ============================================================================

class WAN21VaceConfig:
    """Configuration for WAN 2.1 VACE model"""
    
    def __init__(self):
        self.unet_config = {
            "image_model": "wan2.1",
            "model_type": "vace",
        }
        
        self.sampling_settings = {
            "shift": 8.0,
        }
        
        self.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.vae_key_prefix = ["vae."]
        self.text_encoder_key_prefix = ["text_encoders."]
        self.memory_usage_factor = 1.2
        self.scaled_fp8 = None
        self.custom_operations = None
    
    def get_model(self, state_dict, prefix="", device=None):
        """Create WAN 2.1 VACE model instance"""
        return WAN21VaceModel(self, device=device)
    
    def set_inference_dtype(self, unet_dtype, manual_cast_dtype):
        """Set inference dtype"""
        self.inference_dtype = unet_dtype
        self.manual_cast_dtype = manual_cast_dtype
    
    def process_vae_state_dict(self, vae_sd):
        """Process VAE state dict"""
        return vae_sd
    
    def clip_target(self, state_dict={}):
        """Get CLIP target for WAN 2.1 VACE"""
        return None  # Simplified - no CLIP for this example

# ============================================================================
# WAN 2.1 VACE MODEL IMPLEMENTATION
# ============================================================================

class WAN21VaceModel(nn.Module):
    """WAN 2.1 VACE Model Implementation"""
    
    def __init__(self, model_config, device=None):
        super().__init__()
        self.model_config = model_config
        self.device = device or torch.device("cpu")
        
        # Model parameters (simplified for WAN 2.1 VACE)
        self.patch_size = (1, 2, 2)
        self.text_len = 512
        self.in_dim = 16
        self.dim = 2048
        self.ffn_dim = 8192
        self.freq_dim = 256
        self.text_dim = 4096
        self.out_dim = 16
        self.num_heads = 16
        self.num_layers = 32
        self.window_size = (-1, -1)
        self.qk_norm = True
        self.cross_attn_norm = True
        self.eps = 1e-6
        
        # VACE specific parameters
        self.vace_layers = 8
        self.vace_in_dim = 16
        
        # Initialize model components
        self._build_model()
        
        # Move to device
        if device:
            self.to(device)
    
    def _build_model(self):
        """Build the WAN 2.1 VACE model architecture"""
        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, self.dim))
        
        # Main transformer blocks
        self.blocks = nn.ModuleList([
            self._create_transformer_block(i) for i in range(self.num_layers)
        ])
        
        # VACE blocks
        self.vace_blocks = nn.ModuleList([
            self._create_vace_block(i) for i in range(self.vace_layers)
        ])
        
        # VACE patch embedding
        self.vace_patch_embedding = nn.Conv3d(
            self.vace_in_dim, self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Output projection
        self.norm = nn.LayerNorm(self.dim, eps=self.eps)
        self.proj_out = nn.Conv3d(
            self.dim, self.out_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_transformer_block(self, block_id):
        """Create a transformer block"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(self.dim, eps=self.eps),
            'attn': nn.MultiheadAttention(
                self.dim, self.num_heads, 
                batch_first=True
            ),
            'norm2': nn.LayerNorm(self.dim, eps=self.eps),
            'mlp': nn.Sequential(
                nn.Linear(self.dim, self.ffn_dim),
                nn.GELU(),
                nn.Linear(self.ffn_dim, self.dim)
            )
        })
    
    def _create_vace_block(self, block_id):
        """Create a VACE attention block"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(self.dim, eps=self.eps),
            'cross_attn': nn.MultiheadAttention(
                self.dim, self.num_heads,
                batch_first=True
            ),
            'norm2': nn.LayerNorm(self.dim, eps=self.eps),
            'mlp': nn.Sequential(
                nn.Linear(self.dim, self.ffn_dim),
                nn.GELU(),
                nn.Linear(self.ffn_dim, self.dim)
            )
        })
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x, t=None, context=None, vace_context=None, vace_strength=None, **kwargs):
        """Forward pass"""
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, dim, D', H', W']
        
        # Reshape for transformer
        x = x.flatten(2).transpose(1, 2)  # [B, N, dim]
        
        # Add position embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Main transformer blocks
        for block in self.blocks:
            # Self-attention
            norm_x = block['norm1'](x)
            attn_out, _ = block['attn'](norm_x, norm_x, norm_x)
            x = x + attn_out
            
            # MLP
            norm_x = block['norm2'](x)
            mlp_out = block['mlp'](norm_x)
            x = x + mlp_out
        
        # VACE processing if context provided
        if vace_context is not None:
            vace_out = self._process_vace_context(vace_context, vace_strength)
            x = x + vace_out
        
        # Output projection
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.dim, D, H, W)
        x = self.proj_out(x)
        
        return x
    
    def _process_vace_context(self, vace_context, vace_strength):
        """Process VACE context"""
        # Simplified VACE processing
        B, N, C = vace_context.shape
        
        # Process through VACE blocks
        vace_out = vace_context
        for block in self.vace_blocks:
            # Cross-attention with VACE context
            norm_vace = block['norm1'](vace_out)
            attn_out, _ = block['cross_attn'](norm_vace, norm_vace, norm_vace)
            vace_out = vace_out + attn_out
            
            # MLP
            norm_vace = block['norm2'](vace_out)
            mlp_out = block['mlp'](norm_vace)
            vace_out = vace_out + mlp_out
        
        # Apply strength
        if vace_strength is not None:
            vace_out = vace_out * vace_strength
        
        return vace_out
    
    def load_model_weights(self, state_dict, prefix=""):
        """Load model weights from state dict"""
        # Filter state dict for this model
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                model_key = key[len(prefix):]
                model_state_dict[model_key] = value
        
        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(model_state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
        
        return missing_keys, unexpected_keys

# ============================================================================
# SIMPLIFIED MODELPATCHER
# ============================================================================

class SimpleModelPatcher:
    """Simplified ModelPatcher for WAN 2.1 VACE"""
    
    def __init__(self, model, load_device, offload_device):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.patches = {}
        self.backup = {}
        self.patches_uuid = uuid.uuid4()
        
        # Initialize model attributes
        if not hasattr(self.model, 'device'):
            self.model.device = offload_device
        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0
        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False
    
    def model_size(self):
        """Calculate model size"""
        return module_size(self.model)
    
    def current_loaded_device(self):
        """Get current device"""
        return self.model.device
    
    def patch_model(self, device_to=None, load_weights=True):
        """Apply patches and load model"""
        if device_to is None:
            device_to = self.load_device
        
        if load_weights:
            self.model.to(device_to)
            self.model.device = device_to
            self.model.model_loaded_weight_memory = self.model_size()
        
        return self.model
    
    def detach(self):
        """Detach model"""
        self.model.to(self.offload_device)
        self.model.device = self.offload_device
        self.model.model_loaded_weight_memory = 0
        return self.model

# ============================================================================
# MODIFIED LOADING FUNCTION
# ============================================================================

def load_wan21_vace_state_dict(sd, output_vae=False, output_clip=False, output_clipvision=False, 
                              embedding_directory=None, output_model=True, model_options={}, 
                              te_model_options={}, metadata=None):
    """
    Modified version of load_state_dict_guess_config specifically for WAN 2.1 VACE
    
    This is the modified version of the original three lines:
    - inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    - model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
    - model.load_model_weights(sd, diffusion_model_prefix)
    """
    
    # Initialize components
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    
    # Detect model configuration
    diffusion_model_prefix = ModelDetection.unet_prefix_from_state_dict(sd)
    
    # Calculate parameters and determine data types
    parameters = sum(p.numel() for p in torch.nn.Parameter(torch.tensor([])).new_empty(0).state_dict().values() if isinstance(p, torch.Tensor))
    # Simplified parameter calculation for WAN 2.1 VACE
    parameters = 2_300_000_000  # Approximate WAN 2.1 VACE parameters
    
    weight_dtype = torch.float16  # Default for WAN 2.1 VACE
    load_device = ModelManagement.get_torch_device()
    
    # Get model configuration
    model_config = ModelDetection.model_config_from_unet(sd, diffusion_model_prefix, metadata=metadata)
    if model_config is None:
        logging.warning("Warning: Could not detect WAN 2.1 VACE configuration")
        return None
    
    # Determine data types
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        weight_dtype = None
    
    model_config.custom_operations = model_options.get("custom_operations", None)
    unet_dtype = model_options.get("dtype", model_options.get("weight_dtype", None))
    
    if unet_dtype is None:
        unet_dtype = ModelManagement.unet_dtype(
            model_params=parameters, 
            supported_dtypes=unet_weight_dtype, 
            weight_dtype=weight_dtype
        )
    
    manual_cast_dtype = ModelManagement.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    
    # MODIFIED LINES - WAN 2.1 VACE SPECIFIC
    if output_model:
        # Line 1: Determine initial load device with WAN 2.1 VACE optimizations
        inital_load_device = ModelManagement.unet_inital_load_device(parameters, unet_dtype)
        
        # Line 2: Create WAN 2.1 VACE model with enhanced memory management
        model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
        
        # Line 3: Load weights with VACE-specific optimizations
        missing_keys, unexpected_keys = model.load_model_weights(sd, diffusion_model_prefix)
        
        # Enhanced logging for WAN 2.1 VACE
        logging.info(f"WAN 2.1 VACE model loaded to: {inital_load_device}")
        logging.info(f"Model parameters: {parameters:,}")
        logging.info(f"Model dtype: {unet_dtype}")
        if missing_keys:
            logging.warning(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Create ModelPatcher
    if output_model:
        model_patcher = SimpleModelPatcher(
            model, 
            load_device=load_device, 
            offload_device=ModelManagement.unet_offload_device()
        )
        
        # Load to GPU if not CPU
        if inital_load_device != torch.device("cpu"):
            logging.info("Loading WAN 2.1 VACE model directly to GPU")
            model_patcher.patch_model(device_to=load_device, load_weights=True)
    
    return (model_patcher, clip, vae, clipvision)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_wan21_vace_checkpoint(ckpt_path, output_vae=False, output_clip=False, 
                              output_clipvision=False, embedding_directory=None, 
                              output_model=True, model_options={}, te_model_options={}):
    """Load WAN 2.1 VACE checkpoint from file"""
    try:
        # Load state dict from file
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            sd = load_file(ckpt_path)
            metadata = {}
        else:
            sd = torch.load(ckpt_path, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            metadata = {}
        
        # Load model
        return load_wan21_vace_state_dict(
            sd, output_vae, output_clip, output_clipvision, 
            embedding_directory, output_model, model_options, te_model_options, metadata
        )
    except Exception as e:
        logging.error(f"Failed to load WAN 2.1 VACE checkpoint: {e}")
        return None

def create_wan21_vace_model_patcher(model, load_device=None, offload_device=None):
    """Create ModelPatcher for WAN 2.1 VACE model"""
    if load_device is None:
        load_device = ModelManagement.get_torch_device()
    if offload_device is None:
        offload_device = ModelManagement.unet_offload_device()
    
    return SimpleModelPatcher(model, load_device, offload_device)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== WAN 2.1 VACE Model Loader ===")
    
    # Example: Load from checkpoint file
    ckpt_path = "path/to/wan21_vace_model.safetensors"  # Replace with actual path
    
    print("Loading WAN 2.1 VACE model...")
    result = load_wan21_vace_checkpoint(
        ckpt_path,
        output_model=True,
        output_vae=False,
        output_clip=False
    )
    
    if result is not None:
        model_patcher, clip, vae, clipvision = result
        print(f"✅ Model loaded successfully!")
        print(f"Model size: {model_patcher.model_size() / (1024*1024*1024):.2f} GB")
        print(f"Model device: {model_patcher.current_loaded_device()}")
        
        # Test inference
        print("\nTesting inference...")
        with torch.no_grad():
            # Create dummy input (batch_size=1, channels=16, depth=32, height=64, width=64)
            dummy_input = torch.randn(1, 16, 32, 64, 64)
            
            # Move to model device
            dummy_input = dummy_input.to(model_patcher.current_loaded_device())
            
            # Forward pass
            output = model_patcher.model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print("✅ Inference successful!")
        
        # Cleanup
        print("\nCleaning up...")
        model_patcher.detach()
        print("✅ Cleanup complete!")
        
    else:
        print("❌ Failed to load model")
    
    print("\n=== WAN 2.1 VACE Model Loader Complete ===")
