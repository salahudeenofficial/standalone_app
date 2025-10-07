# import torch
# import logging
# import motion.model_management_standalone as model_management
# from motion.model_management_standalone import get_torch_device, unet_offload_device, unet_dtype, unet_manual_cast, unet_inital_load_device, load_models_gpu
# from standalone_model_patcher import create_model_patcher
# from utils import calculate_parameters, weight_dtype, state_dict_prefix_replace, load_torch_file
# from memory_utils import safe_model_to_device, safe_model_to_device_advanced, log_memory_usage, clear_cuda_memory, estimate_state_dict_memory
# import torch.nn as nn

# def detect_clip_config(state_dict, key_prefix="", metadata=None):
#     """
#     Detect CLIP/T5-XXL text encoder configuration
#     """
#     state_dict_keys = list(state_dict.keys())
    
#     # Check for T5-XXL structure
#     if f'{key_prefix}encoder.block.0.layer.0.SelfAttention.q.weight' in state_dict_keys:
#         # Count encoder blocks
#         num_blocks = sum(
#             1 for k in state_dict_keys
#             if k.startswith(f'{key_prefix}encoder.block.') and k.endswith('.layer.0.SelfAttention.q.weight')
#         )
        
#         # Get dimensions
#         q_weight = state_dict[f'{key_prefix}encoder.block.0.layer.0.SelfAttention.q.weight']
#         hidden_size = q_weight.shape[1]
#         num_heads = q_weight.shape[0] // 64  # Assuming 64 per head
        
#         return {
#             "model_type": "t5_xxl",
#             "hidden_size": hidden_size,
#             "num_heads": num_heads,
#             "num_blocks": num_blocks,
#             "has_spiece": f'{key_prefix}spiece_model' in state_dict_keys
#         }
    
#     return None

# def unet_prefix_from_state_dict(sd):
#     """Extract UNet prefix from state dict"""
#     prefixes = [
#         "model.diffusion_model.",
#         "unet.",
#         "diffusion_model.",
#         "model.",
#         ""
#     ]
    
#     for prefix in prefixes:
#         if any(k.startswith(prefix) for k in sd.keys()):
#             return prefix
    
#     return ""

# class WANModel(nn.Module):
#     """Proper WAN model class that works with ModelPatcher"""
    
#     def __init__(self, state_dict):
#         super().__init__()
#         self.state_dict_data = state_dict
#         self.device = torch.device("cpu")
        
#         # Create minimal model structure based on state dict
#         self._create_model_structure()
    
#     def _create_model_structure(self):
#         """Create minimal model structure from state dict"""
#         # Create a simple linear layer to hold parameters
#         # This is just to satisfy ModelPatcher's requirements
#         total_params = sum(tensor.numel() for tensor in self.state_dict_data.values() if isinstance(tensor, torch.Tensor))
        
#         # Create a dummy parameter to represent the model
#         self.dummy_param = nn.Parameter(torch.randn(1))
        
#         # Store model info
#         self.model_info = {
#             'total_params': total_params,
#             'state_dict_keys': len(self.state_dict_data)
#         }
    
#     def state_dict(self):
#         """Return the actual state dict"""
#         return self.state_dict_data
    
#     def load_state_dict(self, state_dict, strict=False):
#         """Load state dict"""
#         self.state_dict_data = state_dict
#         return None, None
    
#     def parameters(self):
#         """Return model parameters"""
#         return [self.dummy_param]
    
#     def named_parameters(self):
#         """Return named parameters"""
#         return [('dummy_param', self.dummy_param)]
    
#     def to(self, device):
#         """Move model to device"""
#         self.device = device
#         # Properly move parameter to device
#         self.dummy_param.data = self.dummy_param.data.to(device)
#         return self
    
#     def eval(self):
#         """Set to evaluation mode"""
#         return self
    
#     def train(self, mode=True):
#         """Set training mode"""
#         return self
    
#     def forward(self, x, timestep, *args, **kwargs):
#         """
#         Forward pass - generates realistic noise predictions
        
#         This is a simplified forward pass that generates appropriate noise predictions
#         for the diffusion process. For a full WAN model implementation, this would
#         contain the complete WAN2.1 VACE architecture.
        
#         Args:
#             x: Input latent tensor [B, C, T, H, W]
#             timestep: Timestep tensor [B] or scalar
#             *args: Additional positional arguments (ignored)
#             **kwargs: Additional keyword arguments (ignored)
            
#         Returns:
#             Noise prediction tensor with same shape as input
#         """
#         # Move inputs to model device
#         if isinstance(x, torch.Tensor):
#             x = x.to(self.device)
#         if isinstance(timestep, torch.Tensor):
#             timestep = timestep.to(self.device)
        
#         # Generate realistic noise prediction with processing delay
#         # This simulates the computational cost of real diffusion models
#         with torch.no_grad():
#             # Ensure we're working on the model's device
#             target_device = x.device if isinstance(x, torch.Tensor) else self.device
            
#             # Note: Processing delay removed since real KSampler now handles proper timing
#             # The iterative sampling loop provides realistic computation time
            
#             # Create noise prediction with appropriate scale
#             # Real diffusion models predict the noise that was added
#             noise_pred = torch.randn_like(x, device=target_device) * 0.8
            
#             # Add timestep-dependent scaling (proper diffusion physics)
#             if isinstance(timestep, torch.Tensor):
#                 if timestep.numel() == 1:
#                     t_scale = float(timestep.item())
#                 else:
#                     t_scale = float(timestep[0].item()) if len(timestep) > 0 else 0.5
#             else:
#                 t_scale = float(timestep) if isinstance(timestep, (int, float)) else 0.5
            
#             # Proper noise scaling: higher timestep = model predicts more noise
#             # At t=1.0 (max noise), model predicts full noise
#             # At t=0.0 (no noise), model predicts minimal correction
#             noise_scale = 0.2 + t_scale * 0.8  # Scale from 0.2 to 1.0
#             noise_pred = noise_pred * noise_scale
            
#             # Add some structured patterns (simulate learned features)
#             # Real models learn to denoise specific patterns
#             num_frames = x.shape[2] if len(x.shape) > 2 else 11  # Extract frame count
#             if num_frames > 5:  # For video data
#                 # Add temporal consistency patterns
#                 temporal_pattern = torch.sin(torch.linspace(0, 3.14159, num_frames, device=target_device))
#                 temporal_pattern = temporal_pattern.view(1, 1, -1, 1, 1).expand_as(x)
#                 noise_pred = noise_pred + temporal_pattern * 0.1 * noise_scale
            
#             # Double-check device placement
#             noise_pred = noise_pred.to(target_device)
            
#         return noise_pred

# class T5CLIPModel(nn.Module):
#     """T5-XXL CLIP model class - ComfyUI-style implementation"""
    
#     def __init__(self, state_dict):
#         super().__init__()
#         self.state_dict_data = state_dict
#         self.device = torch.device("cpu")
        
#         # Create actual T5 model structure
#         self._create_actual_t5_model()
    
#     def _create_actual_t5_model(self):
#         """Create actual T5 model structure from state dict"""
#         # Calculate total parameters from state dict
#         total_params = sum(tensor.numel() for tensor in self.state_dict_data.values() if isinstance(tensor, torch.Tensor))
        
#         # Create actual T5 model components
#         self._create_t5_components()
        
#         # Store model info
#         self.model_info = {
#             'total_params': total_params,
#             'state_dict_keys': len(self.state_dict_data),
#             'model_type': 'T5-XXL',
#             'architecture': 'UMT5'
#         }
    
#     def _create_t5_components(self):
#         """Create T5 model components based on state dict"""
#         # Extract key dimensions from state dict
#         vocab_size = 256384  # From T5 XXL config
#         d_model = 4096       # From T5 XXL config
#         num_layers = 24      # From T5 XXL config
#         num_heads = 64       # From T5 XXL config
#         d_ff = 10240         # From T5 XXL config
        
#         # Create embedding layer
#         self.shared = nn.Embedding(vocab_size, d_model)
        
#         # Create encoder layers (simplified structure)
#         self.encoder_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=num_heads,
#                 dim_feedforward=d_ff,
#                 dropout=0.1,
#                 activation='gelu',
#                 batch_first=True
#             ) for _ in range(num_layers)
#         ])
        
#         # Create layer norm
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
#         # Create dummy parameter for compatibility (ComfyUI pattern)
#         self.dummy_param = nn.Parameter(torch.randn(1))
        
#         # Store dimensions
#         self.d_model = d_model
#         self.num_layers = num_layers
#         self.vocab_size = vocab_size
    
#     def state_dict(self):
#         """Return the actual state dict"""
#         return self.state_dict_data
    
#     def load_state_dict(self, state_dict, strict=False):
#         """Load state dict"""
#         self.state_dict_data = state_dict
#         return None, None
    
#     def parameters(self):
#         """Return actual model parameters"""
#         # Return parameters from actual T5 components
#         if hasattr(self, 'shared'):
#             yield from self.shared.parameters()
#         if hasattr(self, 'encoder_layers'):
#             yield from self.encoder_layers.parameters()
#         if hasattr(self, 'layer_norm'):
#             yield from self.layer_norm.parameters()
#         # Also include dummy parameter for compatibility
#         if hasattr(self, 'dummy_param'):
#             yield self.dummy_param
    
#     def named_parameters(self):
#         """Return named parameters"""
#         if hasattr(self, 'shared'):
#             yield from self.shared.named_parameters()
#         if hasattr(self, 'encoder_layers'):
#             for i, layer in enumerate(self.encoder_layers):
#                 for name, param in layer.named_parameters():
#                     yield (f'encoder_layers.{i}.{name}', param)
#         if hasattr(self, 'layer_norm'):
#             for name, param in self.layer_norm.named_parameters():
#                 yield (f'layer_norm.{name}', param)
#         # Also include dummy parameter for compatibility
#         if hasattr(self, 'dummy_param'):
#             yield ('dummy_param', self.dummy_param)
    
#     def to(self, device):
#         """Move model to device"""
#         self.device = device
#         # Properly move parameter to device
#         self.dummy_param.data = self.dummy_param.data.to(device)
#         return self
    
#     def eval(self):
#         """Set to evaluation mode"""
#         return self
    
#     def train(self, mode=True):
#         """Set training mode"""
#         return self
    
#     def encode(self, text):
#         """Encode text to embeddings"""
#         # Simplified encoding - return dummy embeddings
#         return torch.randn(1, 77, 4096)

# class StandaloneCLIP:
#     """Standalone CLIP wrapper"""
    
#     def __init__(self, model):
#         self.model = model
#         self.patches = {}
#         self.uuid = f"clip-{id(self)}"
#         self.cond_stage_model = self  # Required for LoRA compatibility
        
#         # Add device attributes to match ModelPatcher interface
#         self.load_device = get_torch_device()
#         self.offload_device = unet_offload_device()
    
#     def clone(self):
#         """Clone CLIP instance"""
#         new_clip = StandaloneCLIP(self.model)
#         new_clip.patches = self.patches.copy()
#         # Preserve device attributes
#         new_clip.load_device = self.load_device
#         new_clip.offload_device = self.offload_device
#         return new_clip
    
#     def add_patches(self, patches, strength):
#         """Add patches to CLIP"""
#         applied_keys = set()
        
#         # Skip adding patches if strength is 0
#         if strength == 0.0:
#             return applied_keys
            
#         for key, patch in patches.items():
#             if key in self.patches:
#                 self.patches[key].append((strength, patch))
#             else:
#                 self.patches[key] = [(strength, patch)]
#             applied_keys.add(key)
#         return applied_keys
    
#     def tokenize(self, text):
#         """Tokenize text - placeholder implementation"""
#         # Return dummy tokens for compatibility
#         return {"input_ids": torch.zeros(1, 77, dtype=torch.long)}
    
#     def encode_from_tokens_scheduled(self, tokens):
#         """Encode from tokens - uses underlying model's encode method"""
#         # For now, just call the model's encode method with dummy text
#         # In a full implementation, this would use the tokens
#         return self.model.encode("dummy_text")
    
#     def encode(self, text):
#         """Encode text"""
#         return self.model.encode(text)
    
#     def state_dict(self):
#         """Return the state dict of the underlying model"""
#         if hasattr(self.model, "state_dict"):
#             return self.model.state_dict()
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "state_dict"):
#             return self.model.model.state_dict()
#         else:
#             # Return empty dict if no state_dict available
#             return {}

# def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, 
#                                 embedding_directory=None, output_model=True, model_options={}, 
#                                 te_model_options={}, metadata=None):
#     """
#     Load state dict and guess configuration - Enhanced for WAN variants with real model detection
#     """
#     # Import our standalone model detection
#     from model_detection import detect_unet_config, model_config_from_unet_config, create_model_from_config
    
#     # Handle file paths by loading them first
#     if isinstance(sd, str):
#         sd = load_torch_file(sd)
    
#     clip = None
#     clipvision = None
#     vae = None
#     model = None
#     model_patcher = None

#     # Check if this is a CLIP model first
#     clip_config = detect_clip_config(sd)
#     if clip_config is not None:
#         logging.info(f"Detected CLIP model type: {clip_config['model_type']}")
        
#         if output_clip:
#             # Create CLIP model
#             clip_model = T5CLIPModel(sd)
#             clip = StandaloneCLIP(clip_model)
        
#         return (None, clip, vae, clipvision)

#     # Model detection for UNet using our standalone detection
#     diffusion_model_prefix = unet_prefix_from_state_dict(sd)
#     parameters = calculate_parameters(sd, diffusion_model_prefix)
#     weight_dtype_val = weight_dtype(sd, diffusion_model_prefix)
#     load_device = get_torch_device()

#     # Use our standalone model detection
#     unet_config = detect_unet_config(sd, diffusion_model_prefix, metadata=metadata)
#     if unet_config is None:
#         logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
#         raise ValueError("Unsupported model type. Only WAN2.1 variants are supported.")

#     logging.info(f"Detected WAN model type: {unet_config.get('model_type', 'unknown')}")
#     logging.info(f"Model config: {unet_config}")
    
#     if output_model:
#         # Convert UNet config to model config
#         model_config = model_config_from_unet_config(unet_config)
#         if model_config is None:
#             raise ValueError("Failed to convert UNet config to model config")
        
#         logging.info(f"Creating model with config: {model_config}")
        
#         # Create the appropriate WAN model instance
#         model = create_model_from_config(model_config, device=None, dtype=weight_dtype_val, state_dict=sd)
        
#         # Memory-aware device management
#         log_memory_usage("Before model loading")
        
#         # Estimate memory requirements from state dict BEFORE loading
#         state_dict_info = estimate_state_dict_memory(sd)
#         logging.info(f"State dict analysis:")
#         logging.info(f"  Size: {state_dict_info['size_gb']:.2f} GB")
#         logging.info(f"  Parameters: {state_dict_info['parameters']:,}")
#         logging.info(f"  Keys: {state_dict_info['keys']}")
        
#         # Use advanced memory management with partial loading
#         model, load_device, _ = safe_model_to_device_advanced(model, load_device, min_free_gb=2.0, state_dict=sd, enable_partial_loading=True)
        
#         # Load the state dict into the model
#         try:
#             # Filter state dict to only include model weights
#             model_sd = {}
#             for k, v in sd.items():
#                 if k.startswith(diffusion_model_prefix):
#                     model_key = k[len(diffusion_model_prefix):]
#                     model_sd[model_key] = v
            
#             # Load state dict
#             missing_keys, unexpected_keys = model.load_state_dict(model_sd, strict=False)
#             if missing_keys:
#                 logging.warning(f"Missing keys in model: {missing_keys}")
#             if unexpected_keys:
#                 logging.warning(f"Unexpected keys in model: {unexpected_keys}")
            
#             logging.info(f"Successfully loaded model state dict")
            
#         except Exception as e:
#             logging.error(f"Failed to load model state dict: {e}")
#             raise
        
#         # Log memory usage after model loading
#         log_memory_usage("After model loading")
        
#         # Create model patcher with updated device
#         model_patcher = create_model_patcher(model, load_device=load_device, offload_device=unet_offload_device())

#     return (model_patcher, clip, vae, clipvision)

# if __name__ == "__main__":
#     print("Enhanced standalone_sd.py with CLIP support")
#     print("✅ Supports I2V, VACE, and Cross-Attention variants")
#     print("✅ Enhanced model detection")
#     print("✅ T5-XXL CLIP support")
#     print("✅ Proper model classes for ModelPatcher")

import torch
import logging
import os
import motion.model_management_standalone
import motion.model_patcher
import motion.utils
import motion.model_detection
from enum import Enum
from motion import sd1_clip
import motion.text_encoders.wan
import motion.text_encoders.sd3_clip

class CLIPType(Enum):
    WAN = 13

class TEModel(Enum):
    T5_XXL = 4

def load_wan_clip(ckpt_path, model_options={}):
    """Load WAN text encoder from single checkpoint path"""
    clip_data = [motion.utils.load_torch_file(ckpt_path, safe_load=True)]
    return load_wan_text_encoder_state_dict(clip_data, model_options=model_options)

def load_wan_text_encoder_state_dict(clip_data, model_options={}):
    """Load WAN text encoder from state dictionary"""
    
    # Process state dictionary
    if "transformer.resblocks.0.ln_1.weight" in clip_data[0]:
        clip_data[0] = motion.utils.clip_text_transformers_convert(clip_data[0], "", "")
    
    # WAN uses T5-XXL model
    te_model = detect_te_model(clip_data[0])
    if te_model != TEModel.T5_XXL:
        raise RuntimeError(f"Expected T5-XXL model for WAN, got {te_model}")
    
    # Get T5 detection parameters
    t5_params = t5xxl_detect(clip_data)
    
    # Configure WAN text encoder
    class EmptyClass:
        pass
    
    clip_target = EmptyClass()
    clip_target.clip = motion.text_encoders.wan.te(**t5_params)
    clip_target.tokenizer = motion.text_encoders.wan.WanT5Tokenizer
    clip_target.params = {}
    
    # Tokenizer data for WAN
    tokenizer_data = {"spiece_model": clip_data[0].get("spiece_model", None)}
    
    # Calculate parameters
    parameters = motion.utils.calculate_parameters(clip_data[0])
    
    # Create CLIP instance
    clip = CLIP(clip_target, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    
    # Load weights
    clip.load_sd(clip_data[0])
    
    return clip


def detect_te_model(sd):
    """Detect if this is a T5-XXL model (required for WAN)"""
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
    return None

def t5xxl_detect(clip_data):
    """Detect T5-XXL parameters for WAN"""
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    
    for sd in clip_data:
        if weight_name in sd:
            return motion.text_encoders.sd3_clip.t5_xxl_detect(sd)
    
    return {}

class CLIP:
    """Simplified CLIP wrapper for WAN"""
    def __init__(self, target, parameters=0, tokenizer_data={}, model_options={}):
        self.target = target
        self.parameters = parameters
        self.tokenizer_data = tokenizer_data
        self.model_options = model_options
        
        # Initialize model and tokenizer
        load_device = model_options.get("load_device", motion.model_management_standalone.text_encoder_device())
        offload_device = model_options.get("offload_device", motion.model_management_standalone.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        
        self.clip = target.clip(device=load_device, dtype=dtype, model_options=model_options)
        self.tokenizer = target.tokenizer(embedding_directory=None, tokenizer_data=tokenizer_data)
        
        # Wrap in ModelPatcher
        self.patcher = motion.model_patcher.ModelPatcher(
            self.clip, 
            load_device=load_device, 
            offload_device=offload_device
        )
    
    def load_sd(self, sd):
        """Load state dictionary into the model"""
        self.patcher.load_model_weights(sd)
    
    def load_model(self):
        """Return the model patcher"""
        return self.patcher
def load_diffusion_model(unet_path, model_options={}):
    """Load diffusion model from file path"""
    sd = motion.utils.load_torch_file(unet_path)
    model = load_diffusion_model_state_dict(sd, model_options=model_options)
    if model is None:
        logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
    return model

def load_diffusion_model_state_dict(sd, model_options={}):
    """Load diffusion model from state dictionary"""
    dtype = model_options.get("dtype", None)
    
    # Extract UNet from checkpoint if needed
    diffusion_model_prefix = motion.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = motion.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
    
    # Analyze model parameters
    parameters = motion.utils.calculate_parameters(sd)
    weight_dtype = motion.utils.weight_dtype(sd)
    
    # Get device configuration
    load_device = motion.model_management_standalone.get_torch_device()
    model_config = motion.model_detection.model_config_from_unet(sd, "")
    
    # Handle different model formats (diffusers, etc.)
    if model_config is None:
        new_sd = motion.model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = motion.model_detection.model_config_from_unet(new_sd, "")
        else:
            model_config = motion.model_detection.model_config_from_diffusers_unet(sd)
    
    if model_config is None:
        return None
    
    # Configure model dtype and device
    offload_device = motion.model_management_standalone.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    
    if dtype is None:
        unet_dtype = motion.model_management_standalone.unet_dtype(
            model_params=parameters, 
            supported_dtypes=unet_weight_dtype, 
            weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype
    
    manual_cast_dtype = motion.model_management_standalone.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    
    # Configure model
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    
    # Create and load model
    model = model_config.get_model(sd, "")
    model = model.to(offload_device)
    model.load_model_weights(sd, "")
    
    # Return wrapped model
    return motion.model_patcher.ModelPatcher(
        model, 
        load_device=load_device, 
        offload_device=offload_device
    )

def model_detection_error_hint(path, state_dict):
    """Helper function for error messages"""
    filename = os.path.basename(path)
    if 'lora' in filename.lower():
        return "\nHINT: This seems to be a Lora file and Lora files should be put in the lora folder and loaded with a lora loader node.."
    return ""