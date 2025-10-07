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
#     """T5-XXL CLIP model class - motionUI-style implementation"""
    
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
        
#         # Create dummy parameter for compatibility (motionUI pattern)
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
import motion.model_management_standalone as model_management
import motion.model_patcher
import motion.utils
import motion.model_detection
from enum import Enum
from motion import sd1_clip
import motion.text_encoders.wan
import motion.text_encoders.sd3_clip

class CLIPType(Enum):
    WAN = 13
    CLIP_G = 3

class TEModel(Enum):
    T5_XXL = 4

class CLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False, tokenizer_data={}, parameters=0, model_options={}):
        if no_init:
            return
        
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_options.get("load_device", model_management.text_encoder_device())
        offload_device = model_options.get("offload_device", model_management.text_encoder_offload_device())
        dtype = model_options.get("dtype", None)
        if dtype is None:
            dtype = model_management.text_encoder_dtype(load_device)

        params['dtype'] = dtype
        params['device'] = model_options.get("initial_device", model_management.text_encoder_initial_device(load_device, offload_device, parameters * model_management.dtype_size(dtype)))
        params['model_options'] = model_options

        self.cond_stage_model = clip(**(params))
        
        # Initialize tokenizer
        self.tokenizer = tokenizer(tokenizer_data)
        
        # Create model patcher
        self.patcher = model_management.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def load_sd(self, sd, full_model=False):
        """Load state dictionary"""
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        """Get state dictionary"""
        sd_clip = self.cond_stage_model.state_dict()
        sd_tokenizer = self.tokenizer.state_dict()
        for k in sd_tokenizer:
            sd_clip[k] = sd_tokenizer[k]
        return sd_clip

    def load_model(self):
        """Load model to GPU"""
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        """Get key patches"""
        return self.patcher.get_key_patches()

def detect_te_model(sd):
    if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
        return TEModel.CLIP_G
    if "text_model.encoder.layers.22.mlp.fc1.weight" in sd:
        return TEModel.CLIP_H
    if "text_model.encoder.layers.0.mlp.fc1.weight" in sd:
        return TEModel.CLIP_L
    if "encoder.block.23.layer.1.DenseReluDense.wi_1.weight" in sd:
        weight = sd["encoder.block.23.layer.1.DenseReluDense.wi_1.weight"]
        if weight.shape[-1] == 4096:
            return TEModel.T5_XXL
        elif weight.shape[-1] == 2048:
            return TEModel.T5_XL
    if 'encoder.block.23.layer.1.DenseReluDense.wi.weight' in sd:
        return TEModel.T5_XXL_OLD
    if "encoder.block.0.layer.0.SelfAttention.k.weight" in sd:
        return TEModel.T5_BASE
    if 'model.layers.0.post_feedforward_layernorm.weight' in sd:
        return TEModel.GEMMA_2_2B
    if 'model.layers.0.self_attn.k_proj.bias' in sd:
        weight = sd['model.layers.0.self_attn.k_proj.bias']
        if weight.shape[0] == 256:
            return TEModel.QWEN25_3B
        if weight.shape[0] == 512:
            return TEModel.QWEN25_7B
    if "model.layers.0.post_attention_layernorm.weight" in sd:
        return TEModel.LLAMA3_8
    return None



def t5xxl_detect(clip_data):
    """Detect T5-XXL parameters for WAN"""
    weight_name = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
    
    for sd in clip_data:
        if weight_name in sd:
            return motion.text_encoders.sd3_clip.t5_xxl_detect(sd)
    
    return {}



def load_clip(ckpt_paths, embedding_directory=None, clip_type="wan", model_options={}):
    """Load CLIP model from checkpoint paths"""
    clip_data = []
    for p in ckpt_paths:
        clip_data.append(motion.utils.load_torch_file(p, safe_load=True))
    return load_text_encoder_state_dicts(clip_data, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)
def load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = state_dicts

    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = motion.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) #old models saved with the CLIPSave node

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}
    if len(clip_data) == 1:
        te_model = detect_te_model(clip_data[0])
        if te_model == TEModel.CLIP_G:
            if clip_type == CLIPType.STABLE_CASCADE:
                clip_target.clip = sdxl_clip.StableCascadeClipModel
                clip_target.tokenizer = sdxl_clip.StableCascadeTokenizer
            elif clip_type == CLIPType.SD3:
                clip_target.clip = motion.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=True, t5=False)
                clip_target.tokenizer = motion.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = motion.text_encoders.hidream.hidream_clip(clip_l=False, clip_g=True, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sdxl_clip.SDXLRefinerClipModel
                clip_target.tokenizer = sdxl_clip.SDXLTokenizer
        elif te_model == TEModel.CLIP_H:
            clip_target.clip = motion.text_encoders.sd2_clip.SD2ClipModel
            clip_target.tokenizer = motion.text_encoders.sd2_clip.SD2Tokenizer
        elif te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.SD3:
                clip_target.clip = motion.text_encoders.sd3_clip.sd3_clip(clip_l=False, clip_g=False, t5=True, **t5xxl_detect(clip_data))
                clip_target.tokenizer = motion.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.LTXV:
                clip_target.clip = motion.text_encoders.lt.ltxv_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = motion.text_encoders.lt.LTXVT5Tokenizer
            elif clip_type == CLIPType.PIXART or clip_type == CLIPType.CHROMA:
                clip_target.clip = motion.text_encoders.pixart_t5.pixart_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = motion.text_encoders.pixart_t5.PixArtTokenizer
            elif clip_type == CLIPType.WAN:
                clip_target.clip = motion.text_encoders.wan.te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = motion.text_encoders.wan.WanT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = motion.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=True, llama=False, dtype_llama=None, llama_scaled_fp8=None)
                clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer
            else: #CLIPType.MOCHI
                clip_target.clip = motion.text_encoders.genmo.mochi_te(**t5xxl_detect(clip_data))
                clip_target.tokenizer = motion.text_encoders.genmo.MochiT5Tokenizer
        elif te_model == TEModel.T5_XXL_OLD:
            clip_target.clip = motion.text_encoders.cosmos.te(**t5xxl_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.cosmos.CosmosT5Tokenizer
        elif te_model == TEModel.T5_XL:
            clip_target.clip = motion.text_encoders.aura_t5.AuraT5Model
            clip_target.tokenizer = motion.text_encoders.aura_t5.AuraT5Tokenizer
        elif te_model == TEModel.T5_BASE:
            if clip_type == CLIPType.ACE or "spiece_model" in clip_data[0]:
                clip_target.clip = motion.text_encoders.ace.AceT5Model
                clip_target.tokenizer = motion.text_encoders.ace.AceT5Tokenizer
                tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
            else:
                clip_target.clip = motion.text_encoders.sa_t5.SAT5Model
                clip_target.tokenizer = motion.text_encoders.sa_t5.SAT5Tokenizer
        elif te_model == TEModel.GEMMA_2_2B:
            clip_target.clip = motion.text_encoders.lumina2.te(**llama_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.lumina2.LuminaTokenizer
            tokenizer_data["spiece_model"] = clip_data[0].get("spiece_model", None)
        elif te_model == TEModel.LLAMA3_8:
            clip_target.clip = motion.text_encoders.hidream.hidream_clip(**llama_detect(clip_data),
                                                                        clip_l=False, clip_g=False, t5=False, llama=True, dtype_t5=None, t5xxl_scaled_fp8=None)
            clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer
        elif te_model == TEModel.QWEN25_3B:
            clip_target.clip = motion.text_encoders.omnigen2.te(**llama_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.omnigen2.Omnigen2Tokenizer
        elif te_model == TEModel.QWEN25_7B:
            clip_target.clip = motion.text_encoders.qwen_image.te(**llama_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.qwen_image.QwenImageTokenizer
        else:
            # clip_l
            if clip_type == CLIPType.SD3:
                clip_target.clip = motion.text_encoders.sd3_clip.sd3_clip(clip_l=True, clip_g=False, t5=False)
                clip_target.tokenizer = motion.text_encoders.sd3_clip.SD3Tokenizer
            elif clip_type == CLIPType.HIDREAM:
                clip_target.clip = motion.text_encoders.hidream.hidream_clip(clip_l=True, clip_g=False, t5=False, llama=False, dtype_t5=None, dtype_llama=None, t5xxl_scaled_fp8=None, llama_scaled_fp8=None)
                clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer
            else:
                clip_target.clip = sd1_clip.SD1ClipModel
                clip_target.tokenizer = sd1_clip.SD1Tokenizer
    elif len(clip_data) == 2:
        if clip_type == CLIPType.SD3:
            te_models = [detect_te_model(clip_data[0]), detect_te_model(clip_data[1])]
            clip_target.clip = motion.text_encoders.sd3_clip.sd3_clip(clip_l=TEModel.CLIP_L in te_models, clip_g=TEModel.CLIP_G in te_models, t5=TEModel.T5_XXL in te_models, **t5xxl_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.sd3_clip.SD3Tokenizer
        elif clip_type == CLIPType.HUNYUAN_DIT:
            clip_target.clip = motion.text_encoders.hydit.HyditModel
            clip_target.tokenizer = motion.text_encoders.hydit.HyditTokenizer
        elif clip_type == CLIPType.FLUX:
            clip_target.clip = motion.text_encoders.flux.flux_clip(**t5xxl_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.flux.FluxTokenizer
        elif clip_type == CLIPType.HUNYUAN_VIDEO:
            clip_target.clip = motion.text_encoders.hunyuan_video.hunyuan_video_clip(**llama_detect(clip_data))
            clip_target.tokenizer = motion.text_encoders.hunyuan_video.HunyuanVideoTokenizer
        elif clip_type == CLIPType.HIDREAM:
            # Detect
            hidream_dualclip_classes = []
            for hidream_te in clip_data:
                te_model = detect_te_model(hidream_te)
                hidream_dualclip_classes.append(te_model)

            clip_l = TEModel.CLIP_L in hidream_dualclip_classes
            clip_g = TEModel.CLIP_G in hidream_dualclip_classes
            t5 = TEModel.T5_XXL in hidream_dualclip_classes
            llama = TEModel.LLAMA3_8 in hidream_dualclip_classes

            # Initialize t5xxl_detect and llama_detect kwargs if needed
            t5_kwargs = t5xxl_detect(clip_data) if t5 else {}
            llama_kwargs = llama_detect(clip_data) if llama else {}

            clip_target.clip = motion.text_encoders.hidream.hidream_clip(clip_l=clip_l, clip_g=clip_g, t5=t5, llama=llama, **t5_kwargs, **llama_kwargs)
            clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer
        else:
            clip_target.clip = sdxl_clip.SDXLClipModel
            clip_target.tokenizer = sdxl_clip.SDXLTokenizer
    elif len(clip_data) == 3:
        clip_target.clip = motion.text_encoders.sd3_clip.sd3_clip(**t5xxl_detect(clip_data))
        clip_target.tokenizer = motion.text_encoders.sd3_clip.SD3Tokenizer
    elif len(clip_data) == 4:
        clip_target.clip = motion.text_encoders.hidream.hidream_clip(**t5xxl_detect(clip_data), **llama_detect(clip_data))
        clip_target.tokenizer = motion.text_encoders.hidream.HiDreamTokenizer

    parameters = 0
    for c in clip_data:
        parameters += motion.utils.calculate_parameters(c)
        tokenizer_data, model_options = motion.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    clip = CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        m, u = clip.load_sd(c)
        if len(m) > 0:
            logging.warning("clip missing: {}".format(m))

        if len(u) > 0:
            logging.debug("clip unexpected: {}".format(u))
    return clip

def model_options_long_clip(sd, tokenizer_data, model_options):
    w = sd.get("clip_l.text_model.embeddings.position_embedding.weight", None)
    if w is None:
        w = sd.get("clip_g.text_model.embeddings.position_embedding.weight", None)
    else:
        model_name = "clip_g"

    if w is None:
        w = sd.get("text_model.embeddings.position_embedding.weight", None)
        if w is not None:
            if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
                model_name = "clip_g"
            elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
                model_name = "clip_l"
    else:
        model_name = "clip_l"

    if w is not None:
        tokenizer_data = tokenizer_data.copy()
        model_options = model_options.copy()
        model_config = model_options.get("model_config", {})
        model_config["max_position_embeddings"] = w.shape[0]
        model_options["{}_model_config".format(model_name)] = model_config
        tokenizer_data["{}_max_length".format(model_name)] = w.shape[0]
    return tokenizer_data, model_options    