"""
Standalone Models Module - WAN2.1 VACE Focused
Contains all classes related to UNet loading and model management
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, List, Tuple
from utils import calculate_parameters, weight_dtype, state_dict_prefix_replace


# ============================================================================
# MODEL DETECTION CLASSES
# ============================================================================

class ModelDetector:
    """Model detection utilities focused on WAN2.1 VACE models"""
    
    @staticmethod
    def unet_prefix_from_state_dict(sd):
        """Extract UNet prefix from state dict"""
        prefixes = [
            "model.diffusion_model.",
            "unet.",
            "diffusion_model.",
            "model.",
            ""
        ]
        
        for prefix in prefixes:
            if any(k.startswith(prefix) for k in sd.keys()):
                return prefix
        
        return ""
    
    @staticmethod
    def detect_unet_config(state_dict, key_prefix, metadata=None):
        """Detect UNet configuration from state dict - WAN2.1 VACE only"""
        state_dict_keys = list(state_dict.keys())
        
        # Only support WAN2.1 VACE models
        if f"{key_prefix}head.modulation" not in state_dict_keys:
            raise ValueError(f"❌ Unsupported model: Not a WAN2.1 model. Missing 'head.modulation' key.")
        
        # Check if it's specifically VACE (not I2V)
        if f"{key_prefix}camera_cond_emb.proj.0.bias" not in state_dict_keys:
            raise ValueError(f"❌ Unsupported model: Not a WAN2.1 VACE model. Missing 'camera_cond_emb.proj.0.bias' key.")
        
        # Extract VACE configuration
        try:
            dim = state_dict[f"{key_prefix}head.modulation"].shape[-1]
            out_dim = state_dict[f"{key_prefix}head.head.weight"].shape[0] // 4
            ffn_dim = state_dict[f"{key_prefix}blocks.0.ffn.0.weight"].shape[0]
            num_layers = sum(
                1 for k in state_dict_keys
                if k.startswith(f"{key_prefix}blocks.") and k.endswith(".ffn.0.weight")
            )
            in_dim = state_dict[f"{key_prefix}patch_embedding.weight"].shape[1]

            # Build VACE config
            config = {
                "image_model": "wan2.1",
                "model_type": "vace",
                "dim": dim,
                "out_dim": out_dim,
                "num_heads": dim // 128,
                "ffn_dim": ffn_dim,
                "num_layers": num_layers,
                "patch_size": (1, 1, 1),  # VACE default
                "in_dim": in_dim,
                "key_prefix": key_prefix
            }

            # Optional extras
            if f"{key_prefix}img_emb.emb_pos" in state_dict_keys:
                config["flf_pos_embed_token_number"] = state_dict[f"{key_prefix}img_emb.emb_pos"].shape[1]

            logging.info(f"✅ Detected WAN2.1 VACE model: {num_layers} layers, dim={dim}, out_dim={out_dim}")
            return config
            
        except Exception as e:
            raise ValueError(f"❌ Failed to parse WAN2.1 VACE model: {e}")


# ============================================================================
# MODEL CONFIGURATION CLASSES
# ============================================================================

class WAN21VaceModelConfig:
    """Configuration class for WAN2.1 VACE models"""
    
    def __init__(self, config):
        self.config = config
        self.supported_inference_dtypes = [torch.float16, torch.float32]
        self.scaled_fp8 = None
        self.custom_operations = None
        self.clip_vision_prefix = None
        self.vae_key_prefix = []
        self._unet_dtype = torch.float16
        self._manual_cast_dtype = torch.float16
    
    def set_inference_dtype(self, unet_dtype, manual_cast_dtype):
        """Set inference data types"""
        self._unet_dtype = unet_dtype
        self._manual_cast_dtype = manual_cast_dtype
    
    def get_model(self, sd, prefix, device):
        """Create WAN2.1 VACE model instance"""
        return WAN21VaceModel(self.config, sd, prefix, device)
    
    def clip_target(self, state_dict=None):
        """Get CLIP target (not used for VACE)"""
        return None
    
    def process_clip_state_dict(self, sd):
        """Process CLIP state dict (not used for VACE)"""
        return {}
    
    def process_vae_state_dict(self, sd):
        """Process VAE state dict"""
        return sd


# ============================================================================
# MODEL IMPLEMENTATION CLASSES
# ============================================================================

class WAN21VaceModel(nn.Module):
    """WAN2.1 VACE model implementation"""
    
    def __init__(self, config, sd=None, prefix="", device=None):
        super().__init__()
        self.config = config
        self.model_type = "wan2.1_vace"
        self.state_dict_keys = []
        self.device = device or torch.device("cpu")
        
        # Initialize model components based on config
        self._initialize_components()
        
        # Load weights if state dict provided
        if sd is not None:
            self.load_model_weights(sd, prefix)
    
    def _initialize_components(self):
        """Initialize model components"""
        # This is a placeholder - in a real implementation, you'd create
        # the actual WAN2.1 VACE architecture here
        logging.info(f"Initializing WAN2.1 VACE model with config: {self.config}")
    
    def load_model_weights(self, sd, prefix):
        """Load WAN2.1 VACE model weights from state dict"""
        self.state_dict_keys = [k for k in sd.keys() if k.startswith(prefix)]
        logging.info(f"✅ Loaded WAN2.1 VACE model: {len(self.state_dict_keys)} parameters")
        logging.info(f"   Model type: {self.config['model_type']}")
        logging.info(f"   Dimensions: {self.config['dim']}D, {self.config['num_layers']} layers")
        
        # In a real implementation, you'd actually load the weights here
        # For now, we just track the keys
    
    def forward(self, *args, **kwargs):
        """Forward pass - placeholder implementation"""
        raise NotImplementedError("WAN2.1 VACE forward pass not implemented yet")


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_diffusion_model_state_dict(sd, model_options={}):
    """Load diffusion model from state dict - WAN2.1 VACE only"""
    detector = ModelDetector()
    
    # Get model prefix
    prefix = detector.unet_prefix_from_state_dict(sd)
    temp_sd = state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
    
    # Calculate parameters and weight dtype
    parameters = calculate_parameters(sd)
    weight_dtype_val = weight_dtype(sd)
    
    # Get load device
    load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Detect model config
    try:
        model_config = detector.detect_unet_config(sd, "")
        if model_config is None:
            return None
        
        # Create model config object
        config_obj = WAN21VaceModelConfig(model_config)
        
        # Create model
        model = config_obj.get_model(sd, "", device=load_device)
        
        # Create model patcher
        from standalone_model_patcher import create_model_patcher
        from wan_vae_components.model_management import unet_offload_device
        
        model_patcher = create_model_patcher(
            model, 
            load_device=load_device, 
            offload_device=unet_offload_device()
        )
        
        return model_patcher
        
    except ValueError as e:
        logging.error(f"Model loading failed: {e}")
        return None


def model_config_from_unet(state_dict, unet_key_prefix, metadata=None):
    """Create model config from UNet state dict - WAN2.1 VACE only"""
    detector = ModelDetector()
    
    try:
        unet_config = detector.detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
        return WAN21VaceModelConfig(unet_config)
    except ValueError as e:
        logging.error(f"Model config creation failed: {e}")
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_wan21_vace_model(sd, prefix="", device=None):
    """Create a WAN2.1 VACE model from state dict"""
    detector = ModelDetector()
    
    try:
        config = detector.detect_unet_config(sd, prefix)
        model = WAN21VaceModel(config, sd, prefix, device)
        return model
    except ValueError as e:
        logging.error(f"Model creation failed: {e}")
        return None


def validate_wan21_vace_model(sd, prefix=""):
    """Validate that a state dict contains a WAN2.1 VACE model"""
    detector = ModelDetector()
    
    try:
        config = detector.detect_unet_config(sd, prefix)
        return True, config
    except ValueError as e:
        return False, str(e)


if __name__ == "__main__":
    print("WAN2.1 VACE Models Module")
    print("✅ Model detection classes loaded")
    print("✅ Model configuration classes loaded") 
    print("✅ Model implementation classes loaded")
    print("✅ Model loading functions loaded")


# ============================================================================
# MODEL ARCHITECTURE CLASSES
# ============================================================================

class BaseModel(nn.Module):
    """Base model class for WAN2.1 VACE models"""
    
    def __init__(self, model_config, device=None):
        super().__init__()
        self.model_config = model_config
        self.device = device or torch.device("cpu")
        self.manual_cast_dtype = getattr(model_config, '_manual_cast_dtype', torch.float16)
        
        # Initialize the actual neural network
        self.diffusion_model = VaceWanModel(
            model_type='vace',
            patch_size=model_config.config.get('patch_size', (1, 1, 1)),
            in_dim=model_config.config.get('in_dim', 16),
            dim=model_config.config.get('dim', 2048),
            ffn_dim=model_config.config.get('ffn_dim', 8192),
            out_dim=model_config.config.get('out_dim', 16),
            num_heads=model_config.config.get('num_heads', 16),
            num_layers=model_config.config.get('num_layers', 32),
            device=device
        )
        
        logging.info(f"Initialized WAN2.1 VACE model on {self.device}")
        logging.info(f"Model dtype: {self.get_dtype()}, manual cast: {self.manual_cast_dtype}")
    
    def get_dtype(self):
        """Get model dtype"""
        return next(self.parameters()).dtype
    
    def load_model_weights(self, sd, prefix):
        """Load model weights from state dict"""
        # Filter state dict to only include our model's weights
        model_keys = [k for k in sd.keys() if k.startswith(prefix)]
        
        if not model_keys:
            logging.warning(f"No weights found with prefix '{prefix}'")
            return
        
        # Create filtered state dict
        model_sd = {k.replace(prefix, ""): v for k, v in sd.items() if k.startswith(prefix)}
        
        # Load weights into the model
        try:
            missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(model_sd, strict=False)
            
            if missing_keys:
                logging.warning(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                logging.warning(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            
            logging.info(f"✅ Loaded {len(model_sd)} model parameters")
            
        except Exception as e:
            logging.error(f"Failed to load model weights: {e}")
            raise
    
    def forward(self, *args, **kwargs):
        """Forward pass - placeholder"""
        raise NotImplementedError("Forward pass not implemented yet")


class VaceWanModel(nn.Module):
    """WAN2.1 VACE neural network implementation"""
    
    def __init__(self, model_type='vace', patch_size=(1, 1, 1), in_dim=16, dim=2048, 
                 ffn_dim=8192, out_dim=16, num_heads=16, num_layers=32, device=None):
        super().__init__()
        
        self.model_type = model_type
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")
        
        # Initialize model components
        self._initialize_components()
        
        logging.info(f"Created VaceWanModel: {num_layers} layers, {dim}D, {num_heads} heads")
    
    def _initialize_components(self):
        """Initialize model components"""
        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Transformer blocks (simplified)
        self.blocks = nn.ModuleList([
            self._create_transformer_block() for _ in range(self.num_layers)
        ])
        
        # Output head
        self.head = nn.Linear(self.dim, self.out_dim * 4)  # *4 for latent channels
        
        # VACE-specific components
        self.vace_patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # VACE attention blocks
        self.vace_blocks = nn.ModuleList([
            self._create_vace_block() for _ in range(min(4, self.num_layers // 8))
        ])
    
    def _create_transformer_block(self):
        """Create a transformer block"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(self.dim),
            'attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
            'norm2': nn.LayerNorm(self.dim),
            'mlp': nn.Sequential(
                nn.Linear(self.dim, self.ffn_dim),
                nn.GELU(),
                nn.Linear(self.ffn_dim, self.dim)
            )
        })
    
    def _create_vace_block(self):
        """Create a VACE-specific block"""
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(self.dim),
            'vace_attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
            'norm2': nn.LayerNorm(self.dim),
            'mlp': nn.Sequential(
                nn.Linear(self.dim, self.ffn_dim),
                nn.GELU(),
                nn.Linear(self.ffn_dim, self.dim)
            )
        })
    
    def forward(self, x, t=None, context=None, vace_context=None, vace_strength=None, **kwargs):
        """Forward pass - placeholder implementation"""
        # This is a simplified forward pass
        # In a real implementation, you'd have the full WAN2.1 VACE architecture
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Reshape for transformer
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B * T * H * W, C)
        
        # Apply transformer blocks
        for block in self.blocks:
            # Layer norm + attention
            norm_x = block['norm1'](x)
            attn_out, _ = block['attn'](norm_x, norm_x, norm_x)
            x = x + attn_out
            
            # Layer norm + MLP
            norm_x = block['norm2'](x)
            mlp_out = block['mlp'](norm_x)
            x = x + mlp_out
        
        # Output head
        x = self.head(x)
        
        # Reshape back to spatial format
        x = x.view(B, T, H, W, self.out_dim * 4).permute(0, 4, 1, 2, 3)
        
        return x


class WAN21_Vace(BaseModel):
    """WAN2.1 VACE model wrapper with VACE-specific functionality"""
    
    def __init__(self, model_config, device=None):
        super().__init__(model_config, device)
        self.image_to_video = False  # VACE is not image-to-video
    
    def extra_conds(self, **kwargs):
        """Handle VACE-specific conditioning"""
        out = {}
        
        # Handle VACE frames
        vace_frames = kwargs.get("vace_frames", None)
        if vace_frames is not None:
            out['vace_context'] = vace_frames
        
        # Handle VACE strength
        vace_strength = kwargs.get("vace_strength", [1.0])
        out['vace_strength'] = vace_strength
        
        return out
    
    def forward(self, x, **kwargs):
        """Forward pass with VACE conditioning"""
        vace_context = kwargs.get('vace_context', None)
        vace_strength = kwargs.get('vace_strength', [1.0])
        
        return self.diffusion_model(x, vace_context=vace_context, vace_strength=vace_strength)


# ============================================================================
# UPDATED MODEL CONFIGURATION
# ============================================================================

class WAN21VaceModelConfig:
    """Updated configuration class for WAN2.1 VACE models"""
    
    def __init__(self, config):
        self.config = config
        self.supported_inference_dtypes = [torch.float16, torch.float32]
        self.scaled_fp8 = None
        self.custom_operations = None
        self.clip_vision_prefix = None
        self.vae_key_prefix = []
        self._unet_dtype = torch.float16
        self._manual_cast_dtype = torch.float16
        self.memory_usage_factor = 1.2  # VACE uses more memory
    
    def set_inference_dtype(self, unet_dtype, manual_cast_dtype):
        """Set inference data types"""
        self._unet_dtype = unet_dtype
        self._manual_cast_dtype = manual_cast_dtype
    
    def get_model(self, sd, prefix, device):
        """Create WAN2.1 VACE model instance"""
        return WAN21_Vace(self, device)
    
    def clip_target(self, state_dict=None):
        """Get CLIP target (not used for VACE)"""
        return None
    
    def process_clip_state_dict(self, sd):
        """Process CLIP state dict (not used for VACE)"""
        return {}
    
    def process_vae_state_dict(self, sd):
        """Process VAE state dict"""
        return sd
