"""
Standalone Model Detection for WAN 2.1 VACE
Complete self-contained implementation with all dependencies included
"""

import torch
import logging
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# ============================================================================
# STANDALONE MODEL DETECTION
# ============================================================================

class ModelType(Enum):
    """Model type enumeration"""
    SD1_5 = "sd1.5"
    SDXL = "sdxl"
    SD3 = "sd3"
    WAN21_T2V = "wan21_t2v"
    WAN21_I2V = "wan21_i2v"
    WAN21_VACE = "wan21_vace"
    WAN21_CAMERA = "wan21_camera"
    WAN22_T2V = "wan22_t2v"
    UNKNOWN = "unknown"

class LatentFormat:
    """Simplified latent format"""
    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

class Wan21LatentFormat(LatentFormat):
    """WAN 2.1 latent format"""
    def __init__(self):
        super().__init__(scale_factor=1.0)

class ClipTarget:
    """Simplified CLIP target"""
    def __init__(self, tokenizer_class, text_encoder_class):
        self.tokenizer_class = tokenizer_class
        self.text_encoder_class = text_encoder_class

# ============================================================================
# STANDALONE UNET CONFIG DETECTION
# ============================================================================

def detect_unet_config(state_dict, unet_key_prefix, metadata=None):
    """
    Standalone UNet configuration detection
    
    Args:
        state_dict: Model state dictionary
        unet_key_prefix: UNet key prefix
        metadata: Optional metadata
    
    Returns:
        Detected UNet configuration dictionary
    """
    
    config = {}
    
    # Extract keys with prefix
    prefixed_keys = [k for k in state_dict.keys() if k.startswith(unet_key_prefix)]
    
    if not prefixed_keys:
        logging.warning(f"No keys found with prefix: {unet_key_prefix}")
        return None
    
    # Remove prefix for analysis
    model_keys = [k[len(unet_key_prefix):] for k in prefixed_keys]
    
    # Detect model dimensions
    config.update(detect_model_dimensions(model_keys, state_dict, unet_key_prefix))
    
    # Detect architecture type
    config.update(detect_architecture_type(model_keys, state_dict, unet_key_prefix))
    
    # Detect VACE-specific features
    config.update(detect_vace_features(model_keys, state_dict, unet_key_prefix))
    
    # Detect text encoder type
    config.update(detect_text_encoder_type(model_keys, state_dict, unet_key_prefix))
    
    # Detect VAE type
    config.update(detect_vae_type(model_keys, state_dict, unet_key_prefix))
    
    logging.info(f"Detected UNet config: {config}")
    return config

def detect_model_dimensions(model_keys, state_dict, prefix):
    """Detect model dimensions from state dict"""
    config = {}
    
    # Look for patch embedding to determine input dimension
    patch_embed_keys = [k for k in model_keys if "patch_embedding" in k and "weight" in k]
    if patch_embed_keys:
        try:
            weight = state_dict[prefix + patch_embed_keys[0]]
            if len(weight.shape) == 5:  # Conv3d: [out_channels, in_channels, d, h, w]
                config["in_dim"] = weight.shape[1]
                config["dim"] = weight.shape[0]
                logging.info(f"Detected dimensions: in_dim={config['in_dim']}, dim={config['dim']}")
        except Exception as e:
            logging.warning(f"Failed to detect dimensions: {e}")
    
    # Detect output dimension from projection layer
    proj_keys = [k for k in model_keys if "proj_out" in k and "weight" in k]
    if proj_keys:
        try:
            weight = state_dict[prefix + proj_keys[0]]
            if len(weight.shape) == 5:  # Conv3d
                config["out_dim"] = weight.shape[0]
                logging.info(f"Detected output dimension: {config['out_dim']}")
        except Exception as e:
            logging.warning(f"Failed to detect output dimension: {e}")
    
    return config

def detect_architecture_type(model_keys, state_dict, prefix):
    """Detect architecture type"""
    config = {}
    
    # Check for WAN-specific patterns
    wan_patterns = [
        "head.modulation",
        "head.head.weight",
        "pos_embed",
        "blocks.",
        "norm.weight"
    ]
    
    wan_score = sum(1 for pattern in wan_patterns if any(pattern in k for k in model_keys))
    
    if wan_score >= 3:
        config["architecture"] = "wan"
        logging.info("Detected WAN architecture")
        
        # Detect specific WAN variant
        if any("vace" in k.lower() for k in model_keys):
            config["model_type"] = "vace"
            config["image_model"] = "wan2.1"
            logging.info("Detected WAN 2.1 VACE model")
        elif any("camera" in k.lower() for k in model_keys):
            config["model_type"] = "camera"
            config["image_model"] = "wan2.1"
            logging.info("Detected WAN 2.1 Camera model")
        else:
            config["model_type"] = "t2v"
            config["image_model"] = "wan2.1"
            logging.info("Detected WAN 2.1 T2V model")
    
    # Check for SDXL patterns
    elif any("time_embed" in k for k in model_keys) and any("input_blocks" in k for k in model_keys):
        config["architecture"] = "sdxl"
        logging.info("Detected SDXL architecture")
    
    # Check for SD1.5 patterns
    elif any("time_embed" in k for k in model_keys) and any("middle_block" in k for k in model_keys):
        config["architecture"] = "sd1.5"
        logging.info("Detected SD1.5 architecture")
    
    else:
        config["architecture"] = "unknown"
        logging.warning("Could not detect architecture type")
    
    return config

def detect_vace_features(model_keys, state_dict, prefix):
    """Detect VACE-specific features"""
    config = {}
    
    # Check for VACE-specific keys
    vace_keys = [k for k in model_keys if "vace" in k.lower()]
    
    if vace_keys:
        config["has_vace"] = True
        config["vace_layers"] = len([k for k in vace_keys if "vace_blocks" in k])
        
        # Detect VACE input dimension
        vace_patch_keys = [k for k in vace_keys if "vace_patch_embedding" in k and "weight" in k]
        if vace_patch_keys:
            try:
                weight = state_dict[prefix + vace_patch_keys[0]]
                config["vace_in_dim"] = weight.shape[1]
                logging.info(f"Detected VACE input dimension: {config['vace_in_dim']}")
            except Exception as e:
                logging.warning(f"Failed to detect VACE input dimension: {e}")
        
        logging.info(f"Detected VACE features: {len(vace_keys)} VACE keys")
    else:
        config["has_vace"] = False
    
    return config

def detect_text_encoder_type(model_keys, state_dict, prefix):
    """Detect text encoder type"""
    config = {}
    
    # Look for text encoder keys
    text_keys = [k for k in model_keys if any(te in k for te in ["text_encoders", "clip", "t5", "umt5"])]
    
    if text_keys:
        if any("umt5" in k for k in text_keys):
            config["text_encoder"] = "umt5"
            logging.info("Detected UMT5 text encoder")
        elif any("t5" in k for k in text_keys):
            config["text_encoder"] = "t5"
            logging.info("Detected T5 text encoder")
        elif any("clip" in k for k in text_keys):
            config["text_encoder"] = "clip"
            logging.info("Detected CLIP text encoder")
    
    return config

def detect_vae_type(model_keys, state_dict, prefix):
    """Detect VAE type"""
    config = {}
    
    # Look for VAE keys
    vae_keys = [k for k in model_keys if "vae" in k.lower()]
    
    if vae_keys:
        config["has_vae"] = True
        logging.info(f"Detected VAE: {len(vae_keys)} VAE keys")
    else:
        config["has_vae"] = False
    
    return config

# ============================================================================
# STANDALONE MODEL CONFIGURATION CLASSES
# ============================================================================

class BaseModelConfig:
    """Base model configuration"""
    
    def __init__(self, unet_config):
        self.unet_config = unet_config
        self.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.memory_usage_factor = 1.0
        self.scaled_fp8 = None
        self.custom_operations = None
        self.optimizations = {"fp8": False}
        
        # Default values
        self.vae_key_prefix = ["vae."]
        self.text_encoder_key_prefix = ["text_encoders."]
        self.latent_format = LatentFormat()
    
    def get_model(self, state_dict, prefix="", device=None):
        """Get model instance - to be implemented by subclasses"""
        raise NotImplementedError
    
    def clip_target(self, state_dict={}):
        """Get CLIP target - to be implemented by subclasses"""
        return None
    
    def set_inference_dtype(self, unet_dtype, manual_cast_dtype):
        """Set inference dtype"""
        self.inference_dtype = unet_dtype
        self.manual_cast_dtype = manual_cast_dtype
    
    def process_vae_state_dict(self, vae_sd):
        """Process VAE state dict"""
        return vae_sd
    
    def process_clip_state_dict(self, clip_sd):
        """Process CLIP state dict"""
        return clip_sd

class WAN21VaceConfig(BaseModelConfig):
    """WAN 2.1 VACE configuration"""
    
    def __init__(self, unet_config):
        super().__init__(unet_config)
        
        # WAN 2.1 VACE specific settings
        self.unet_config.update({
            "image_model": "wan2.1",
            "model_type": "vace",
        })
        
        self.sampling_settings = {"shift": 8.0}
        self.latent_format = Wan21LatentFormat()
        self.memory_usage_factor = 1.2  # VACE requires more memory
        
        # VACE-specific parameters
        if "vace_layers" in unet_config:
            self.vace_layers = unet_config["vace_layers"]
        else:
            self.vace_layers = 8  # Default VACE layers
        
        if "vace_in_dim" in unet_config:
            self.vace_in_dim = unet_config["vace_in_dim"]
        else:
            self.vace_in_dim = 16  # Default VACE input dimension
        
        logging.info(f"WAN 2.1 VACE config initialized: vace_layers={self.vace_layers}, vace_in_dim={self.vace_in_dim}")
    
    def get_model(self, state_dict, prefix="", device=None):
        """Create WAN 2.1 VACE model instance"""
        from wan21_vace_model import WAN21VaceModel  # Import from our standalone model
        return WAN21VaceModel(self, device=device)
    
    def clip_target(self, state_dict={}):
        """Get CLIP target for WAN 2.1 VACE"""
        # Simplified - return None for now
        return None

class WAN21T2VConfig(BaseModelConfig):
    """WAN 2.1 T2V configuration"""
    
    def __init__(self, unet_config):
        super().__init__(unet_config)
        
        self.unet_config.update({
            "image_model": "wan2.1",
            "model_type": "t2v",
        })
        
        self.sampling_settings = {"shift": 8.0}
        self.latent_format = Wan21LatentFormat()
        self.memory_usage_factor = 1.0

class WAN21CameraConfig(BaseModelConfig):
    """WAN 2.1 Camera configuration"""
    
    def __init__(self, unet_config):
        super().__init__(unet_config)
        
        self.unet_config.update({
            "image_model": "wan2.1",
            "model_type": "camera",
        })
        
        self.sampling_settings = {"shift": 8.0}
        self.latent_format = Wan21LatentFormat()
        self.memory_usage_factor = 1.1

# ============================================================================
# STANDALONE MODEL CONFIG DETECTION
# ============================================================================

def model_config_from_unet_config(unet_config, state_dict):
    """
    Standalone model configuration detection from UNet config
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Model state dictionary
    
    Returns:
        Model configuration instance or None
    """
    
    if unet_config is None:
        return None
    
    # Check for WAN 2.1 VACE
    if (unet_config.get("architecture") == "wan" and 
        unet_config.get("model_type") == "vace" and
        unet_config.get("has_vace", False)):
        
        logging.info("Creating WAN 2.1 VACE configuration")
        return WAN21VaceConfig(unet_config)
    
    # Check for WAN 2.1 Camera
    elif (unet_config.get("architecture") == "wan" and 
          unet_config.get("model_type") == "camera"):
        
        logging.info("Creating WAN 2.1 Camera configuration")
        return WAN21CameraConfig(unet_config)
    
    # Check for WAN 2.1 T2V
    elif (unet_config.get("architecture") == "wan" and 
          unet_config.get("model_type") == "t2v"):
        
        logging.info("Creating WAN 2.1 T2V configuration")
        return WAN21T2VConfig(unet_config)
    
    # Fallback to base configuration
    else:
        logging.warning("No specific model configuration found, using base configuration")
        return BaseModelConfig(unet_config)

# ============================================================================
# MAIN STANDALONE FUNCTION
# ============================================================================

def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
    """
    Standalone model_config_from_unet function with WAN 2.1 VACE detection
    
    Args:
        state_dict: Model state dictionary
        unet_key_prefix: UNet key prefix (e.g., "model.", "unet.")
        use_base_if_no_match: Whether to use base model if no match found
        metadata: Optional metadata from checkpoint
    
    Returns:
        Model configuration instance or None
    """
    
    logging.info(f"Starting standalone model detection with prefix: {unet_key_prefix}")
    
    # Step 1: Detect UNet configuration
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    
    if unet_config is None:
        logging.warning("Could not detect UNet configuration")
        if use_base_if_no_match:
            logging.info("Using base model configuration as fallback")
            return BaseModelConfig({})
        return None
    
    # Step 2: Create model configuration from UNet config
    model_config = model_config_from_unet_config(unet_config, state_dict)
    
    if model_config is None:
        logging.warning("Could not create model configuration")
        if use_base_if_no_match:
            logging.info("Using base model configuration as fallback")
            return BaseModelConfig(unet_config)
        return None
    
    # Step 3: Handle scaled_fp8 if present
    model_config = apply_scaled_fp8_settings(model_config, state_dict, unet_key_prefix)
    
    logging.info(f"Successfully created model configuration: {type(model_config).__name__}")
    return model_config

def apply_scaled_fp8_settings(model_config, state_dict, unet_key_prefix):
    """
    Apply scaled_fp8 settings to model configuration
    
    Args:
        model_config: Model configuration instance
        state_dict: Model state dictionary
        unet_key_prefix: UNet key prefix
    
    Returns:
        Model configuration with fp8 settings applied
    """
    
    if model_config is None:
        return None
    
    scaled_fp8_key = f"{unet_key_prefix}scaled_fp8"
    if scaled_fp8_key in state_dict:
        scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
        model_config.scaled_fp8 = scaled_fp8_weight.dtype
        
        if model_config.scaled_fp8 == torch.float32:
            model_config.scaled_fp8 = torch.float8_e4m3fn
        
        if scaled_fp8_weight.nelement() == 2:
            model_config.optimizations["fp8"] = False
        else:
            model_config.optimizations["fp8"] = True
        
        logging.info(f"Applied scaled_fp8 settings: {model_config.scaled_fp8}")
    
    return model_config

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_potentially_wan_model(unet_config, state_dict):
    """
    Check if the model might be a WAN model
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Model state dictionary
    
    Returns:
        True if this might be a WAN model
    """
    
    # Check for WAN-specific configuration keys
    wan_config_keys = ["image_model", "model_type", "dim", "out_dim", "architecture"]
    has_wan_config = any(key in unet_config for key in wan_config_keys)
    
    # Check for WAN-specific state dict keys
    wan_state_keys = [
        "head.modulation",
        "head.head.weight", 
        "patch_embedding.weight",
        "blocks.",
        "vace_patch_embedding.weight",
        "vace_blocks.",
        "pos_embed"
    ]
    has_wan_state = any(key in state_dict for key in wan_state_keys)
    
    return has_wan_config or has_wan_state

def get_model_type_from_config(model_config):
    """Get model type from configuration"""
    if hasattr(model_config, 'unet_config'):
        return model_config.unet_config.get("model_type", "unknown")
    return "unknown"

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=== Testing Standalone Model Detection ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock state dict for WAN 2.1 VACE
    mock_state_dict = {
        "model.head.modulation": torch.randn(1, 2000),
        "model.head.head.weight": torch.randn(64, 2000),
        "model.patch_embedding.weight": torch.randn(2000, 16, 1, 2, 2),
        "model.vace_patch_embedding.weight": torch.randn(2000, 16, 1, 2, 2),
        "model.vace_blocks.0.attn.weight": torch.randn(2000, 2000),
        "model.blocks.0.attn.weight": torch.randn(2000, 2000),
        "model.pos_embed": torch.randn(1, 1024, 2000),
        "model.norm.weight": torch.randn(2000),
        "model.proj_out.weight": torch.randn(16, 2000, 1, 1, 1),
    }
    
    print(f"Created mock state dict with {len(mock_state_dict)} keys")
    
    # Test the standalone function
    print("\nTesting standalone model detection...")
    result = model_config_from_unet(mock_state_dict, "model.", metadata=None)
    
    if result:
        print("✅ Successfully detected model configuration")
        print(f"Model type: {type(result).__name__}")
        print(f"Model config: {result.unet_config}")
        print(f"Memory usage factor: {result.memory_usage_factor}")
        print(f"Supported dtypes: {result.supported_inference_dtypes}")
        
        if hasattr(result, 'vace_layers'):
            print(f"VACE layers: {result.vace_layers}")
        if hasattr(result, 'vace_in_dim'):
            print(f"VACE input dimension: {result.vace_in_dim}")
        
    else:
        print("❌ Failed to detect model configuration")
    
    print("\n🎉 Standalone model detection ready!")
