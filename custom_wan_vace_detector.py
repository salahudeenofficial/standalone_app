"""
Custom Model Detection for WAN2.1 Vace
Replacement for model_config_from_unet_config function specifically for WAN2.1 Vace models
"""

import torch
import logging
from typing import Dict, Any, Optional


class CustomWAN21VaceConfig:
    """
    Custom configuration class for WAN2.1 Vace models
    """
    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = {"shift": 8.0}
        self.latent_format = None  # Will be set based on detected format
        self.memory_usage_factor = 1.2  # Vace has higher memory usage
        self.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        self.vae_key_prefix = ["vae."]
        self.text_encoder_key_prefix = ["text_encoders."]
        
        # Calculate memory usage factor based on model dimension
        dim = self.unet_config.get("dim", 2000)
        self.memory_usage_factor = 1.2 * (dim / 2000)
    
    def get_model(self, state_dict, prefix="", device=None):
        """Create the actual model instance"""
        # Import here to avoid circular imports
        from comfy.model_base import WAN21_Vace
        return WAN21_Vace(self, image_to_video=False, device=device)
    
    def clip_target(self, state_dict={}):
        """Detect and return CLIP target for WAN2.1 Vace"""
        pref = self.text_encoder_key_prefix[0]
        
        # Detect T5 configuration
        from comfy.text_encoders.sd3_clip import t5_xxl_detect
        t5_detect = t5_xxl_detect(state_dict, "{}umt5xxl.transformer.".format(pref))
        
        # Return CLIP target
        from comfy.text_encoders.wan import WanT5Tokenizer, te
        from comfy.supported_models_base import ClipTarget
        
        return ClipTarget(WanT5Tokenizer, te(**t5_detect))
    
    def process_clip_state_dict(self, state_dict):
        """Process CLIP state dict for WAN2.1 Vace"""
        from comfy.utils import state_dict_prefix_replace
        return state_dict_prefix_replace(
            state_dict, 
            {k: "" for k in self.text_encoder_key_prefix}, 
            filter_keys=True
        )
    
    def process_vae_state_dict(self, state_dict):
        """Process VAE state dict for WAN2.1 Vace"""
        from comfy.utils import state_dict_prefix_replace
        return state_dict_prefix_replace(
            state_dict, 
            {k: "" for k in self.vae_key_prefix}, 
            filter_keys=True
        )


def custom_model_config_from_unet_config(unet_config: Dict[str, Any], state_dict: Optional[Dict] = None):
    """
    Custom replacement for model_config_from_unet_config specifically for WAN2.1 Vace
    
    Args:
        unet_config: Detected UNet configuration dictionary
        state_dict: Optional state dictionary for additional validation
    
    Returns:
        CustomWAN21VaceConfig instance if it's a WAN2.1 Vace model, None otherwise
    """
    
    # Check if this is a WAN2.1 Vace model
    if is_wan21_vace_model(unet_config, state_dict):
        logging.info("Detected WAN2.1 Vace model - using custom configuration")
        return CustomWAN21VaceConfig(unet_config)
    
    # If not WAN2.1 Vace, fall back to original detection
    logging.info("Not a WAN2.1 Vace model - falling back to original detection")
    return fallback_to_original_detection(unet_config, state_dict)


def is_wan21_vace_model(unet_config: Dict[str, Any], state_dict: Optional[Dict] = None) -> bool:
    """
    Check if the detected configuration matches WAN2.1 Vace model
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Optional state dictionary
    
    Returns:
        True if this is a WAN2.1 Vace model
    """
    
    # Required keys for WAN2.1 Vace
    required_config_keys = {
        "image_model": "wan2.1",
        "model_type": "vace",
    }
    
    # Check if all required config keys match
    for key, expected_value in required_config_keys.items():
        if key not in unet_config:
            return False
        if unet_config[key] != expected_value:
            return False
    
    # Additional validation using state dict if provided
    if state_dict is not None:
        # Check for WAN2.1 Vace specific keys in state dict
        wan_vace_keys = [
            "vace_patch_embedding.weight",
            "vace_blocks.",
            "head.modulation",
            "head.head.weight",
        ]
        
        # Check if any of these keys exist in the state dict
        has_wan_keys = any(key in state_dict for key in wan_vace_keys)
        if not has_wan_keys:
            return False
    
    return True


def fallback_to_original_detection(unet_config: Dict[str, Any], state_dict: Optional[Dict] = None):
    """
    Fallback to original ComfyUI model detection
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Optional state dictionary
    
    Returns:
        Model configuration from original detection system
    """
    try:
        # Import original detection function
        from comfy.model_detection import model_config_from_unet_config as original_detector
        
        # Call original detection
        return original_detector(unet_config, state_dict)
        
    except ImportError:
        logging.error("Could not import original model detection function")
        return None
    except Exception as e:
        logging.error(f"Error in fallback detection: {e}")
        return None


def enhanced_wan21_vace_detection(unet_config: Dict[str, Any], state_dict: Optional[Dict] = None):
    """
    Enhanced detection with additional WAN2.1 Vace specific checks
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Optional state dictionary
    
    Returns:
        CustomWAN21VaceConfig instance with enhanced configuration
    """
    
    if not is_wan21_vace_model(unet_config, state_dict):
        return fallback_to_original_detection(unet_config, state_dict)
    
    # Create enhanced configuration
    config = CustomWAN21VaceConfig(unet_config)
    
    # Add additional WAN2.1 Vace specific configurations
    if state_dict is not None:
        # Detect Vace-specific parameters
        config.vace_layers = detect_vace_layers(state_dict)
        config.vace_in_dim = detect_vace_input_dim(state_dict)
        
        # Detect model dimensions
        config.dim = detect_model_dimension(state_dict)
        config.num_layers = detect_num_layers(state_dict)
        
        # Detect patch size
        config.patch_size = detect_patch_size(state_dict)
    
    logging.info(f"Enhanced WAN2.1 Vace configuration: {config.unet_config}")
    return config


def detect_vace_layers(state_dict: Dict[str, Any]) -> int:
    """Detect number of Vace layers from state dict"""
    vace_layer_keys = [k for k in state_dict.keys() if "vace_blocks." in k and ".weight" in k]
    if vace_layer_keys:
        # Count unique layer numbers
        layer_numbers = set()
        for key in vace_layer_keys:
            # Extract layer number from key like "vace_blocks.0.attn.weight"
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_numbers.add(int(parts[1]))
        return len(layer_numbers)
    return 0


def detect_vace_input_dim(state_dict: Dict[str, Any]) -> int:
    """Detect Vace input dimension from state dict"""
    vace_patch_key = "vace_patch_embedding.weight"
    if vace_patch_key in state_dict:
        return state_dict[vace_patch_key].shape[1]
    return 16  # Default


def detect_model_dimension(state_dict: Dict[str, Any]) -> int:
    """Detect model dimension from state dict"""
    head_modulation_key = "head.modulation"
    if head_modulation_key in state_dict:
        return state_dict[head_modulation_key].shape[-1]
    return 2000  # Default


def detect_num_layers(state_dict: Dict[str, Any]) -> int:
    """Detect number of layers from state dict"""
    block_keys = [k for k in state_dict.keys() if "blocks." in k and ".weight" in k]
    if block_keys:
        layer_numbers = set()
        for key in block_keys:
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_numbers.add(int(parts[1]))
        return len(layer_numbers)
    return 32  # Default


def detect_patch_size(state_dict: Dict[str, Any]) -> tuple:
    """Detect patch size from state dict"""
    patch_embedding_key = "patch_embedding.weight"
    if patch_embedding_key in state_dict:
        weight_shape = state_dict[patch_embedding_key].shape
        if len(weight_shape) >= 3:
            return (1, weight_shape[2], weight_shape[3])
    return (1, 2, 2)  # Default


# Example usage and testing
if __name__ == "__main__":
    # Example WAN2.1 Vace configuration
    example_unet_config = {
        "image_model": "wan2.1",
        "model_type": "vace",
        "dim": 2000,
        "out_dim": 16,
        "num_heads": 16,
        "ffn_dim": 8192,
        "num_layers": 32,
        "patch_size": (1, 2, 2),
        "freq_dim": 256,
        "window_size": (-1, -1),
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "in_dim": 16,
        "vace_in_dim": 16,
        "vace_layers": 8,
    }
    
    # Test the custom detection
    config = custom_model_config_from_unet_config(example_unet_config)
    
    if config:
        print("✅ Successfully detected WAN2.1 Vace model")
        print(f"Memory usage factor: {config.memory_usage_factor}")
        print(f"Model config: {config.unet_config}")
    else:
        print("❌ Failed to detect WAN2.1 Vace model")
    
    print("\nCustom WAN2.1 Vace detector ready!")
