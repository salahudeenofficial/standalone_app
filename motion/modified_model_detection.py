"""
Modified model_config_from_unet function with custom WAN2.1 Vace detection
"""

import torch
import logging
from custom_wan_vace_detector import custom_model_config_from_unet_config, enhanced_wan21_vace_detection


def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
    """
    Modified version of model_config_from_unet with custom WAN2.1 Vace detection
    
    Args:
        state_dict: Model state dictionary
        unet_key_prefix: UNet key prefix (e.g., "model.", "unet.")
        use_base_if_no_match: Whether to use base model if no match found
        metadata: Optional metadata from checkpoint
    
    Returns:
        Model configuration instance or None
    """
    
    # Detect UNet configuration first
    from comfy.model_detection import detect_unet_config
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    
    if unet_config is None:
        logging.warning("Could not detect UNet configuration")
        return None
    
    # Try custom WAN2.1 Vace detection first
    logging.info("Attempting custom WAN2.1 Vace detection...")
    model_config = enhanced_wan21_vace_detection(unet_config, state_dict)
    
    # If custom detection succeeded, return it
    if model_config is not None:
        logging.info("✅ Successfully detected WAN2.1 Vace model with custom detector")
        
        # Handle scaled_fp8 if present
        scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
        if scaled_fp8_key in state_dict:
            scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
            model_config.scaled_fp8 = scaled_fp8_weight.dtype
            if model_config.scaled_fp8 == torch.float32:
                model_config.scaled_fp8 = torch.float8_e4m3fn
            if scaled_fp8_weight.nelement() == 2:
                model_config.optimizations["fp8"] = False
            else:
                model_config.optimizations["fp8"] = True
        
        return model_config
    
    # Fall back to original ComfyUI detection
    logging.info("Falling back to original ComfyUI model detection...")
    try:
        from comfy.model_detection import model_config_from_unet_config as original_detector
        model_config = original_detector(unet_config, state_dict)
        
        if model_config is None and use_base_if_no_match:
            logging.info("Using base model configuration as fallback")
            from comfy.supported_models_base import BASE
            model_config = BASE(unet_config)
        
        # Handle scaled_fp8 if present
        if model_config is not None:
            scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
            if scaled_fp8_key in state_dict:
                scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
                model_config.scaled_fp8 = scaled_fp8_weight.dtype
                if model_config.scaled_fp8 == torch.float32:
                    model_config.scaled_fp8 = torch.float8_e4m3fn
                if scaled_fp8_weight.nelement() == 2:
                    model_config.optimizations["fp8"] = False
                else:
                    model_config.optimizations["fp8"] = True
        
        return model_config
        
    except Exception as e:
        logging.error(f"Error in fallback detection: {e}")
        return None


def model_config_from_unet_with_wan_priority(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
    """
    Alternative version that prioritizes WAN models detection
    
    Args:
        state_dict: Model state dictionary
        unet_key_prefix: UNet key prefix
        use_base_if_no_match: Whether to use base model if no match found
        metadata: Optional metadata
    
    Returns:
        Model configuration instance or None
    """
    
    # Detect UNet configuration
    from comfy.model_detection import detect_unet_config
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    
    if unet_config is None:
        return None
    
    # Check if this looks like a WAN model first
    if is_potentially_wan_model(unet_config, state_dict):
        logging.info("Detected potential WAN model - using custom detection")
        
        # Try WAN2.1 Vace detection
        model_config = enhanced_wan21_vace_detection(unet_config, state_dict)
        if model_config is not None:
            return apply_scaled_fp8_settings(model_config, state_dict, unet_key_prefix)
        
        # Try other WAN variants if Vace detection failed
        # (You can add more WAN variants here)
        
    # Fall back to original detection
    logging.info("Using original ComfyUI detection")
    try:
        from comfy.model_detection import model_config_from_unet_config as original_detector
        model_config = original_detector(unet_config, state_dict)
        
        if model_config is None and use_base_if_no_match:
            from comfy.supported_models_base import BASE
            model_config = BASE(unet_config)
        
        return apply_scaled_fp8_settings(model_config, state_dict, unet_key_prefix)
        
    except Exception as e:
        logging.error(f"Error in detection: {e}")
        return None


def is_potentially_wan_model(unet_config, state_dict):
    """
    Check if the model might be a WAN model based on configuration and state dict
    
    Args:
        unet_config: Detected UNet configuration
        state_dict: Model state dictionary
    
    Returns:
        True if this might be a WAN model
    """
    
    # Check for WAN-specific configuration keys
    wan_config_keys = ["image_model", "model_type", "dim", "out_dim"]
    has_wan_config = any(key in unet_config for key in wan_config_keys)
    
    # Check for WAN-specific state dict keys
    wan_state_keys = [
        "head.modulation",
        "head.head.weight", 
        "patch_embedding.weight",
        "blocks.",
        "vace_patch_embedding.weight",  # Vace specific
        "vace_blocks.",                 # Vace specific
    ]
    has_wan_state = any(key in state_dict for key in wan_state_keys)
    
    return has_wan_config or has_wan_state


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
    
    scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
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


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Modified model_config_from_unet ===")
    
    # Create a mock state dict for testing
    mock_state_dict = {
        "head.modulation": torch.randn(1, 2000),
        "head.head.weight": torch.randn(64, 2000),
        "patch_embedding.weight": torch.randn(2000, 16, 1, 2, 2),
        "vace_patch_embedding.weight": torch.randn(2000, 16, 1, 2, 2),
        "vace_blocks.0.attn.weight": torch.randn(2000, 2000),
        "blocks.0.attn.weight": torch.randn(2000, 2000),
    }
    
    # Test the modified function
    result = model_config_from_unet(mock_state_dict, "model.", metadata=None)
    
    if result:
        print("✅ Successfully detected model configuration")
        print(f"Model type: {type(result).__name__}")
        if hasattr(result, 'unet_config'):
            print(f"UNet config keys: {list(result.unet_config.keys())}")
    else:
        print("❌ Failed to detect model configuration")
    
    print("\n🎉 Modified model_config_from_unet ready for WAN2.1 Vace!")
