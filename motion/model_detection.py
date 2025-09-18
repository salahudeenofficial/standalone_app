# Standalone Model Detection Implementation
# Based on ComfyUI's comfy.model_detection but with all dependencies resolved

import math
import json
import logging
import torch
from typing import Dict, Any, Optional, List

def count_blocks(state_dict_keys: List[str], prefix_string: str) -> int:
    """Count the number of blocks with the given prefix"""
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def detect_unet_config(state_dict: Dict[str, torch.Tensor], key_prefix: str = "", metadata: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """
    Detect UNet configuration from state dict
    Based on ComfyUI's detect_unet_config but standalone
    """
    state_dict_keys = list(state_dict.keys())
    
    # Helper function to check if a key exists with or without prefix
    def has_key(key_name):
        # Try with prefix first
        prefixed_key = '{}{}'.format(key_prefix, key_name)
        if prefixed_key in state_dict_keys:
            return prefixed_key
        # Try without prefix (for mixed key formats)
        if key_name in state_dict_keys:
            return key_name
        return None
    
    # Check for WAN 2.1 models (using ComfyUI's exact detection logic)
    head_modulation_key = has_key('head.modulation')
    if head_modulation_key:  # Wan 2.1
        dit_config = {}
        dit_config["image_model"] = "wan2.1"
        dim = state_dict[head_modulation_key].shape[-1]
        
        head_weight_key = has_key('head.head.weight')
        if not head_weight_key:
            return None
        out_dim = state_dict[head_weight_key].shape[0] // 4
        
        dit_config["dim"] = dim
        dit_config["out_dim"] = out_dim
        dit_config["num_heads"] = dim // 128
        
        ffn_weight_key = has_key('blocks.0.ffn.0.weight')
        if not ffn_weight_key:
            return None
        dit_config["ffn_dim"] = state_dict[ffn_weight_key].shape[0]
        
        # Count blocks - look for any key that starts with blocks.X.
        block_count = 0
        while True:
            found_block = False
            for key in state_dict_keys:
                if key.startswith(f'blocks.{block_count}.'):
                    found_block = True
                    break
            if found_block:
                block_count += 1
            else:
                break
        dit_config["num_layers"] = block_count
        dit_config["patch_size"] = (1, 2, 2)
        
        # Get freq_dim from time_embed.0.weight if available
        time_embed_key = has_key('time_embed.0.weight')
        if time_embed_key:
            dit_config["freq_dim"] = state_dict[time_embed_key].shape[1]
        else:
            dit_config["freq_dim"] = 256  # Default
        dit_config["window_size"] = (-1, -1)
        dit_config["qk_norm"] = True
        dit_config["cross_attn_norm"] = True
        dit_config["eps"] = 1e-6
        
        patch_embed_key = has_key('patch_embedding.weight')
        if not patch_embed_key:
            return None
        dit_config["in_dim"] = state_dict[patch_embed_key].shape[1]
        
        # Determine model type based on specific keys (ComfyUI's exact logic)
        vace_patch_key = has_key('vace_patch_embedding.weight')
        if vace_patch_key:
            dit_config["model_type"] = "vace"
            dit_config["vace_in_dim"] = state_dict[vace_patch_key].shape[1]
            dit_config["vace_layers"] = count_blocks(state_dict_keys, '{}vace_blocks.'.format(key_prefix) + '{}.')
        elif has_key('control_adapter.conv.weight'):
            if has_key('img_emb.proj.0.bias'):
                dit_config["model_type"] = "camera"
            else:
                dit_config["model_type"] = "camera_2.2"
        else:
            if has_key('img_emb.proj.0.bias'):
                dit_config["model_type"] = "i2v"
            else:
                dit_config["model_type"] = "t2v"
        
        # Check for additional features (ComfyUI's exact logic)
        flf_pos_key = has_key('img_emb.emb_pos')
        if flf_pos_key:
            dit_config["flf_pos_embed_token_number"] = state_dict[flf_pos_key].shape[1]

        ref_conv_key = has_key('ref_conv.weight')
        if ref_conv_key:
            dit_config["in_dim_ref_conv"] = state_dict[ref_conv_key].shape[1]

        return dit_config
    
    # Check for other model types (simplified versions)
    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys:  # MMDiT model
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        patch_size = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (patch_size * patch_size)
        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        unet_config["input_size"] = None
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]
        return unet_config
    
    # Check for Stable Cascade
    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys:  # stable cascade
        unet_config = {}
        text_mapper_name = '{}clip_txt_mapper.weight'.format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'c'
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536:  # stage c lite
                unet_config['c_cond'] = 1536
                unet_config['c_hidden'] = [1536, 1536]
                unet_config['nhead'] = [24, 24]
                unet_config['blocks'] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048:  # stage c full
                unet_config['c_cond'] = 2048
        elif '{}clip_mapper.weight'.format(key_prefix) in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'b'
            w = state_dict['{}down_blocks.1.0.channelwise.0.weight'.format(key_prefix)]
            if w.shape[-1] == 640:
                unet_config['c_hidden'] = [320, 640, 1280, 1280]
                unet_config['nhead'] = [-1, -1, 20, 20]
                unet_config['blocks'] = [[2, 6, 28, 6], [6, 28, 6, 2]]
        return unet_config
    
    # If no known model type detected, return None
    return None

def model_config_from_unet_config(unet_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert UNet config to model config
    Based on ComfyUI's model_config_from_unet_config but standalone
    """
    if unet_config is None:
        return None
    
    # Handle WAN 2.1 models
    if unet_config.get("image_model") == "wan2.1":
        model_config = {
            "image_model": "wan2.1",
            "model_type": unet_config.get("model_type", "t2v"),
            "patch_size": unet_config.get("patch_size", (1, 2, 2)),
            "text_len": 512,
            "in_dim": unet_config.get("in_dim", 16),
            "dim": unet_config.get("dim", 2048),
            "ffn_dim": unet_config.get("ffn_dim", 8192),
            "freq_dim": unet_config.get("freq_dim", 256),
            "text_dim": 4096,
            "out_dim": unet_config.get("out_dim", 16),
            "num_heads": unet_config.get("num_heads", 16),
            "num_layers": unet_config.get("num_layers", 32),
            "window_size": unet_config.get("window_size", (-1, -1)),
            "qk_norm": unet_config.get("qk_norm", True),
            "cross_attn_norm": unet_config.get("cross_attn_norm", True),
            "eps": unet_config.get("eps", 1e-6),
        }
        
        # Add VACE specific config
        if unet_config.get("model_type") == "vace":
            model_config["vace_layers"] = unet_config.get("vace_layers")
            model_config["vace_in_dim"] = unet_config.get("vace_in_dim")
        
        # Add other optional configs
        if "flf_pos_embed_token_number" in unet_config:
            model_config["flf_pos_embed_token_number"] = unet_config["flf_pos_embed_token_number"]
        if "in_dim_ref_conv" in unet_config:
            model_config["in_dim_ref_conv"] = unet_config["in_dim_ref_conv"]
        
        return model_config
    
    # Handle other model types (simplified)
    if unet_config.get("image_model") == "mmdit":
        return {
            "image_model": "mmdit",
            "in_channels": unet_config.get("in_channels", 4),
            "patch_size": unet_config.get("patch_size", 2),
            "out_channels": unet_config.get("out_channels", 4),
            "depth": unet_config.get("depth", 24),
            "input_size": unet_config.get("input_size"),
            "adm_in_channels": unet_config.get("adm_in_channels"),
        }
    
    if unet_config.get("stable_cascade_stage") == "c":
        return {
            "stable_cascade_stage": "c",
            "c_cond": unet_config.get("c_cond", 1536),
            "c_hidden": unet_config.get("c_hidden", [1536, 1536]),
            "nhead": unet_config.get("nhead", [24, 24]),
            "blocks": unet_config.get("blocks", [[4, 12], [12, 4]]),
        }
    
    if unet_config.get("stable_cascade_stage") == "b":
        return {
            "stable_cascade_stage": "b",
            "c_hidden": unet_config.get("c_hidden", [320, 640, 1280, 1280]),
            "nhead": unet_config.get("nhead", [-1, -1, 20, 20]),
            "blocks": unet_config.get("blocks", [[2, 6, 28, 6], [6, 28, 6, 2]]),
        }
    
    return None

def detect_model_type_from_state_dict(state_dict: Dict[str, torch.Tensor], key_prefix: str = "") -> str:
    """
    Detect the model type from state dict
    Returns: 'wan21_vace', 'wan21_t2v', 'wan21_i2v', 'wan21_camera', 'unknown'
    """
    state_dict_keys = list(state_dict.keys())
    
    # Check for WAN 2.1 models
    if '{}head.modulation'.format(key_prefix) in state_dict_keys:
        if '{}vace_patch_embedding.weight'.format(key_prefix) in state_dict_keys:
            return 'wan21_vace'
        elif '{}control_adapter.conv.weight'.format(key_prefix) in state_dict_keys:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                return 'wan21_camera'
            else:
                return 'wan21_camera_22'
        else:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                return 'wan21_i2v'
            else:
                return 'wan21_t2v'
    
    # Check for other model types
    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys:
        return 'mmdit'
    
    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys:
        return 'stable_cascade'
    
    return 'unknown'

def get_model_class_for_type(model_type: str):
    """
    Get the appropriate model class for the detected type
    """
    from wan_model import WanModel, VaceWanModel, CameraWanModel
    
    if model_type == 'wan21_vace':
        return VaceWanModel
    elif model_type == 'wan21_camera':
        return CameraWanModel
    elif model_type in ['wan21_t2v', 'wan21_i2v']:
        return WanModel
    else:
        # Fallback to base WanModel
        return WanModel

def create_model_from_config(state_dict: Dict[str, torch.Tensor], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    """
    Create a model instance from configuration
    Following ComfyUI's pattern: detect_unet_config -> model_config_from_unet_config -> get_model
    """
    try:
        # Debug: Check what we're receiving
        logging.info(f"create_model_from_config received: type={type(state_dict)}")
        if isinstance(state_dict, dict):
            logging.info(f"State dict keys: {len(state_dict)}")
        else:
            logging.error(f"Expected dict, got {type(state_dict)}: {state_dict}")
            return None
        
        # First, detect the UNet config from state dict
        unet_config = detect_unet_config(state_dict)
        if unet_config is None:
            logging.error("Failed to detect UNet config from state dict")
            return None
        
        # Create model config object (following ComfyUI pattern)
        model_config_obj = model_config_from_unet_config(unet_config)
        if model_config_obj is None:
            logging.error("Failed to create model config object")
            return None
        
        # Auto-detect dtype from state dict if not provided
        if dtype is None:
            # Get dtype from first tensor in state dict
            first_tensor = next(iter(state_dict.values()))
            if isinstance(first_tensor, torch.Tensor):
                dtype = first_tensor.dtype
                logging.info(f"Auto-detected dtype from state dict: {dtype}")
        
        # Default to float32 if still None
        if dtype is None:
            dtype = torch.float32
        
        # Create model using ComfyUI-style approach
        # For now, we'll create a simple dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self, config, device=None, dtype=None):
                super().__init__()
                # Create a simple linear layer for testing
                self.linear = torch.nn.Linear(10, 10)
                if dtype is not None:
                    self.linear = self.linear.to(dtype=dtype)
                if device is not None:
                    self.linear = self.linear.to(device=device)
            
            def forward(self, x):
                return self.linear(x)
        
        model = DummyModel(model_config_obj, device=device, dtype=dtype)
        
        # Load dummy state dict for testing
        dummy_state_dict = {
            'linear.weight': torch.randn(10, 10, dtype=dtype),
            'linear.bias': torch.randn(10, dtype=dtype)
        }
        model.load_state_dict(dummy_state_dict)
        logging.info("Loaded dummy state dict for testing")
        
        return model
        
    except Exception as e:
        logging.error(f"Failed to create model from config: {e}")
        import traceback
        traceback.print_exc()
        return None
