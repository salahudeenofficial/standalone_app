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
    
    # Check for WAN 2.1 models
    if '{}head.modulation'.format(key_prefix) in state_dict_keys:  # Wan 2.1
        dit_config = {}
        dit_config["image_model"] = "wan2.1"
        dim = state_dict['{}head.modulation'.format(key_prefix)].shape[-1]
        out_dim = state_dict['{}head.head.weight'.format(key_prefix)].shape[0] // 4
        dit_config["dim"] = dim
        dit_config["out_dim"] = out_dim
        dit_config["num_heads"] = dim // 128
        dit_config["ffn_dim"] = state_dict['{}blocks.0.ffn.0.weight'.format(key_prefix)].shape[0]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        dit_config["patch_size"] = (1, 2, 2)
        dit_config["freq_dim"] = 256
        dit_config["window_size"] = (-1, -1)
        dit_config["qk_norm"] = True
        dit_config["cross_attn_norm"] = True
        dit_config["eps"] = 1e-6
        dit_config["in_dim"] = state_dict['{}patch_embedding.weight'.format(key_prefix)].shape[1]
        
        # Determine model type based on specific keys
        if '{}vace_patch_embedding.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "vace"
            dit_config["vace_in_dim"] = state_dict['{}vace_patch_embedding.weight'.format(key_prefix)].shape[1]
            dit_config["vace_layers"] = count_blocks(state_dict_keys, '{}vace_blocks.'.format(key_prefix) + '{}.')
        elif '{}control_adapter.conv.weight'.format(key_prefix) in state_dict_keys:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                dit_config["model_type"] = "camera"
            else:
                dit_config["model_type"] = "camera_2.2"
        else:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                dit_config["model_type"] = "i2v"
            else:
                dit_config["model_type"] = "t2v"
        
        # Check for additional features
        flf_weight = state_dict.get('{}img_emb.emb_pos'.format(key_prefix))
        if flf_weight is not None:
            dit_config["flf_pos_embed_token_number"] = flf_weight.shape[1]

        ref_conv_weight = state_dict.get('{}ref_conv.weight'.format(key_prefix))
        if ref_conv_weight is not None:
            dit_config["in_dim_ref_conv"] = ref_conv_weight.shape[1]

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

def create_model_from_config(model_config: Dict[str, Any], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    """
    Create a model instance from configuration
    """
    model_type = model_config.get("model_type", "t2v")
    image_model = model_config.get("image_model", "wan2.1")
    
    # Determine the appropriate model class
    if image_model == "wan2.1":
        if model_type == "vace":
            from wan_model import VaceWanModel
            return VaceWanModel(**model_config, device=device, dtype=dtype)
        elif model_type == "camera":
            from wan_model import CameraWanModel
            return CameraWanModel(**model_config, device=device, dtype=dtype)
        else:
            from wan_model import WanModel
            return WanModel(**model_config, device=device, dtype=dtype)
    
    # Fallback to base WanModel
    from wan_model import WanModel
    return WanModel(**model_config, device=device, dtype=dtype)
