import torch
import logging
from wan_vae_components.model_management import get_torch_device, unet_offload_device, unet_dtype, unet_manual_cast, unet_inital_load_device, load_models_gpu
from standalone_model_patcher import create_model_patcher
from utils import calculate_parameters, weight_dtype, state_dict_prefix_replace, load_torch_file
import torch.nn as nn

def detect_unet_config(state_dict, key_prefix, metadata=None):
    """
    Enhanced UNet config detector for Wan2.1 models.
    Handles both I2V/VACE variants and cross-attention variants.
    """
    state_dict_keys = list(state_dict.keys())

    # --- Basic Wan2.1 check ---
    if f'{key_prefix}head.modulation' not in state_dict_keys:
        return None

    # Get dimensions from actual model
    head_modulation = state_dict[f'{key_prefix}head.modulation']
    head_weight = state_dict[f'{key_prefix}head.head.weight']
    
    # Handle different modulation shapes
    if len(head_modulation.shape) == 3:  # [1, 2, dim] format
        dim = head_modulation.shape[-1]
    else:  # [dim] format
        dim = head_modulation.shape[-1]
    
    out_dim = head_weight.shape[0] // 4
    ffn_dim = state_dict[f'{key_prefix}blocks.0.ffn.0.weight'].shape[0]
    num_layers = sum(
        1 for k in state_dict_keys
        if k.startswith(f'{key_prefix}blocks.') and k.endswith('.ffn.0.weight')
    )
    
    # Get patch embedding info
    patch_embedding = state_dict[f'{key_prefix}patch_embedding.weight']
    in_dim = patch_embedding.shape[1]
    patch_size = patch_embedding.shape[2:]  # [1, 2, 2] or [1, 1, 1]

    # --- Build base config ---
    dit_config = {
        "image_model": "wan2.1",
        "dim": dim,
        "out_dim": out_dim,
        "num_heads": dim // 128,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "patch_size": patch_size,
        "in_dim": in_dim,
    }

    # --- Detect I2V ---
    if f'{key_prefix}img_emb.proj.0.bias' in state_dict_keys:
        dit_config["model_type"] = "i2v"
        if f'{key_prefix}img_emb.emb_pos' in state_dict_keys:
            dit_config["flf_pos_embed_token_number"] = state_dict[f'{key_prefix}img_emb.emb_pos'].shape[1]
        if f'{key_prefix}ref_conv.weight' in state_dict_keys:
            dit_config["in_dim_ref_conv"] = state_dict[f'{key_prefix}ref_conv.weight'].shape[1]
        return dit_config

    # --- Detect VACE (original) ---
    if f'{key_prefix}camera_cond_emb.proj.0.bias' in state_dict_keys:
        dit_config["model_type"] = "vace"
        dit_config["patch_size"] = (1, 1, 1)  # VACE default
        if f'{key_prefix}img_emb.emb_pos' in state_dict_keys:
            dit_config["flf_pos_embed_token_number"] = state_dict[f'{key_prefix}img_emb.emb_pos'].shape[1]
        return dit_config

    # --- Detect Cross-Attention Variant (new) ---
    if f'{key_prefix}blocks.0.cross_attn.q.weight' in state_dict_keys:
        dit_config["model_type"] = "cross_attn"  # New variant with cross-attention
        dit_config["has_cross_attention"] = True
        dit_config["has_self_attention"] = f'{key_prefix}blocks.0.self_attn.q.weight' in state_dict_keys
        return dit_config

    # --- Fallback: Generic WAN2.1 ---
    dit_config["model_type"] = "generic"
    return dit_config

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

def load_state_dict_guess_config(sd, output_vae=True, output_clip=True, output_clipvision=False, 
                                embedding_directory=None, output_model=True, model_options={}, 
                                te_model_options={}, metadata=None):
    """
    Load state dict and guess configuration - Enhanced for WAN variants
    """
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None

    # Model detection
    diffusion_model_prefix = unet_prefix_from_state_dict(sd)
    parameters = calculate_parameters(sd, diffusion_model_prefix)
    weight_dtype_val = weight_dtype(sd, diffusion_model_prefix)
    load_device = get_torch_device()

    unet_config = detect_unet_config(sd, diffusion_model_prefix, metadata=metadata)
    if unet_config is None:
        logging.warning("Warning, This is not a checkpoint file, trying to load it as a diffusion model only.")
        raise ValueError("Unsupported model type. Only WAN2.1 variants are supported.")

    logging.info(f"Detected WAN model type: {unet_config['model_type']}")
    
    # For now, just return a dummy model patcher for testing
    if output_model:
        # Create a dummy model patcher
        class DummyModel:
            def __init__(self):
                self.device = torch.device("cpu")
                self.patches = {}
                self.uuid = "dummy-unet-uuid"
            
            def state_dict(self):
                return sd
            
            def load_state_dict(self, state_dict, strict=False):
                pass
        
        dummy_model = DummyModel()
        model_patcher = create_model_patcher(dummy_model, load_device=load_device, offload_device=unet_offload_device())

    return (model_patcher, clip, vae, clipvision)

if __name__ == "__main__":
    print("Simplified standalone_sd.py with WAN variant support")
    print("✅ Supports I2V, VACE, and Cross-Attention variants")
    print("✅ Enhanced model detection")
    print("✅ Simplified implementation")
