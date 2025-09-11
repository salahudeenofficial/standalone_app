#!/usr/bin/env python3
"""
Detailed debug script to understand the LoRA key mapping issue
"""

import torch
import logging
from lora import load_lora_for_models, convert_lora, model_lora_keys_unet, load_lora
from standalone_sd import WANModel
from standalone_model_patcher import create_model_patcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def debug_real_lora_issue():
    """Debug the real LoRA issue with actual key patterns"""
    print("🔍 DEBUGGING REAL LORA ISSUE")
    print("="*60)
    
    # Create a model that matches your WAN structure (no prefix)
    model_sd = {
        "blocks.0.self_attn.q.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.k.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.v.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.o.weight": torch.randn(1024, 1024),
        "blocks.0.cross_attn.q.weight": torch.randn(1024, 1024),
        "blocks.0.cross_attn.k.weight": torch.randn(1024, 1024),
        "blocks.0.cross_attn.v.weight": torch.randn(1024, 1024),
        "blocks.0.cross_attn.o.weight": torch.randn(1024, 1024),
    }
    
    # Create model and ModelPatcher
    unet_model = WANModel(model_sd)
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    print(f"Model keys (no prefix):")
    for key in model_sd.keys():
        print(f"  {key}")
    
    # Create LoRA that matches your real LoRA pattern
    lora_sd = {
        # These are the actual patterns from your error messages
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.alpha": torch.tensor(32.0),
        
        "lora_unet__diffusion_model_blocks_0_self_attn_k.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_self_attn_k.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_self_attn_k.alpha": torch.tensor(32.0),
        
        "lora_unet__diffusion_model_blocks_0_cross_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_cross_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_cross_attn_q.alpha": torch.tensor(32.0),
    }
    
    print(f"\nLoRA keys (with double underscore and prefix):")
    for key in lora_sd.keys():
        print(f"  {key}")
    
    # Test conversion
    print(f"\n🔧 TESTING CONVERSION:")
    converted_lora = convert_lora(lora_sd)
    print(f"After conversion:")
    for key in converted_lora.keys():
        print(f"  {key}")
    
    # Test key mapping
    print(f"\n🔧 TESTING KEY MAPPING:")
    key_map = model_lora_keys_unet(unet_patcher)
    
    print(f"Generated key mappings (showing relevant ones):")
    for lora_key, model_key in key_map.items():
        if "blocks_0" in lora_key or "diffusion_model" in lora_key:
            print(f"  {lora_key} -> {model_key}")
    
    # Test load_lora function directly
    print(f"\n🔧 TESTING LOAD_LORA FUNCTION:")
    loaded_patches = load_lora(converted_lora, key_map)
    
    print(f"Loaded patches:")
    for patch_key, patch_value in loaded_patches.items():
        print(f"  {patch_key}: {type(patch_value).__name__}")
    
    # Test add_patches
    print(f"\n🔧 TESTING ADD_PATCHES:")
    new_unet = unet_patcher.clone()
    applied_keys = new_unet.add_patches(loaded_patches, 1.0)
    
    print(f"Applied keys: {applied_keys}")
    print(f"Patches in model: {len(new_unet.patches)}")
    
    if len(new_unet.patches) > 0:
        print("Patches:")
        for key, patches in new_unet.patches.items():
            print(f"  {key}: {len(patches)} patch(es)")
    else:
        print("❌ No patches applied!")
        
        # Debug why no patches were applied
        print("\n🔍 DEBUGGING WHY NO PATCHES APPLIED:")
        model_sd_keys = set(unet_patcher.model.state_dict().keys())
        patch_keys = set(loaded_patches.keys())
        
        print(f"Model state dict keys: {len(model_sd_keys)}")
        print(f"Patch keys: {len(patch_keys)}")
        
        matching_keys = model_sd_keys.intersection(patch_keys)
        print(f"Matching keys: {len(matching_keys)}")
        
        if len(matching_keys) == 0:
            print("❌ No matching keys found!")
            print("Model keys:")
            for key in list(model_sd_keys)[:5]:
                print(f"  {key}")
            print("Patch keys:")
            for key in list(patch_keys)[:5]:
                print(f"  {key}")
        else:
            print("✅ Found matching keys:")
            for key in matching_keys:
                print(f"  {key}")

if __name__ == "__main__":
    debug_real_lora_issue()
