#!/usr/bin/env python3
"""
Test the prefix mismatch fix
"""

import torch
from lora import load_lora_for_models, convert_lora, model_lora_keys_unet
from standalone_sd import WANModel
from standalone_model_patcher import create_model_patcher

def test_prefix_mismatch():
    """Test LoRA with diffusion_model prefix on model without prefix"""
    print("🧪 TESTING PREFIX MISMATCH FIX")
    print("="*50)
    
    # Create model WITHOUT prefix (like your WAN model)
    model_sd = {
        "blocks.0.self_attn.q.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.k.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.v.weight": torch.randn(1024, 1024),
    }
    
    # Create model and ModelPatcher
    unet_model = WANModel(model_sd)
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    print(f"Model keys (no prefix): {list(model_sd.keys())}")
    
    # Create LoRA WITH diffusion_model prefix (like your real LoRA)
    lora_sd = {
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.alpha": torch.tensor(32.0),
    }
    
    print(f"LoRA keys (with prefix): {list(lora_sd.keys())}")
    
    # Test conversion
    converted_lora = convert_lora(lora_sd)
    print(f"\nAfter conversion:")
    for key in converted_lora.keys():
        print(f"  {key}")
    
    # Test key mapping
    key_map = model_lora_keys_unet(unet_patcher)
    print(f"\nKey mappings:")
    for lora_key, model_key in key_map.items():
        if "blocks_0_self_attn_q" in lora_key:
            print(f"  {lora_key} -> {model_key}")
    
    # Apply LoRA
    new_unet, new_clip = load_lora_for_models(unet_patcher, None, lora_sd)
    
    if new_unet is not None:
        patches_count = len(new_unet.patches)
        print(f"\n✅ Patches applied: {patches_count}")
        
        if patches_count > 0:
            print("🎉 SUCCESS: Prefix mismatch resolved!")
            print("Patches:")
            for key, patches in new_unet.patches.items():
                print(f"  {key}: {len(patches)} patch(es)")
        else:
            print("❌ FAILED: Still no patches applied")
            print("Available mappings:")
            for k, v in key_map.items():
                if "diffusion_model" in k:
                    print(f"  {k} -> {v}")
    else:
        print("❌ FAILED: Model is None")

if __name__ == "__main__":
    test_prefix_mismatch()
