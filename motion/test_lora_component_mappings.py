#!/usr/bin/env python3
"""
Test if the new LoRA component mappings are being created
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import model_lora_keys_unet
from utils import load_torch_file

def test_lora_component_mappings():
    """Test if LoRA component mappings are being created"""
    print("🔍 TESTING LORA COMPONENT MAPPINGS")
    print("="*50)
    
    # Load UNet model
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    unet_sd = load_torch_file(unet_path)
    
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    # Generate key mappings
    key_map = model_lora_keys_unet(unet_model, {})
    
    print(f"📊 Total key mappings: {len(key_map)}")
    
    # Check for specific LoRA component mappings
    test_keys = [
        "blocks.0.cross_attn.k.lora_down.weight",
        "blocks.0.cross_attn.k.lora_up.weight", 
        "blocks.0.cross_attn.k.alpha",
        "blocks.0.cross_attn.k.diff",
        "blocks.0.cross_attn.k.diff_b"
    ]
    
    print(f"\n🔍 CHECKING SPECIFIC LORA COMPONENT MAPPINGS:")
    found_mappings = 0
    for test_key in test_keys:
        if test_key in key_map:
            found_mappings += 1
            print(f"  ✅ {test_key} -> {key_map[test_key]}")
        else:
            print(f"  ❌ {test_key} -> NOT FOUND")
    
    print(f"\n📊 Found {found_mappings}/{len(test_keys)} expected mappings")
    
    # Check what mappings we do have for blocks.0.cross_attn.k
    print(f"\n🔍 ALL MAPPINGS FOR blocks.0.cross_attn.k:")
    related_mappings = []
    for mapping_key, model_key in key_map.items():
        if "blocks.0.cross_attn.k" in mapping_key and "blocks.0.cross_attn.k" in model_key:
            related_mappings.append((mapping_key, model_key))
    
    print(f"  Found {len(related_mappings)} related mappings:")
    for mapping_key, model_key in related_mappings:
        print(f"    {mapping_key} -> {model_key}")
    
    # Check if the model actually has the expected keys
    print(f"\n🔍 MODEL KEY CHECK:")
    expected_weight_key = "blocks.0.cross_attn.k.weight"
    expected_bias_key = "blocks.0.cross_attn.k.bias"
    
    if expected_weight_key in unet_sd:
        print(f"  ✅ {expected_weight_key} exists in model")
    else:
        print(f"  ❌ {expected_weight_key} missing from model")
    
    if expected_bias_key in unet_sd:
        print(f"  ✅ {expected_bias_key} exists in model")
    else:
        print(f"  ❌ {expected_bias_key} missing from model")

if __name__ == "__main__":
    test_lora_component_mappings()
