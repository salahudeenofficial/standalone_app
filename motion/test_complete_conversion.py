#!/usr/bin/env python3
"""
Test the complete LoRA conversion fix
"""

import torch
from lora import convert_lora, convert_lora_wan

def test_complete_conversion():
    """Test the complete LoRA conversion process"""
    print("🧪 TESTING COMPLETE LORA CONVERSION")
    print("="*50)
    
    # Create LoRA with the exact pattern from your error messages
    lora_sd = {
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_self_attn_q.alpha": torch.tensor(32.0),
        
        "lora_unet__diffusion_model_blocks_0_cross_attn_v.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__diffusion_model_blocks_0_cross_attn_v.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__diffusion_model_blocks_0_cross_attn_v.alpha": torch.tensor(32.0),
    }
    
    print("Original LoRA keys:")
    for key in lora_sd.keys():
        print(f"  {key}")
    
    # Test convert_lora_wan directly
    print(f"\nAfter convert_lora_wan():")
    converted_wan = convert_lora_wan(lora_sd)
    for key in converted_wan.keys():
        print(f"  {key}")
    
    # Test convert_lora (which should call convert_lora_wan)
    print(f"\nAfter convert_lora():")
    converted = convert_lora(lora_sd)
    for key in converted.keys():
        print(f"  {key}")
    
    # Check if conversion worked correctly
    print(f"\n🔍 CHECKING CONVERSION:")
    
    # Check for double underscore removal
    has_double_underscore = any("lora_unet__" in key for key in converted.keys())
    print(f"Double underscore removed: {not has_double_underscore}")
    
    # Check for diffusion_model prefix removal
    has_diffusion_prefix = any("diffusion_model" in key for key in converted.keys())
    print(f"Diffusion_model prefix removed: {not has_diffusion_prefix}")
    
    # Check if keys match expected model format
    expected_keys = [
        "lora_unet_blocks_0_self_attn_q.lora_down.weight",
        "lora_unet_blocks_0_self_attn_q.lora_up.weight", 
        "lora_unet_blocks_0_self_attn_q.alpha",
        "lora_unet_blocks_0_cross_attn_v.lora_down.weight",
        "lora_unet_blocks_0_cross_attn_v.lora_up.weight",
        "lora_unet_blocks_0_cross_attn_v.alpha",
    ]
    
    converted_keys = set(converted.keys())
    expected_keys_set = set(expected_keys)
    
    print(f"Keys match expected format: {converted_keys == expected_keys_set}")
    
    if converted_keys == expected_keys_set:
        print("🎉 SUCCESS: Conversion working correctly!")
    else:
        print("❌ FAILED: Conversion not working correctly")
        print("Missing keys:")
        for key in expected_keys_set - converted_keys:
            print(f"  {key}")
        print("Extra keys:")
        for key in converted_keys - expected_keys_set:
            print(f"  {key}")

if __name__ == "__main__":
    test_complete_conversion()
