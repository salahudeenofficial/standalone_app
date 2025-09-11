#!/usr/bin/env python3
"""
Test the fix for direct diffusion_model prefix removal
"""

import torch
from lora import convert_lora, convert_lora_wan

def test_direct_diffusion_fix():
    """Test the fix for LoRA with direct diffusion_model prefix"""
    print("🧪 TESTING DIRECT DIFFUSION MODEL PREFIX FIX")
    print("="*60)
    
    # Create LoRA with the exact pattern from your real LoRA
    lora_sd = {
        "diffusion_model.blocks.0.cross_attn.k.lora_down.weight": torch.randn(32, 1024),
        "diffusion_model.blocks.0.cross_attn.k.lora_up.weight": torch.randn(1024, 32),
        "diffusion_model.blocks.0.cross_attn.k.diff_b": torch.randn(1024),
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.randn(32, 1024),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.randn(1024, 32),
        "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(32.0),
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
    
    # Check for diffusion_model prefix removal
    has_diffusion_prefix = any("diffusion_model." in key for key in converted.keys())
    print(f"Diffusion_model prefix removed: {not has_diffusion_prefix}")
    
    # Check if keys match expected model format
    expected_keys = [
        "blocks.0.cross_attn.k.lora_down.weight",
        "blocks.0.cross_attn.k.lora_up.weight",
        "blocks.0.cross_attn.k.diff_b",
        "blocks.0.self_attn.q.lora_down.weight",
        "blocks.0.self_attn.q.lora_up.weight",
        "blocks.0.self_attn.q.alpha",
    ]
    
    converted_keys = set(converted.keys())
    expected_keys_set = set(expected_keys)
    
    print(f"Keys match expected format: {converted_keys == expected_keys_set}")
    
    if converted_keys == expected_keys_set:
        print("🎉 SUCCESS: Direct diffusion_model prefix removal working!")
    else:
        print("❌ FAILED: Conversion not working correctly")
        print("Missing keys:")
        for key in expected_keys_set - converted_keys:
            print(f"  {key}")
        print("Extra keys:")
        for key in converted_keys - expected_keys_set:
            print(f"  {key}")

if __name__ == "__main__":
    test_direct_diffusion_fix()
