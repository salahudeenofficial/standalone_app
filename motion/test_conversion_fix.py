#!/usr/bin/env python3
"""
Test the fixed LoRA conversion
"""

import torch
from lora import convert_lora, convert_lora_wan

def test_wan_conversion():
    """Test the WAN LoRA conversion fix"""
    print("🧪 TESTING WAN LORA CONVERSION FIX")
    print("="*50)
    
    # Create test LoRA with double underscores
    test_sd = {
        "lora_unet__blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__blocks_0_self_attn_q.alpha": torch.tensor(32.0),
        "lora_unet__blocks_0_cross_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__blocks_0_cross_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__blocks_0_cross_attn_q.alpha": torch.tensor(32.0),
    }
    
    print("Original keys:")
    for key in test_sd.keys():
        print(f"  {key}")
    
    # Test convert_lora_wan directly
    print("\nAfter convert_lora_wan():")
    converted_wan = convert_lora_wan(test_sd)
    for key in converted_wan.keys():
        print(f"  {key}")
    
    # Test convert_lora (which should call convert_lora_wan)
    print("\nAfter convert_lora():")
    converted = convert_lora(test_sd)
    for key in converted.keys():
        print(f"  {key}")
    
    # Check if conversion worked
    has_double_underscore = any("lora_unet__" in key for key in converted.keys())
    print(f"\n✅ Conversion successful: {not has_double_underscore}")
    
    if not has_double_underscore:
        print("🎉 Double underscores successfully removed!")
    else:
        print("❌ Double underscores still present - conversion failed")

if __name__ == "__main__":
    test_wan_conversion()
