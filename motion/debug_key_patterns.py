#!/usr/bin/env python3
"""
Debug the exact key patterns
"""

import torch
from utils import state_dict_prefix_replace

def debug_key_patterns():
    """Debug the exact key patterns"""
    print("🔍 DEBUGGING KEY PATTERNS")
    print("="*50)
    
    # Test data
    sd = {
        "lora_unet_diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
    }
    
    key = list(sd.keys())[0]
    print(f"Key: {key}")
    
    # Test different patterns
    patterns_to_test = [
        "lora_unet_diffusion_model_",
        "diffusion_model_",
        "diffusion_model.",
        "lora_unet_diffusion_model",
    ]
    
    for pattern in patterns_to_test:
        if key.startswith(pattern):
            print(f"✅ Key starts with: '{pattern}'")
            new_key = pattern + key[len(pattern):]
            print(f"   Would become: {new_key}")
        else:
            print(f"❌ Key does NOT start with: '{pattern}'")
    
    # Test the actual replacement
    print(f"\n🔧 TESTING ACTUAL REPLACEMENT:")
    replacements = {"lora_unet_diffusion_model_": "lora_unet_"}
    result = state_dict_prefix_replace(sd, replacements)
    
    print(f"Before: {list(sd.keys())[0]}")
    print(f"After:  {list(result.keys())[0]}")
    
    # Test manual replacement
    print(f"\n🔧 TESTING MANUAL REPLACEMENT:")
    manual_key = key.replace("lora_unet_diffusion_model_", "lora_unet_")
    print(f"Manual replacement: {manual_key}")

if __name__ == "__main__":
    debug_key_patterns()
