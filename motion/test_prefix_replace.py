#!/usr/bin/env python3
"""
Debug the state_dict_prefix_replace function
"""

import torch
from utils import state_dict_prefix_replace

def test_prefix_replace():
    """Test the prefix replacement function"""
    print("🧪 TESTING PREFIX REPLACEMENT")
    print("="*50)
    
    # Test data
    sd = {
        "lora_unet_diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet_diffusion_model_blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet_diffusion_model_blocks_0_self_attn_q.alpha": torch.tensor(32.0),
    }
    
    print("Original keys:")
    for key in sd.keys():
        print(f"  {key}")
    
    # Test removing diffusion_model prefix
    print(f"\nTesting removal of 'diffusion_model.' prefix:")
    replacements = {"diffusion_model.": ""}
    result = state_dict_prefix_replace(sd, replacements)
    
    print("After replacement:")
    for key in result.keys():
        print(f"  {key}")
    
    # Check if it worked
    has_diffusion = any("diffusion_model." in key for key in result.keys())
    print(f"\nDiffusion_model prefix removed: {not has_diffusion}")
    
    # Test with different approach - remove from the middle
    print(f"\nTesting removal from middle of key:")
    sd2 = {
        "lora_unet_diffusion_model_blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
    }
    
    # Try to replace the specific pattern
    replacements2 = {"lora_unet_diffusion_model_": "lora_unet_"}
    result2 = state_dict_prefix_replace(sd2, replacements2)
    
    print("After middle replacement:")
    for key in result2.keys():
        print(f"  {key}")

if __name__ == "__main__":
    test_prefix_replace()
