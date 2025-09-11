#!/usr/bin/env python3
"""
Check the actual LoRA file structure
"""

import torch
from utils import load_torch_file
from lora import convert_lora

def check_real_lora_structure():
    """Check the actual structure of your real LoRA file"""
    print("🔍 CHECKING REAL LORA STRUCTURE")
    print("="*60)
    
    # Load your real LoRA file
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    try:
        print("📥 Loading real LoRA file...")
        lora_sd = load_torch_file(lora_path)
        
        print(f"📊 Loaded LoRA with {len(lora_sd)} keys")
        
        # Check the first few keys to understand the structure
        print(f"\n🔍 FIRST 20 KEYS:")
        for i, key in enumerate(list(lora_sd.keys())[:20]):
            print(f"  {i+1:2d}. {key}")
        
        # Check for different patterns
        print(f"\n🔍 PATTERN ANALYSIS:")
        
        # Check for lora_unet__ pattern
        double_underscore_keys = [k for k in lora_sd.keys() if "lora_unet__" in k]
        print(f"Keys with lora_unet__: {len(double_underscore_keys)}")
        if double_underscore_keys:
            print("First 5 double underscore keys:")
            for i, key in enumerate(double_underscore_keys[:5]):
                print(f"  {i+1}. {key}")
        
        # Check for lora_unet_ pattern
        single_underscore_keys = [k for k in lora_sd.keys() if "lora_unet_" in k and "lora_unet__" not in k]
        print(f"Keys with lora_unet_ (single): {len(single_underscore_keys)}")
        if single_underscore_keys:
            print("First 5 single underscore keys:")
            for i, key in enumerate(single_underscore_keys[:5]):
                print(f"  {i+1}. {key}")
        
        # Check for diffusion_model pattern
        diffusion_keys = [k for k in lora_sd.keys() if "diffusion_model" in k]
        print(f"Keys with diffusion_model: {len(diffusion_keys)}")
        if diffusion_keys:
            print("First 5 diffusion_model keys:")
            for i, key in enumerate(diffusion_keys[:5]):
                print(f"  {i+1}. {key}")
        
        # Test conversion
        print(f"\n🔧 TESTING CONVERSION:")
        converted_lora = convert_lora(lora_sd)
        
        print(f"After conversion:")
        print(f"Original keys: {len(lora_sd)}")
        print(f"Converted keys: {len(converted_lora)}")
        
        # Check if conversion changed anything
        if lora_sd.keys() == converted_lora.keys():
            print("❌ Conversion did not change any keys!")
        else:
            print("✅ Conversion changed some keys")
            print("First 5 converted keys:")
            for i, key in enumerate(list(converted_lora.keys())[:5]):
                print(f"  {i+1}. {key}")
        
        # Check for keys that should be converted
        print(f"\n🔍 CHECKING FOR CONVERSION CANDIDATES:")
        
        # Look for keys that have both lora_unet and diffusion_model
        conversion_candidates = [k for k in lora_sd.keys() if "lora_unet" in k and "diffusion_model" in k]
        print(f"Keys that should be converted: {len(conversion_candidates)}")
        if conversion_candidates:
            print("First 5 conversion candidates:")
            for i, key in enumerate(conversion_candidates[:5]):
                print(f"  {i+1}. {key}")
        
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        print("💡 Make sure the LoRA file exists and is accessible")

if __name__ == "__main__":
    check_real_lora_structure()
