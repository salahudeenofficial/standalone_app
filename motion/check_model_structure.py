#!/usr/bin/env python3
"""
Check the actual model structure from your real WAN model
"""

import torch
from standalone_sd import load_state_dict_guess_config
from utils import load_torch_file

def check_real_model_structure():
    """Check the actual structure of your real WAN model"""
    print("🔍 CHECKING REAL MODEL STRUCTURE")
    print("="*60)
    
    # Load your real UNet model
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    
    try:
        print("📥 Loading real UNet model...")
        unet_sd = load_torch_file(unet_path)
        
        print(f"📊 Loaded state dict with {len(unet_sd)} keys")
        
        # Check the first few keys to understand the structure
        print(f"\n🔍 FIRST 20 KEYS:")
        for i, key in enumerate(list(unet_sd.keys())[:20]):
            print(f"  {i+1:2d}. {key}")
        
        # Check for different prefix patterns
        print(f"\n🔍 PREFIX ANALYSIS:")
        prefixes = {}
        for key in unet_sd.keys():
            if '.' in key:
                prefix = key.split('.')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        print("Prefix counts:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} keys")
        
        # Check for weight keys specifically
        print(f"\n🔍 WEIGHT KEYS ANALYSIS:")
        weight_keys = [k for k in unet_sd.keys() if k.endswith('.weight')]
        print(f"Total weight keys: {len(weight_keys)}")
        
        # Show first 10 weight keys
        print("First 10 weight keys:")
        for i, key in enumerate(weight_keys[:10]):
            print(f"  {i+1:2d}. {key}")
        
        # Check for blocks pattern
        print(f"\n🔍 BLOCKS ANALYSIS:")
        block_keys = [k for k in unet_sd.keys() if 'blocks.' in k]
        print(f"Total block keys: {len(block_keys)}")
        
        if block_keys:
            print("First 10 block keys:")
            for i, key in enumerate(block_keys[:10]):
                print(f"  {i+1:2d}. {key}")
        
        # Check for attention keys specifically
        print(f"\n🔍 ATTENTION KEYS ANALYSIS:")
        attn_keys = [k for k in unet_sd.keys() if 'attn' in k and k.endswith('.weight')]
        print(f"Total attention weight keys: {len(attn_keys)}")
        
        if attn_keys:
            print("First 10 attention keys:")
            for i, key in enumerate(attn_keys[:10]):
                print(f"  {i+1:2d}. {key}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model file exists and is accessible")

if __name__ == "__main__":
    check_real_model_structure()
