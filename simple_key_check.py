#!/usr/bin/env python3
"""
Simple script to check what keys are actually in the UNET state dict.
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def simple_key_check():
    """Simply check what keys exist in the state dict"""
    print("🔍 SIMPLE KEY CHECK")
    print("="*30)
    
    try:
        import comfy.utils
        
        # Load the UNET file
        unet_path = Path(__file__).parent / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        
        if not unet_path.exists():
            print(f"❌ UNET file not found: {unet_path}")
            return
            
        print(f"📁 Loading: {unet_path}")
        
        # Load state dict
        sd = comfy.utils.load_torch_file(str(unet_path))
        print(f"✅ Loaded state dict with {len(sd)} keys")
        
        # Show all keys (first 50)
        print("\n📋 First 50 keys:")
        all_keys = list(sd.keys())
        for i, key in enumerate(all_keys[:50]):
            print(f"  {i+1:2d}. {key}")
            
        # Show last 10 keys
        if len(all_keys) > 50:
            print(f"\n📋 Last 10 keys:")
            for i, key in enumerate(all_keys[-10:]):
                print(f"  {len(all_keys)-10+i+1:2d}. {key}")
                
        # Check for any keys containing specific patterns
        print("\n🔍 Looking for specific patterns:")
        patterns = ['head', 'patch', 'block', 'embed', 'vace', 'modulation']
        
        for pattern in patterns:
            matching = [k for k in all_keys if pattern.lower() in k.lower()]
            if matching:
                print(f"  '{pattern}': {len(matching)} keys found")
                for key in matching[:5]:  # Show first 5
                    print(f"    {key}")
            else:
                print(f"  '{pattern}': 0 keys found")
                
        # Check if there are any keys at all
        print(f"\n📊 Summary:")
        print(f"  Total keys: {len(all_keys)}")
        print(f"  First key: {all_keys[0] if all_keys else 'None'}")
        print(f"  Last key: {all_keys[-1] if all_keys else 'None'}")
        
        # Check if keys are strings
        if all_keys:
            first_key = all_keys[0]
            print(f"  First key type: {type(first_key)}")
            print(f"  First key value: {repr(first_key)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_key_check() 