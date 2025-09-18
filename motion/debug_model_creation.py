#!/usr/bin/env python3
"""
Debug script to isolate the model creation issue
"""

import torch
import logging
from safetensors import safe_open

# Set up logging
logging.basicConfig(level=logging.INFO)

def debug_model_creation():
    """Debug the model creation process step by step"""
    
    print("🔍 Debugging Model Creation Process")
    print("=" * 50)
    
    # Load state dict
    model_path = "./models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    print(f"📁 Loading state dict from: {model_path}")
    
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    
    print(f"✅ State dict loaded: {len(state_dict)} keys")
    print(f"📊 Sample keys: {list(state_dict.keys())[:5]}")
    
    # Test detect_unet_config
    print(f"\n🔍 Testing detect_unet_config...")
    try:
        from model_detection import detect_unet_config
        unet_config = detect_unet_config(state_dict)
        print(f"✅ UNet config detected: {type(unet_config)}")
        if isinstance(unet_config, dict):
            print(f"📊 Config keys: {list(unet_config.keys())}")
        else:
            print(f"❌ UNet config is not a dict: {unet_config}")
            return
    except Exception as e:
        print(f"❌ Error in detect_unet_config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model_config_from_unet_config
    print(f"\n🔍 Testing model_config_from_unet_config...")
    try:
        from model_detection import model_config_from_unet_config
        model_config = model_config_from_unet_config(unet_config)
        print(f"✅ Model config created: {type(model_config)}")
        if isinstance(model_config, dict):
            print(f"📊 Model config keys: {list(model_config.keys())}")
        else:
            print(f"❌ Model config is not a dict: {model_config}")
            return
    except Exception as e:
        print(f"❌ Error in model_config_from_unet_config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test create_model_from_config
    print(f"\n🔍 Testing create_model_from_config...")
    try:
        from model_detection import create_model_from_config
        model = create_model_from_config(
            model_config=model_config,
            device=None,
            dtype=None,
            state_dict=state_dict
        )
        print(f"✅ Model created: {type(model)}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"❌ Error in create_model_from_config: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_creation()
