#!/usr/bin/env python3
"""
Test script to debug WAN model detection and see why UNET is not being detected as WAN.
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_wan_model_detection():
    """Test WAN model detection to understand why UNET is not compatible"""
    print("🧪 TESTING WAN MODEL DETECTION")
    print("="*50)
    
    try:
        # Import ComfyUI modules
        import comfy.sd
        import comfy.model_management
        import comfy.utils
        import comfy.model_detection
        
        print("✅ ComfyUI modules imported successfully")
        
        # Check the UNET model
        unet_path = Path(__file__).parent / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        
        if not unet_path.exists():
            print(f"❌ UNET file not found: {unet_path}")
            return
            
        print(f"📁 Testing UNET: {unet_path}")
        
        # Load the state dict
        print("🔄 Loading UNET state dict...")
        sd = comfy.utils.load_torch_file(str(unet_path))
        print(f"✅ State dict loaded with {len(sd)} keys")
        
        # Check for WAN-specific keys
        print("\n🔍 Checking for WAN-specific keys...")
        wan_keys = [k for k in sd.keys() if 'wan' in k.lower() or 'head.modulation' in k or 'patch_embedding' in k]
        print(f"WAN-related keys found: {len(wan_keys)}")
        for key in wan_keys[:10]:  # Show first 10
            print(f"  {key}")
            
        # Check for specific WAN architecture keys
        print("\n🔍 Checking for specific WAN architecture...")
        key_indicators = {
            'head.modulation': 'WAN modulation head',
            'patch_embedding.weight': 'WAN patch embedding',
            'blocks.0.ffn.0.weight': 'WAN transformer blocks',
            'img_emb.proj.0.bias': 'WAN image embedding',
            'vace_patch_embedding.weight': 'WAN VACE variant'
        }
        
        for key, description in key_indicators.items():
            if key in sd:
                print(f"✅ {description}: {key}")
                if 'weight' in key:
                    shape = sd[key].shape
                    print(f"   Shape: {shape}")
            else:
                print(f"❌ {description}: {key} - NOT FOUND")
        
        # Try model detection
        print("\n🔍 Testing model detection...")
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        print(f"Detected prefix: {diffusion_model_prefix}")
        
        # Try to detect model config
        model_config = comfy.model_detection.model_config_from_unet(sd, diffusion_model_prefix)
        if model_config:
            print(f"✅ Model config detected: {type(model_config).__name__}")
            print(f"   Model type: {getattr(model_config, 'model_type', 'Unknown')}")
            print(f"   Image model: {getattr(model_config, 'image_model', 'Unknown')}")
        else:
            print("❌ No model config detected!")
            
        # Try loading with state dict directly
        print("\n🔄 Testing direct state dict loading...")
        try:
            model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
            if model:
                print("✅ Model loaded successfully with load_diffusion_model_state_dict!")
                print(f"   Model type: {type(model)}")
                print(f"   Model device: {model.model.device if hasattr(model, 'model') else 'Unknown'}")
            else:
                print("❌ load_diffusion_model_state_dict returned None")
        except Exception as e:
            print(f"❌ Error loading with state dict: {e}")
            
        # Try the regular loading function
        print("\n🔄 Testing regular load_diffusion_model...")
        try:
            model = comfy.sd.load_diffusion_model(str(unet_path))
            if model:
                print("✅ Model loaded successfully with load_diffusion_model!")
                print(f"   Model type: {type(model)}")
                print(f"   Model device: {model.model.device if hasattr(model, 'model') else 'Unknown'}")
            else:
                print("❌ load_diffusion_model returned None")
        except Exception as e:
            print(f"❌ Error loading with regular function: {e}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wan_model_detection() 