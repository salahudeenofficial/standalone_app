#!/usr/bin/env python3
"""
Debug script to compare model detection between ComfyUI and standalone app.
This will help us understand why the same function works differently.
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def debug_model_detection_step_by_step():
    """Debug model detection step by step to find the root cause"""
    print("🔍 DEBUGGING MODEL DETECTION STEP BY STEP")
    print("="*60)
    
    try:
        # Import all necessary modules
        print("📦 Importing modules...")
        import comfy.sd
        import comfy.model_management
        import comfy.utils
        import comfy.model_detection
        import comfy.supported_models
        import comfy.model_base
        
        print("✅ All modules imported successfully")
        
        # Test the UNET file
        unet_path = Path(__file__).parent / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        
        if not unet_path.exists():
            print(f"❌ UNET file not found: {unet_path}")
            return
            
        print(f"📁 Testing UNET: {unet_path}")
        
        # STEP 1: Load the state dict (same as load_diffusion_model does)
        print("\n🔍 STEP 1: Loading state dict...")
        sd = comfy.utils.load_torch_file(str(unet_path))
        print(f"✅ State dict loaded: {len(sd)} keys")
        
        # STEP 2: Check what prefix is detected (same as load_diffusion_model_state_dict)
        print("\n🔍 STEP 2: Detecting diffusion model prefix...")
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        print(f"✅ Detected prefix: '{diffusion_model_prefix}'")
        
        # STEP 3: Check if model config can be detected (this is where it fails!)
        print("\n🔍 STEP 3: Detecting model config...")
        model_config = comfy.model_detection.model_config_from_unet(sd, diffusion_model_prefix)
        
        if model_config:
            print(f"✅ Model config detected: {type(model_config).__name__}")
            print(f"   Model type: {getattr(model_config, 'model_type', 'Unknown')}")
            print(f"   Image model: {getattr(model_config, 'image_model', 'Unknown')}")
            print(f"   Supported dtypes: {getattr(model_config, 'supported_inference_dtypes', 'Unknown')}")
        else:
            print("❌ No model config detected!")
            print("   This is where the failure happens!")
            
        # STEP 4: Check what happens in load_diffusion_model_state_dict
        print("\n🔍 STEP 4: Testing load_diffusion_model_state_dict...")
        try:
            model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
            if model:
                print("✅ load_diffusion_model_state_dict succeeded!")
                print(f"   Model type: {type(model)}")
                print(f"   Model device: {model.model.device if hasattr(model, 'model') else 'Unknown'}")
            else:
                print("❌ load_diffusion_model_state_dict returned None")
        except Exception as e:
            print(f"❌ Error in load_diffusion_model_state_dict: {e}")
            
        # STEP 5: Check what happens in the full load_diffusion_model
        print("\n🔍 STEP 5: Testing full load_diffusion_model...")
        try:
            model = comfy.sd.load_diffusion_model(str(unet_path))
            if model:
                print("✅ load_diffusion_model succeeded!")
                print(f"   Model type: {type(model)}")
                print(f"   Model device: {model.model.device if hasattr(model, 'model') else 'Unknown'}")
            else:
                print("❌ load_diffusion_model returned None")
        except Exception as e:
            print(f"❌ Error in load_diffusion_model: {e}")
            
        # STEP 6: Debug why model detection is failing
        print("\n🔍 STEP 6: Debugging model detection failure...")
        print("   Let's check what keys are available for detection...")
        
        # Check for specific detection patterns
        detection_patterns = {
            'head.modulation': 'WAN modulation head',
            'patch_embedding.weight': 'WAN patch embedding', 
            'blocks.0.ffn.0.weight': 'WAN transformer blocks',
            'img_emb.proj.0.bias': 'WAN image embedding',
            'vace_patch_embedding.weight': 'WAN VACE variant'
        }
        
        print("   Checking detection patterns:")
        for key, description in detection_patterns.items():
            if key in sd:
                print(f"     ✅ {description}: {key}")
                if 'weight' in key:
                    shape = sd[key].shape
                    print(f"        Shape: {shape}")
            else:
                print(f"     ❌ {description}: {key} - NOT FOUND")
                
        # STEP 7: Check if there are any missing dependencies
        print("\n🔍 STEP 7: Checking for missing dependencies...")
        try:
            # Check if supported_models is properly loaded
            wan21_class = getattr(comfy.supported_models, 'WAN21_T2V', None)
            if wan21_class:
                print("✅ WAN21_T2V class found in supported_models")
            else:
                print("❌ WAN21_T2V class NOT found in supported_models")
                
            # Check if model_base is properly loaded
            wan21_base = getattr(comfy.model_base, 'WAN21', None)
            if wan21_base:
                print("✅ WAN21 class found in model_base")
            else:
                print("❌ WAN21 class NOT found in model_base")
                
        except Exception as e:
            print(f"❌ Error checking dependencies: {e}")
            
        # STEP 8: Check the exact error in model detection
        print("\n🔍 STEP 8: Detailed model detection debugging...")
        try:
            # Let's see what happens when we call model_config_from_unet
            print("   Calling model_config_from_unet with debug info...")
            
            # Check what's in the state dict that might help detection
            print(f"   State dict keys starting with '{diffusion_model_prefix}':")
            prefix_keys = [k for k in sd.keys() if k.startswith(diffusion_model_prefix)]
            print(f"   Found {len(prefix_keys)} keys with prefix")
            
            # Show first few keys
            for key in prefix_keys[:5]:
                print(f"     {key}")
                
            # Try to understand why detection fails
            print("   Attempting to understand detection failure...")
            
            # Check if it's a prefix issue
            if diffusion_model_prefix != "model.":
                print(f"   ⚠️  Prefix mismatch: expected 'model.' but got '{diffusion_model_prefix}'")
                print("   This might be causing detection issues")
            
        except Exception as e:
            print(f"❌ Error in detailed debugging: {e}")
            
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_detection_step_by_step() 