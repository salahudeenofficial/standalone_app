#!/usr/bin/env python3
"""
Fix for ComfyUI's WAN model detection issue.

The problem: WAN models use no prefix (keys like 'head.modulation') but ComfyUI's
unet_prefix_from_state_dict() function defaults to 'model.' prefix, causing WAN
models to be incorrectly detected as FLOW models.

This script patches the detection logic to recognize WAN models correctly.
"""

import sys
import os
from pathlib import Path

def patch_comfyui_wan_detection():
    """Patch ComfyUI's model detection to properly recognize WAN models"""
    
    # Path to ComfyUI's model_detection.py
    comfyui_path = Path(__file__).parent / "comfy"
    model_detection_path = comfyui_path / "model_detection.py"
    
    if not model_detection_path.exists():
        print(f"❌ ComfyUI model_detection.py not found at: {model_detection_path}")
        return False
    
    print(f"🔧 Patching ComfyUI model detection at: {model_detection_path}")
    
    # Read the current file
    with open(model_detection_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "candidates = [\"\", \"model.diffusion_model.\", \"model.model.\", \"net.\"]" in content:
        print("✅ Already patched!")
        return True
    
    # Find the candidates list - current format
    old_pattern = 'candidates = ["model.diffusion_model.", #ldm/sgm models\n                  "model.model.", #audio models\n                  "net.", #cosmos\n                  ]'
    
    new_pattern = 'candidates = ["", #wan models (no prefix)\n                  "model.diffusion_model.", #ldm/sgm models\n                  "model.model.", #audio models\n                  "net.", #cosmos\n                  ]'
    
    if old_pattern in content:
        # Replace the pattern
        new_content = content.replace(old_pattern, new_pattern)
        
        # Write the patched file
        with open(model_detection_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully patched ComfyUI model detection!")
        print("   Added empty prefix \"\" to recognize WAN models")
        return True
    else:
        print("❌ Could not find the candidates pattern to patch")
        print("   The file structure may have changed")
        return False

def test_wan_detection():
    """Test if WAN model detection now works"""
    
    print("\n🧪 Testing WAN model detection after patch...")
    
    try:
        # Import the patched modules
        import comfy.model_detection
        import comfy.utils
        
        # Test with the WAN model
        unet_path = Path(__file__).parent / "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        
        if not unet_path.exists():
            print(f"❌ UNET file not found: {unet_path}")
            return False
        
        # Load the state dict
        sd = comfy.utils.load_torch_file(str(unet_path))
        
        # Test prefix detection
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
        print(f"   Detected prefix: '{diffusion_model_prefix}'")
        
        # Test model config detection
        model_config = comfy.model_detection.model_config_from_unet(sd, diffusion_model_prefix)
        
        if model_config:
            print(f"   ✅ Model config detected: {type(model_config).__name__}")
            if hasattr(model_config, 'image_model'):
                print(f"   ✅ Image model: {model_config.image_model}")
            if hasattr(model_config, 'model_type'):
                print(f"   ✅ Model type: {model_config.model_type}")
            
            # Check if it's detected as WAN
            if hasattr(model_config, 'image_model') and model_config.image_model == 'wan2.1':
                print("   🎯 SUCCESS: WAN model correctly detected!")
                
                # Now test if we can load the model with the correct type
                print("   🧪 Testing model loading with detected config...")
                try:
                    model = model_config.get_model(sd, device=None)
                    if model:
                        print(f"   ✅ Model loaded successfully: {type(model)}")
                        if hasattr(model, 'model_type'):
                            print(f"   ✅ Final model type: {model.model_type}")
                            if hasattr(model.model_type, 'name'):
                                print(f"   ✅ Model type name: {model.model_type.name}")
                        return True
                    else:
                        print("   ❌ Model loading failed")
                        return False
                except Exception as e:
                    print(f"   ❌ Model loading error: {e}")
                    return False
            else:
                print("   ⚠️  Model detected but not as WAN")
                return False
        else:
            print("   ❌ No model config detected")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 ComfyUI WAN Model Detection Fix")
    print("=" * 50)
    
    # Patch the detection logic
    if patch_comfyui_wan_detection():
        # Test if it works
        if test_wan_detection():
            print("\n🎉 SUCCESS: WAN model detection is now working!")
            print("   You can now run the pipeline and it should detect WAN models correctly.")
        else:
            print("\n⚠️  Patch applied but WAN detection still not working.")
            print("   There may be additional issues to resolve.")
    else:
        print("\n❌ Failed to patch ComfyUI model detection.")
        print("   Manual intervention may be required.")

if __name__ == "__main__":
    main() 