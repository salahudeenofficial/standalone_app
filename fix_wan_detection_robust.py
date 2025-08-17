#!/usr/bin/env python3
"""
Robust fix for ComfyUI's WAN model detection issue.

This script directly edits the file to add the empty prefix for WAN models.
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
        lines = f.readlines()
    
    # Find the line with candidates
    candidates_line_index = None
    for i, line in enumerate(lines):
        if 'candidates = [' in line:
            candidates_line_index = i
            break
    
    if candidates_line_index is None:
        print("❌ Could not find candidates line")
        return False
    
    print(f"✅ Found candidates line at line {candidates_line_index + 1}")
    
    # Check if already patched
    if '["", #wan models (no prefix)' in lines[candidates_line_index]:
        print("✅ Already patched!")
        return True
    
    # Get the current candidates line
    current_line = lines[candidates_line_index]
    print(f"   Current: {current_line.strip()}")
    
    # Create the new line with empty prefix added
    if '["model.diffusion_model.", #ldm/sgm models' in current_line:
        # Replace the first quote with empty prefix
        new_line = current_line.replace('["model.diffusion_model.', '["", #wan models (no prefix)\n                  "model.diffusion_model.')
        print(f"   New: {new_line.strip()}")
        
        # Update the line
        lines[candidates_line_index] = new_line
        
        # Write the patched file
        with open(model_detection_path, 'w') as f:
            f.writelines(lines)
        
        print("✅ Successfully patched ComfyUI model detection!")
        print("   Added empty prefix \"\" to recognize WAN models")
        return True
    else:
        print("❌ Unexpected candidates line format")
        print(f"   Line content: {current_line}")
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
    print("🔧 ComfyUI WAN Model Detection Fix (Robust)")
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