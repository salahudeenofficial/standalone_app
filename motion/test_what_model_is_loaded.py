#!/usr/bin/env python3
"""
Test script to see what model is actually being loaded in the pipeline
"""

import torch
import time
import sys
import os

# Add motion directory to path
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

def test_what_model_is_loaded():
    """Test what model is actually being loaded"""
    print("🧪 Testing What Model Is Actually Loaded")
    print("="*60)
    
    from pipeline import WanVideoPipeline
    
    # Initialize pipeline
    pipeline = WanVideoPipeline()
    
    # Try to load the UNet model
    unet_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    
    if os.path.exists(unet_path):
        print("✅ UNet model file exists, loading...")
        try:
            from utils import load_torch_file
            from standalone_sd import load_state_dict_guess_config
            
            # Load UNet state dict
            unet_state_dict = load_torch_file(unet_path)
            print(f"   📊 Loaded UNet state dict with {len(unet_state_dict)} keys")
            
            # Load UNet model
            result = load_state_dict_guess_config(
                unet_state_dict,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is not None:
                model_patcher, _, _, _ = result
                print(f"✅ UNet loaded successfully")
                print(f"   ModelPatcher type: {type(model_patcher).__name__}")
                print(f"   Model type: {type(model_patcher.model).__name__}")
                print(f"   Model device: {model_patcher.load_device}")
                
                # Test the model's forward method
                print("\n🔍 Testing model's forward method...")
                test_x = torch.randn(1, 16, 11, 104, 60, device=model_patcher.load_device)
                test_timestep = torch.tensor([0.5], device=model_patcher.load_device)
                
                try:
                    result = model_patcher.model.forward(test_x, test_timestep)
                    print(f"✅ Model forward method works")
                    print(f"   Result shape: {result.shape}")
                    print(f"   Result range: [{result.min():.3f}, {result.max():.3f}]")
                    
                    # Check if result is all zeros (indicating broken model)
                    if torch.allclose(result, torch.zeros_like(result), atol=1e-6):
                        print("⚠️  Model is returning all zeros - this is the problem!")
                    else:
                        print("✅ Model is returning realistic values")
                        
                except Exception as e:
                    print(f"❌ Model forward method failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
            else:
                print("❌ Failed to load UNet model")
                
        except Exception as e:
            print(f"❌ Failed to load UNet model: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("❌ UNet model file not found")
        print("   This explains why the pipeline is not working properly")
        print("   The pipeline is probably loading a dummy/broken model")

if __name__ == "__main__":
    test_what_model_is_loaded()
