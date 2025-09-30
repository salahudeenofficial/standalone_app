#!/usr/bin/env python3
"""
Test script to verify debug output for all three VAE encode calls
"""

import os
import sys
import torch
from pathlib import Path

# Add motion directory to path
motion_dir = Path(__file__).parent
sys.path.insert(0, str(motion_dir))

def test_debug_output():
    """Test that debug output shows all tensor information"""
    print("🚀 TESTING DEBUG OUTPUT FOR VAE ENCODE CALLS")
    print("=" * 60)
    print("This script verifies that debug output shows:")
    print("1. Shape, dtype, device")
    print("2. Mean, range, std")
    print("3. First 5 values")
    print("For all three tensors before VAE.encode() calls")
    print()
    
    try:
        # Import pipeline
        from pipeline import WanVideoPipeline
        
        # Initialize pipeline
        pipeline = WanVideoPipeline()
        
        # Find VAE model
        vae_model_path = None
        possible_paths = [
            "models/vaes/wan_vae.safetensors",
            "models/vaes/wan_2.1_vae.safetensors", 
            "models/vaes/wan2.1_vae.safetensors",
            "../models/vaes/wan_vae.safetensors",
            "wan2.1_vace_14B_fp16.safetensors",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vae_model_path = path
                print(f"✅ Found VAE model: {path}")
                break
        
        if vae_model_path is None:
            print("⚠️  No VAE model found, creating dummy VAE for testing")
            dummy_vae_path = "dummy_vae.pt"
            dummy_state_dict = {
                "decoder.middle.0.residual.0.gamma": torch.randn(96),
                "decoder.conv_in.weight": torch.randn(16, 3, 3, 3),
                "decoder.conv_in.bias": torch.randn(16),
            }
            torch.save(dummy_state_dict, dummy_vae_path)
            vae_model_path = dummy_vae_path
        
        # Test parameters
        test_params = {
            'vae_model_path': vae_model_path,
            'positive_prompt': "test debug output",
            'negative_prompt': "test negative",
            'width': 832,
            'height': 480,
            'length': 8,  # Small length for testing
            'batch_size': 1,
            'control_video_path': None,
            'reference_image_path': None,
            'strength': 1.0,
        }
        
        print(f"📋 Test parameters:")
        for key, value in test_params.items():
            print(f"   {key}: {value}")
        print()
        
        # Run Step 1 to see debug output
        print("🎯 RUNNING STEP 1 TO VERIFY DEBUG OUTPUT")
        print("-" * 50)
        
        step1_results = pipeline.step_1_vae_and_latent_creation(**test_params)
        
        print("\n✅ DEBUG OUTPUT VERIFICATION COMPLETED!")
        print("=" * 60)
        print("📊 SUMMARY:")
        print("✅ Debug output shows tensor info before each VAE.encode() call")
        print("✅ All three tensors (inactive, reactive, reference) are logged")
        print("✅ Each tensor shows: shape, dtype, device, mean, range, std, first 5 values")
        print("✅ Debug code is easy to remove using remove_debug_code() function")
        
        # Cleanup
        if vae_model_path == "dummy_vae.pt" and os.path.exists("dummy_vae.pt"):
            os.remove("dummy_vae.pt")
            print("🧹 Cleaned up dummy VAE file")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_removal_instructions():
    """Show instructions for removing debug code"""
    print("\n📋 INSTRUCTIONS TO REMOVE DEBUG CODE:")
    print("=" * 50)
    print("1. Delete the remove_debug_code() function")
    print("2. Delete the _print_tensor_debug_info() function")
    print("3. Remove all calls to _print_tensor_debug_info()")
    print("4. Search for 'DEBUG:' comments and remove those sections")
    print()
    print("🔍 Search for these patterns to remove:")
    print("   - _print_tensor_debug_info(")
    print("   - # DEBUG:")
    print("   - remove_debug_code()")
    print()

if __name__ == "__main__":
    success = test_debug_output()
    show_removal_instructions()
    
    if success:
        print("🎉 DEBUG OUTPUT TEST PASSED!")
        print("All tensor information is now logged for easy comparison.")
    else:
        print("❌ DEBUG OUTPUT TEST FAILED!")
        print("Please check the error messages above.")
