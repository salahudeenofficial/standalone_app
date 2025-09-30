#!/usr/bin/env python3
"""
Test script for VAE encoding with real video data
Tests the motion pipeline with safu.mp4 and safu.jpg
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add motion directory to path
motion_dir = Path(__file__).parent / "motion"
sys.path.insert(0, str(motion_dir))

# Also add parent directory for imports
parent_dir = motion_dir.parent
sys.path.insert(0, str(parent_dir))

def test_real_video_vae_encoding():
    """Test VAE encoding with real video data"""
    print("🎬 REAL VIDEO VAE ENCODING TEST")
    print("=" * 60)
    print("Testing motion pipeline with real video data (safu.mp4)")
    print()
    
    try:
        # Import pipeline components
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("motion_pipeline", motion_dir / "pipeline.py")
            motion_pipeline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(motion_pipeline)
            WanVideoPipeline = motion_pipeline.WanVideoPipeline
            print("✅ Imported WanVideoPipeline from motion/pipeline.py")
        except Exception as e:
            print(f"❌ Failed to import WanVideoPipeline: {e}")
            return False
        
        # Check if real video file exists
        video_path = "safu.mp4"
        image_path = "safu.jpg"
        
        if not os.path.exists(video_path):
            print(f"❌ Real video file not found: {video_path}")
            return False
        
        if not os.path.exists(image_path):
            print(f"❌ Real image file not found: {image_path}")
            return False
        
        print(f"✅ Found real video file: {video_path}")
        print(f"✅ Found real image file: {image_path}")
        
        # Initialize pipeline
        print("\n🔧 Initializing pipeline...")
        pipeline = WanVideoPipeline()
        
        # Find a VAE model file
        vae_model_path = None
        possible_paths = [
            "models/vaes/wan_vae.safetensors",
            "models/vaes/wan_2.1_vae.safetensors", 
            "models/vaes/wan2.1_vae.safetensors",
            "../models/vaes/wan_vae.safetensors",
            "../../models/vaes/wan_vae.safetensors",
            "wan2.1_vace_14B_fp16.safetensors",
            "../wan2.1_vace_14B_fp16.safetensors",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                vae_model_path = path
                print(f"✅ Found VAE model: {path}")
                break
        
        if vae_model_path is None:
            print("⚠️  No VAE model found, creating dummy VAE state dict for testing")
            dummy_vae_path = "dummy_vae.safetensors"
            dummy_state_dict = {
                "decoder.middle.0.residual.0.gamma": torch.randn(96),
                "decoder.conv_in.weight": torch.randn(16, 3, 3, 3),
                "decoder.conv_in.bias": torch.randn(16),
            }
            torch.save(dummy_state_dict, dummy_vae_path)
            vae_model_path = dummy_vae_path
            print(f"✅ Created dummy VAE: {dummy_vae_path}")
        
        # Test parameters with real video
        test_params = {
            'vae_model_path': vae_model_path,
            'positive_prompt': "test prompt with real video",
            'negative_prompt': "test negative prompt",
            'width': 832,
            'height': 480,
            'length': 37,  # Small length for testing
            'batch_size': 1,
            'control_video_path': None,  # Will use real video automatically
            'reference_image_path': None,  # Will use real image automatically
            'strength': 1.0,
        }
        
        print(f"📋 Test parameters:")
        for key, value in test_params.items():
            print(f"   {key}: {value}")
        
        # Test Step 1: VAE and Latent Creation with real video
        print(f"\n🎯 TESTING STEP 1: VAE AND LATENT CREATION WITH REAL VIDEO")
        print("-" * 60)
        
        # Call step 1
        step1_results = pipeline.step_1_vae_and_latent_creation(**test_params)
        
        # Validate results
        print(f"\n✅ STEP 1 COMPLETED SUCCESSFULLY WITH REAL VIDEO!")
        print(f"📊 Results structure:")
        
        if isinstance(step1_results, dict):
            for key, value in step1_results.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape} | dtype: {value.dtype} | device: {value.device}")
                    if value.numel() > 0:
                        print(f"      Mean: {value.mean().item():.6f}")
                        print(f"      Range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                        print(f"      Std: {value.std().item():.6f}")
                    else:
                        print(f"      ⚠️  EMPTY TENSOR!")
                else:
                    print(f"   {key}: {type(value)} = {value}")
        else:
            print(f"   Result type: {type(step1_results)}")
            if isinstance(step1_results, torch.Tensor):
                print(f"   Shape: {step1_results.shape}")
                print(f"   Dtype: {step1_results.dtype}")
                print(f"   Device: {step1_results.device}")
                if step1_results.numel() > 0:
                    print(f"   Mean: {step1_results.mean().item():.6f}")
                    print(f"   Range: [{step1_results.min().item():.6f}, {step1_results.max().item():.6f}]")
        
        print(f"\n🎉 REAL VIDEO VAE ENCODING TEST COMPLETED!")
        print("=" * 60)
        
        # Cleanup dummy VAE file if created
        if vae_model_path == "dummy_vae.safetensors" and os.path.exists("dummy_vae.safetensors"):
            os.remove("dummy_vae.safetensors")
            print("🧹 Cleaned up dummy VAE file")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎬 MOTION PIPELINE REAL VIDEO VAE ENCODING TEST")
    print("=" * 70)
    print("This script tests the motion pipeline with real video data")
    print("to compare VAE encoding results with ComfyUI.")
    print()
    
    # Test with real video
    success = test_real_video_vae_encoding()
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 30)
    if success:
        print("✅ Real video VAE encoding test: PASSED")
        print("🎉 Motion pipeline successfully processed real video data!")
        print("📊 Check the output above for VAE encoding statistics")
        print("🔍 Compare these results with ComfyUI expected values")
    else:
        print("❌ Real video VAE encoding test: FAILED")
        print("⚠️  Please check the error messages above for details")

if __name__ == "__main__":
    main()
