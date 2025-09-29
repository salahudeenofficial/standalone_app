#!/usr/bin/env python3
"""
Test Script for Step 3: Model Sampling + Text Encoding

This script tests the updated Step 3 implementation that works with the new model loading structure.
It runs Steps 1, 2, and 3 sequentially to ensure proper integration.
"""

import os
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test Step 3: Model Sampling + Text Encoding"""
    print("🚀 Testing Step 3: Model Sampling + Text Encoding")
    print("="*80)
    print("🎯 This test runs Steps 1, 2, and 3 sequentially")
    print("="*80)
    
    # Import the pipeline
    from pipeline import WanVideoPipeline
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Model file paths
    vae_model_path = "models/vaes/wan_vae.safetensors"
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    # Check available model files
    available_models = []
    missing_models = []
    
    if os.path.exists(vae_model_path):
        available_models.append("VAE")
    else:
        missing_models.append(f"VAE: {vae_model_path}")
    
    if os.path.exists(unet_model_path):
        available_models.append("UNet")
    else:
        missing_models.append(f"UNet: {unet_model_path}")
    
    if os.path.exists(clip_model_path):
        available_models.append("CLIP")
    else:
        missing_models.append(f"CLIP: {clip_model_path}")
    
    print(f"\n📊 MODEL AVAILABILITY:")
    print(f"   Available: {', '.join(available_models) if available_models else 'None'}")
    if missing_models:
        print(f"   Missing: {', '.join(missing_models)}")
    
    # Check if we can run all three steps
    can_run_all = "VAE" in available_models and "UNet" in available_models and "CLIP" in available_models
    
    if not can_run_all:
        print(f"\n❌ Cannot run Step 3 test - missing required models")
        print(f"   Required: VAE, UNet, CLIP")
        print(f"   Available: {', '.join(available_models)}")
        return
    
    # Prepare parameters for all three steps
    step_1_params = {
        'vae_model_path': vae_model_path,
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'control_video_path': "safu.mp4" if os.path.exists("safu.mp4") else None,
        'reference_image_path': "safu.jpg" if os.path.exists("safu.jpg") else None,
        'width': 480,
        'height': 832,
        'length': 37,
        'batch_size': 1,
        'strength': 1.0
    }
    
    step_2_params = {
        'unet_model_path': unet_model_path,
        'clip_model_path': clip_model_path,
        'lora_model_path': None,  # No LoRA for this test
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_3_params = {
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    try:
        # Run Steps 1, 2, and 3 sequentially
        print(f"\n🚀 RUNNING STEPS 1, 2 & 3 SEQUENTIALLY")
        print("="*60)
        
        # Step 1: VAE Loading and Latent Creation
        print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
        step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
        
        # Step 2: UNet + CLIP Loading
        print("\n🧠 STEP 2: UNET + CLIP LOADING")
        step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
        
        # Step 3: Model Sampling + Text Encoding
        print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
        step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
        
        print(f"\n🎉 STEPS 1, 2 & 3 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display Step 3 specific results
        if step_3_results:
            print(f"\n📝 STEP 3 DETAILED RESULTS:")
            model_info = step_3_results.get('model_info', {})
            print(f"   UNet Original Type: {model_info.get('original_type', 'Unknown')}")
            print(f"   UNet Patched Type: {model_info.get('patched_type', 'Unknown')}")
            print(f"   Sampling Patch Applied: {'Yes' if step_3_results.get('sampling_applied', False) else 'No'}")
            print(f"   Shift Parameter: {model_info.get('shift', 'Unknown')}")
            print(f"   Multiplier Parameter: {model_info.get('multiplier', 'Unknown')}")
            
            conditioning_info = step_3_results.get('conditioning_info', {})
            print(f"   Positive Prompt: '{conditioning_info.get('positive_prompt', 'Unknown')}'")
            print(f"   Negative Prompt: '{conditioning_info.get('negative_prompt', 'Unknown')}'")
            print(f"   Positive Shape: {conditioning_info.get('positive_shape', 'Unknown')}")
            print(f"   Negative Shape: {conditioning_info.get('negative_shape', 'Unknown')}")
            print(f"   Positive Device: {conditioning_info.get('positive_device', 'Unknown')}")
            
            timing = step_3_results.get('timing', {})
            print(f"   Sampling Time: {timing.get('sampling_time', 0.0):.2f}s")
            print(f"   Positive Encoding Time: {timing.get('positive_encoding', 0.0):.3f}s")
            print(f"   Negative Encoding Time: {timing.get('negative_encoding', 0.0):.3f}s")
            print(f"   Total Step Time: {timing.get('total_step_time', 0.0):.2f}s")
            
            # Verify conditioning
            positive_cond = step_3_results.get('positive_conditioning')
            negative_cond = step_3_results.get('negative_conditioning')
            print(f"   Positive Conditioning Status: {'✅ Generated' if positive_cond is not None else '❌ Failed'}")
            print(f"   Negative Conditioning Status: {'✅ Generated' if negative_cond is not None else '❌ Failed'}")
            
            # Check step completion status
            step_status = pipeline.get_step_status()
            completed_steps = sum(1 for completed in step_status.values() if completed)
            
            print(f"\n📊 FINAL STEP STATUS:")
            print(f"   Steps Completed: {completed_steps}/7")
            print(f"   Step 1 (VAE): {'✅ Completed' if step_status.get(1, False) else '❌ Failed'}")
            print(f"   Step 2 (UNet+CLIP): {'✅ Completed' if step_status.get(2, False) else '❌ Failed'}")
            print(f"   Step 3 (Sampling+Encoding): {'✅ Completed' if step_status.get(3, False) else '❌ Failed'}")
            
            if completed_steps >= 3:
                print(f"\n🎉 SUCCESS: Step 3 integration test completed!")
                print(f"🎯 Pipeline is ready for Step 4 (KSampler Denoising)")
            else:
                print(f"\n⚠️  Some steps failed - check the output above")
        
    except Exception as e:
        print(f"\n❌ STEP 3 TEST FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Check the error details above and ensure:")
        print(f"   - Model files exist and are valid")
        print(f"   - All required dependencies are installed")
        print(f"   - The standalone_sd.py fixes are properly applied")
        print(f"   - Step 3 model sampling and text encoding are working correctly")

if __name__ == "__main__":
    main()
