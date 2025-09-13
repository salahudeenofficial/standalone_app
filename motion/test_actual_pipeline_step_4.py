#!/usr/bin/env python3
"""
Test script to run actual pipeline Step 4 and debug the issue
"""

import torch
import time
import sys
import os

# Add motion directory to path
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

def test_actual_pipeline_step_4():
    """Test the actual pipeline Step 4 to see what's happening"""
    print("🧪 Testing Actual Pipeline Step 4")
    print("="*60)
    
    from pipeline import WanVideoPipeline
    
    # Initialize pipeline
    pipeline = WanVideoPipeline()
    
    # Run Steps 1, 2, 3 first
    print("🚀 Running Steps 1, 2, 3...")
    
    try:
        # Step 1: VAE + Latent Creation
        step_1_results = pipeline.step_1_vae_and_latent_creation(
            vae_model_path="models/vaes/wan_vae.safetensors",
            reference_image_path="test_images/reference.jpg",
            control_video_path="test_videos/control.mp4"
        )
        print("✅ Step 1 completed")
        
        # Step 2: UNet + CLIP + LoRA
        step_2_results = pipeline.step_2_unet_clip_lora_loading(
            unet_model_path="models/diffusion_models/wan_2.1_diffusion_model.safetensors",
            clip_model_path="models/text_encoders/wan_clip_model.safetensors"
        )
        print("✅ Step 2 completed")
        
        # Step 3: Model Sampling + Text Encoding
        step_3_results = pipeline.step_3_model_sampling_text_encoding(
            positive_prompt="very cinematic video of a beautiful landscape",
            negative_prompt="blurry, low quality, distorted"
        )
        print("✅ Step 3 completed")
        
        # Now test Step 4
        print("\n🚀 Testing Step 4: KSampler Denoising...")
        
        # Prepare Step 4 parameters
        step_4_params = {
            'initial_latent': step_1_results['output_latent'],
            'positive_conditioning': step_3_results['positive_conditioning'],
            'negative_conditioning': step_3_results['negative_conditioning'],
            'seed': 42,
            'steps': 20,
            'cfg': 7.0,
            'sampler_name': 'euler',
            'scheduler': 'normal',
            'denoise': 1.0
        }
        
        # Run Step 4
        step_4_start = time.time()
        step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
        step_4_time = time.time() - step_4_start
        
        print(f"\n📊 STEP 4 RESULTS:")
        print(f"   Total time: {step_4_time:.2f}s")
        print(f"   Denoised latent shape: {step_4_results['denoised_latent'].shape}")
        print(f"   Denoised latent range: [{step_4_results['denoised_latent'].min():.3f}, {step_4_results['denoised_latent'].max():.3f}]")
        
        # Check if the results look realistic
        if step_4_time < 0.5:
            print("⚠️  Step 4 was too fast - might not be doing real work")
        else:
            print("✅ Step 4 took realistic time")
        
        # Check the model that was used
        if hasattr(pipeline.unet, 'model') and hasattr(pipeline.unet.model, 'call_count'):
            print(f"   Model calls: {pipeline.unet.model.call_count}")
            if pipeline.unet.model.call_count < 10:
                print("⚠️  Too few model calls - might not be doing real sampling")
            else:
                print("✅ Realistic number of model calls")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_actual_pipeline_step_4()
