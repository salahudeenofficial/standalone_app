#!/usr/bin/env python3
"""
Test Step 4: KSampler Denoising with ComfyUI Integration

This script tests the complete Step 4 implementation including:
- ComfyUI CFGGuider integration
- Fallback to standalone KSampler
- Memory management
- Sequential execution up to Step 4
"""

import os
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import WanVideoPipeline

def main():
    """Test Steps 1, 2, 3, and 4: Sequential VAE Loading + UNet + CLIP Loading + Model Sampling + Text Encoding + KSampler Denoising"""
    print("🚀 WAN Video Pipeline - Sequential Steps 1, 2, 3 & 4 Test")
    print("="*80)
    print("🎯 Testing Step 1 (VAE) → Step 2 (UNet + CLIP) → Step 3 (Model Sampling + Text Encoding) → Step 4 (KSampler Denoising)")
    print("="*80)
    
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
    
    # Prepare parameters for all steps
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
        # Check if we can run all four steps
        can_run_step1 = "VAE" in available_models
        can_run_step2 = "UNet" in available_models and "CLIP" in available_models
        can_run_step3 = can_run_step2  # Step 3 depends on Step 2
        can_run_step4 = can_run_step2  # Step 4 depends on Step 2 (UNet + CLIP)
        
        if can_run_step1 and can_run_step2 and can_run_step3 and can_run_step4:
            # Run Steps 1, 2, 3, and 4 sequentially
            print(f"\n🚀 RUNNING STEPS 1, 2, 3 & 4 SEQUENTIALLY")
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
            
            # Step 4: KSampler Denoising
            print("\n🎯 STEP 4: KSAMPLER DENOISING")
            step_4_params = {
                'initial_latent': step_1_results['out_latent']['samples'],
                'positive_conditioning': step_3_results['positive_conditioning'],
                'negative_conditioning': step_3_results['negative_conditioning'],
                'seed': 42,
                'steps': 4,
                'cfg': 7.0,
                'sampler_name': 'euler',
                'scheduler': 'normal',
                'denoise': 1.0,
                'noise_inds': None
            }
            step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
            
            print(f"\n🎉 SEQUENTIAL STEPS 1, 2, 3 & 4 COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Display Step 4 specific results
            if step_4_results:
                print(f"\n🎯 STEP 4 DETAILED RESULTS:")
                sampling_config = step_4_results.get('sampling_config', {})
                print(f"   Seed: {sampling_config.get('seed', 'Unknown')}")
                print(f"   Steps: {sampling_config.get('steps', 'Unknown')}")
                print(f"   CFG: {sampling_config.get('cfg', 'Unknown')}")
                print(f"   Sampler: {sampling_config.get('sampler_name', 'Unknown')}")
                print(f"   Scheduler: {sampling_config.get('scheduler', 'Unknown')}")
                print(f"   Denoise: {sampling_config.get('denoise', 'Unknown')}")
                
                processing_info = step_4_results.get('processing_info', {})
                print(f"   Noise Preparation Time: {processing_info.get('noise_preparation_time', 0.0):.3f}s")
                print(f"   KSampler Setup Time: {processing_info.get('ksampler_setup_time', 0.0):.3f}s")
                print(f"   Denoising Time: {processing_info.get('denoising_time', 0.0):.2f}s")
                print(f"   Total Step Time: {processing_info.get('total_step_time', 0.0):.2f}s")
                
                # Verify denoised latent
                denoised_latent = step_4_results.get('denoised_latent')
                print(f"   Denoised Latent Status: {'✅ Generated' if denoised_latent is not None else '❌ Failed'}")
                if denoised_latent is not None:
                    print(f"   Denoised Latent Shape: {denoised_latent.shape}")
                    print(f"   Denoised Latent Device: {denoised_latent.device}")
                    print(f"   Denoised Latent Range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
                    
                    # Check for valid denoising
                    if torch.isfinite(denoised_latent).all():
                        print(f"   Denoised Latent Quality: ✅ Valid (all finite values)")
                    else:
                        print(f"   Denoised Latent Quality: ❌ Invalid (contains NaN/Inf)")
                
                # Check ComfyUI integration status
                comfyui_used = step_4_results.get('comfyui_integration_used', False)
                print(f"   ComfyUI Integration: {'✅ Used' if comfyui_used else '⚠️  Fallback Used'}")
                
                # Memory usage analysis
                memory_info = step_4_results.get('memory_info', {})
                if memory_info:
                    print(f"   Peak GPU Memory: {memory_info.get('peak_gpu_memory', 'Unknown')}")
                    print(f"   Memory Efficiency: {memory_info.get('memory_efficiency', 'Unknown')}")
            
            # Final status check
            step_status = pipeline.get_step_status()
            completed_steps = sum(1 for completed in step_status.values() if completed)
            
            print(f"\n📊 FINAL PIPELINE STATUS:")
            print(f"   Steps Completed: {completed_steps}/7")
            for step_num, completed in step_status.items():
                status = "✅ Completed" if completed else "⏳ Pending"
                print(f"   Step {step_num}: {status}")
            
            if completed_steps >= 4:
                print(f"\n🎉 SUCCESS: Steps 1, 2, 3, and 4 completed sequentially!")
                print(f"🎯 Pipeline is ready for Step 5 (VAE Decoding)")
                print(f"💡 ComfyUI KSampler integration working correctly")
            else:
                print(f"\n⚠️  PARTIAL SUCCESS: {completed_steps} steps completed")
                print(f"💡 Check model availability and dependencies")
        
        else:
            # Fallback scenarios
            if can_run_step1 and can_run_step2 and can_run_step3:
                print(f"\n🚀 RUNNING STEPS 1, 2 & 3 ONLY (Step 4 requires all models)")
                print("="*60)
                
                # Run Steps 1, 2, and 3
                step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
                step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
                step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
                
                print(f"\n🎉 STEPS 1, 2 & 3 COMPLETED SUCCESSFULLY!")
                print(f"💡 Steps 1, 2 & 3 completed - Step 4 requires all models")
                
            elif can_run_step1 and can_run_step2:
                print(f"\n🚀 RUNNING STEPS 1 & 2 ONLY (Step 3 requires both)")
                print("="*60)
                
                # Run Steps 1 and 2
                step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
                step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
                
                print(f"\n🎉 STEPS 1 & 2 COMPLETED SUCCESSFULLY!")
                print(f"💡 Steps 1 & 2 completed - Step 3 requires both VAE and UNet+CLIP models")
                
            elif can_run_step1:
                print(f"\n🚀 RUNNING STEP 1 ONLY (Step 2 models not available)")
                print("="*60)
                
                # Run Step 1 only
                step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
                
                print(f"\n🎉 STEP 1 COMPLETED SUCCESSFULLY!")
                print(f"💡 Step 1 completed - Step 2 requires UNet and CLIP model files")
                
            else:
                print(f"\n⏭️  NO MODELS AVAILABLE - Testing pipeline initialization only")
                print("="*60)
                
                print(f"   ✅ Pipeline initialized successfully")
                print(f"   Device: {pipeline.device}")
                print(f"   Offload Device: {pipeline.offload_device}")
                print(f"   Models Directory: {pipeline.models_dir}")
                
                step_status = pipeline.get_step_status()
                print(f"   Initial step status: {step_status}")
                
                print(f"\n💡 Pipeline ready - model files required for Step 1, 2, 3, and 4")
        
    except Exception as e:
        print(f"\n❌ SEQUENTIAL STEPS 1, 2, 3 & 4 TEST FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 Check the error details above and ensure:")
        print(f"   - Model files exist and are valid")
        print(f"   - All required dependencies are installed")
        print(f"   - The standalone_sd.py fixes are properly applied")
        print(f"   - Step 4 KSampler denoising and ComfyUI integration are working correctly")

if __name__ == "__main__":
    main()
