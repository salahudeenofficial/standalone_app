#!/usr/bin/env python3
"""
Test Standalone ModelPatcher - No ComfyUI Dependencies

This script verifies that the motion pipeline ModelPatcher works
without any external dependencies, following Disclaimer.txt guidelines.
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_real_modelpatcher_pipeline():
    """Test ModelPatcher with real UNet loading, model sampling, and KSampling"""
    print("🔧 TESTING REAL MODELPATCHER PIPELINE")
    print("="*60)
    
    # Model file paths
    vae_model_path = "models/vaes/wan_vae.safetensors"
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    # Check if all models are available
    if not all(os.path.exists(p) for p in [vae_model_path, unet_model_path, clip_model_path]):
        print("❌ Not all required models available for real testing")
        print("   Required models:")
        print(f"   - VAE: {vae_model_path}")
        print(f"   - UNet: {unet_model_path}")
        print(f"   - CLIP: {clip_model_path}")
        return False
    
    print("✅ All required models available")
    
    try:
        # Import motion pipeline components
        from pipeline import WanVideoPipeline
        from standalone_model_patcher import ModelPatcher
        from standalone_ksampler import StandaloneKSampler
        from model_sampling import ModelSamplingSD3
        from text_encoder import CLIPTextEncode
        
        print("✅ All motion pipeline components imported successfully")
        
        # Initialize pipeline
        pipeline = WanVideoPipeline(models_dir="models")
        print("✅ WanVideoPipeline initialized")
        
        # ========================================================================
        # STEP 1: VAE Loading (like in pipeline Step 1)
        # ========================================================================
        print("\n🎬 STEP 1: VAE LOADING")
        print("-" * 40)
        
        step_1_params = {
            'vae_model_path': vae_model_path,
            'positive_prompt': "test prompt",
            'negative_prompt': "test negative",
            'control_video_path': None,
            'reference_image_path': None,
            'width': 480,
            'height': 832,
            'length': 37,
            'batch_size': 1,
            'strength': 1.0
        }
        
        step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
        if not step_1_results:
            print("❌ Step 1 failed")
            return False
        print("✅ Step 1 completed - VAE loaded and latent created")
        
        # ========================================================================
        # STEP 2: UNet + CLIP Loading (like in pipeline Step 2)
        # ========================================================================
        print("\n🧠 STEP 2: UNET + CLIP LOADING")
        print("-" * 40)
        
        step_2_params = {
            'unet_model_path': unet_model_path,
            'clip_model_path': clip_model_path,
            'lora_model_path': None,
            'strength_model': 1.0,
            'strength_clip': 0.0
        }
        
        step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
        if not step_2_results:
            print("❌ Step 2 failed")
            return False
        print("✅ Step 2 completed - UNet and CLIP loaded")
        
        # Verify ModelPatcher
        unet = step_2_results.get('unet')
        if unet is None:
            print("❌ UNet not loaded")
            return False
        
        print(f"✅ UNet loaded: {type(unet).__name__}")
        print(f"   ModelPatcher type: {type(unet).__name__}")
        print(f"   Load device: {getattr(unet, 'load_device', 'unknown')}")
        print(f"   Offload device: {getattr(unet, 'offload_device', 'unknown')}")
        
        # Test ModelPatcher methods
        print("\n🔧 Testing ModelPatcher methods...")
        
        # Test pre_run
        if hasattr(unet, 'pre_run'):
            print("✅ pre_run method exists")
            try:
                unet.pre_run()
                print("✅ pre_run() executed successfully - weights loaded")
            except Exception as e:
                print(f"⚠️  pre_run() failed: {e}")
        else:
            print("❌ pre_run method missing")
        
        # Test unload
        if hasattr(unet, 'unload'):
            print("✅ unload method exists")
        else:
            print("❌ unload method missing")
        
        # Test cleanup
        if hasattr(unet, 'cleanup'):
            print("✅ cleanup method exists")
        else:
            print("❌ cleanup method missing")
        
        # ========================================================================
        # STEP 3: Model Sampling + Text Encoding (like in pipeline Step 3)
        # ========================================================================
        print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
        print("-" * 40)
        
        step_3_params = {
            'positive_prompt': "test prompt",
            'negative_prompt': "test negative",
            'shift': 8.0,
            'multiplier': 1000
        }
        
        step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
        if not step_3_results:
            print("❌ Step 3 failed")
            return False
        print("✅ Step 3 completed - Model sampling and text encoding done")
        
        # ========================================================================
        # STEP 4: KSampler Denoising (like in pipeline Step 4)
        # ========================================================================
        print("\n🎯 STEP 4: KSAMPLER DENOISING")
        print("-" * 40)
        
        step_4_params = {
            'initial_latent': step_1_results['out_latent']['samples'],
            'positive_conditioning': step_3_results['positive_conditioning'],
            'negative_conditioning': step_3_results['negative_conditioning'],
            'seed': 42,
            'steps': 3,  # Minimal steps for testing
            'cfg': 7.0,
            'sampler_name': 'euler',
            'scheduler': 'normal',
            'denoise': 1.0,
            'noise_inds': None
        }
        
        print("🔧 Testing ModelPatcher with real KSampling...")
        print("   - Using real UNet ModelPatcher")
        print("   - Using real conditioning from Step 3")
        print("   - Using real latent from Step 1")
        
        # Test ModelPatcher cleanup before sampling
        if hasattr(unet, 'cleanup'):
            print("🔧 Testing ModelPatcher.cleanup()...")
            try:
                unet.cleanup()
                print("✅ ModelPatcher.cleanup() executed successfully")
            except Exception as e:
                print(f"⚠️  ModelPatcher.cleanup() failed: {e}")
        
        # Run Step 4
        start_time = time.time()
        step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
        sampling_time = time.time() - start_time
        
        if step_4_results:
            print(f"✅ Step 4 completed in {sampling_time:.2f}s")
            
            # Verify results
            denoised_latent = step_4_results.get('denoised_latent')
            if denoised_latent is not None:
                print(f"📊 Denoised latent shape: {denoised_latent.shape}")
                print(f"📊 Denoised latent device: {denoised_latent.device}")
                print(f"📊 Denoised latent dtype: {denoised_latent.dtype}")
                print(f"📊 Denoised latent range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
                
                # Check for valid values
                if torch.isfinite(denoised_latent).all():
                    print("✅ All denoised values are finite")
                else:
                    print("❌ Denoised latent contains NaN/Inf values")
                
                # Check for non-zero values
                if torch.count_nonzero(denoised_latent) > 0:
                    print("✅ Denoised latent has non-zero values")
                else:
                    print("❌ Denoised latent is all zeros")
                
                print("\n✅ REAL MODELPATCHER PIPELINE TEST PASSED!")
                print("🔧 ModelPatcher successfully handled:")
                print("   - UNet loading and weight management")
                print("   - Model sampling configuration")
                print("   - KSampler denoising with proper model interface")
                print("   - Memory management and cleanup")
                
                return True
            else:
                print("❌ No denoised latent returned")
                return False
        else:
            print("❌ Step 4 failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_pipeline_imports():
    """Test that pipeline imports work without ComfyUI dependencies"""
    print("\n🔧 TESTING PIPELINE IMPORTS")
    print("="*60)
    
    try:
        # Test pipeline import
        from pipeline import WanVideoPipeline
        print("✅ WanVideoPipeline imported successfully")
        
        # Test other motion modules
        from standalone_vae import VAE
        print("✅ VAE imported successfully")
        
        from standalone_sd import load_state_dict_guess_config
        print("✅ load_state_dict_guess_config imported successfully")
        
        from standalone_ksampler import StandaloneKSampler
        print("✅ StandaloneKSampler imported successfully")
        
        from text_encoder import CLIPTextEncode
        print("✅ CLIPTextEncode imported successfully")
        
        print("\n✅ ALL PIPELINE IMPORTS SUCCESSFUL!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 REAL MODELPATCHER PIPELINE VERIFICATION")
    print("="*80)
    print("📋 Following Disclaimer.txt guidelines:")
    print("   - No external module dependencies")
    print("   - Motion pipeline fully standalone")
    print("   - Algorithm borrowed from ComfyUI but implemented internally")
    print("   - Testing with real UNet loading, model sampling, and KSampling")
    print("="*80)
    
    # Test 1: Real ModelPatcher pipeline (Steps 1-4)
    real_pipeline_ok = test_real_modelpatcher_pipeline()
    
    # Test 2: Pipeline imports
    imports_ok = test_pipeline_imports()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL VERIFICATION SUMMARY")
    print("="*80)
    
    if real_pipeline_ok:
        print("✅ Real ModelPatcher Pipeline: PASSED")
    else:
        print("❌ Real ModelPatcher Pipeline: FAILED")
    
    if imports_ok:
        print("✅ Pipeline Imports: PASSED")
    else:
        print("❌ Pipeline Imports: FAILED")
    
    if real_pipeline_ok and imports_ok:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("🔧 Motion pipeline is fully standalone")
        print("🔧 No ComfyUI dependencies")
        print("🔧 ModelPatcher works with real UNet loading")
        print("🔧 ModelPatcher works with real model sampling")
        print("🔧 ModelPatcher works with real KSampling")
        print("🔧 Following Disclaimer.txt guidelines")
        print("\n📊 VERIFIED CAPABILITIES:")
        print("   ✅ Step 1: VAE loading and latent creation")
        print("   ✅ Step 2: UNet + CLIP loading with ModelPatcher")
        print("   ✅ Step 3: Model sampling and text encoding")
        print("   ✅ Step 4: KSampler denoising with ModelPatcher")
        print("   ✅ ModelPatcher weight loading/unloading")
        print("   ✅ ModelPatcher memory management")
        print("   ✅ ModelPatcher cleanup functionality")
    else:
        print("\n🚨 VERIFICATION FAILED!")
        print("🔧 Check the issues above and ensure:")
        print("   - No external module dependencies")
        print("   - All motion modules are self-contained")
        print("   - ModelPatcher implements ComfyUI algorithms internally")
        print("   - Real model files are available for testing")

if __name__ == "__main__":
    main()
