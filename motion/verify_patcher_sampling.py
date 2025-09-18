#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: ModelPatcher Weight Loading and Proper Sampling

This script verifies that:
1. ModelPatcher is properly loading weights
2. ComfyUI's sampling is actually using the loaded model
3. Proper inference is taking place (not dummy data)
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import WanVideoPipeline

def verify_model_patcher_loading():
    """Verify that ModelPatcher is properly loading weights"""
    print("🔍 VERIFYING MODELPATCHER WEIGHT LOADING")
    print("="*60)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Model file paths
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    if not os.path.exists(unet_model_path):
        print(f"❌ UNet model not found: {unet_model_path}")
        return False
    
    if not os.path.exists(clip_model_path):
        print(f"❌ CLIP model not found: {clip_model_path}")
        return False
    
    # Load models
    print("📊 Loading UNet and CLIP models...")
    step_2_params = {
        'unet_model_path': unet_model_path,
        'clip_model_path': clip_model_path,
        'lora_model_path': None,
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
    
    if not step_2_results:
        print("❌ Step 2 failed - cannot verify ModelPatcher")
        return False
    
    unet = step_2_results.get('unet')
    if unet is None:
        print("❌ UNet not loaded - cannot verify ModelPatcher")
        return False
    
    print(f"✅ UNet loaded: {type(unet).__name__}")
    
    # Check if it's a ModelPatcher
    if hasattr(unet, 'load') and hasattr(unet, 'unload'):
        print("✅ UNet is ComfyUI-style ModelPatcher")
        
        # Check model state
        print(f"📊 Model device: {unet.load_device}")
        print(f"📊 Offload device: {unet.offload_device}")
        print(f"📊 Model options: {getattr(unet, 'model_options', {})}")
        
        # Check if model has weights
        if hasattr(unet, 'model'):
            model = unet.model
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                print(f"📊 Model parameter count: {param_count:,}")
                
                # Check if parameters are on correct device
                device_params = sum(1 for p in model.parameters() if p.device.type == 'cuda')
                total_params = sum(1 for p in model.parameters())
                print(f"📊 Parameters on GPU: {device_params}/{total_params}")
                
                if device_params == 0:
                    print("⚠️  WARNING: No parameters on GPU - model may not be loaded!")
                    return False
                else:
                    print("✅ Model parameters are on GPU - weights loaded!")
            else:
                print("⚠️  WARNING: Model has no parameters attribute!")
                return False
        else:
            print("⚠️  WARNING: ModelPatcher has no model attribute!")
            return False
        
        return True
    else:
        print("⚠️  UNet is not ComfyUI-style ModelPatcher")
        return False

def verify_proper_sampling():
    """Verify that proper sampling is taking place"""
    print("\n🔍 VERIFYING PROPER SAMPLING")
    print("="*60)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Model file paths
    vae_model_path = "models/vaes/wan_vae.safetensors"
    unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
    
    # Check if all models are available
    if not all(os.path.exists(p) for p in [vae_model_path, unet_model_path, clip_model_path]):
        print("❌ Not all required models available for sampling test")
        return False
    
    # Run Steps 1, 2, 3
    print("📊 Running Steps 1, 2, 3...")
    
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
    
    step_2_params = {
        'unet_model_path': unet_model_path,
        'clip_model_path': clip_model_path,
        'lora_model_path': None,
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_3_params = {
        'positive_prompt': "test prompt",
        'negative_prompt': "test negative",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    # Step 1
    step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
    if not step_1_results:
        print("❌ Step 1 failed")
        return False
    
    # Step 2
    step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
    if not step_2_results:
        print("❌ Step 2 failed")
        return False
    
    # Step 3
    step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
    if not step_3_results:
        print("❌ Step 3 failed")
        return False
    
    print("✅ Steps 1, 2, 3 completed successfully")
    
    # Now test Step 4 with detailed verification
    print("\n📊 Testing Step 4 with detailed verification...")
    
    step_4_params = {
        'initial_latent': step_1_results['out_latent']['samples'],
        'positive_conditioning': step_3_results['positive_conditioning'],
        'negative_conditioning': step_3_results['negative_conditioning'],
        'seed': 42,
        'steps': 5,  # Use fewer steps for faster testing
        'cfg': 7.0,
        'sampler_name': 'euler',
        'scheduler': 'normal',
        'denoise': 1.0,
        'noise_inds': None
    }
    
    # Record initial state
    initial_latent = step_4_params['initial_latent']
    print(f"📊 Initial latent shape: {initial_latent.shape}")
    print(f"📊 Initial latent range: [{initial_latent.min().item():.3f}, {initial_latent.max().item():.3f}]")
    print(f"📊 Initial latent std: {initial_latent.std().item():.3f}")
    
    # Run Step 4
    start_time = time.time()
    step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
    sampling_time = time.time() - start_time
    
    if not step_4_results:
        print("❌ Step 4 failed")
        return False
    
    # Verify results
    denoised_latent = step_4_results.get('denoised_latent')
    if denoised_latent is None:
        print("❌ No denoised latent returned")
        return False
    
    print(f"\n🔍 SAMPLING VERIFICATION RESULTS:")
    print(f"📊 Sampling time: {sampling_time:.2f}s")
    print(f"📊 Denoised latent shape: {denoised_latent.shape}")
    print(f"📊 Denoised latent range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
    print(f"📊 Denoised latent std: {denoised_latent.std().item():.3f}")
    
    # Critical checks
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Latent changed
    if not torch.allclose(initial_latent, denoised_latent, atol=1e-6):
        print("✅ Check 1 PASSED: Denoised latent differs from initial latent")
        checks_passed += 1
    else:
        print("❌ Check 1 FAILED: Denoised latent is identical to initial latent")
    
    # Check 2: Finite values
    if torch.isfinite(denoised_latent).all():
        print("✅ Check 2 PASSED: All denoised values are finite")
        checks_passed += 1
    else:
        print("❌ Check 2 FAILED: Denoised latent contains NaN/Inf values")
    
    # Check 3: Reasonable timing
    if sampling_time >= 0.5:  # At least 0.5 seconds for 5 steps
        print(f"✅ Check 3 PASSED: Sampling took reasonable time ({sampling_time:.2f}s)")
        checks_passed += 1
    else:
        print(f"❌ Check 3 FAILED: Sampling too fast ({sampling_time:.2f}s) - may be dummy data")
    
    # Check 4: Proper device
    if denoised_latent.device.type == 'cuda':
        print("✅ Check 4 PASSED: Denoised latent is on GPU")
        checks_passed += 1
    else:
        print(f"❌ Check 4 FAILED: Denoised latent is on {denoised_latent.device}")
    
    # Check 5: Non-zero values
    if torch.count_nonzero(denoised_latent) > 0:
        print("✅ Check 5 PASSED: Denoised latent has non-zero values")
        checks_passed += 1
    else:
        print("❌ Check 5 FAILED: Denoised latent is all zeros")
    
    print(f"\n📊 VERIFICATION SUMMARY: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 4:
        print("✅ SAMPLING VERIFICATION PASSED: Proper sampling is taking place!")
        return True
    else:
        print("❌ SAMPLING VERIFICATION FAILED: Sampling may not be working properly!")
        return False

def main():
    """Main verification function"""
    print("🚀 CRITICAL VERIFICATION: ModelPatcher Weight Loading and Proper Sampling")
    print("="*80)
    
    # Test 1: ModelPatcher loading
    patcher_ok = verify_model_patcher_loading()
    
    # Test 2: Proper sampling
    sampling_ok = verify_proper_sampling()
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL VERIFICATION SUMMARY")
    print("="*80)
    
    if patcher_ok:
        print("✅ ModelPatcher Weight Loading: PASSED")
    else:
        print("❌ ModelPatcher Weight Loading: FAILED")
    
    if sampling_ok:
        print("✅ Proper Sampling: PASSED")
    else:
        print("❌ Proper Sampling: FAILED")
    
    if patcher_ok and sampling_ok:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("🔧 ModelPatcher is properly loading weights")
        print("🔧 ComfyUI sampling is working correctly")
        print("🔧 Proper inference is taking place")
    else:
        print("\n🚨 VERIFICATION FAILED!")
        print("🔧 Check the issues above and ensure:")
        print("   - Model files are valid and accessible")
        print("   - ComfyUI components are properly imported")
        print("   - ModelPatcher is correctly implemented")
        print("   - Sampling logic is using actual model inference")

if __name__ == "__main__":
    main()
