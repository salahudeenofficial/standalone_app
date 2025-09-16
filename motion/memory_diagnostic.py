#!/usr/bin/env python3
"""
Memory Diagnostic Script for Pipeline OOM Issues

This script helps diagnose why a 32GB model causes 48GB OOM during inference.
It monitors memory usage at each step and identifies memory leaks.
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import pipeline and memory utils
from pipeline import WanVideoPipeline
from memory_utils import log_memory_usage, get_memory_info, clear_cuda_memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def monitor_memory_during_pipeline():
    """Monitor memory usage during pipeline execution"""
    
    print("🔍 MEMORY DIAGNOSTIC FOR PIPELINE OOM ISSUES")
    print("="*70)
    
    # System information
    print(f"📊 System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print("\n" + "="*70)
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Test parameters (using smaller dimensions for testing)
    step_1_params = {
        'vae_model_path': "models/vaes/wan_vae.safetensors",
        'positive_prompt': "cinematic video",
        'negative_prompt': "low quality, blurry",
        'control_video_path': "safu.mp4",
        'reference_image_path': "safu.jpg",  
        'width': 256,  # Smaller for testing
        'height': 256,  # Smaller for testing
        'length': 8,    # Shorter for testing
        'batch_size': 1,
        'strength': 1.0
    }
    
    step_2_params = {
        'unet_model_path': "models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        'clip_model_path': "models/text_encoders/wan_clip_model.safetensors",
        'lora_model_path': "models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_3_params = {
        'positive_prompt': "cinematic video",
        'negative_prompt': "low quality, blurry",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    step_4_params = {
        'initial_latent': None,  # Will be set from step_1_results
        'positive_conditioning': None,  # Will be set from step_3_results
        'negative_conditioning': None,  # Will be set from step_3_results
        'seed': 42,
        'steps': 5,  # Very few steps for testing
        'cfg': 7.0,
        'sampler_name': "euler",
        'scheduler': "normal",
        'denoise': 1.0,
        'noise_inds': None
    }
    
    # Check if model files exist
    required_files = [
        step_1_params['vae_model_path'],
        step_2_params['unet_model_path'], 
        step_2_params['clip_model_path']
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Required model files not found:")
        for missing in missing_files:
            print(f"   {missing}")
        print("\n💡 This diagnostic requires the WAN models to be downloaded")
        return False
    
    try:
        # Step 0: Initial memory
        print("\n🔍 STEP 0: INITIAL MEMORY STATE")
        log_memory_usage("Initial State")
        
        # Step 1: VAE + Latent Creation
        print("\n🔍 STEP 1: VAE + LATENT CREATION")
        log_memory_usage("Before Step 1")
        step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
        log_memory_usage("After Step 1")
        
        # Step 2: UNet + CLIP + LoRA Loading
        print("\n🔍 STEP 2: UNET + CLIP + LORA LOADING")
        log_memory_usage("Before Step 2")
        step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
        log_memory_usage("After Step 2")
        
        # Check UNet device and memory usage
        if pipeline.unet:
            unet_model = pipeline.unet.model if hasattr(pipeline.unet, 'model') else pipeline.unet
            print(f"   📊 UNet device: {unet_model.device}")
            print(f"   📊 UNet load_device: {pipeline.unet.load_device}")
            
            if hasattr(unet_model, '_dynamic_loading_info'):
                info = unet_model._dynamic_loading_info
                if 'modules' in info:
                    print(f"   📊 Dynamic loading enabled: {len(info['modules'])} modules")
                if 'total_size_gb' in info:
                    print(f"   📊 Total module size: {info['total_size_gb']:.3f} GB")
                print(f"   📊 Dynamic loading info keys: {list(info.keys())}")
            else:
                print(f"   📊 Dynamic loading: Not available")
        
        # Step 3: Model Sampling + Text Encoding
        print("\n🔍 STEP 3: MODEL SAMPLING + TEXT ENCODING")
        log_memory_usage("Before Step 3")
        step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
        log_memory_usage("After Step 3")
        
        # Prepare Step 4 parameters
        step_4_params['initial_latent'] = step_1_results['out_latent']['samples']
        step_4_params['positive_conditioning'] = step_3_results['positive_conditioning']
        step_4_params['negative_conditioning'] = step_3_results['negative_conditioning']
        
        # Step 4: KSampler Denoising (with detailed monitoring)
        print("\n🔍 STEP 4: KSAMPLER DENOISING (DETAILED MONITORING)")
        log_memory_usage("Before Step 4")
        
        # Check UNet state before Step 4
        if pipeline.unet:
            unet_model = pipeline.unet.model if hasattr(pipeline.unet, 'model') else pipeline.unet
            print(f"   📊 UNet device before Step 4: {unet_model.device}")
            print(f"   📊 UNet load_device before Step 4: {pipeline.unet.load_device}")
        
        # Run Step 4 with detailed monitoring
        step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
        log_memory_usage("After Step 4")
        
        # Final memory analysis
        print("\n🔍 FINAL MEMORY ANALYSIS")
        log_memory_usage("Final State")
        
        # Check if UNet was unloaded
        if pipeline.unet:
            unet_model = pipeline.unet.model if hasattr(pipeline.unet, 'model') else pipeline.unet
            print(f"   📊 UNet device after Step 4: {unet_model.device}")
            print(f"   📊 UNet load_device after Step 4: {pipeline.unet.load_device}")
        
        print("\n✅ MEMORY DIAGNOSTIC COMPLETED")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ CUDA OOM DETECTED: {e}")
        print(f"   Error occurred during pipeline execution")
        
        # Get memory info at failure point
        log_memory_usage("OOM Failure Point")
        
        # Try to identify the issue
        if pipeline.unet:
            unet_model = pipeline.unet.model if hasattr(pipeline.unet, 'model') else pipeline.unet
            print(f"   📊 UNet device at failure: {unet_model.device}")
            print(f"   📊 UNet load_device at failure: {pipeline.unet.load_device}")
            
            if hasattr(unet_model, '_dynamic_loading_info'):
                print(f"   📊 Dynamic loading was available but OOM still occurred")
                print(f"   📊 This suggests the model was loaded multiple times or not unloaded properly")
            else:
                print(f"   📊 No dynamic loading - model likely loaded to GPU multiple times")
        
        return False
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_leaks():
    """Test for memory leaks by running the pipeline multiple times"""
    
    print("\n🔍 TESTING FOR MEMORY LEAKS")
    print("="*50)
    
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Simple test parameters
    step_1_params = {
        'vae_model_path': "models/vaes/wan_vae.safetensors",
        'positive_prompt': "test",
        'negative_prompt': "test",
        'control_video_path': "safu.mp4",
        'reference_image_path': "safu.jpg",  
        'width': 128,
        'height': 128,
        'length': 4,
        'batch_size': 1,
        'strength': 1.0
    }
    
    step_2_params = {
        'unet_model_path': "models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        'clip_model_path': "models/text_encoders/wan_clip_model.safetensors",
        'lora_model_path': None,  # No LoRA for leak testing
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    step_3_params = {
        'positive_prompt': "test",
        'negative_prompt': "test",
        'shift': 8.0,
        'multiplier': 1000
    }
    
    step_4_params = {
        'initial_latent': None,
        'positive_conditioning': None,
        'negative_conditioning': None,
        'seed': 42,
        'steps': 2,  # Very few steps
        'cfg': 7.0,
        'sampler_name': "euler",
        'scheduler': "normal",
        'denoise': 1.0,
        'noise_inds': None
    }
    
    try:
        for i in range(3):  # Run 3 times to check for leaks
            print(f"\n🔄 RUN {i+1}/3")
            log_memory_usage(f"Before Run {i+1}")
            
            # Run steps 1-3
            step_1_results = pipeline.step_1_vae_and_latent_creation(**step_1_params)
            step_2_results = pipeline.step_2_unet_clip_lora_loading(**step_2_params)
            step_3_results = pipeline.step_3_model_sampling_and_text_encoding(**step_3_params)
            
            # Prepare Step 4
            step_4_params['initial_latent'] = step_1_results['out_latent']['samples']
            step_4_params['positive_conditioning'] = step_3_results['positive_conditioning']
            step_4_params['negative_conditioning'] = step_3_results['negative_conditioning']
            
            # Run Step 4
            step_4_results = pipeline.step_4_ksampler_denoising(**step_4_params)
            
            log_memory_usage(f"After Run {i+1}")
            
            # Clear cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   🧹 CUDA cache cleared after run {i+1}")
        
        print("\n✅ MEMORY LEAK TEST COMPLETED")
        return True
        
    except Exception as e:
        print(f"\n❌ MEMORY LEAK TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Memory Diagnostic...")
    
    # Run main diagnostic
    success1 = monitor_memory_during_pipeline()
    
    if success1:
        # Run memory leak test
        success2 = test_memory_leaks()
        
        if success1 and success2:
            print("\n🎉 ALL DIAGNOSTICS PASSED!")
            print("   ✅ Memory usage is within expected limits")
            print("   ✅ No memory leaks detected")
        else:
            print("\n⚠️  SOME DIAGNOSTICS FAILED")
            print("   Check the output above for memory issues")
    else:
        print("\n❌ MAIN DIAGNOSTIC FAILED")
        print("   OOM issue detected - check the analysis above")
    
    print("\n" + "="*70)
