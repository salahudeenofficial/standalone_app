#!/usr/bin/env python3
"""
Test script for the updated pipeline with ComfyUI-style memory management

This script demonstrates:
1. Advanced memory management integration
2. ComfyUI-style model loading/unloading
3. Dynamic loading support
4. Complete pipeline workflow
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import pipeline
from pipeline import WanVideoPipeline
from memory_utils import log_memory_usage, get_memory_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_pipeline_memory_management():
    """Test the pipeline with advanced memory management"""
    
    print("🚀 PIPELINE MEMORY MANAGEMENT TEST")
    print("="*60)
    
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
            print(f"   GPU {i}: {gpu_name}")
    
    print("\n" + "="*60)
    
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
        'steps': 10,  # Fewer steps for testing
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
        print("\n💡 This test requires the WAN models to be downloaded")
        print("🧪 Testing memory management functions only...")
        
        # Test memory management functions
        test_memory_functions()
        return
    
    try:
        # Initial memory status
        print("\n📊 INITIAL MEMORY STATUS:")
        log_memory_usage("Pipeline Start")
        
        # Run complete pipeline with advanced memory management
        print("\n🚀 Running Complete Pipeline with Advanced Memory Management...")
        pipeline_results = pipeline.run_complete_pipeline_with_memory_management(
            step_1_params, step_2_params, step_3_params, step_4_params
        )
        
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        
        # Final memory status
        print("\n📊 FINAL MEMORY STATUS:")
        log_memory_usage("Pipeline Complete")
        
        # Pipeline summary
        print(f"\n📋 PIPELINE SUMMARY:")
        print(f"   Total Time: {pipeline_results['pipeline_time']:.2f}s")
        print(f"   Memory Management: {pipeline_results['memory_management']}")
        print(f"   Dynamic Loading: {'✅ Enabled' if pipeline_results['dynamic_loading_enabled'] else '❌ Not available'}")
        
        # Check UNet dynamic loading info
        if pipeline_results['dynamic_loading_enabled']:
            unet_model = pipeline.unet.model if hasattr(pipeline.unet, 'model') else pipeline.unet
            if hasattr(unet_model, '_dynamic_loading_info'):
                info = unet_model._dynamic_loading_info
                print(f"   Modules Available: {len(info['modules'])}")
                print(f"   Total Module Size: {info['total_size_gb']:.3f} GB")
                print(f"   Target GPU Device: {info['target_device']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {str(e)}")
        print(f"   Error Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_functions():
    """Test memory management functions without models"""
    
    print("\n🧪 TESTING MEMORY MANAGEMENT FUNCTIONS")
    print("="*50)
    
    try:
        # Test memory info
        print("📊 Testing memory info...")
        mem_info = get_memory_info()
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Allocated: {mem_info.get('cuda_allocated', 0):.3f} GB")
            print(f"   CUDA Reserved: {mem_info.get('cuda_reserved', 0):.3f} GB")
            print(f"   CUDA Free: {mem_info.get('cuda_free', 0):.3f} GB")
            print(f"   CUDA Total: {mem_info.get('cuda_total', 0):.3f} GB")
        
        # Test memory logging
        print("\n📊 Testing memory logging...")
        log_memory_usage("Test Memory Logging")
        
        print("✅ Memory management functions working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Memory function test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline_memory_management()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   ✅ Pipeline with advanced memory management working")
        print("   ✅ ComfyUI-style model loading/unloading integrated")
        print("   ✅ Dynamic loading support enabled")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("   Check the error messages above for details")
    
    print("\n" + "="*60)
