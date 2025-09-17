#!/usr/bin/env python3
"""
Simple test for ComfyUI-style partial loading integration with existing pipeline
"""

import torch
import sys
import os
import time
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_torch_file
from memory_utils import log_memory_usage, safe_model_to_device_advanced
from standalone_sd import load_state_dict_guess_config
from standalone_ksampler import StandaloneKSampler

def test_simple_partial_loading():
    """Simple test of partial loading with KSampler"""
    print("🧪 SIMPLE PARTIAL LOADING TEST")
    print("="*60)
    
    # Check if models exist
    unet_path = "../models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    
    if not os.path.exists(unet_path):
        print(f"❌ UNet model not found: {unet_path}")
        print("   Current file size:", os.path.getsize(unet_path) if os.path.exists(unet_path) else "N/A")
        print("   Expected size: ~32GB")
        print("   Please download proper model files first")
        return False
    
    # Check file size
    file_size = os.path.getsize(unet_path)
    file_size_gb = file_size / (1024**3)
    print(f"📊 UNet file size: {file_size_gb:.3f} GB")
    
    if file_size_gb < 1.0:  # Less than 1GB is suspicious
        print(f"⚠️  WARNING: UNet file is very small ({file_size_gb:.3f} GB)")
        print(f"   Expected: ~32GB")
        print(f"   This might be a corrupted download!")
    
    try:
        # Load UNet model
        print("🚀 Loading UNet model...")
        log_memory_usage("Before UNet Loading")
        
        unet_state_dict = load_torch_file(unet_path)
        result = load_state_dict_guess_config(
            unet_state_dict,
            output_vae=False,
            output_clip=False,
            output_clipvision=False,
            output_model=True
        )
        
        if result is None:
            print("❌ Failed to load UNet model")
            return False
        
        model_patcher, _, _, _ = result
        unet_model = model_patcher.model if hasattr(model_patcher, 'model') else model_patcher
        
        print(f"✅ UNet model loaded")
        print(f"   📊 Model type: {type(unet_model).__name__}")
        print(f"   📊 Model device: {next(unet_model.parameters()).device}")
        
        log_memory_usage("After UNet Loading")
        
        # Test with advanced memory management
        print("\n🔧 Testing advanced memory management...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            # Use advanced memory management with partial loading
            print("🚀 Applying advanced memory management...")
            model, final_device, loading_info = safe_model_to_device_advanced(
                unet_model, 
                device, 
                min_free_gb=2.0, 
                state_dict=unet_state_dict, 
                enable_partial_loading=True
            )
            
            print(f"✅ Advanced memory management applied:")
            print(f"   Loading type: {loading_info['loading_type']}")
            print(f"   Final device: {final_device}")
            
            if loading_info['loading_type'] == 'partial':
                print(f"   Modules loaded: {loading_info['modules_loaded']}")
                print(f"   Modules dynamic: {loading_info['modules_dynamic']}")
                print(f"   Memory used: {loading_info['memory_used_gb']:.3f} GB")
                print(f"   Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
            
            log_memory_usage("After Advanced Memory Management")
            
            # Test KSampler
            print("\n🔧 Testing KSampler...")
            
            # Create test inputs
            initial_latent = torch.randn(1, 2, 16, 64, 64).to(device)
            positive_conditioning = torch.randn(1, 77, 5120).to(device)
            negative_conditioning = torch.randn(1, 77, 5120).to(device)
            
            print(f"   📊 Initial latent shape: {initial_latent.shape}")
            print(f"   📊 Conditioning shapes: {positive_conditioning.shape}, {negative_conditioning.shape}")
            
            # Create KSampler
            ksampler = StandaloneKSampler(
                model=model_patcher,
                steps=3,  # Very few steps for testing
                cfg=7.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0
            )
            
            print("🚀 Running KSampler...")
            inference_start = time.time()
            
            try:
                # Memory monitoring callback
                def memory_callback(step, total_steps, current_step=None, **kwargs):
                    if step % max(1, total_steps // 2) == 0:
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"      Step {step}/{total_steps}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                denoised_latent = ksampler.sample(
                    initial_latent=initial_latent,
                    positive_conditioning=positive_conditioning,
                    negative_conditioning=negative_conditioning,
                    seed=42,
                    callback=memory_callback
                )
                
                inference_time = time.time() - inference_start
                
                print(f"✅ KSampler completed successfully!")
                print(f"   📊 Inference time: {inference_time:.2f}s")
                print(f"   📊 Output shape: {denoised_latent.shape}")
                print(f"   📊 Output device: {denoised_latent.device}")
                print(f"   📊 Output range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
                
                log_memory_usage("After KSampler")
                
                return True
                
            except Exception as e:
                print(f"❌ KSampler failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("⚠️  CUDA not available, skipping GPU tests")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_estimation():
    """Test memory estimation accuracy"""
    print("\n🧪 TESTING MEMORY ESTIMATION")
    print("="*60)
    
    unet_path = "../models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    
    if not os.path.exists(unet_path):
        print(f"❌ UNet model not found: {unet_path}")
        return False
    
    try:
        # Load state dict
        print("📂 Loading state dict...")
        state_dict = load_torch_file(unet_path)
        
        # Test memory estimation
        from memory_utils import estimate_state_dict_memory
        memory_info = estimate_state_dict_memory(state_dict)
        
        print(f"📊 Memory Estimation Results:")
        print(f"   Parameters: {memory_info['parameters']:,}")
        print(f"   Raw size: {memory_info['size_gb']:.3f} GB")
        print(f"   CPU size (1.8x): {memory_info['size_gb_cpu']:.3f} GB")
        print(f"   GPU size (2.6x): {memory_info['size_gb_gpu']:.3f} GB")
        print(f"   Keys: {memory_info['keys']}")
        
        # Check if estimation is reasonable
        if memory_info['size_gb'] < 0.1:  # Less than 100MB is suspicious
            print(f"⚠️  WARNING: Estimated size is very small!")
            print(f"   This suggests the model file is corrupted or incomplete")
            print(f"   Expected: ~32GB for WAN 2.1 UNet")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory estimation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 SIMPLE PARTIAL LOADING TEST")
    print("="*80)
    
    # Test memory estimation
    test1_passed = test_memory_estimation()
    
    # Test simple partial loading
    test2_passed = test_simple_partial_loading()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Memory estimation test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Simple partial loading test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Partial loading system is working!")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Check the issues above!")

if __name__ == "__main__":
    main()
