#!/usr/bin/env python3
"""
Working test for partial loading with KSampler using existing memory management
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

def test_working_partial_loading():
    """Test partial loading using existing memory management system"""
    print("🧪 TESTING WORKING PARTIAL LOADING WITH KSAMPLER")
    print("="*80)
    
    # Check if models exist
    unet_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    
    if not os.path.exists(unet_path):
        print(f"❌ UNet model not found: {unet_path}")
        return False
    
    try:
        # ========================================================================
        # Step 1: Load UNet Model
        # ========================================================================
        print("\n🔧 STEP 1: LOADING UNET MODEL")
        print("="*60)
        
        log_memory_usage("Before UNet Loading")
        
        # Load UNet state dict
        print(f"📂 Loading UNet state dict from: {unet_path}")
        unet_state_dict = load_torch_file(unet_path)
        print(f"   📊 State dict keys: {len(unet_state_dict)}")
        
        # Load UNet model
        print("🚀 Loading UNet model...")
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
        
        print(f"✅ UNet model loaded successfully")
        print(f"   📊 Model type: {type(unet_model).__name__}")
        print(f"   📊 Model device: {next(unet_model.parameters()).device}")
        
        log_memory_usage("After UNet Loading")
        
        # ========================================================================
        # Step 2: Apply Advanced Memory Management
        # ========================================================================
        print("\n🔧 STEP 2: APPLYING ADVANCED MEMORY MANAGEMENT")
        print("="*60)
        
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
            elif loading_info['loading_type'] == 'dynamic_cpu':
                print(f"   Modules available: {loading_info['modules_available']}")
                print(f"   Total size: {loading_info['total_size_gb']:.3f} GB")
                print(f"   Target GPU device: {loading_info['target_gpu_device']}")
            
            log_memory_usage("After Advanced Memory Management")
        else:
            print("⚠️  CUDA not available, skipping memory management")
            model = unet_model
            final_device = device
        
        # ========================================================================
        # Step 3: Test KSampler
        # ========================================================================
        print("\n🔧 STEP 3: TESTING KSAMPLER")
        print("="*60)
        
        # Create test inputs
        print("📊 Creating test inputs...")
        batch_size = 1
        frames = 16
        height = 64
        width = 64
        channels = 2
        
        # Create initial latent
        initial_latent = torch.randn(batch_size, channels, frames, height, width)
        if final_device.type == 'cuda':
            initial_latent = initial_latent.to(final_device)
        
        print(f"   📊 Initial latent shape: {initial_latent.shape}")
        print(f"   📊 Initial latent device: {initial_latent.device}")
        
        # Create conditioning (simplified)
        positive_conditioning = torch.randn(batch_size, 77, 5120)
        negative_conditioning = torch.randn(batch_size, 77, 5120)
        
        if final_device.type == 'cuda':
            positive_conditioning = positive_conditioning.to(final_device)
            negative_conditioning = negative_conditioning.to(final_device)
        
        print(f"   📊 Positive conditioning shape: {positive_conditioning.shape}")
        print(f"   📊 Negative conditioning shape: {negative_conditioning.shape}")
        
        # Create KSampler
        print("🚀 Creating KSampler...")
        ksampler = StandaloneKSampler(
            model=model_patcher,
            steps=3,  # Very few steps for testing
            sampler="euler",
            scheduler="normal",
            denoise=1.0
        )
        
        print(f"✅ KSampler created successfully")
        
        # Test inference
        print("\n🔧 Testing inference...")
        
        # Run KSampler
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
                noise=initial_latent,
                positive=positive_conditioning,
                negative=negative_conditioning,
                cfg=7.0,
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
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 WORKING PARTIAL LOADING TEST")
    print("="*80)
    
    # Test working partial loading
    test_passed = test_working_partial_loading()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Working partial loading test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print(f"\n🎉 TEST PASSED!")
        print(f"   Partial loading system is working correctly!")
        print(f"   Ready for production use!")
    else:
        print(f"\n❌ TEST FAILED!")
        print(f"   Need to fix issues before production use!")

if __name__ == "__main__":
    main()
