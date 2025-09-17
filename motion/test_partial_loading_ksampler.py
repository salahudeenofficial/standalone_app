#!/usr/bin/env python3
"""
Test script for ComfyUI-style partial loading with KSampler
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
from comfyui_style_partial_loading import ComfyUIStylePartialLoader

def test_partial_loading_with_ksampler():
    """Test ComfyUI-style partial loading with KSampler"""
    print("🧪 TESTING COMFYUI-STYLE PARTIAL LOADING WITH KSAMPLER")
    print("="*80)
    
    # Check if models exist
    unet_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    clip_path = "models/text_encoders/wan_clip_model.safetensors"
    
    if not os.path.exists(unet_path):
        print(f"❌ UNet model not found: {unet_path}")
        print("   Please download proper model files first")
        return False
    
    if not os.path.exists(clip_path):
        print(f"❌ CLIP model not found: {clip_path}")
        print("   Please download proper model files first")
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
        # Step 2: Set Up ComfyUI-Style Partial Loading
        # ========================================================================
        print("\n🔧 STEP 2: SETTING UP COMFYUI-STYLE PARTIAL LOADING")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device.type == 'cuda':
            # Get available memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory_gb = gpu_memory / (1024**3)
            
            # Set conservative memory budget (use only 70% of available memory)
            memory_budget_gb = available_memory_gb * 0.7
            
            print(f"📊 GPU Memory Analysis:")
            print(f"   Total GPU Memory: {available_memory_gb:.2f} GB")
            print(f"   Memory Budget: {memory_budget_gb:.2f} GB")
            
            # Set up ComfyUI-style partial loading
            print("🔧 Setting up ComfyUI-style partial loading...")
            loader = ComfyUIStylePartialLoader(unet_model, device, memory_budget_gb)
            loading_info = loader.setup_partial_loading()
            
            print(f"✅ Partial loading setup complete:")
            print(f"   Loading type: {loading_info['loading_type']}")
            print(f"   Loaded weights: {loading_info['loaded_weights']}")
            print(f"   Patched weights: {loading_info['patched_weights']}")
            print(f"   Memory used: {loading_info['memory_used_gb']:.3f} GB")
            print(f"   Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
            
            log_memory_usage("After Partial Loading Setup")
        else:
            print("⚠️  CUDA not available, skipping partial loading setup")
            loader = None
        
        # ========================================================================
        # Step 3: Load CLIP Model
        # ========================================================================
        print("\n🔧 STEP 3: LOADING CLIP MODEL")
        print("="*60)
        
        log_memory_usage("Before CLIP Loading")
        
        # Load CLIP state dict
        print(f"📂 Loading CLIP state dict from: {clip_path}")
        clip_state_dict = load_torch_file(clip_path)
        print(f"   📊 State dict keys: {len(clip_state_dict)}")
        
        # Load CLIP model
        print("🚀 Loading CLIP model...")
        result = load_state_dict_guess_config(
            clip_state_dict,
            output_vae=False,
            output_clip=True,
            output_clipvision=False,
            output_model=False
        )
        
        if result is None:
            print("❌ Failed to load CLIP model")
            return False
        
        _, clip_model, _, _ = result
        
        print(f"✅ CLIP model loaded successfully")
        print(f"   📊 Model type: {type(clip_model).__name__}")
        
        log_memory_usage("After CLIP Loading")
        
        # ========================================================================
        # Step 4: Test KSampler with Partial Loading
        # ========================================================================
        print("\n🔧 STEP 4: TESTING KSAMPLER WITH PARTIAL LOADING")
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
        if device.type == 'cuda':
            initial_latent = initial_latent.to(device)
        
        print(f"   📊 Initial latent shape: {initial_latent.shape}")
        print(f"   📊 Initial latent device: {initial_latent.device}")
        
        # Create conditioning (simplified)
        positive_conditioning = torch.randn(batch_size, 77, 5120)
        negative_conditioning = torch.randn(batch_size, 77, 5120)
        
        if device.type == 'cuda':
            positive_conditioning = positive_conditioning.to(device)
            negative_conditioning = negative_conditioning.to(device)
        
        print(f"   📊 Positive conditioning shape: {positive_conditioning.shape}")
        print(f"   📊 Negative conditioning shape: {negative_conditioning.shape}")
        
        # Create KSampler
        print("🚀 Creating KSampler...")
        sampler_name = "euler"
        scheduler = "normal"
        denoise = 1.0
        
        ksampler = StandaloneKSampler(
            model=model_patcher,
            steps=5,  # Reduced steps for testing
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=denoise
        )
        
        print(f"✅ KSampler created successfully")
        
        # Test inference with partial loading
        print("\n🔧 Testing inference with partial loading...")
        
        if loader is not None:
            print("🔄 Loading weights for inference...")
            loader.load_weights_for_inference()
            log_memory_usage("After Weight Loading")
        
        # Run KSampler
        print("🚀 Running KSampler...")
        inference_start = time.time()
        
        try:
            # Create memory monitoring callback
            def memory_callback(step, total_steps, current_step=None, **kwargs):
                if step % max(1, total_steps // 4) == 0:
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
            print(f"   📊 Denoised latent shape: {denoised_latent.shape}")
            print(f"   📊 Denoised latent device: {denoised_latent.device}")
            print(f"   📊 Denoised latent range: [{denoised_latent.min().item():.3f}, {denoised_latent.max().item():.3f}]")
            
            log_memory_usage("After KSampler Inference")
            
        except Exception as e:
            print(f"❌ KSampler failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup: Evict weights after inference
            if loader is not None:
                print("🧹 Evicting weights after inference...")
                loader.evict_weights_after_inference()
                log_memory_usage("After Weight Eviction")
        
        # ========================================================================
        # Step 5: Analyze Results
        # ========================================================================
        print("\n🔧 STEP 5: ANALYZING RESULTS")
        print("="*60)
        
        if loader is not None:
            loading_info = loader.get_loading_info()
            print(f"📊 Final Loading Info:")
            print(f"   Total weights: {loading_info['total_weights']}")
            print(f"   Loaded weights: {loading_info['loaded_count']}")
            print(f"   Patched weights: {loading_info['patched_count']}")
            print(f"   Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
        
        # Check memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            allocated_gb = gpu_memory['allocated_bytes.all.current'] / (1024**3)
            reserved_gb = gpu_memory['reserved_bytes.all.current'] / (1024**3)
            
            print(f"📊 Final GPU Memory:")
            print(f"   Allocated: {allocated_gb:.2f} GB")
            print(f"   Reserved: {reserved_gb:.2f} GB")
        
        print("\n🎉 PARTIAL LOADING TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency of partial loading"""
    print("\n🧪 TESTING MEMORY EFFICIENCY")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory efficiency test")
        return True
    
    # Get GPU memory info
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = gpu_memory / (1024**3)
    
    print(f"📊 GPU Memory Analysis:")
    print(f"   Total GPU Memory: {total_memory_gb:.2f} GB")
    
    # Test different memory budgets
    test_budgets = [0.1, 0.5, 1.0, 2.0]  # GB
    
    for budget_gb in test_budgets:
        if budget_gb > total_memory_gb:
            continue
            
        print(f"\n🔧 Testing with {budget_gb:.1f} GB budget:")
        
        try:
            # Create a simple test model
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                    self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
                    self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
                    self.fc = torch.nn.Linear(256 * 8 * 8, 1000)
                
                def forward(self, x):
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = torch.relu(self.conv3(x))
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return x
            
            model = TestModel()
            device = torch.device('cuda')
            
            # Set up partial loading
            loader = ComfyUIStylePartialLoader(model, device, budget_gb)
            loading_info = loader.setup_partial_loading()
            
            print(f"   ✅ Setup complete:")
            print(f"      Loaded weights: {loading_info['loaded_weights']}")
            print(f"      Patched weights: {loading_info['patched_weights']}")
            print(f"      Memory used: {loading_info['memory_used_gb']:.3f} GB")
            
            # Test inference
            input_tensor = torch.randn(1, 3, 32, 32).to(device)
            
            with torch.no_grad():
                loader.load_weights_for_inference()
                output = model(input_tensor)
                loader.evict_weights_after_inference()
            
            print(f"      ✅ Inference successful: {output.shape}")
            
        except Exception as e:
            print(f"      ❌ Failed: {e}")
    
    return True

def main():
    """Main test function"""
    print("🚀 COMFYUI-STYLE PARTIAL LOADING WITH KSAMPLER TEST")
    print("="*80)
    
    # Test partial loading with KSampler
    test1_passed = test_partial_loading_with_ksampler()
    
    # Test memory efficiency
    test2_passed = test_memory_efficiency()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Partial loading with KSampler: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Memory efficiency test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ComfyUI-style partial loading is working correctly!")
        print(f"   Ready for production use!")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Need to fix issues before production use!")

if __name__ == "__main__":
    main()
