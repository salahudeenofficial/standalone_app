#!/usr/bin/env python3
"""
Test script to debug actual KSampler with real model
"""

import torch
import time
import sys
import os

# Add motion directory to path
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

from standalone_ksampler import StandaloneKSampler
from wan_vae_components.model_management import get_torch_device

def test_real_ksampler():
    """Test KSampler with actual model from pipeline"""
    print("🧪 Testing Real KSampler with Pipeline Model")
    print("="*60)
    
    # Import pipeline to get the actual model
    from pipeline import WanVideoPipeline
    
    # Initialize pipeline
    pipeline = WanVideoPipeline()
    
    # Check if models are available
    model_files_exist = (
        os.path.exists("models/diffusion_models/wan_2.1_diffusion_model.safetensors") and
        os.path.exists("models/text_encoders/wan_clip_model.safetensors") and
        os.path.exists("models/vaes/wan_vae.safetensors")
    )
    
    if not model_files_exist:
        print("❌ Model files not available, using mock model")
        
        # Create a mock model that simulates real processing
        class RealisticMockModel:
            def __init__(self):
                self.call_count = 0
                self.device = get_torch_device()
                
            def forward(self, x, timestep, *args, **kwargs):
                self.call_count += 1
                print(f"   🔍 Mock Model.forward() called #{self.call_count}")
                print(f"      Input shape: {x.shape}")
                print(f"      Timestep: {timestep}")
                
                # Simulate realistic processing time
                time.sleep(0.05)  # 50ms per call
                
                # Generate realistic noise prediction
                noise_pred = torch.randn_like(x, device=x.device) * 0.8
                
                # Add timestep scaling
                if isinstance(timestep, torch.Tensor):
                    t_scale = float(timestep.item()) if timestep.numel() == 1 else float(timestep[0].item())
                else:
                    t_scale = float(timestep)
                
                noise_scale = 0.2 + t_scale * 0.8
                noise_pred = noise_pred * noise_scale
                
                print(f"      Output range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
                
                return noise_pred
        
        # Create mock model patcher
        class MockModelPatcher:
            def __init__(self, model):
                self.model = model
                self.load_device = get_torch_device()
                self.offload_device = torch.device("cpu")
        
        mock_model = RealisticMockModel()
        model_patcher = MockModelPatcher(mock_model)
        
    else:
        print("✅ Model files available, loading real model...")
        # Load real model (this will fail if files don't exist)
        try:
            step_2_results = pipeline.step_2_unet_clip_lora_loading(
                unet_model_path="models/diffusion_models/wan_2.1_diffusion_model.safetensors",
                clip_model_path="models/text_encoders/wan_clip_model.safetensors"
            )
            model_patcher = step_2_results['unet']
        except Exception as e:
            print(f"❌ Failed to load real model: {e}")
            print("Using mock model instead...")
            
            # Create mock model
            class RealisticMockModel:
                def __init__(self):
                    self.call_count = 0
                    self.device = get_torch_device()
                    
                def forward(self, x, timestep, *args, **kwargs):
                    self.call_count += 1
                    print(f"   🔍 Mock Model.forward() called #{self.call_count}")
                    print(f"      Input shape: {x.shape}")
                    print(f"      Timestep: {timestep}")
                    
                    # Simulate realistic processing time
                    time.sleep(0.05)  # 50ms per call
                    
                    # Generate realistic noise prediction
                    noise_pred = torch.randn_like(x, device=x.device) * 0.8
                    
                    # Add timestep scaling
                    if isinstance(timestep, torch.Tensor):
                        t_scale = float(timestep.item()) if timestep.numel() == 1 else float(timestep[0].item())
                    else:
                        t_scale = float(timestep)
                    
                    noise_scale = 0.2 + t_scale * 0.8
                    noise_pred = noise_pred * noise_scale
                    
                    print(f"      Output range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
                    
                    return noise_pred
            
            # Create mock model patcher
            class MockModelPatcher:
                def __init__(self, model):
                    self.model = model
                    self.load_device = get_torch_device()
                    self.offload_device = torch.device("cpu")
            
            mock_model = RealisticMockModel()
            model_patcher = MockModelPatcher(mock_model)
    
    # Create KSampler
    print("\n🔧 Creating KSampler...")
    ksampler = StandaloneKSampler(
        model=model_patcher,
        steps=20,
        device=get_torch_device(),
        sampler="euler",
        scheduler="simple",
        denoise=1.0
    )
    
    # Prepare test data
    print("\n🔧 Preparing test data...")
    noise = torch.randn(1, 16, 11, 104, 60, device=get_torch_device())
    positive_cond = torch.randn(1, 77, 4096)
    negative_cond = torch.randn(1, 77, 4096)
    
    print(f"   Noise shape: {noise.shape}")
    print(f"   Positive conditioning shape: {positive_cond.shape}")
    print(f"   Negative conditioning shape: {negative_cond.shape}")
    
    # Run sampling
    print("\n🚀 Running KSampler sampling...")
    start_time = time.time()
    
    try:
        denoised_latent = ksampler.sample(
            noise=noise,
            positive=positive_cond,
            negative=negative_cond,
            cfg=7.0,
            seed=42
        )
        
        sampling_time = time.time() - start_time
        
        print(f"\n📊 SAMPLING RESULTS:")
        print(f"   Sampling time: {sampling_time:.2f}s")
        print(f"   Denoised latent shape: {denoised_latent.shape}")
        print(f"   Denoised latent range: [{denoised_latent.min():.3f}, {denoised_latent.max():.3f}]")
        
        # Check if we got realistic results
        if sampling_time < 0.5:
            print("⚠️  Sampling was too fast - might not be doing real work")
        else:
            print("✅ Sampling took realistic time")
            
        if hasattr(model_patcher.model, 'call_count'):
            print(f"   Model calls: {model_patcher.model.call_count}")
            if model_patcher.model.call_count < 10:
                print("⚠️  Too few model calls - might not be doing real sampling")
            else:
                print("✅ Realistic number of model calls")
        
        return True
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_ksampler()
