#!/usr/bin/env python3
"""
Test script to debug the actual pipeline model
"""

import torch
import time
import sys
import os

# Add motion directory to path
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

def test_pipeline_model():
    """Test the actual model loaded by the pipeline"""
    print("🧪 Testing Pipeline Model")
    print("="*50)
    
    from pipeline import WanVideoPipeline
    from standalone_sd import load_state_dict_guess_config, load_torch_file
    
    # Initialize pipeline
    pipeline = WanVideoPipeline()
    
    # Check if we have model files
    unet_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    if not os.path.exists(unet_path):
        print("❌ UNet model file not found, creating mock model...")
        
        # Create a mock model that behaves like the real one
        class MockWANModel:
            def __init__(self):
                self.call_count = 0
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            def forward(self, x, timestep, *args, **kwargs):
                self.call_count += 1
                print(f"   🔍 MockWANModel.forward() called #{self.call_count}")
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
                self.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.offload_device = torch.device("cpu")
        
        mock_model = MockWANModel()
        model_patcher = MockModelPatcher(mock_model)
        
    else:
        print("✅ UNet model file found, loading real model...")
        try:
            # Load UNet state dict
            unet_state_dict = load_torch_file(unet_path)
            print(f"   📊 Loaded UNet state dict with {len(unet_state_dict)} keys")
            
            # Load UNet model using standalone_sd
            result = load_state_dict_guess_config(
                unet_state_dict,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is None:
                raise RuntimeError("Failed to load UNet model")
            
            model_patcher, _, _, _ = result
            
            print(f"✅ UNet loaded successfully")
            print(f"   Type: {type(model_patcher).__name__}")
            print(f"   Model type: {type(model_patcher.model).__name__}")
            print(f"   Device: {model_patcher.load_device}")
            
        except Exception as e:
            print(f"❌ Failed to load real model: {e}")
            print("Using mock model instead...")
            
            # Create mock model
            class MockWANModel:
                def __init__(self):
                    self.call_count = 0
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                def forward(self, x, timestep, *args, **kwargs):
                    self.call_count += 1
                    print(f"   🔍 MockWANModel.forward() called #{self.call_count}")
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
                    self.load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.offload_device = torch.device("cpu")
            
            mock_model = MockWANModel()
            model_patcher = MockModelPatcher(mock_model)
    
    # Test the model directly
    print("\n🔍 Testing model directly...")
    test_x = torch.randn(1, 16, 11, 104, 60, device=model_patcher.load_device)
    test_timestep = torch.tensor([0.5], device=model_patcher.load_device)
    
    start_time = time.time()
    try:
        result = model_patcher.model.forward(test_x, test_timestep)
        direct_time = time.time() - start_time
        
        print(f"✅ Direct model call successful")
        print(f"   Time: {direct_time:.3f}s")
        print(f"   Result shape: {result.shape}")
        print(f"   Result range: [{result.min():.3f}, {result.max():.3f}]")
        
        if hasattr(model_patcher.model, 'call_count'):
            print(f"   Model calls: {model_patcher.model.call_count}")
        
    except Exception as e:
        print(f"❌ Direct model call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test through CFG Guider
    print("\n🔍 Testing model through CFG Guider...")
    from standalone_ksampler import StandaloneCFGGuider
    
    cfg_guider = StandaloneCFGGuider(model_patcher)
    positive_cond = torch.randn(1, 77, 4096)
    negative_cond = torch.randn(1, 77, 4096)
    cfg_guider.set_conds(positive_cond, negative_cond)
    cfg_guider.set_cfg(7.0)
    
    start_time = time.time()
    try:
        cfg_result = cfg_guider.predict_noise(test_x, test_timestep)
        cfg_time = time.time() - start_time
        
        print(f"✅ CFG Guider call successful")
        print(f"   Time: {cfg_time:.3f}s")
        print(f"   Result shape: {cfg_result.shape}")
        print(f"   Result range: [{cfg_result.min():.3f}, {cfg_result.max():.3f}]")
        
        if hasattr(model_patcher.model, 'call_count'):
            print(f"   Total model calls: {model_patcher.model.call_count}")
        
    except Exception as e:
        print(f"❌ CFG Guider call failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_pipeline_model()
