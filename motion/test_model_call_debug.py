#!/usr/bin/env python3
"""
Test script to debug KSampler model calls
"""

import torch
import time
import sys
import os

# Add motion directory to path
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

from standalone_ksampler import StandaloneKSampler, StandaloneCFGGuider
from wan_vae_components.model_management import get_torch_device

def test_model_call():
    """Test if the model is actually being called"""
    print("🧪 Testing Model Call Debug")
    print("="*50)
    
    # Create a mock model that logs calls
    class DebugModel:
        def __init__(self):
            self.call_count = 0
            self.device = get_torch_device()
            
        def forward(self, x, timestep, *args, **kwargs):
            self.call_count += 1
            print(f"   🔍 Model.forward() called #{self.call_count}")
            print(f"      Input shape: {x.shape}")
            print(f"      Timestep: {timestep}")
            print(f"      Device: {x.device}")
            
            # Generate realistic noise prediction
            noise_pred = torch.randn_like(x, device=x.device) * 0.8
            
            # Add timestep scaling
            if isinstance(timestep, torch.Tensor):
                t_scale = float(timestep.item()) if timestep.numel() == 1 else float(timestep[0].item())
            else:
                t_scale = float(timestep)
            
            noise_scale = 0.2 + t_scale * 0.8
            noise_pred = noise_pred * noise_scale
            
            print(f"      Output shape: {noise_pred.shape}")
            print(f"      Output range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
            
            return noise_pred
    
    # Create mock model patcher
    class MockModelPatcher:
        def __init__(self, model):
            self.model = model
            self.load_device = get_torch_device()
            self.offload_device = torch.device("cpu")
    
    # Test setup
    debug_model = DebugModel()
    model_patcher = MockModelPatcher(debug_model)
    
    # Create CFG Guider
    cfg_guider = StandaloneCFGGuider(model_patcher)
    
    # Set up conditioning
    positive_cond = torch.randn(1, 77, 4096)
    negative_cond = torch.randn(1, 77, 4096)
    cfg_guider.set_conds(positive_cond, negative_cond)
    cfg_guider.set_cfg(7.0)
    
    # Test noise prediction
    print("\n🔍 Testing noise prediction...")
    test_x = torch.randn(1, 16, 11, 104, 60, device=get_torch_device())
    test_timestep = torch.tensor([0.5], device=get_torch_device())
    
    start_time = time.time()
    noise_pred = cfg_guider.predict_noise(test_x, test_timestep)
    prediction_time = time.time() - start_time
    
    print(f"\n📊 RESULTS:")
    print(f"   Model calls: {debug_model.call_count}")
    print(f"   Prediction time: {prediction_time:.3f}s")
    print(f"   Output shape: {noise_pred.shape}")
    print(f"   Output range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
    
    if debug_model.call_count == 0:
        print("❌ Model was never called!")
        return False
    else:
        print("✅ Model was called successfully!")
        return True

if __name__ == "__main__":
    test_model_call()
