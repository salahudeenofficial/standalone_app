#!/usr/bin/env python3
"""
Test ComfyUI-style memory management with your actual model
"""

import torch
import logging
from memory_utils import safe_model_to_device_advanced, estimate_state_dict_memory, get_memory_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_comfyui_memory_management():
    """Test ComfyUI-style memory management with realistic model"""
    print("🧪 Testing ComfyUI-style Memory Management")
    print("=" * 60)
    
    # Get system info
    info = get_memory_info()
    print(f"📊 System Information:")
    print(f"  Available GPU memory: {info['cuda_free']:.2f} GB")
    print(f"  Total VRAM: {info['cuda_total']:.2f} GB")
    print(f"  Available RAM: {info.get('ram_free', 0):.2f} GB")
    
    # Determine VRAM state based on ComfyUI logic
    total_vram_gb = info['cuda_total']
    if total_vram_gb < 4:
        vram_state = "NO_VRAM"
    elif total_vram_gb < 8:
        vram_state = "LOW_VRAM"
    elif total_vram_gb < 16:
        vram_state = "NORMAL_VRAM"
    else:
        vram_state = "HIGH_VRAM"
        
    print(f"  Detected VRAM state: {vram_state}")
    
    # Test with your 32GB WAN model
    print(f"\n🎯 Testing with WAN 2.1 VACE 16B Model:")
    print(f"  Parameters: 17,337,592,896")
    print(f"  Dtype: torch.float16")
    print(f"  Expected size: ~32.29 GB")
    
    # Create a dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            # Create a model with approximately the right number of parameters
            self.layers = torch.nn.ModuleList()
            for i in range(10):
                layer_size = params // 10
                self.layers.append(torch.nn.Linear(layer_size, layer_size))
                
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Create dummy model with 17B parameters
    dummy_model = DummyModel(17_337_592_896)
    
    # Test memory estimation
    state_dict = dummy_model.state_dict()
    model_info = estimate_state_dict_memory(state_dict)
    
    print(f"\n📏 Memory Estimation:")
    print(f"  Raw model size: {model_info['size_gb']:.2f} GB")
    print(f"  Parameters: {model_info['parameters']:,}")
    print(f"  Keys: {model_info['keys']}")
    
    # Test loading strategy
    print(f"\n🚀 Testing Loading Strategy:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model, final_device, loading_info = safe_model_to_device_advanced(
            dummy_model, 
            device, 
            state_dict=state_dict
        )
        
        print(f"  Loading type: {loading_info['loading_type']}")
        print(f"  Final device: {final_device}")
        print(f"  VRAM state: {loading_info.get('vram_state', 'N/A')}")
        
        if loading_info['loading_type'] == 'full_gpu':
            print(f"  ✅ SUCCESS: Model loaded to GPU!")
            print(f"  Memory used: {loading_info.get('memory_used_gb', 0):.2f} GB")
        else:
            print(f"  📱 Model loaded to CPU with dynamic loading")
            print(f"  Reason: {loading_info.get('reason', 'N/A')}")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    # Test with different model sizes
    print(f"\n🔬 Testing with different model sizes:")
    test_sizes = [1, 5, 10, 20, 30, 40]  # GB
    
    for size_gb in test_sizes:
        # Create dummy state dict with approximate size
        dummy_state_dict = {}
        params_per_gb = 1_000_000_000  # Approximate parameters per GB in FP16
        total_params = int(size_gb * params_per_gb)
        
        # Create a few large tensors
        for i in range(5):
            tensor_size = total_params // 5
            dummy_state_dict[f'layer_{i}'] = torch.randn(tensor_size, dtype=torch.float16)
        
        model_info = estimate_state_dict_memory(dummy_state_dict)
        
        # Test loading decision
        if model_info['size_gb'] < info['cuda_free'] - 1.0:  # Reserve 1GB
            decision = "GPU"
        else:
            decision = "CPU"
            
        print(f"  {size_gb:2d}GB model: {model_info['size_gb']:5.2f}GB -> {decision}")

if __name__ == "__main__":
    test_comfyui_memory_management()
