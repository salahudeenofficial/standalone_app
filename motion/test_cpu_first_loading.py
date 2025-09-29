#!/usr/bin/env python3
"""
Test ComfyUI-style memory management with CPU-first loading and patcher assignment
"""

import torch
import logging
from memory_utils import safe_model_to_device_advanced, estimate_state_dict_memory, get_memory_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_comfyui_cpu_first_loading():
    """Test ComfyUI-style CPU-first loading with patcher assignment"""
    print("🧪 Testing ComfyUI-style CPU-first Loading with Patcher Assignment")
    print("=" * 70)
    
    # Get system info
    info = get_memory_info()
    print(f"📊 System Information:")
    print(f"  Available GPU memory: {info['cuda_free']:.2f} GB")
    print(f"  Total VRAM: {info['cuda_total']:.2f} GB")
    
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
    
    # Create a simple dummy model for testing
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(1000, 1000)
            self.layer2 = torch.nn.Linear(1000, 1000)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            return x
    
    # Create dummy model
    dummy_model = DummyModel()
    
    # Test memory estimation
    state_dict = dummy_model.state_dict()
    model_info = estimate_state_dict_memory(state_dict)
    
    print(f"\n📏 Memory Estimation:")
    print(f"  Raw model size: {model_info['size_gb']:.2f} GB")
    print(f"  Parameters: {model_info['parameters']:,}")
    print(f"  Keys: {model_info['keys']}")
    
    # Test loading strategy
    print(f"\n🚀 Testing ComfyUI-style Loading Strategy:")
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
        print(f"  Patcher type: {loading_info.get('patcher_type', 'N/A')}")
        
        if loading_info['loading_type'] == 'full_gpu':
            print(f"  ✅ SUCCESS: Model loaded to GPU with complete patcher!")
            print(f"  Memory used: {loading_info.get('memory_used_gb', 0):.2f} GB")
        elif loading_info['loading_type'] == 'cpu_first_partial':
            print(f"  🔧 SUCCESS: Model loaded to CPU with partial patcher!")
            print(f"  Modules available: {loading_info.get('modules_available', 0)}")
            print(f"  Memory budget: {loading_info.get('memory_budget_gb', 0):.2f} GB")
        else:
            print(f"  📱 Model loaded to CPU with CPU-only patcher")
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
            decision = "GPU (complete patcher)"
        else:
            decision = "CPU (partial patcher)"
            
        print(f"  {size_gb:2d}GB model: {model_info['size_gb']:5.2f}GB -> {decision}")
    
    # Test what would happen on VAST AI with 44GB VRAM
    print(f"\n🌐 Simulating VAST AI (44GB VRAM):")
    vast_ai_info = {
        'cuda_free': 44.0,
        'cuda_total': 44.0
    }
    
    # Determine VRAM state for VAST AI
    if vast_ai_info['cuda_total'] < 4:
        vast_vram_state = "NO_VRAM"
    elif vast_ai_info['cuda_total'] < 8:
        vast_vram_state = "LOW_VRAM"
    elif vast_ai_info['cuda_total'] < 16:
        vast_vram_state = "NORMAL_VRAM"
    else:
        vast_vram_state = "HIGH_VRAM"
        
    print(f"  VRAM state: {vast_vram_state}")
    print(f"  Available memory: {vast_ai_info['cuda_free']:.2f} GB")
    
    # Test 32GB model on VAST AI
    wan_model_size = 32.29  # GB
    if vast_vram_state == "HIGH_VRAM":
        if wan_model_size < vast_ai_info['cuda_free'] - 1.0:
            decision = "GPU (complete patcher)"
        else:
            decision = "CPU (partial patcher)"
    elif wan_model_size < vast_ai_info['cuda_free'] - 1.0:
        decision = "GPU (complete patcher)"
    else:
        decision = "CPU (partial patcher)"
        
    print(f"  32GB WAN model: {wan_model_size:.2f}GB -> {decision}")
    
    print(f"\n🎯 Key Benefits of ComfyUI-style CPU-first Loading:")
    print(f"  ✅ Models always start on CPU (ComfyUI approach)")
    print(f"  ✅ Automatic patcher type assignment based on memory")
    print(f"  ✅ Complete patcher for models that fit in GPU")
    print(f"  ✅ Partial patcher for large models with dynamic loading")
    print(f"  ✅ CPU-only patcher when GPU not available")
    print(f"  ✅ Memory-efficient with aggressive CUDA cache clearing")

if __name__ == "__main__":
    test_comfyui_cpu_first_loading()
