#!/usr/bin/env python3
"""
Test the fixed memory calculation for large models
"""

from memory_utils import estimate_state_dict_memory, get_memory_info, safe_model_to_device_advanced
import torch

def test_memory_estimation():
    """Test memory estimation for 32GB model"""
    print("🧪 Testing Fixed Memory Estimation")
    print("=" * 50)
    
    # Simulate your 32GB model state dict
    fake_state_dict = {}
    
    # Create a realistic 32GB model simulation
    # WAN 2.1 VACE has ~17B parameters in FP16 = ~32GB
    total_params_target = 17_000_000_000  # 17B parameters
    param_size_fp16 = 2  # 2 bytes per FP16 parameter
    
    # Create a few large tensors to simulate the model
    params_per_tensor = total_params_target // 10  # Split into 10 tensors
    for i in range(10):
        tensor_shape = (params_per_tensor,)
        fake_state_dict[f'large_tensor_{i}'] = torch.randn(tensor_shape, dtype=torch.float16)
    
    # Test the estimation
    print("📊 Model Simulation:")
    total_params = sum(t.numel() for t in fake_state_dict.values())
    total_size_gb = sum(t.numel() * t.element_size() for t in fake_state_dict.values()) / 1024**3
    print(f"   Total parameters: {total_params:,}")
    print(f"   Raw size: {total_size_gb:.2f} GB")
    print()
    
    # Test the new estimation function
    estimation = estimate_state_dict_memory(fake_state_dict)
    
    print("🔍 Memory Estimation Results:")
    print(f"   Raw model size: {estimation['size_gb']:.2f} GB")
    print(f"   GPU overhead multiplier: {estimation['gpu_multiplier']:.1f}x")
    print(f"   Estimated GPU memory: {estimation['size_gb_gpu']:.2f} GB")
    print(f"   CPU overhead multiplier: {estimation['cpu_multiplier']:.1f}x")
    print(f"   Estimated CPU memory: {estimation['size_gb_cpu']:.2f} GB")
    print()
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_info = get_memory_info()
        print("🎮 GPU Information:")
        print(f"   Available GPU memory: {gpu_info['cuda_free']:.2f} GB")
        print(f"   Total GPU memory: {gpu_info['cuda_total']:.2f} GB")
        print()
        
        # Test loading decision
        memory_budget = gpu_info['cuda_free'] - 3.0  # 3GB buffer
        will_fit = estimation['size_gb_gpu'] <= memory_budget
        
        print("🎯 Loading Decision:")
        print(f"   Memory budget: {memory_budget:.2f} GB")
        print(f"   Required memory: {estimation['size_gb_gpu']:.2f} GB")
        print(f"   Will fit on GPU: {'✅ YES' if will_fit else '❌ NO'}")
        print()
        
        if will_fit:
            print("🚀 RESULT: Model should load on GPU!")
        else:
            print("📱 RESULT: Model will load on CPU")
            print(f"   Need: {estimation['size_gb_gpu'] - memory_budget:.2f} GB more")
    else:
        print("❌ CUDA not available")

if __name__ == "__main__":
    test_memory_estimation()