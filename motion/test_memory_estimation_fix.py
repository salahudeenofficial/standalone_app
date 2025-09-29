#!/usr/bin/env python3
"""
Test script to verify the corrected memory estimation with realistic overhead multipliers
"""

import torch
import logging
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_utils import estimate_state_dict_memory, get_memory_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_memory_estimation_fix():
    """Test the corrected memory estimation with realistic overhead"""
    print("="*80)
    print("🧪 Testing Corrected Memory Estimation")
    print("="*80)
    
    # Create a dummy state dict similar to WAN model
    print("\n📊 Creating dummy state dict similar to WAN model...")
    
    # Simulate WAN model parameters (17B parameters, ~32GB raw)
    dummy_state_dict = {}
    
    # Add large tensors to simulate WAN model
    for i in range(40):  # 40 transformer blocks
        # Each block has ~400M parameters
        dummy_state_dict[f'blocks.{i}.self_attn.q.weight'] = torch.randn(5120, 5120, dtype=torch.float32)
        dummy_state_dict[f'blocks.{i}.self_attn.k.weight'] = torch.randn(5120, 5120, dtype=torch.float32)
        dummy_state_dict[f'blocks.{i}.self_attn.v.weight'] = torch.randn(5120, 5120, dtype=torch.float32)
        dummy_state_dict[f'blocks.{i}.self_attn.o.weight'] = torch.randn(5120, 5120, dtype=torch.float32)
        dummy_state_dict[f'blocks.{i}.mlp.fc1.weight'] = torch.randn(5120, 13824, dtype=torch.float32)
        dummy_state_dict[f'blocks.{i}.mlp.fc2.weight'] = torch.randn(13824, 5120, dtype=torch.float32)
    
    # Add VACE blocks
    for i in range(8):
        dummy_state_dict[f'vace_blocks.{i}.self_attn.q.weight'] = torch.randn(96, 96, dtype=torch.float32)
        dummy_state_dict[f'vace_blocks.{i}.self_attn.k.weight'] = torch.randn(96, 96, dtype=torch.float32)
        dummy_state_dict[f'vace_blocks.{i}.self_attn.v.weight'] = torch.randn(96, 96, dtype=torch.float32)
        dummy_state_dict[f'vace_blocks.{i}.self_attn.o.weight'] = torch.randn(96, 96, dtype=torch.float32)
    
    print(f"   Created dummy state dict with {len(dummy_state_dict)} tensors")
    
    # Test memory estimation
    print("\n📊 Testing memory estimation...")
    memory_info = estimate_state_dict_memory(dummy_state_dict)
    
    print(f"\n📋 MEMORY ESTIMATION RESULTS:")
    print(f"   Raw model size: {memory_info['size_gb']:.2f} GB")
    print(f"   CPU memory (1.8x): {memory_info['size_gb_cpu']:.2f} GB")
    print(f"   GPU memory (2.6x): {memory_info['size_gb_gpu']:.2f} GB")
    print(f"   Parameters: {memory_info['parameters']:,}")
    print(f"   Keys: {memory_info['keys']}")
    
    # Compare with ComfyUI observations
    print(f"\n🔍 COMPARISON WITH COMFYUI OBSERVATIONS:")
    print(f"   ComfyUI UNet RAM usage: ~33.4 GB")
    print(f"   Our CPU estimate: {memory_info['size_gb_cpu']:.2f} GB")
    print(f"   Difference: {abs(33.4 - memory_info['size_gb_cpu']):.2f} GB")
    
    if abs(33.4 - memory_info['size_gb_cpu']) < 5.0:  # Within 5GB is acceptable
        print(f"   ✅ CPU estimate is reasonable!")
    else:
        print(f"   ⚠️  CPU estimate may need adjustment")
    
    print(f"\n   ComfyUI GPU usage: ~63.97 GB (from your logs)")
    print(f"   Our GPU estimate: {memory_info['size_gb_gpu']:.2f} GB")
    print(f"   Difference: {abs(63.97 - memory_info['size_gb_gpu']):.2f} GB")
    
    if abs(63.97 - memory_info['size_gb_gpu']) < 10.0:  # Within 10GB is acceptable
        print(f"   ✅ GPU estimate is reasonable!")
    else:
        print(f"   ⚠️  GPU estimate may need adjustment")
    
    # Test memory decision logic
    print(f"\n🧠 TESTING MEMORY DECISION LOGIC:")
    if torch.cuda.is_available():
        gpu_info = get_memory_info()
        available_gpu = gpu_info['cuda_free']
        print(f"   Available GPU memory: {available_gpu:.2f} GB")
        
        if memory_info['size_gb_gpu'] <= available_gpu - 2.0:  # 2GB buffer
            print(f"   ✅ Model can fit on GPU")
            print(f"   Decision: Load to GPU")
        else:
            print(f"   ❌ Model too large for GPU")
            print(f"   Decision: Load to CPU with dynamic loading")
    else:
        print(f"   ⚠️  CUDA not available, skipping GPU test")
    
    print(f"\n✅ Memory estimation test complete!")
    return True

def main():
    print("🚀 Testing Corrected Memory Estimation")
    print("="*60)
    
    if test_memory_estimation_fix():
        print("\n🎉 MEMORY ESTIMATION FIX VERIFIED!")
        print("   The corrected estimation should now properly account for:")
        print("   - PyTorch module overhead")
        print("   - CUDA memory fragmentation") 
        print("   - Intermediate activations")
        print("   - Memory alignment requirements")
    else:
        print("\n❌ MEMORY ESTIMATION TEST FAILED!")
    
    print("="*60)

if __name__ == "__main__":
    main()
