#!/usr/bin/env python3
"""
Quick test to verify memory calculation accuracy
"""

import torch
import sys
import os

# Add motion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_torch_file
from memory_utils import estimate_state_dict_memory

def test_memory_calculation():
    """Test memory calculation accuracy"""
    print("🧪 TESTING MEMORY CALCULATION ACCURACY")
    print("="*60)
    
    # Load the WAN model state dict
    model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"📊 Loading model from: {model_path}")
    state_dict = load_torch_file(model_path)
    
    print(f"📊 State dict keys: {len(state_dict)}")
    
    # Calculate memory usage
    memory_info = estimate_state_dict_memory(state_dict)
    
    print(f"\n📊 MEMORY ANALYSIS:")
    print(f"   Parameters: {memory_info['parameters']:,}")
    print(f"   Size: {memory_info['size_gb']:.3f} GB")
    print(f"   Keys: {memory_info['keys']}")
    
    # Calculate expected size (17B parameters * 4 bytes per float32)
    expected_size_gb = (17_337_592_896 * 4) / (1024**3)
    print(f"\n📊 EXPECTED SIZE:")
    print(f"   17.337B parameters * 4 bytes = {expected_size_gb:.3f} GB")
    
    # Check if our calculation is reasonable
    ratio = memory_info['size_gb'] / expected_size_gb
    print(f"\n📊 CALCULATION RATIO:")
    print(f"   Actual / Expected = {ratio:.3f}")
    
    if 0.8 <= ratio <= 1.2:
        print("   ✅ Memory calculation looks accurate")
        return True
    else:
        print("   ❌ Memory calculation seems off")
        return False

if __name__ == "__main__":
    success = test_memory_calculation()
    sys.exit(0 if success else 1)
