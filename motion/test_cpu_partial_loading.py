#!/usr/bin/env python3
"""
Test ComfyUI-style partial loading on CPU (no CUDA required)
Focuses on the core logic and weight_function setup
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_style_partial_loading import ComfyUIStylePartialLoader
from model_aware_patcher import ModelAwarePatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_test_model():
    """Create a simple test model"""
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 32 * 32, 1000)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return TestModel()

def test_comfyui_style_partial_loading():
    """Test ComfyUI-style partial loading on CPU"""
    print("🧪 TESTING COMFYUI-STYLE PARTIAL LOADING (CPU)")
    print("=" * 60)
    
    # Create test model
    model = create_test_model()
    print(f"📊 Created test model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test with CPU device (no CUDA required)
    device = torch.device('cpu')
    memory_budget_gb = 0.1  # Small budget for testing
    
    try:
        # Initialize ComfyUI-style partial loader
        print(f"🚀 Initializing ComfyUI-style partial loader...")
        print(f"   Device: {device}")
        print(f"   Memory budget: {memory_budget_gb} GB")
        
        loader = ComfyUIStylePartialLoader(model, device, memory_budget_gb)
        
        # Get loading info
        info = loader.get_loading_info()
        print(f"✅ Partial loading setup complete:")
        print(f"   Loading type: {info['loading_type']}")
        print(f"   Loaded weights: {info['loaded_weights_count']}")
        print(f"   Patched weights: {info['patched_weights_count']}")
        print(f"   Memory used: {info['memory_used_gb']:.3f} GB")
        print(f"   Memory budget: {info['memory_budget_gb']:.3f} GB")
        
        # Test weight_function setup
        print(f"\n🔧 Testing weight_function setup...")
        weight_function_count = 0
        bias_function_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight_function') and module.weight_function:
                weight_function_count += len(module.weight_function)
                print(f"   ✅ {name}: {len(module.weight_function)} weight functions")
            if hasattr(module, 'bias_function') and module.bias_function:
                bias_function_count += len(module.bias_function)
                print(f"   ✅ {name}: {len(module.bias_function)} bias functions")
        
        print(f"📊 Total weight functions: {weight_function_count}")
        print(f"📊 Total bias functions: {bias_function_count}")
        
        # Test inference preparation (should NOT move model to GPU)
        print(f"\n🔄 Testing inference preparation...")
        loader.load_weights_for_inference()
        
        # Verify model is still on CPU
        model_device = next(model.parameters()).device
        print(f"   Model device: {model_device}")
        
        if str(model_device) == 'cpu':
            print(f"   ✅ Model correctly stays on CPU (ComfyUI-style)")
        else:
            print(f"   ❌ Model moved to {model_device} (not ComfyUI-style)")
            return False
        
        # Test cleanup
        print(f"\n🧹 Testing cleanup...")
        loader.evict_weights_after_inference()
        
        # Verify weight_function lists are cleared
        remaining_weight_functions = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight_function') and module.weight_function:
                remaining_weight_functions += len(module.weight_function)
            if hasattr(module, 'bias_function') and module.bias_function:
                remaining_weight_functions += len(module.bias_function)
        
        print(f"   Remaining weight functions: {remaining_weight_functions}")
        
        if remaining_weight_functions == 0:
            print(f"   ✅ Weight functions properly cleared")
        else:
            print(f"   ❌ {remaining_weight_functions} weight functions not cleared")
            return False
        
        print(f"\n🎉 ComfyUI-style partial loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_aware_patching():
    """Test model-aware patching"""
    print(f"\n🧪 TESTING MODEL-AWARE PATCHING")
    print("=" * 60)
    
    # Create test model
    model = create_test_model()
    
    try:
        # Test model-aware patching
        patcher = ModelAwarePatcher()
        patched_model = patcher.patch_model(model)
        
        # Count patched layers
        patched_count = 0
        for name, module in patched_model.named_modules():
            if hasattr(module, 'weight_function'):
                patched_count += 1
        
        print(f"✅ Model-aware patching complete:")
        print(f"   Patched layers: {patched_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model-aware patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMFYUI-STYLE PARTIAL LOADING TEST (CPU)")
    print("=" * 80)
    
    # Test ComfyUI-style partial loading
    test1_passed = test_comfyui_style_partial_loading()
    
    # Test model-aware patching
    test2_passed = test_model_aware_patching()
    
    # Final results
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   ComfyUI-style partial loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Model-aware patching: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ComfyUI-style partial loading is working correctly!")
        print(f"   Model stays on CPU, weights loaded on-demand via weight_function")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Need to fix issues before production use!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
