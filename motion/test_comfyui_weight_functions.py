#!/usr/bin/env python3
"""
Test ComfyUI-style partial loading with weight_function approach
"""

import torch
import torch.nn as nn
import logging
from comfyui_style_partial_loading import ComfyUIStylePartialLoader, LowVramPatch
from comfyui_ops import ComfyUILinear, ComfyUIConv2d

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_comfyui_weight_functions():
    """Test ComfyUI's weight_function approach"""
    print("🧪 Testing ComfyUI-style weight_function approach...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ComfyUIConv2d(3, 64, 3, padding=1)
            self.linear = ComfyUILinear(64, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=[2, 3])  # Global average pooling
            x = self.linear(x)
            return x
    
    model = SimpleModel()
    print(f"✅ Created model with ComfyUI-style layers")
    
    # Test input
    input_tensor = torch.randn(1, 3, 32, 32)
    print(f"📊 Input shape: {input_tensor.shape}")
    
    # Test without weight_function (should work normally)
    print("\n🔧 Testing normal forward pass...")
    output1 = model(input_tensor)
    print(f"✅ Normal forward pass: {output1.shape}")
    
    # Test with weight_function (ComfyUI approach)
    print("\n🔧 Testing with weight_function...")
    
    # Create LowVramPatch for conv weight
    conv_weight = model.conv.weight.data.clone()
    conv_patch = LowVramPatch("conv.weight", conv_weight, torch.device('cuda'))
    
    # Add to weight_function list
    model.conv.weight_function.append(conv_patch)
    print(f"✅ Added LowVramPatch to conv.weight_function")
    
    # Test forward pass with weight_function
    input_cuda = input_tensor.cuda()
    model = model.cuda()
    
    print("🚀 Running forward pass with weight_function...")
    output2 = model(input_cuda)
    print(f"✅ Forward pass with weight_function: {output2.shape}")
    
    # Verify weights were loaded to GPU
    if conv_patch.is_loaded:
        print("✅ Weight was loaded to GPU during forward pass!")
    else:
        print("❌ Weight was not loaded to GPU")
    
    # Test eviction
    print("\n🧹 Testing eviction...")
    conv_patch.evict()
    if not conv_patch.is_loaded:
        print("✅ Weight was evicted from GPU!")
    else:
        print("❌ Weight was not evicted")
    
    print("\n🎉 ComfyUI-style weight_function test completed!")

def test_partial_loading_integration():
    """Test integration with partial loading system"""
    print("\n🧪 Testing partial loading integration...")
    
    # Create a simple model
    model = nn.Sequential(
        ComfyUIConv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        ComfyUILinear(64, 10)
    )
    
    # Create partial loader
    loader = ComfyUIStylePartialLoader(model, torch.device('cuda'), memory_budget_gb=0.1)
    
    # Test input
    input_tensor = torch.randn(1, 3, 8, 8).cuda()
    
    # Test inference
    print("🚀 Testing inference with partial loading...")
    loader.load_weights_for_inference()
    
    model = model.cuda()
    output = model(input_tensor)
    print(f"✅ Inference completed: {output.shape}")
    
    # Cleanup
    loader.evict_weights_after_inference()
    print("✅ Cleanup completed")
    
    print("\n🎉 Partial loading integration test completed!")

if __name__ == "__main__":
    print("🚀 ComfyUI-style Weight Function Tests")
    print("=" * 50)
    
    try:
        test_comfyui_weight_functions()
        test_partial_loading_integration()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
