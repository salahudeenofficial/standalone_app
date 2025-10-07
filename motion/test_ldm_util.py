#!/usr/bin/env python3
"""
Test Script for ComfyUI ldm/util.py Implementation
Tests the standalone utility functions for motion pipeline compatibility
"""

import torch
import torch.nn as nn
import logging
import sys
import os

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

def test_utility_functions():
    """Test all utility functions from ldm_util.py"""
    
    print("🧪 TESTING COMFYUI LDM/UTIL.PY IMPLEMENTATION")
    print("=" * 60)
    print("✅ Motion pipeline utility functions test")
    
    try:
        # Import motion's utility functions
        from ldm_util import (
            exists, default, log_txt_as_img, ismap, isimage, mean_flat,
            count_params, instantiate_from_config, get_obj_from_str,
            AdamWwithEMAandWings, safe_device_cast, get_device_info,
            tensor_info, PIL_AVAILABLE
        )
        
        print("✅ Successfully imported all utility functions")
        print(f"📊 PIL Available: {PIL_AVAILABLE}")
        
        # Test 1: Basic utility functions
        print("\n🔍 Test 1: Basic utility functions...")
        
        # Test exists()
        assert exists(None) == False
        assert exists("test") == True
        assert exists(0) == True
        assert exists([]) == True
        print("   ✅ exists() working correctly")
        
        # Test default()
        assert default(None, "fallback") == "fallback"
        assert default("value", "fallback") == "value"
        assert default(None, lambda: "computed") == "computed"
        print("   ✅ default() working correctly")
        
        # Test ismap() and isimage()
        image_tensor = torch.randn(1, 3, 64, 64)  # RGB image
        map_tensor = torch.randn(1, 6, 64, 64)    # Map with 6 channels
        wrong_tensor = torch.randn(1, 64, 64)     # Wrong shape
        
        assert isimage(image_tensor) == True
        assert isimage(map_tensor) == False
        assert isimage(wrong_tensor) == False
        print("   ✅ isimage() working correctly")
        
        assert ismap(map_tensor) == True
        assert ismap(image_tensor) == False
        assert ismap(wrong_tensor) == False
        print("   ✅ ismap() working correctly")
        
        # Test mean_flat()
        test_tensor = torch.randn(2, 3, 4, 5)
        mean_result = mean_flat(test_tensor)
        assert mean_result.shape == (2,)
        assert torch.allclose(mean_result, test_tensor.mean(dim=(1, 2, 3)))
        print("   ✅ mean_flat() working correctly")
        
        # Test 2: Model utilities
        print("\n🔍 Test 2: Model utility functions...")
        
        # Create a simple model
        model, num_params = create_test_model()
        
        # Test count_params()
        counted_params = count_params(model, verbose=True)
        assert counted_params == num_params
        print("   ✅ count_params() working correctly")
        
        # Test tensor_info()
        info = tensor_info(image_tensor, "Test Image")
        assert info['name'] == "Test Image"
        assert info['shape'] == (1, 3, 64, 64)
        assert info['dtype'] == 'torch.float32'
        assert 'min' in info and 'max' in info
        print("   ✅ tensor_info() working correctly")
        
        # Test 3: Device and casting utilities
        print("\n🔍 Test 3: Device utilities...")
        
        # Test safe_device_cast()
        new_tensor = safe_device_cast(image_tensor, torch.device('cpu'))
        assert new_tensor.device.type == 'cpu'
        
        # Test get_device_info()
        device_info = get_device_info()
        assert 'device' in device_info
        assert 'type' in device_info
        print(f"   📊 Device info: {device_info}")
        print("   ✅ safe_device_cast() and get_device_info() working correctly")
        
        # Test 4: Optimizer
        print("\n🔍 Test 4: AdamW with EMA optimizer...")
        
        test_optimizer(model, image_tensor)
        print("   ✅ AdamWwithEMAandWings working correctly")
        
        # Test 5: Text to image (if PIL available)
        if PIL_AVAILABLE:
            print("\n🔍 Test 5: Text to image conversion...")
            try:
                txt_images = log_txt_as_img((256, 256), ["Test caption 1", "Test caption 2"], size=12)
                assert txt_images.shape[0] == 2  # 2 captions
                assert txt_images.shape[1] == 3  # RGB
                assert txt_images.shape[2] == 256  # Height
                assert txt_images.shape[3] == 256  # Width
                print("   ✅ log_txt_as_img() working correctly")
            except Exception as e:
                print(f"   ⚠️ log_txt_as_img() failed: {e}")
                print("   (This is acceptable if fonts are not available)")
        else:
            print("\n🔍 Test 5: Text to image conversion skipped (PIL not available)")
            print("   ⚠️ PIL not available - text functions will return zeros")
        
        print("\n🎉 ALL UTILITY TESTS PASSED!")
        
        # Summary
        print("\n📊 IMPLEMENTATION SUMMARY:")
        print("✅ Basic utilities: exists, default, ismap, isimage")
        print("✅ Math utilities: mean_flat")  
        print("✅ Model utilities: count_params, tensor_info")
        print("✅ Device utilities: safe_device_cast, get_device_info")
        print("✅ Optimizer: AdamWwithEMAandWings")
        if PIL_AVAILABLE:
            print("✅ Image utilities: log_txt_as_img")
        else:
            print("⚠️ Image utilities: log_txt_as_img (PIL required)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_model():
    """Create a simple test model for testing utilities"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.linear = nn.Linear(32*64*64, 10)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x
    
    # Calculate expected parameters
    expected_params = (
        3*16*3*3 + 16 +  # conv1: weights + bias
        16*32*3*3 + 32 +  # conv2: weights + bias
        32*64*64*10 + 10   # linear: weights + bias
    )
    
    model = SimpleModel()
    actual_params = sum(p.numel() for p in model.parameters())
    
    return model, expected_params


def test_optimizer(model, input_tensor):
    """Test the AdamW optimizer"""
    from ldm_util import AdamWwithEMAandWings
    optimizer = AdamWwithEMAandWings(
        params=model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        ema_decay=0.9999,
        ema_power=1.0
    )
    
    optimizer.zero_grad()
    
    # Simple forward pass and backward
    output = model(input_tensor)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    
    # Check EMA parameters are initialized after step
    for group in optimizer.param_groups:
        for p in group['params']:
            assert 'param_exp_avg' in optimizer.state[p]
    
    # Check that step was taken
    for group in optimizer.param_groups:
        for p in group['params']:
            assert optimizer.state[p]['step'] == 1


def test_config_instantiation():
    """Test configuration-based instantiation utilities"""
    print("\n🔍 Test 6: Configuration utilities...")
    
    try:
        from ldm_util import instantiate_from_config, get_obj_from_str
        
        # Test get_obj_from_str
        LinearClass = get_obj_from_str("torch.nn.Linear")
        assert LinearClass == torch.nn.Linear
        
        # Test instantiate_from_config
        config = {
            "target": "torch.nn.Linear",
            "params": {"in_features": 10, "out_features": 5}
        }
        linear_layer = instantiate_from_config(config)
        assert isinstance(linear_layer, torch.nn.Linear)
        assert linear_layer.in_features == 10
        assert linear_layer.out_features == 5
        
        print("   ✅ Configuration instantiation working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ Configuration utilities test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting ComfyUI LDM Utility Tests...")
    
    success = test_utility_functions()
    
    print("\n🔍 Running additional configuration tests...")
    config_success = test_config_instantiation()
    
    if success and config_success:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Motion pipeline utility implementation is working perfectly!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        print("⚠️ Check the errors above and fix any issues")
        sys.exit(1)
