#!/usr/bin/env python3
"""
Test script for the new real WAN model loading
"""

import sys
import os
import torch
import logging

# Add motion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_detection():
    """Test the new model detection functionality"""
    print("🔍 Testing Model Detection...")
    
    try:
        from model_detection import detect_unet_config, model_config_from_unet_config, detect_model_type_from_state_dict
        
        # Create a dummy state dict that looks like a WAN21_Vace model
        dummy_state_dict = {
            'head.modulation': torch.randn(1, 2, 2048),
            'head.head.weight': torch.randn(64, 2048),  # 16 * 4 = 64
            'blocks.0.ffn.0.weight': torch.randn(8192, 2048),
            'blocks.1.ffn.0.weight': torch.randn(8192, 2048),
            'patch_embedding.weight': torch.randn(2048, 16, 1, 2, 2),
            'vace_patch_embedding.weight': torch.randn(2048, 32, 1, 2, 2),  # VACE specific
            'vace_blocks.0.norm1.weight': torch.randn(2048),
            'vace_blocks.1.norm1.weight': torch.randn(2048),
        }
        
        # Test detection
        unet_config = detect_unet_config(dummy_state_dict)
        print(f"   ✅ Detected UNet config: {unet_config}")
        
        if unet_config:
            model_config = model_config_from_unet_config(unet_config)
            print(f"   ✅ Converted to model config: {model_config}")
            
            model_type = detect_model_type_from_state_dict(dummy_state_dict)
            print(f"   ✅ Detected model type: {model_type}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test creating a real WAN model"""
    print("\n🔧 Testing Model Creation...")
    
    try:
        from model_detection import create_model_from_config
        from wan_model import VaceWanModel
        
        # Test config for VaceWanModel
        model_config = {
            "image_model": "wan2.1",
            "model_type": "vace",
            "patch_size": (1, 2, 2),
            "text_len": 512,
            "in_dim": 16,
            "dim": 2048,
            "ffn_dim": 8192,
            "freq_dim": 256,
            "text_dim": 4096,
            "out_dim": 16,
            "num_heads": 16,
            "num_layers": 32,
            "window_size": (-1, -1),
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
            "vace_layers": 2,
            "vace_in_dim": 32,
        }
        
        # Create model
        model = create_model_from_config(model_config)
        print(f"   ✅ Created model: {type(model).__name__}")
        
        # Test forward pass with dummy inputs
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy inputs
        x = torch.randn(1, 16, 11, 104, 60, device=device)  # [B, C, T, H, W]
        t = torch.tensor([0.5], device=device)
        context = torch.randn(1, 77, 4096, device=device)  # Text conditioning
        vace_context = torch.randn(1, 32, 11, 104, 60, device=device)  # VACE conditioning
        vace_strength = [1.0, 1.0]  # VACE strength
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, t, context, vace_context=vace_context, vace_strength=vace_strength)
            print(f"   ✅ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test the updated model loading function"""
    print("\n📦 Testing Model Loading...")
    
    try:
        from standalone_sd import load_state_dict_guess_config
        
        # Create a dummy state dict
        dummy_state_dict = {
            'head.modulation': torch.randn(1, 2, 2048),
            'head.head.weight': torch.randn(64, 2048),
            'blocks.0.ffn.0.weight': torch.randn(8192, 2048),
            'blocks.1.ffn.0.weight': torch.randn(8192, 2048),
            'patch_embedding.weight': torch.randn(2048, 16, 1, 2, 2),
            'vace_patch_embedding.weight': torch.randn(2048, 32, 1, 2, 2),
            'vace_blocks.0.norm1.weight': torch.randn(2048),
            'vace_blocks.1.norm1.weight': torch.randn(2048),
        }
        
        # Test loading
        model_patcher, clip, vae, clipvision = load_state_dict_guess_config(
            dummy_state_dict, 
            output_model=True,
            output_clip=False,
            output_vae=False
        )
        
        print(f"   ✅ Model loaded: {type(model_patcher).__name__}")
        print(f"   ✅ Model patcher created: {model_patcher is not None}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Real WAN Model Implementation")
    print("="*50)
    
    tests = [
        ("Model Detection", test_model_detection),
        ("Model Creation", test_model_creation),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Real WAN model implementation is working correctly")
        print("✅ Ready to test with real model files")
    else:
        print("❌ Some tests failed - check the errors above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
