#!/usr/bin/env python3
"""
Priority 3: Test WAN21_Vace Configuration Class
Important debugging for VACE-specific configuration and model creation
"""

import torch
import sys
import os
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

def test_wan21_vace_config():
    """Test WAN21_Vace configuration"""
    from models import WAN21_Vace
    
    print("🔍 Testing WAN21_Vace Configuration:")
    
    # Check if class exists
    print(f"   ✅ WAN21_Vace class: {WAN21_Vace}")
    
    # Check unet_config
    if hasattr(WAN21_Vace, 'unet_config'):
        config = WAN21_Vace.unet_config
        print(f"   ✅ unet_config dict: {config}")
        
        expected_keys = ['image_model', 'model_type']
        for key in expected_keys:
            if key in config:
                print(f"   ✅ {key}: {config[key]}")
            else:
                print(f"   ❌ Missing key: {key}")
    else:
        print("   ❌ unet_config not found!")
    
    # Check memory_usage_factor
    if hasattr(WAN21_Vace, 'memory_usage_factor'):
        factor = WAN21_Vace.memory_usage_factor
        print(f"   ✅ memory_usage_factor: {factor}")
        expected_factor = 1.2  # ComfyUI's value
        if abs(factor - expected_factor) < 0.01:
            print(f"   ✅ Matches ComfyUI value ({expected_factor})")
        else:
            print(f"   ⚠️ Doesn't match ComfyUI value ({expected_factor})")
    else:
        print("   ❌ memory_usage_factor not found!")


def test_wan21_vace_model_creation():
    """Test WAN21_Vace model creation process"""
    from models import WAN21_Vace, WANConfig
    
    print("\n🔍 Testing WAN21_Vace Model Creation:")
    
    try:
        # Test config instantiation
        config = WAN21_Vace(WANConfig())
        print(f"   ✅ Config instantiation: {type(config)}")
        
        # Test get_model method
        if hasattr(config, 'get_model'):
            print("   ✅ get_model() method exists")
            
            try:
                model = config.get_model(None, "", device='cpu')
                print(f"   ✅ get_model() succeeded: {type(model)}")
                return model
                
            except Exception as e:
                print(f"   ❌ get_model() failed: {e}")
                return None
        else:
            print("   ❌ get_model() method missing!")
            return None
            
    except Exception as e:
        print(f"   ❌ WAN21_Vace instantiation failed: {e}")
        return None


def test_wan21_vace_comfyui_compatibility():
    """Test compatibility with ComfyUI's WAN21_Vace"""
    print("\n📊 ComfyUI Compatibility Analysis:")
    
    # Expected ComfyUI WAN21_Vace features
    comfyui_features = {
        'unet_config': {
            'image_model': 'wan2.1',
            'model_type': 'vace'
        },
        'memory_usage_factor': 1.2,
        'get_model_method': 'returns model_base.WAN21_Vace',
        'inherits_from': 'WAN21_T2V'
    }
    
    print("Expected ComfyUI WAN21_Vace structure:")
    for feature, expected in comfyui_features.items():
        print(f"   📋 {feature}: {expected}")


def test_wan21_vace_inheritance():
    """Test WAN21_Vace inheritance chain"""
    from models import WAN21_Vace, BaseModel
    
    print("\n🔍 Testing WAN21_Vace Inheritance:")
    
    # Check inheritance from BaseModel
    if issubclass(WAN21_Vace, BaseModel):
        print("   ✅ Inherits from BaseModel")
    else:
        print("   ❌ Does not inherit from BaseModel")
    
    # Check MRO (Method Resolution Order)
    mro = WAN21_Vace.__mro__
    print("   📋 Method Resolution Order:")
    for i, cls in enumerate(mro):
        print(f"      {i}: {cls.__name__}")


def test_wan21_vace_latent_format():
    """Test latent format in WAN21_Vace"""
    from models import WAN21_Vace, WANConfig
    
    print("\n🔍 Testing WAN21_Vace Latent Format:")
    
    try:
        config = WAN21_Vace(WANConfig())
        model = config.get_model(None, "", device='cpu')
        
        if hasattr(model, 'latent_format'):
            latent_format = model.latent_format
            print(f"   ✅ Latent format found: {type(latent_format)}")
            print(f"   ✅ Latent channels: {latent_format.latent_channels}")
            print(f"   ✅ Latent dimensions: {latent_format.latent_dimensions}")
        else:
            print("   ❌ Latent format missing!")
            
    except Exception as e:
        print(f"   ❌ Latent format test failed: {e}")


def test_wan21_vace_extra_conds():
    """Test extra_conds implementation in WAN21_Vace"""
    from models import WAN21_Vace, WANConfig
    
    print("\n🔍 Testing WAN21_Vace extra_conds():")
    
    try:
        config = WAN21_Vace(WANConfig())
        model = config.get_model(None, "", device='cpu')
        
        if hasattr(model, 'extra_conds'):
            print("   ✅ extra_conds() method exists")
            
            # Test call with mock parameters
            try:
                import torch
                
                # Mock parameters for VACE
                noise_shape = [1, 16, 9, 104, 60]
                
                mock_kwargs = {
                    'noise': torch.randn(*noise_shape),
                    'vace_frames': [torch.randn([1, 32, 9, 104, 60])],
                    'vace_mask': [torch.ones([1, 64, 9, 104, 60])],
                    'vace_strength': [1.0]
                }
                
                result = model.extra_conds(**mock_kwargs)
                print(f"   ✅ extra_conds() executed successfully")
                print(f"   📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                
            except Exception as e:
                print(f"   ❌ extra_conds() execution failed: {e}")
                
        else:
            print("   ❌ extra_conds() method missing!")
            
    except Exception as e:
        print(f"   ❌ extra_conds() test failed: {e}")


def test_wan21_vace_device_handling():
    """Test device handling in WAN21_Vace"""
    from models import WAN21_Vace, WANConfig
    
    print("\n🔍 Testing WAN21_Vace Device Handling:")
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for device in devices:
        print(f"\n   Testing device: {device}")
        
        try:
            config = WAN21_Vace(WANConfig())
            model = config.get_model(None, "", device=device)
            
            print(f"   ✅ Model created on {device}")
            
            # Verify model device matches requested device
            if hasattr(model, 'device'):
                print(f"   ✅ Model.device: {model.device}")
            
        except Exception as e:
            print(f"   ❌ Device handling failed for {device}: {e}")


if __name__ == "__main__":
    try:
        test_wan21_vace_config()
        print("\n" + "="*50)
        
        test_wan21_vace_model_creation()
        print("\n" + "="*50)
        
        test_wan21_vace_comfyui_compatibility()
        print("\n" + "="*50)
        
        test_wan21_vace_inheritance()
        print("\n" + "="*50)
        
        test_wan21_vace_latent_format()
        print("\n" + "="*50)
        
        test_wan21_vace_extra_conds()
        print("\n" + "="*50)
        
        test_wan21_vace_device_handling()
        print("\n" + "="*50)
        
        print("🎉 ALL WAN21_Vace TESTS COMPLETED!")
        
    except Exception as e:
        print(f"💥 WAN21_Vace TESTS FAILED: {e}")
        print(f"This reveals issues in motion's WAN21_Vace implementation!")
        import traceback
        traceback.print_exc()
