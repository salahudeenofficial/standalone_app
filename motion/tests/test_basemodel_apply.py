#!/usr/bin/env python3
"""
Priority 2: Test BaseModel apply_model() Implementation
Critical debugging for the core diffusion inference method
"""

import torch
import sys
import os
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

def test_basemodel_import():
    """Test that we can import BaseModel from motion"""
    from models import BaseModel
    
    print("✅ BaseModel import successful!")


def test_basemodel_methods():
    """Test BaseModel methods exist"""
    from models import BaseModel
    
    # Get all methods
    methods = [method for method in dir(BaseModel) if not method.startswith('_')]
    
    print("📋 BaseModel methods found:")
    for method in methods:
        print(f"   - {method}")
    
    # Check for critical methods
    critical_methods = ['apply_model', '_apply_model', 'get_dtype', 'load_model_weights']
    missing_methods = []
    
    print("\n🔍 Checking critical methods:")
    for method in critical_methods:
        if hasattr(BaseModel, method):
            print(f"   ✅ {method}: EXISTS")
        else:
            print(f"   ❌ {method}: MISSING")
            missing_methods.append(method)
    
    if missing_methods:
        print(f"\n⚠️ Missing critical methods: {missing_methods}")
        print(f"   These are required for diffusion inference!")


def test_basemodel_instantiation():
    """Test BaseModel instantiation"""
    from models import BaseModel, WANConfig
    
    # Create config
    config = WANConfig()
    
    # Create model
    model = BaseModel(config, device='cpu')
    
    # Check basic properties
    assert hasattr(model, 'device'), "Missing device attribute"
    assert hasattr(model, 'model_config'), "Missing model_config attribute"
    
    print("✅ BaseModel instantiation successful!")
    print(f"   Device: {model.device}")
    print(f"   Config type: {type(model.model_config)}")


def test_basemodel_apply_model_exists():
    """Test if apply_model method exists and works"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    model = BaseModel(config, device='cpu')
    
    if hasattr(model, 'apply_model'):
        print("✅ apply_model method EXISTS")
        
        # Try to call it to see what error we get
        try:
            # Test with minimal inputs
            x = torch.randn([1, 16, 11, 104, 60])
            t = torch.randn([1])
            
            result = model.apply_model(x, t)
            print("🎉 apply_model works!")
            
        except Exception as e:
            print(f"⚠️ apply_model exists but fails: {e}")
            print(f"   This suggests incomplete implementation")
            
    else:
        print("❌ apply_model method MISSING!")
        print("   This is a CRITICAL issue!")


def test_basemodel_get_dtype():
    """Test get_dtype method"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    
    # Test CPU
    model_cpu = BaseModel(config, device='cpu')
    
    if hasattr(model_cpu, 'get_dtype'):
        dtype = model_cpu.get_dtype()
        print(f"✅ get_dtype() works: {dtype}")
    else:
        print("❌ get_dtype() method missing!")
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_gpu = BaseModel(config, device='cuda')
        if hasattr(model_gpu, 'get_dtype'):
            dtype_gpu = model_gpu.get_dtype()
            print(f"✅ GPU get_dtype() works: {dtype_gpu}")


def test_basemodel_device_management():
    """Test device management capabilities"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    
    print("🔍 Testing device management:")
    
    # Test CPU device
    model_cpu = BaseModel(config, device='cpu')
    print(f"   ✅ CPU device: {model_cpu.device}")
    
    # Test GPU device if available
    if torch.cuda.is_available():
        model_gpu = BaseModel(config, device='cuda')
        print(f"   ✅ GPU device: {model_gpu.device}")
        
        # Test automatic device detection
        gpu_device = torch.device('cuda')
        model_auto = BaseModel(config, device=gpu_device)
        print(f"   ✅ Auto device: {model_auto.device}")
    else:
        print("   ⏭️ Skipping GPU tests - no CUDA available")


def test_basemodel_memory_factors():
    """Test memory usage factors"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    model = BaseModel(config, device='cpu')
    
    print("🔍 Checking memory management:")
    
    # Check if memory factors exist
    memory_attrs = ['memory_usage_factor', 'memory_usage_factor_conds']
    
    for attr in memory_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            print(f"   ✅ {attr}: {value}")
        else:
            print(f"   ❌ {attr}: MISSING")


def test_basemodel_latent_format():
    """Test latent format integration"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    model = BaseModel(config, device='cpu')
    
    print("🔍 Checking latent format integration:")
    
    if hasattr(model, 'latent_format'):
        latent_format = model.latent_format
        print(f"   ✅ Latent format: {type(latent_format)}")
        print(f"   ✅ Latent channels: {latent_format.latent_channels}")
        print(f"   ✅ Latent dimensions: {latent_format.latent_dimensions}")
    else:
        print("   ❌ Latent format: MISSING!")


def test_basemodel_comfyui_compatibility():
    """Test compatibility with ComfyUI's BaseModel patterns"""
    from models import BaseModel, WANConfig
    
    config = WANConfig()
    model = BaseModel(config, device='cpu')
    
    print("🔍 Checking ComfyUI compatibility:")
    
    # Expected ComfyUI attributes
    comfyui_attrs = [
        'current_patcher',
        'model_type',
        'manual_cast_dtype',
        'adm_channels',
        'concat_keys',
        'latent_format'
    ]
    
    compatible_count = 0
    for attr in comfyui_attrs:
        if hasattr(model, attr):
            print(f"   ✅ {attr}: {getattr(model, attr)}")
            compatible_count += 1
        else:
            print(f"   ❌ {attr}: MISSING")
    
    compatibility_score = (compatible_count / len(comfyui_attrs)) * 100
    print(f"\n📊 ComfyUI Compatibility Score: {compatibility_score:.1f}%")


if __name__ == "__main__":
    try:
        test_basemodel_import()
        print("\n" + "="*50)
        
        test_basemodel_methods()
        print("\n" + "="*50)
        
        test_basemodel_instantiation()
        print("\n" + "="*50)
        
        test_basemodel_apply_model_exists()
        print("\n" + "="*50)
        
        test_basemodel_get_dtype()
        print("\n" + "="*50)
        
        test_basemodel_device_management()
        print("\n" + "="*50)
        
        test_basemodel_memory_factors()
        print("\n" + "="*50)
        
        test_basemodel_latent_format()
        print("\n" + "="*50)
        
        test_basemodel_comfyui_compatibility()
        print("\n" + "="*50)
        
        print("🎉 ALL BaseModel TESTS COMPLETED!")
        
    except Exception as e:
        print(f"💥 BaseModel TESTS FAILED: {e}")
        print(f"This reveals critical issues in motion's BaseModel implementation!")
        import traceback
        traceback.print_exc()
