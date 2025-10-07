#!/usr/bin/env python3
"""
Priority 1: Test VaceWanModel Core Implementation
Critical debugging for the lowest-level diffusion model
"""

import torch
import sys
import os
sys.path.append('/home/fashionx/v_pipe/standalone_app/motion')

import pytest

def test_vacewmodel_import():
    """Test that we can import VaceWanModel"""
    from pure_wan_models import PureVaceWanModel
    
    # Test basic instantiation
    model = PureVaceWanModel(
        model_type='vace',
        dim=2048,
        num_heads=16,
        num_layers=32,
        device='cpu'
    )
    
    assert hasattr(model, 'latent_format'), "Missing latent_format attribute"
    assert model.latent_format.latent_channels == 16, "Wrong latent channels"
    assert model.latent_format.latent_dimensions == 3, "Welcome latent dims"
    
    print("✅ VaceWanModel import and basic instantiation successful!")


def test_vacewmodel_forward_basic():
    """Test basic forward pass without conditioning"""
    from pure_wan_models import PureVaceWanModel
    
    model = PureVaceWanModel(
        model_type='vace',
        patch_size=(1, 2, 2),
        dim=2048,
        out_dim=16,
        num_heads=16,
        num_layers=4,  # Use fewer layers for testing
        device='cpu'
    )
    
    # Test input shapes
    input_tensor = torch.randn([1, 16, 11, 104, 60])  # [B, C, T, H, W]
    timestep = torch.randint(0, 1000, (1,))
    context = torch.randn([1, 4096])  # Basic context
    
    try:
        output = model(input_tensor, timestep, context=context)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        assert output.shape == input_tensor.shape, f"Shape mismatch: {output.shape} vs {input_tensor.shape}"
        
        print(f"✅ Output statistics:")
        print(f"   Mean: {output.mean().item():.6f}")
        print(f"   Std: {output.std().item():.6f}")
        print(f"   Range: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_vacewmodel_device_compatibility():
    """Test model device handling"""
    from pure_wan_models import PureVaceWanModel
    
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🔄 Testing on GPU ({device})")
    else:
        device = 'cpu'
        print(f"🔄 Testing on CPU")
    
    model = PureVaceWanModel(
        model_type='vace',
        dim=512,  # Smaller model for testing
        num_heads=8,
        num_layers=2,
        device=device
    )
    
    model = model.to(device)
    
    # Test tensors on same device as model
    input_tensor = torch.randn([1, 16, 5, 32, 32], device=device)
    timestep = torch.randint(0, 100, (1,), device=device)
    context = torch.randn([1, 2048], device=device)
    
    try:
        with torch.no_grad():
            output = model(input_tensor, timestep, context=context)
        
        print(f"✅ Device compatibility test successful!")
        print(f"   Input device: {input_tensor.device}")
        print(f"   Output device: {output.device}")
        print(f"   Model device: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ Device compatibility test failed: {e}")
        raise


def test_vacewmodel_memory_usage():
    """Test memory usage of forward pass"""
    from pure_wan_models import PureVaceWanModel
    
    if not torch.cuda.is_available():
        print("⏭️ Skipping memory test - no CUDA available")
        return
    
    model = PureVaceWanModel(
        model_type='vace',
        dim=2048,
        num_heads=16,
        num_layers=8,  # Medium size for memory testing
        device='cuda'
    )
    
    # Memory tracking
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()
    
    input_tensor = torch.randn([1, 16, 11, 104, 60], device='cuda')
    timestep = torch.randint(0, 1000, (1,), device='cuda')
    context = torch.randn([1, 4096], device='cuda')
    
    with torch.no_grad():
        output = model(input_tensor, timestep, context=context)
    
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"✅ Memory usage test completed!")
    print(f"   Memory increase: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
    print(f"   Peak memory: {peak_memory / 1024 / 1024:.2f} MB")


def test_vacewmodel_vace_specific_features():
    """Test VACE-specific features if implemented"""
    from pure_wan_models import PureVaceWanModel
    
    model = PureVaceWanModel(
        model_type='vace',
        dim=512,
        num_heads=8,
        num_layers=2,
        device='cpu'
    )
    
    # Check for VACE-specific attributes
    vace_features = {}
    
    # Check for VACE blocks
    if hasattr(model, 'vace_blocks'):
        vace_features['vace_blocks'] = len(model.vace_blocks)
    
    # Check for VACE patch embedding
    if hasattr(model, 'vace_patch_embedding'):
        vace_features['vace_patch_embedding'] = True
    
    # Check for VACE layers mapping
    if hasattr(model, 'vace_layers_mapping'):
        vace_features['vace_layers_mapping'] = model.vace_layers_mapping
    
    print(f"🔍 VACE-specific features found:")
    for feature, value in vace_features.items():
        print(f"   {feature}: {value}")
    
    if not vace_features:
        print(f"⚠️ Warning: No VACE-specific features detected!")
        print(f"   This suggests motion's PureVaceWanModel is generic,")
        print(f"   not VACE-specific like ComfyUI's VaceWanModel")


if __name__ == "__main__":
    try:
        test_vacewmodel_import()
        print("\n" + "="*50)
        
        test_vacewmodel_forward_basic()
        print("\n" + "="*50)
        
        test_vacewmodel_device_compatibility()
        print("\n" + "="*50)
        
        test_vacewmodel_memory_usage()
        print("\n" + "="*50)
        
        test_vacewmodel_vace_specific_features()
        print("\n" + "="*50)
        
        print("🎉 ALL VaceWanModel TESTS PASSED!")
        
    except Exception as e:
        print(f"💥 TESTS FAILED: {e}")
        print(f"This reveals critical issues in motion's VaceWanModel implementation!")
        import traceback
        traceback.print_exc()
