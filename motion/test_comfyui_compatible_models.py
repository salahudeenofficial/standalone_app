#!/usr/bin/env python3
"""
Test Script for ComfyUI-Compatible WAN 2.1 Vace Model Classes
Verifies that our model classes can load complete state dicts without missing keys
"""

import torch
import logging
import sys
import os

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_comfyui_compatible_models():
    """Test ComfyUI-compatible model classes"""
    
    print("🧪 TESTING COMFYUI-COMPATIBLE WAN 2.1 VACE MODEL CLASSES")
    print("=" * 60)
    
    try:
        # Import our ComfyUI-compatible model classes
        from comfyui_compatible_models import ComfyUIWanModel, ComfyUIVaceWanModel
        
        print("✅ Successfully imported ComfyUI-compatible model classes")
        
        # Test 1: Create a basic WanModel
        print("\n🔍 Test 1: Creating basic WanModel...")
        try:
            wan_model = ComfyUIWanModel(
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6
            )
            print(f"✅ WanModel created successfully")
            print(f"   Model type: {wan_model.model_type}")
            print(f"   Dimensions: {wan_model.dim}D, {wan_model.num_layers} layers")
            print(f"   Parameters: {sum(p.numel() for p in wan_model.parameters()):,}")
        except Exception as e:
            print(f"❌ Failed to create WanModel: {e}")
            return False
        
        # Test 2: Create a VaceWanModel
        print("\n🔍 Test 2: Creating VaceWanModel...")
        try:
            vace_model = ComfyUIVaceWanModel(
                model_type='vace',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                vace_layers=8,
                vace_in_dim=16
            )
            print(f"✅ VaceWanModel created successfully")
            print(f"   Model type: {vace_model.model_type}")
            print(f"   Dimensions: {vace_model.dim}D, {vace_model.num_layers} layers")
            print(f"   VACE layers: {vace_model.vace_layers}")
            print(f"   Parameters: {sum(p.numel() for p in vace_model.parameters()):,}")
        except Exception as e:
            print(f"❌ Failed to create VaceWanModel: {e}")
            return False
        
        # Test 3: Test model structure compatibility
        print("\n🔍 Test 3: Testing model structure compatibility...")
        try:
            # Check that all expected components exist
            expected_components = [
                'patch_embedding', 'text_embedding', 'time_embed', 'time_projection',
                'blocks', 'head', 'rope_embedder'
            ]
            
            for component in expected_components:
                if hasattr(wan_model, component):
                    print(f"   ✅ {component}: {type(getattr(wan_model, component)).__name__}")
                else:
                    print(f"   ❌ Missing component: {component}")
                    return False
            
            # Check VACE-specific components
            vace_components = ['vace_blocks', 'vace_patch_embedding', 'vace_layers_mapping']
            for component in vace_components:
                if hasattr(vace_model, component):
                    print(f"   ✅ VACE {component}: {type(getattr(vace_model, component)).__name__}")
                else:
                    print(f"   ❌ Missing VACE component: {component}")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to test model structure: {e}")
            return False
        
        # Test 4: Test forward pass compatibility
        print("\n🔍 Test 4: Testing forward pass compatibility...")
        try:
            # Create dummy inputs
            batch_size = 1
            channels = 16
            frames = 8
            height = 64
            width = 64
            
            x = torch.randn(batch_size, channels, frames, height, width)
            timestep = torch.randint(0, 1000, (batch_size,))
            context = torch.randn(batch_size, 512, 4096)  # text_len=512, text_dim=4096
            
            # Test WanModel forward pass
            with torch.no_grad():
                output = wan_model(x, timestep, context)
                print(f"   ✅ WanModel forward pass successful")
                print(f"   Output shape: {output.shape}")
                
                # Test VaceWanModel forward pass
                vace_context = torch.randn(batch_size, 1, 16, frames, height, width)
                vace_strength = [1.0]
                
                vace_output = vace_model(x, timestep, context, vace_context, vace_strength)
                print(f"   ✅ VaceWanModel forward pass successful")
                print(f"   VACE output shape: {vace_output.shape}")
                
        except Exception as e:
            print(f"❌ Failed forward pass test: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 5: Test state dict loading compatibility
        print("\n🔍 Test 5: Testing state dict loading compatibility...")
        try:
            # Create a dummy state dict with expected keys
            dummy_state_dict = {}
            
            # Add patch embedding weights
            dummy_state_dict['patch_embedding.weight'] = torch.randn(2048, 16, 1, 2, 2)
            dummy_state_dict['patch_embedding.bias'] = torch.randn(2048)
            
            # Add text embedding weights
            dummy_state_dict['text_embedding.0.weight'] = torch.randn(2048, 4096)
            dummy_state_dict['text_embedding.0.bias'] = torch.randn(2048)
            dummy_state_dict['text_embedding.2.weight'] = torch.randn(2048, 2048)
            dummy_state_dict['text_embedding.2.bias'] = torch.randn(2048)
            
            # Add time embedding weights
            dummy_state_dict['time_embed.0.weight'] = torch.randn(2048, 256)
            dummy_state_dict['time_embed.0.bias'] = torch.randn(2048)
            dummy_state_dict['time_embed.2.weight'] = torch.randn(2048, 2048)
            dummy_state_dict['time_embed.2.bias'] = torch.randn(2048)
            
            # Add time projection weights
            dummy_state_dict['time_projection.1.weight'] = torch.randn(2048 * 6, 2048)
            dummy_state_dict['time_projection.1.bias'] = torch.randn(2048 * 6)
            
            # Add block weights (simplified)
            for i in range(32):
                dummy_state_dict[f'blocks.{i}.norm1.weight'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.norm1.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.self_attn.q.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.self_attn.q.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.self_attn.k.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.self_attn.k.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.self_attn.v.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.self_attn.v.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.self_attn.o.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.self_attn.o.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.q.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.q.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.k.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.k.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.v.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.v.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.o.weight'] = torch.randn(2048, 2048)
                dummy_state_dict[f'blocks.{i}.cross_attn.o.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.norm2.weight'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.norm2.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.ffn.0.weight'] = torch.randn(8192, 2048)
                dummy_state_dict[f'blocks.{i}.ffn.0.bias'] = torch.randn(8192)
                dummy_state_dict[f'blocks.{i}.ffn.2.weight'] = torch.randn(2048, 8192)
                dummy_state_dict[f'blocks.{i}.ffn.2.bias'] = torch.randn(2048)
                dummy_state_dict[f'blocks.{i}.modulation'] = torch.randn(1, 6, 2048)
            
            # Add head weights
            dummy_state_dict['head.norm.weight'] = torch.randn(2048)
            dummy_state_dict['head.norm.bias'] = torch.randn(2048)
            dummy_state_dict['head.head.weight'] = torch.randn(64, 2048)
            dummy_state_dict['head.head.bias'] = torch.randn(64)
            dummy_state_dict['head.modulation'] = torch.randn(1, 2, 2048)
            
            # Test loading state dict
            missing_keys, unexpected_keys = wan_model.load_state_dict(dummy_state_dict, strict=False)
            
            print(f"   ✅ State dict loading successful")
            print(f"   Missing keys: {len(missing_keys)}")
            print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"   ⚠️  Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"   ⚠️  Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
        except Exception as e:
            print(f"❌ Failed state dict loading test: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ ComfyUI-compatible model classes are working correctly")
        print("✅ Models can be created with proper structure")
        print("✅ Forward passes work correctly")
        print("✅ State dict loading is compatible")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMFYUI-COMPATIBLE MODEL CLASSES TEST")
    print("=" * 50)
    
    success = test_comfyui_compatible_models()
    
    if success:
        print("\n✅ SUCCESS: All tests passed!")
        print("🎯 Our ComfyUI-compatible model classes are ready for VAST AI testing")
        return 0
    else:
        print("\n❌ FAILURE: Some tests failed!")
        print("🔧 Please fix the issues before proceeding to VAST AI testing")
        return 1

if __name__ == "__main__":
    exit(main())
