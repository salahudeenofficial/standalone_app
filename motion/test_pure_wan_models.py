#!/usr/bin/env python3
"""
Test Script for Pure PyTorch WAN 2.1 Vace Model Classes
Tests the standalone implementation without any ComfyUI dependencies
"""

import torch
import logging
import sys
import os

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def test_pure_wan_models():
    """Test pure PyTorch WAN model classes"""
    
    print("🧪 TESTING PURE PYTORCH WAN 2.1 VACE MODEL CLASSES")
    print("=" * 60)
    print("✅ No ComfyUI dependencies - Pure PyTorch only!")
    
    try:
        # Import our pure PyTorch model classes
        from pure_wan_models import PureWanModel, PureVaceWanModel
        
        print("✅ Successfully imported pure PyTorch model classes")
        
        # Test 1: Create a basic WanModel
        print("\n🔍 Test 1: Creating basic PureWanModel...")
        try:
            wan_model = PureWanModel(
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=100,  # Test special case
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6
            )
            print(f"✅ PureWanModel created successfully")
            print(f"   Model type: {wan_model.model_type}")
            print(f"   Dimensions: {wan_model.dim}D, {wan_model.num_layers} layers")
            print(f"   Freq dim: {wan_model.freq_dim} (special case handling)")
            print(f"   Parameters: {sum(p.numel() for p in wan_model.parameters()):,}")
            
            # Check time_embed structure for special case
            time_embed_layers = list(wan_model.time_embed.children())
            first_layer = time_embed_layers[0]
            print(f"   Time embed first layer: {first_layer.in_features} -> {first_layer.out_features}")
            if wan_model.freq_dim == 100 and wan_model.dim == 2048:
                expected_shape = (100, 100)
                actual_shape = (first_layer.in_features, first_layer.out_features)
                if actual_shape == expected_shape:
                    print(f"   ✅ Special case handling correct: {actual_shape}")
                else:
                    print(f"   ❌ Special case handling failed: {actual_shape} != {expected_shape}")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to create PureWanModel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 2: Create a VaceWanModel
        print("\n🔍 Test 2: Creating PureVaceWanModel...")
        try:
            vace_model = PureVaceWanModel(
                model_type='vace',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=100,  # Test special case
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
            print(f"✅ PureVaceWanModel created successfully")
            print(f"   Model type: {vace_model.model_type}")
            print(f"   Dimensions: {vace_model.dim}D, {vace_model.num_layers} layers")
            print(f"   VACE layers: {vace_model.vace_layers}")
            print(f"   Parameters: {sum(p.numel() for p in vace_model.parameters()):,}")
        except Exception as e:
            print(f"❌ Failed to create PureVaceWanModel: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 3: Test model structure compatibility
        print("\n🔍 Test 3: Testing pure model structure...")
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
        print("\n🔍 Test 4: Testing pure PyTorch forward pass...")
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
                print(f"   ✅ PureWanModel forward pass successful")
                print(f"   Output shape: {output.shape}")
                
                # Test VaceWanModel forward pass
                vace_context = torch.randn(batch_size, 1, 16, frames, height, width)
                vace_strength = [1.0]
                
                vace_output = vace_model(x, timestep, context, vace_context, vace_strength)
                print(f"   ✅ PureVaceWanModel forward pass successful")
                print(f"   VACE output shape: {vace_output.shape}")
                
        except Exception as e:
            print(f"❌ Failed forward pass test: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 5: Test state dict compatibility  
        print("\n🔍 Test 5: Testing state dict compatibility...")
        try:
            # Create a minimal state dict with correct dimensions
            dummy_state_dict = {}
            
            # Add patch embedding weights
            dummy_state_dict['patch_embedding.weight'] = torch.randn(2048, 16, 1, 2, 2)
            dummy_state_dict['patch_embedding.bias'] = torch.randn(2048)
            
            # Add text embedding weights
            dummy_state_dict['text_embedding.0.weight'] = torch.randn(2048, 4096)
            dummy_state_dict['text_embedding.0.bias'] = torch.randn(2048)
            dummy_state_dict['text_embedding.2.weight'] = torch.randn(2048, 2048)
            dummy_state_dict['text_embedding.2.bias'] = torch.randn(2048)
            
            # Add time embedding weights (special case: freq_dim=100)
            dummy_state_dict['time_embed.0.weight'] = torch.randn(100, 100)  # [100, 100] not [2048, 100]
            dummy_state_dict['time_embed.0.bias'] = torch.randn(100)
            dummy_state_dict['time_embed.2.weight'] = torch.randn(2048, 100)
            dummy_state_dict['time_embed.2.bias'] = torch.randn(2048)
            
            # Add time projection weights
            dummy_state_dict['time_projection.1.weight'] = torch.randn(2048 * 6, 2048)
            dummy_state_dict['time_projection.1.bias'] = torch.randn(2048 * 6)
            
            # Add head weights
            dummy_state_dict['head.norm.weight'] = torch.randn(2048)
            dummy_state_dict['head.head.weight'] = torch.randn(64, 2048)
            dummy_state_dict['head.head.bias'] = torch.randn(64)
            dummy_state_dict['head.modulation'] = torch.randn(1, 2, 2048)
            
            # Test loading state dict
            missing_keys, unexpected_keys = wan_model.load_state_dict(dummy_state_dict, strict=False)
            
            print(f"   ✅ State dict loading successful")
            print(f"   Missing keys: {len(missing_keys)}")
            print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            # Check critical dimensions were loaded correctly
            loaded_time_embed_weight = wan_model.time_embed[0].weight
            expected_shape = torch.Size([100, 100])
            if loaded_time_embed_weight.shape == expected_shape:
                print(f"   ✅ Critical dimension fix verified: time_embed.0.weight shape = {loaded_time_embed_weight.shape}")
            else:
                print(f"   ❌ Critical dimension fix failed: expected {expected_shape}, got {loaded_time_embed_weight.shape}")
                return False
                
        except Exception as e:
            print(f"❌ Failed state dict loading test: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Pure PyTorch WAN model classes are working correctly")
        print("✅ No ComfyUI dependencies required")
        print("✅ Special dimension handling works correctly")
        print("✅ Models can be created with proper structure")
        print("✅ Forward passes work correctly")
        print("✅ State dict loading is compatible")
        print("✅ Critical dimension fix (time_embed [100,100]) verified")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 PURE PYTORCH WAN MODEL CLASSES TEST")
    print("=" * 50)
    
    success = test_pure_wan_models()
    
    if success:
        print("\n✅ SUCCESS: All tests passed!")
        print("🎯 Pure PyTorch WAN models are ready for VAST AI testing")
        print("🔧 Motion pipeline now has ZERO ComfyUI dependencies")
        return 0
    else:
        print("\n❌ FAILURE: Some tests failed!")
        print("🔧 Please fix the issues before proceeding")
        return 1

if __name__ == "__main__":
    exit(main())

