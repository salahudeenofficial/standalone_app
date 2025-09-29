#!/usr/bin/env python3
"""
VAE Compatibility Comparison Test

This test compares the motion pipeline's standalone VAE with ComfyUI's VAE
using the same WAN 2.1 VAE model and input tensor to verify they perform
the same function.
"""

import torch
import sys
import os
from pathlib import Path
import safetensors
import traceback

# Add motion pipeline to path
sys.path.insert(0, str(Path('.').absolute() / 'motion'))

# Add ComfyUI to path (assuming it's in the expected location)
comfy_path = Path('/home/fashionx/comfy/ComfyUI')
if comfy_path.exists():
    sys.path.insert(0, str(comfy_path))

def load_model_state_dict(model_path):
    """Load model state dict from safetensors file"""
    print(f"📁 Loading model: {model_path}")
    with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    print(f"   Loaded {len(sd)} parameters")
    return sd

def create_test_tensor():
    """Create a test tensor compatible with WAN VAE"""
    # Create a test tensor: batch=1, channels=3, frames=8, height=64, width=64
    # Using dimensions that are multiples of 8 (WAN VAE requirement)
    # WAN VAE downscale ratio is 8, so we need at least 8 frames and 64x64 resolution
    test_tensor = torch.randn(1, 3, 8, 64, 64)
    print(f"🧪 Test tensor created:")
    print(f"   Shape: {test_tensor.shape}")
    print(f"   Dtype: {test_tensor.dtype}")
    print(f"   Range: [{test_tensor.min().item():.6f}, {test_tensor.max().item():.6f}]")
    print(f"   Mean: {test_tensor.mean().item():.6f}")
    print(f"   Std: {test_tensor.std().item():.6f}")
    return test_tensor

def test_motion_pipeline_vae(sd, test_tensor):
    """Test motion pipeline's standalone VAE"""
    print("\n" + "="*60)
    print("🧪 TESTING MOTION PIPELINE VAE")
    print("="*60)
    
    try:
        from standalone_vae import VAE
        
        print("✅ Importing motion pipeline VAE...")
        vae = VAE(sd=sd)
        print(f"✅ VAE initialized: {type(vae.first_stage_model).__name__}")
        
        # Check VAE properties
        print(f"🔍 VAE Properties:")
        print(f"   Latent channels: {vae.latent_channels}")
        print(f"   Latent dim: {vae.latent_dim}")
        print(f"   Downscale ratio: {vae.downscale_ratio}")
        print(f"   Upscale ratio: {vae.upscale_ratio}")
        
        # Test encoding
        print(f"\n🔧 Testing encoding...")
        with torch.no_grad():
            try:
                encoded = vae.encode(test_tensor)
                print(f"✅ Encoding successful!")
                print(f"   Output shape: {encoded.shape}")
                print(f"   Output dtype: {encoded.dtype}")
                print(f"   Output range: [{encoded.min().item():.6f}, {encoded.max().item():.6f}]")
                print(f"   Output mean: {encoded.mean().item():.6f}")
                print(f"   Output std: {encoded.std().item():.6f}")
                
                # Test decoding
                print(f"\n🔧 Testing decoding...")
                decoded = vae.decode(encoded)
                print(f"✅ Decoding successful!")
                print(f"   Decoded shape: {decoded.shape}")
                print(f"   Decoded dtype: {decoded.dtype}")
                print(f"   Decoded range: [{decoded.min().item():.6f}, {decoded.max().item():.6f}]")
                print(f"   Decoded mean: {decoded.mean().item():.6f}")
                print(f"   Decoded std: {decoded.std().item():.6f}")
                
                return {
                    'success': True,
                    'encoded': encoded,
                    'decoded': decoded,
                    'vae': vae
                }
                
            except Exception as encode_error:
                print(f"❌ Encoding/Decoding failed: {encode_error}")
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(encode_error),
                    'encoded': None,
                    'decoded': None,
                    'vae': vae
                }
                
    except Exception as e:
        print(f"❌ Motion pipeline VAE test failed: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'encoded': None,
            'decoded': None,
            'vae': None
        }

def test_comfyui_vae(sd, test_tensor):
    """Test ComfyUI's VAE"""
    print("\n" + "="*60)
    print("🧪 TESTING COMFYUI VAE")
    print("="*60)
    
    try:
        # Try to import ComfyUI VAE
        try:
            from comfy.sd import VAE as ComfyVAE
            print("✅ Importing ComfyUI VAE...")
        except ImportError:
            print("❌ ComfyUI not found, trying alternative import...")
            # Try alternative import path
            sys.path.insert(0, '/home/fashionx/comfy/ComfyUI')
            from comfy.sd import VAE as ComfyVAE
            print("✅ ComfyUI VAE imported successfully!")
        
        # Initialize ComfyUI VAE
        print("🔧 Initializing ComfyUI VAE...")
        comfy_vae = ComfyVAE(sd=sd)
        print(f"✅ ComfyUI VAE initialized: {type(comfy_vae.first_stage_model).__name__}")
        
        # Check VAE properties
        print(f"🔍 ComfyUI VAE Properties:")
        print(f"   Latent channels: {comfy_vae.latent_channels}")
        print(f"   Latent dim: {comfy_vae.latent_dim}")
        print(f"   Downscale ratio: {comfy_vae.downscale_ratio}")
        print(f"   Upscale ratio: {comfy_vae.upscale_ratio}")
        
        # Test encoding
        print(f"\n🔧 Testing ComfyUI encoding...")
        with torch.no_grad():
            try:
                encoded = comfy_vae.encode(test_tensor)
                print(f"✅ ComfyUI encoding successful!")
                print(f"   Output shape: {encoded.shape}")
                print(f"   Output dtype: {encoded.dtype}")
                print(f"   Output range: [{encoded.min().item():.6f}, {encoded.max().item():.6f}]")
                print(f"   Output mean: {encoded.mean().item():.6f}")
                print(f"   Output std: {encoded.std().item():.6f}")
                
                # Test decoding
                print(f"\n🔧 Testing ComfyUI decoding...")
                decoded = comfy_vae.decode(encoded)
                print(f"✅ ComfyUI decoding successful!")
                print(f"   Decoded shape: {decoded.shape}")
                print(f"   Decoded dtype: {decoded.dtype}")
                print(f"   Decoded range: [{decoded.min().item():.6f}, {decoded.max().item():.6f}]")
                print(f"   Decoded mean: {decoded.mean().item():.6f}")
                print(f"   Decoded std: {decoded.std().item():.6f}")
                
                return {
                    'success': True,
                    'encoded': encoded,
                    'decoded': decoded,
                    'vae': comfy_vae
                }
                
            except Exception as encode_error:
                print(f"❌ ComfyUI encoding/decoding failed: {encode_error}")
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(encode_error),
                    'encoded': None,
                    'decoded': None,
                    'vae': comfy_vae
                }
                
    except Exception as e:
        print(f"❌ ComfyUI VAE test failed: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'encoded': None,
            'decoded': None,
            'vae': None
        }

def compare_results(motion_result, comfy_result):
    """Compare the results from both VAE implementations"""
    print("\n" + "="*60)
    print("🔍 COMPARING RESULTS")
    print("="*60)
    
    if not motion_result['success'] or not comfy_result['success']:
        print("❌ Cannot compare - one or both tests failed")
        if not motion_result['success']:
            print(f"   Motion pipeline error: {motion_result['error']}")
        if not comfy_result['success']:
            print(f"   ComfyUI error: {comfy_result['error']}")
        return
    
    motion_encoded = motion_result['encoded']
    comfy_encoded = comfy_result['encoded']
    motion_decoded = motion_result['decoded']
    comfy_decoded = comfy_result['decoded']
    
    print("📊 ENCODED OUTPUT COMPARISON:")
    print(f"   Motion shape: {motion_encoded.shape}")
    print(f"   ComfyUI shape: {comfy_encoded.shape}")
    print(f"   Shapes match: {motion_encoded.shape == comfy_encoded.shape}")
    
    if motion_encoded.shape == comfy_encoded.shape:
        # Compare tensor values
        diff = torch.abs(motion_encoded - comfy_encoded)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Values identical: {torch.allclose(motion_encoded, comfy_encoded, atol=1e-6)}")
        print(f"   Values very close: {torch.allclose(motion_encoded, comfy_encoded, atol=1e-3)}")
    
    print("\n📊 DECODED OUTPUT COMPARISON:")
    print(f"   Motion shape: {motion_decoded.shape}")
    print(f"   ComfyUI shape: {comfy_decoded.shape}")
    print(f"   Shapes match: {motion_decoded.shape == comfy_decoded.shape}")
    
    if motion_decoded.shape == comfy_decoded.shape:
        # Compare tensor values
        diff = torch.abs(motion_decoded - comfy_decoded)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Values identical: {torch.allclose(motion_decoded, comfy_decoded, atol=1e-6)}")
        print(f"   Values very close: {torch.allclose(motion_decoded, comfy_decoded, atol=1e-3)}")
    
    print("\n🎯 COMPATIBILITY ASSESSMENT:")
    if motion_encoded.shape == comfy_encoded.shape and motion_decoded.shape == comfy_decoded.shape:
        if torch.allclose(motion_encoded, comfy_encoded, atol=1e-6):
            print("   ✅ PERFECT COMPATIBILITY - Identical outputs!")
        elif torch.allclose(motion_encoded, comfy_encoded, atol=1e-3):
            print("   ✅ EXCELLENT COMPATIBILITY - Very close outputs!")
        else:
            print("   ⚠️  PARTIAL COMPATIBILITY - Some differences detected")
    else:
        print("   ❌ INCOMPATIBLE - Different output shapes")

def main():
    """Main test function"""
    print("🚀 VAE COMPATIBILITY COMPARISON TEST")
    print("="*60)
    print("This test compares motion pipeline VAE with ComfyUI VAE")
    print("using the same WAN 2.1 VAE model and input tensor.")
    print()
    
    # Check if model exists
    model_path = 'models/vaes/wan_2.1_vae.safetensors'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please download the model first using the previous command.")
        return
    
    try:
        # Load model state dict
        sd = load_model_state_dict(model_path)
        
        # Create test tensor
        test_tensor = create_test_tensor()
        
        # Test motion pipeline VAE
        motion_result = test_motion_pipeline_vae(sd, test_tensor)
        
        # Test ComfyUI VAE
        comfy_result = test_comfyui_vae(sd, test_tensor)
        
        # Compare results
        compare_results(motion_result, comfy_result)
        
        print("\n" + "="*60)
        print("🎉 TEST COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
