#!/usr/bin/env python3
"""
Direct VAE Comparison Test

This test compares the motion pipeline's standalone VAE with ComfyUI's VAE
by testing the core functionality directly, bypassing the crop_pixels issue.
"""

import torch
import sys
import os
from pathlib import Path
import safetensors
import traceback

# Add motion pipeline to path
sys.path.insert(0, str(Path('.').absolute() / 'motion'))

# Add ComfyUI to path
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

def test_direct_vae_functionality(sd):
    """Test VAE functionality directly without crop_pixels"""
    print("\n" + "="*60)
    print("🧪 TESTING DIRECT VAE FUNCTIONALITY")
    print("="*60)
    
    try:
        from standalone_vae import VAE
        from wan_vae_components.vae import WanVAE
        
        print("✅ Importing motion pipeline VAE...")
        vae = VAE(sd=sd)
        print(f"✅ VAE initialized: {type(vae.first_stage_model).__name__}")
        
        # Test direct WAN VAE functionality
        print(f"\n🔧 Testing direct WAN VAE functionality...")
        
        # Create a properly sized tensor for WAN VAE
        # WAN VAE expects: batch, channels, frames, height, width
        # Use dimensions that work with the downscale ratio
        test_tensor = torch.randn(1, 3, 16, 128, 128)  # Large enough for downscale ratio 8
        print(f"   Test tensor shape: {test_tensor.shape}")
        
        # Test process_input
        print(f"\n🔧 Testing process_input...")
        processed = vae.process_input(test_tensor)
        print(f"   Processed shape: {processed.shape}")
        print(f"   Processed range: [{processed.min().item():.6f}, {processed.max().item():.6f}]")
        print(f"   Processed mean: {processed.mean().item():.6f}")
        
        # Test direct WAN VAE encode (bypassing crop_pixels)
        print(f"\n🔧 Testing direct WAN VAE encode...")
        try:
            # Get the WAN VAE model directly
            wan_vae = vae.first_stage_model
            
            # Test the encode method directly
            with torch.no_grad():
                # Convert to the right dtype and device
                input_tensor = processed.to(torch.float16)
                
                # Test encode
                encoded_result = wan_vae.encode(input_tensor)
                print(f"✅ Direct encode successful!")
                print(f"   Encoded shape: {encoded_result.shape}")
                print(f"   Encoded dtype: {encoded_result.dtype}")
                print(f"   Encoded range: [{encoded_result.min().item():.6f}, {encoded_result.max().item():.6f}]")
                print(f"   Encoded mean: {encoded_result.mean().item():.6f}")
                
                # Test decode
                print(f"\n🔧 Testing direct WAN VAE decode...")
                decoded_result = wan_vae.decode(encoded_result)
                print(f"✅ Direct decode successful!")
                print(f"   Decoded shape: {decoded_result.shape}")
                print(f"   Decoded dtype: {decoded_result.dtype}")
                print(f"   Decoded range: [{decoded_result.min().item():.6f}, {decoded_result.max().item():.6f}]")
                print(f"   Decoded mean: {decoded_result.mean().item():.6f}")
                
                return {
                    'success': True,
                    'encoded': encoded_result,
                    'decoded': decoded_result,
                    'vae': vae
                }
                
        except Exception as encode_error:
            print(f"❌ Direct encode/decode failed: {encode_error}")
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

def test_comfyui_direct_functionality(sd):
    """Test ComfyUI VAE functionality directly"""
    print("\n" + "="*60)
    print("🧪 TESTING COMFYUI DIRECT FUNCTIONALITY")
    print("="*60)
    
    try:
        # Try to import ComfyUI VAE
        try:
            from comfy.sd import VAE as ComfyVAE
            print("✅ Importing ComfyUI VAE...")
        except ImportError:
            print("❌ ComfyUI not found, trying alternative import...")
            sys.path.insert(0, '/home/fashionx/comfy/ComfyUI')
            from comfy.sd import VAE as ComfyVAE
            print("✅ ComfyUI VAE imported successfully!")
        
        # Initialize ComfyUI VAE
        print("🔧 Initializing ComfyUI VAE...")
        comfy_vae = ComfyVAE(sd=sd)
        print(f"✅ ComfyUI VAE initialized: {type(comfy_vae.first_stage_model).__name__}")
        
        # Test direct WAN VAE functionality
        print(f"\n🔧 Testing ComfyUI direct WAN VAE functionality...")
        
        # Create the same test tensor
        test_tensor = torch.randn(1, 3, 16, 128, 128)
        print(f"   Test tensor shape: {test_tensor.shape}")
        
        # Test process_input
        print(f"\n🔧 Testing ComfyUI process_input...")
        processed = comfy_vae.process_input(test_tensor)
        print(f"   Processed shape: {processed.shape}")
        print(f"   Processed range: [{processed.min().item():.6f}, {processed.max().item():.6f}]")
        print(f"   Processed mean: {processed.mean().item():.6f}")
        
        # Test direct WAN VAE encode
        print(f"\n🔧 Testing ComfyUI direct WAN VAE encode...")
        try:
            # Get the WAN VAE model directly
            wan_vae = comfy_vae.first_stage_model
            
            # Test the encode method directly
            with torch.no_grad():
                # Convert to the right dtype and device
                input_tensor = processed.to(torch.float16)
                
                # Test encode
                encoded_result = wan_vae.encode(input_tensor)
                print(f"✅ ComfyUI direct encode successful!")
                print(f"   Encoded shape: {encoded_result.shape}")
                print(f"   Encoded dtype: {encoded_result.dtype}")
                print(f"   Encoded range: [{encoded_result.min().item():.6f}, {encoded_result.max().item():.6f}]")
                print(f"   Encoded mean: {encoded_result.mean().item():.6f}")
                
                # Test decode
                print(f"\n🔧 Testing ComfyUI direct WAN VAE decode...")
                decoded_result = wan_vae.decode(encoded_result)
                print(f"✅ ComfyUI direct decode successful!")
                print(f"   Decoded shape: {decoded_result.shape}")
                print(f"   Decoded dtype: {decoded_result.dtype}")
                print(f"   Decoded range: [{decoded_result.min().item():.6f}, {decoded_result.max().item():.6f}]")
                print(f"   Decoded mean: {decoded_result.mean().item():.6f}")
                
                return {
                    'success': True,
                    'encoded': encoded_result,
                    'decoded': decoded_result,
                    'vae': comfy_vae
                }
                
        except Exception as encode_error:
            print(f"❌ ComfyUI direct encode/decode failed: {encode_error}")
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

def compare_direct_results(motion_result, comfy_result):
    """Compare the direct results from both VAE implementations"""
    print("\n" + "="*60)
    print("🔍 COMPARING DIRECT RESULTS")
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
    print("🚀 DIRECT VAE COMPATIBILITY COMPARISON TEST")
    print("="*60)
    print("This test compares motion pipeline VAE with ComfyUI VAE")
    print("by testing the core WAN VAE functionality directly.")
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
        
        # Test motion pipeline VAE directly
        motion_result = test_direct_vae_functionality(sd)
        
        # Test ComfyUI VAE directly
        comfy_result = test_comfyui_direct_functionality(sd)
        
        # Compare results
        compare_direct_results(motion_result, comfy_result)
        
        print("\n" + "="*60)
        print("🎉 DIRECT TEST COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
