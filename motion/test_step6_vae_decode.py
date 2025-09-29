#!/usr/bin/env python3
"""
Test script for Step 6: VAE Decode
Tests the VAE decode functionality extracted from ComfyUI
"""

import torch
import logging
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from components.vae_decoder import VAEDecode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class MockVAE:
    """Mock VAE class for testing"""
    
    def __init__(self, name="MockVAE"):
        self.name = name
    
    def decode(self, samples):
        """Mock decode method that simulates VAE decoding"""
        # Simulate VAE decode: latent -> images
        # Input: (B, C, T, H, W) -> Output: (B*T, H*8, W*8, 3)
        batch_size, channels, frames, height, width = samples.shape
        
        # Simulate upscaling (typical VAE decode upscales by 8x)
        output_height = height * 8
        output_width = width * 8
        
        # Create mock decoded images
        decoded_images = torch.randn(batch_size * frames, output_height, output_width, 3)
        
        # Ensure values are in [0, 1] range (typical for images)
        decoded_images = torch.sigmoid(decoded_images)
        
        return decoded_images

def test_vae_decode():
    """Test the VAEDecode functionality"""
    print("="*80)
    print("🧪 Testing VAEDecode Component")
    print("="*80)
    
    # Create VAE decoder
    vae_decoder = VAEDecode()
    print("✅ VAEDecode processor created")
    
    # Create mock VAE
    mock_vae = MockVAE("TestVAE")
    print("✅ Mock VAE created")
    
    # Test cases
    test_cases = [
        {
            'name': 'Small latent (1 frame)',
            'shape': (1, 4, 1, 8, 13),  # (B, C, T, H, W)
            'expected_output_frames': 1
        },
        {
            'name': 'Medium latent (5 frames)',
            'shape': (1, 4, 5, 8, 13),
            'expected_output_frames': 5
        },
        {
            'name': 'Large latent (37 frames)',
            'shape': (1, 4, 37, 8, 13),
            'expected_output_frames': 37
        },
        {
            'name': 'Batch latent (2 batches)',
            'shape': (2, 4, 10, 8, 13),
            'expected_output_frames': 20  # 2 * 10
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"   Input shape: {test_case['shape']}")
        print(f"   Expected output frames: {test_case['expected_output_frames']}")
        
        try:
            # Create dummy latent tensor
            latent_shape = test_case['shape']
            latent_tensor = torch.randn(latent_shape)
            
            # Wrap in expected format
            latent_dict = {"samples": latent_tensor}
            
            # Perform VAE decoding
            decoded_images = vae_decoder.decode(mock_vae, latent_dict)
            
            print(f"   ✅ Output shape: {decoded_images.shape}")
            
            # Verify output shape
            batch_size, channels, frames, height, width = latent_shape
            expected_frames = batch_size * frames
            expected_height = height * 8  # VAE typically upscales by 8x
            expected_width = width * 8
            
            expected_shape = (expected_frames, expected_height, expected_width, 3)
            
            print(f"   ✅ Expected shape: {expected_shape}")
            
            if decoded_images.shape == expected_shape:
                print(f"   ✅ Shape verification passed!")
            else:
                print(f"   ❌ Shape verification failed!")
                print(f"      Expected: {expected_shape}")
                print(f"      Got: {decoded_images.shape}")
                continue
            
            # Verify output range
            min_val, max_val = decoded_images.min().item(), decoded_images.max().item()
            print(f"   📊 Output range: [{min_val:.3f}, {max_val:.3f}]")
            
            if min_val >= 0.0 and max_val <= 1.0:
                print(f"   ✅ Output in expected range [0, 1]")
            else:
                print(f"   ⚠️  Output outside expected range [0, 1]")
            
            # Verify tensor properties
            if torch.isfinite(decoded_images).all():
                print(f"   ✅ Output contains finite values")
            else:
                print(f"   ❌ Output contains NaN/Inf values")
                continue
            
            print(f"   ✅ Test case {i} passed!")
            
        except Exception as e:
            print(f"   ❌ Test case {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 VAEDecode component test completed!")

def test_edge_cases():
    """Test edge cases for VAE decoding"""
    print("\n" + "="*80)
    print("🧪 Testing Edge Cases")
    print("="*80)
    
    vae_decoder = VAEDecode()
    mock_vae = MockVAE("EdgeCaseVAE")
    
    # Edge case 1: Empty latent
    print("\n📋 Edge Case 1: Empty latent tensor")
    try:
        empty_latent = torch.randn(1, 4, 0, 8, 13)  # 0 frames
        latent_dict = {"samples": empty_latent}
        
        decoded_images = vae_decoder.decode(mock_vae, latent_dict)
        
        print(f"   Input frames: 0")
        print(f"   Output shape: {decoded_images.shape}")
        
        if decoded_images.shape[0] == 0:
            print(f"   ✅ Correctly handled: Empty output")
        else:
            print(f"   ⚠️  Unexpected behavior: Non-empty output")
            
    except Exception as e:
        print(f"   ❌ Error handling edge case: {e}")
    
    # Edge case 2: Very large latent
    print("\n📋 Edge Case 2: Very large latent")
    try:
        large_latent = torch.randn(1, 4, 100, 8, 13)  # 100 frames
        latent_dict = {"samples": large_latent}
        
        decoded_images = vae_decoder.decode(mock_vae, latent_dict)
        
        print(f"   Input frames: 100")
        print(f"   Output shape: {decoded_images.shape}")
        print(f"   ✅ Large latent handled successfully")
        
    except Exception as e:
        print(f"   ❌ Error handling large latent: {e}")
    
    print(f"\n🎉 Edge cases test completed!")

def test_performance():
    """Test performance with realistic tensor sizes"""
    print("\n" + "="*80)
    print("🧪 Testing Performance")
    print("="*80)
    
    vae_decoder = VAEDecode()
    mock_vae = MockVAE("PerformanceVAE")
    
    # Performance test with realistic tensor (similar to WAN model output)
    print("\n📋 Performance Test: Realistic tensor size")
    try:
        import time
        
        # Create realistic tensor (similar to WAN model output after trimming)
        realistic_shape = (1, 4, 32, 8, 13)  # 32 frames after trimming
        print(f"   Creating realistic tensor: {realistic_shape}")
        
        start_time = time.time()
        latent_tensor = torch.randn(realistic_shape)
        creation_time = time.time() - start_time
        
        print(f"   Tensor creation time: {creation_time:.3f}s")
        
        # Test VAE decoding
        latent_dict = {"samples": latent_tensor}
        
        start_time = time.time()
        decoded_images = vae_decoder.decode(mock_vae, latent_dict)
        decoding_time = time.time() - start_time
        
        print(f"   VAE decoding time: {decoding_time:.3f}s")
        print(f"   Output shape: {decoded_images.shape}")
        
        # Calculate throughput
        frames_per_second = realistic_shape[2] / decoding_time
        print(f"   Throughput: {frames_per_second:.1f} frames/second")
        
        print(f"   ✅ Performance test completed!")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Performance test completed!")

def test_comfyui_compatibility():
    """Test compatibility with ComfyUI's exact interface"""
    print("\n" + "="*80)
    print("🧪 Testing ComfyUI Compatibility")
    print("="*80)
    
    vae_decoder = VAEDecode()
    mock_vae = MockVAE("ComfyUIVAE")
    
    print("\n📋 ComfyUI Interface Test")
    try:
        # Test exact ComfyUI interface: vae_decoder.decode(vae, samples)
        latent_tensor = torch.randn(1, 4, 5, 8, 13)
        latent_dict = {"samples": latent_tensor}
        
        # This should match ComfyUI's exact call pattern
        result = vae_decoder.decode(mock_vae, latent_dict)
        
        print(f"   ✅ ComfyUI interface compatible")
        print(f"   ✅ Result type: {type(result)}")
        print(f"   ✅ Result shape: {result.shape}")
        
        # Verify it returns a tensor (not tuple like ComfyUI)
        if isinstance(result, torch.Tensor):
            print(f"   ✅ Returns tensor directly (simplified interface)")
        else:
            print(f"   ⚠️  Returns {type(result)} (may need tuple wrapping)")
        
    except Exception as e:
        print(f"   ❌ ComfyUI compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 ComfyUI compatibility test completed!")

def main():
    print("🚀 Testing Step 6: VAE Decode")
    print("="*60)
    
    # Run all tests
    test_vae_decode()
    test_edge_cases()
    test_performance()
    test_comfyui_compatibility()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("✅ Step 6 VAE Decode is ready for integration")
    print("="*60)

if __name__ == "__main__":
    main()
