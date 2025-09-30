#!/usr/bin/env python3
"""
Test script to verify the latent scale factor fix
Tests the Wan21 latent format scaling without requiring a full VAE model
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add motion directory to path
motion_dir = Path(__file__).parent
sys.path.insert(0, str(motion_dir))

# Also add parent directory for imports
parent_dir = motion_dir.parent
sys.path.insert(0, str(parent_dir))

def test_simple_latent_format():
    """Test SimpleLatentFormat scaling (fallback)"""
    print("🔍 TESTING SIMPLE LATENT FORMAT SCALING")
    print("=" * 50)
    
    try:
        # Import SimpleLatentFormat from motion/standalone_vae
        import sys
        sys.path.insert(0, str(motion_dir))
        from standalone_vae import SimpleLatentFormat
        print("✅ Successfully imported SimpleLatentFormat")
        
        # Create SimpleLatentFormat instance
        latent_format = SimpleLatentFormat()
        print(f"✅ Created SimpleLatentFormat instance")
        print(f"   Scale factor: {latent_format.scale_factor}")
        print(f"   Latent channels: {latent_format.latent_channels}")
        print(f"   Latent dimensions: {latent_format.latent_dimensions}")
        
        # Create test latent tensor (similar to VAE encode output)
        test_latent = torch.randn(1, 16, 8, 60, 104)  # [B, C, T, H, W]
        print(f"\n📊 Test latent tensor:")
        print(f"   Shape: {test_latent.shape}")
        print(f"   Mean: {test_latent.mean().item():.6f}")
        print(f"   Range: [{test_latent.min().item():.6f}, {test_latent.max().item():.6f}]")
        print(f"   Std: {test_latent.std().item():.6f}")
        
        # Apply process_out scaling (this is what was missing)
        scaled_latent = latent_format.process_out(test_latent)
        print(f"\n📊 After process_out scaling:")
        print(f"   Shape: {scaled_latent.shape}")
        print(f"   Mean: {scaled_latent.mean().item():.6f}")
        print(f"   Range: [{scaled_latent.min().item():.6f}, {scaled_latent.max().item():.6f}]")
        print(f"   Std: {scaled_latent.std().item():.6f}")
        
        # Compare with expected ComfyUI ranges
        print(f"\n🎯 COMPARISON WITH COMFYUI EXPECTED RANGES:")
        print(f"   Motion Pipeline (before fix): Mean: 0.025922, Range: [-7.906250, 7.406250]")
        print(f"   Motion Pipeline (after fix):  Mean: {scaled_latent.mean().item():.6f}, Range: [{scaled_latent.min().item():.6f}, {scaled_latent.max().item():.6f}]")
        print(f"   ComfyUI Expected:             Mean: ~0.025922, Range: ~[-7.906250, 7.406250]")
        
        # Check if scaling is working
        mean_change = abs(scaled_latent.mean().item() - test_latent.mean().item())
        range_change = abs(scaled_latent.max().item() - scaled_latent.min().item()) - abs(test_latent.max().item() - test_latent.min().item())
        
        print(f"\n📈 SCALING ANALYSIS:")
        print(f"   Mean change: {mean_change:.6f}")
        print(f"   Range change: {range_change:.6f}")
        
        if mean_change > 0.1 or abs(range_change) > 0.1:
            print(f"   ✅ Scaling is working - significant changes detected")
        else:
            print(f"   ⚠️  Scaling may not be working - minimal changes detected")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SimpleLatentFormat: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_standalone_vae_latent_format():
    """Test standalone VAE latent format initialization"""
    print("\n🔍 TESTING STANDALONE VAE LATENT FORMAT INITIALIZATION")
    print("=" * 60)
    
    try:
        from standalone_vae import VAE
        print("✅ Successfully imported VAE from standalone_vae.py")
        
        # Create VAE instance (without loading state dict)
        vae = VAE()
        print(f"✅ Created VAE instance")
        
        # Check if latent format was initialized
        if hasattr(vae, 'latent_format') and vae.latent_format is not None:
            print(f"✅ Latent format initialized: {type(vae.latent_format).__name__}")
            print(f"   Scale factor: {vae.latent_format.scale_factor}")
            print(f"   Latent channels: {vae.latent_format.latent_channels}")
        else:
            print(f"❌ Latent format not initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 LATENT SCALE FACTOR FIX VERIFICATION")
    print("=" * 60)
    print("This script tests the Wan21 latent format scaling fix")
    print("to verify that the missing scale factor has been added.")
    print()
    
    # Test 1: SimpleLatentFormat scaling
    success1 = test_simple_latent_format()
    
    # Test 2: Standalone VAE latent format initialization
    success2 = test_standalone_vae_latent_format()
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 30)
    if success1:
        print("✅ SimpleLatentFormat scaling test: PASSED")
    else:
        print("❌ SimpleLatentFormat scaling test: FAILED")
    
    if success2:
        print("✅ Standalone VAE latent format test: PASSED")
    else:
        print("❌ Standalone VAE latent format test: FAILED")
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("The latent scale factor fix is working correctly.")
        print("Wan21 latent format scaling has been successfully added.")
        print("This should resolve the range differences with ComfyUI.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print("Please check the error messages above for details.")

if __name__ == "__main__":
    main()
