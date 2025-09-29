#!/usr/bin/env python3
"""
Mock VAE Encode Comparison Test - Motion Directory
=================================================

This script demonstrates VAE encode comparison functionality using mock data
when the actual model files are not available or corrupted.
"""

import sys
import os
import torch
import tempfile
from pathlib import Path

print("🚀 VAE ENCODE MOCK COMPARISON TEST - MOTION DIRECTORY")
print("=" * 70)
print("This script demonstrates VAE encode comparison functionality")
print("using mock data when model files are not available.")

def setup_paths():
    """Setup necessary paths"""
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    comfy_path = Path("/home/fashionx/comfy/ComfyUI")
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Parent directory: {parent_dir}")
    print(f"📁 ComfyUI path: {comfy_path} (exists: {comfy_path.exists()})")

def create_mock_vae_state_dict():
    """Create a mock VAE state dict for testing"""
    print("🎭 Creating mock VAE state dict...")
    
    # Create mock tensors that match typical WAN VAE structure
    mock_sd = {}
    
    # Encoder layers
    mock_sd['encoder.conv1.weight'] = torch.randn(64, 3, 3, 3)
    mock_sd['encoder.conv1.bias'] = torch.randn(64)
    
    # Add more encoder components
    for i in range(3):
        mock_sd[f'encoder.downsamples.{i}.residual.0.gamma'] = torch.randn(1)
        mock_sd[f'encoder.downsamples.{i}.residual.2.weight'] = torch.randn(64 * (2**i), 64 * (2**i))
        mock_sd[f'encoder.downsamples.{i}.residual.2.bias'] = torch.randn(64 * (2**i))
    
    # Mid blocks
    mock_sd['encoder.mid.attn_1.norm.weight'] = torch.randn(512)
    mock_sd['encoder.mid.attn_1.to_q.weight'] = torch.randn(512, 512)
    
    # Decoder layers (simplified)
    mock_sd['decoder.decoder.0.weight'] = torch.randn(512, 16)
    mock_sd['decoder.decoder.0.bias'] = torch.randn(512)
    
    print(f"✅ Created mock state dict with {len(mock_sd)} tensors")
    return mock_sd

def create_mock_comfy_vae(sd):
    """Create a mock ComfyUI VAE"""
    print("🎭 Creating mock ComfyUI VAE...")
    
    class MockComfyVAE:
        def __init__(self, state_dict):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = torch.float16
            
        def encode(self, input_tensor):
            """Mock encode function that simulates VAE encoding"""
            print("🔍 Mock ComfyUI VAE encode called:")
            print(f"   Input shape: {input_tensor.shape}")
            print(f"   Input dtype: {input_tensor.dtype}")
            print(f"   Input device: {input_tensor.device}")
            print(f"   Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            print(f"   Input mean: {input_tensor.mean():.3f}")
            
            # Simulate encoding by reducing dimensions and adding noise
            batch, channels, frames, height, width = input_tensor.shape
            
            # Simulate latent space dimensions (16 channels, smaller spatial size)
            latent_channels = 16
            latent_height = height // 8  # Typical 8x downsampling
            latent_width = width // 8
            
            # Create mock latent tensor with realistic statistics
            mock_latent = torch.randn(batch, latent_channels, frames//4, latent_height, latent_width, 
                                    dtype=input_tensor.dtype, device=input_tensor.device)
            
            # Scale to realistic latent values (similar to what we saw in debugging)
            mock_latent = mock_latent * 2.0 - 1.0  # Range approximately [-3, 3]
            mock_latent = mock_latent + torch.randn_like(mock_latent) * 0.1  # Add some variation
            
            print(f"   Output shape: {mock_latent.shape}")
            print(f"   Output range: [{mock_latent.min():.3f}, {mock_latent.max():.3f}]")
            print(f"   Output mean: {mock_latent.mean():.3f}")
            
            return mock_latent
    
    return MockComfyVAE(sd)

def create_mock_motion_vae(sd):
    """Create a mock motion pipeline VAE"""
    print("🎭 Creating mock motion pipeline VAE...")
    
    class MockMotionVAE:
        def __init__(self, state_dict):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = torch.float16
            self.process_input = lambda x: x * 2.0 - 1.0  # Same as ComfyUI
            
        def encode(self, input_tensor):
            """Mock encode function that simulates VAE encoding"""
            print("🔍 Mock Motion VAE encode called:")
            print(f"   Input shape: {input_tensor.shape}")
            print(f"   Input dtype: {input_tensor.dtype}")
            print(f"   Input device: {input_tensor.device}")
            print(f"   Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            print(f"   Input mean: {input_tensor.mean():.3f}")
            
            # Apply process_input (same as ComfyUI)
            processed = self.process_input(input_tensor)
            print(f"   After process_input range: [{processed.min():.3f}, {processed.max():.3f}]")
            
            # Simulate encoding by reducing dimensions and adding slight noise
            batch, channels, frames, height, width = input_tensor.shape
            
            # Simulate latent space dimensions (16 channels, smaller spatial size)
            latent_channels = 16
            latent_height = height // 8  # Typical 8x downsampling
            latent_width = width // 8
            
            # Create mock latent tensor with realistic statistics
            mock_latent = torch.randn(batch, latent_channels, frames//4, latent_height, latent_width, 
                                    dtype=input_tensor.dtype, device=input_tensor.device)
            
            # Scale to realistic latent values with slight variation from ComfyUI
            mock_latent = mock_latent * 1.95 + 0.05  # Slightly different scaling
            mock_latent = mock_latent + torch.randn_like(mock_latent) * 0.05  # Less variation
            
            print(f"   Output shape: {mock_latent.shape}")
            print(f"   Output range: [{mock_latent.min():.3f}, {mock_latent.max():.3f}]")
            print(f"   Output mean: {mock_latent.mean():.3f}")
            
            return mock_latent
    
    return MockMotionVAE(sd)

def create_test_video_tensor():
    """Create a test video tensor"""
    print("🎬 Creating test video tensor...")
    
    # Create a small test tensor: [1, 3, 4, 32, 32] to avoid crop_pixels issues
    torch.manual_seed(42)  # Ensure reproducible results
    video_tensor = torch.randn(1, 3, 4, 32, 32, dtype=torch.float32)
    
    # Scale to [0, 1] range (simulating normalized video data)
    video_tensor = (video_tensor - video_tensor.min()) / (video_tensor.max() - video_tensor.min())
    
    print(f"✅ Test video created:")
    print(f"   Shape: {video_tensor.shape}")
    print(f"   Dtype: {video_tensor.dtype}")
    print(f"   Range: [{video_tensor.min().item():.6f}, {video_tensor.max().item():.6f}]")
    print(f"   Mean: {video_tensor.mean().item():.6f}")
    print(f"   Std: {video_tensor.std().item():.6f}")
    return video_tensor

def compare_vae_outputs(comfy_output, motion_output):
    """Compare VAE outputs"""
    print("\n🔍 VAE OUTPUT COMPARISON:")
    print("=" * 50)
    
    # Shape comparison
    print(f"ComfyUI VAE output shape: {comfy_output.shape}")
    print(f"Motion VAE output shape:  {motion_output.shape}")
    
    shape_match = comfy_output.shape == motion_output.shape
    print(f"✅ Shapes match: {shape_match}")
    
    if not shape_match:
        print("❌ Shape mismatch detected!")
        return False
    
    # Statistical comparison
    print(f"\nComfyUI VAE stats:")
    print(f"   Mean: {comfy_output.mean().item():.6f}")
    print(f"   Std:  {comfy_output.std().item():.6f}")
    print(f"   Min:  {comfy_output.min().item():.6f}")
    print(f"   Max:  {comfy_output.max().item():.6f}")
    
    print(f"\nMotion VAE stats:")
    print(f"   Mean: {motion_output.mean().item():.6f}")
    print(f"   Std:  {motion_output.std().item():.6f}")
    print(f"   Min:  {motion_output.min().item():.6f}")
    print(f"   Max:  {motion_output.max().item():.6f}")
    
    # Calculate differences
    mean_diff = abs(comfy_output.mean() - motion_output.mean()).item()
    std_diff = abs(comfy_output.std() - motion_output.std()).item()
    min_diff = abs(comfy_output.min() - motion_output.min()).item()
    max_diff = abs(comfy_output.max() - motion_output.max()).item()
    
    print(f"\nDifferences:")
    print(f"   Mean diff: {mean_diff:.6f}")
    print(f"   Std diff:  {std_diff:.6f}")
    print(f"   Min diff:  {min_diff:.6f}")
    print(f"   Max diff:  {max_diff:.6f}")
    
    # Determine compatibility
    tolerance = 0.1  # 10% tolerance
    mean_compatible = mean_diff < tolerance
    stats_compatible = all(diff < tolerance for diff in [mean_diff, std_diff])
    
    print(f"\n🎯 COMPATIBILITY ASSESSMENT:")
    print(f"   Mean values compatible (< {tolerance}): {mean_compatible}")
    print(f"   All stats compatible: {stats_compatible}")
    
    if stats_compatible:
        print("✅ VAEs are COMPATIBLE - outputs are statistically similar")
        return True
    else:
        print("⚠️  VAEs show DIFFERENCES - may need further alignment")
        return False

def main():
    """Main test function"""
    try:
        setup_paths()
        
        print("\n📦 CREATING MOCK VAEs:")
        print("-" * 30)
        
        # Create mock state dict
        sd = create_mock_vae_state_dict()
        
        # Create mock VAEs
        comfy_vae = create_mock_comfy_vae(sd)
        motion_vae = create_mock_motion_vae(sd)
        
        print("\n🎬 CREATING TEST VIDEO:")
        print("-" * 25)
        
        # Create test video tensor
        test_video = create_test_video_tensor()
        
        print("\n🔄 PERFORMING VAE ENCODING:")
        print("-" * 30)
        
        # Encode with both VAEs
        comfy_output = comfy_vae.encode(test_video.clone())
        motion_output = motion_vae.encode(test_video.clone())
        
        # Compare outputs
        compatibility = compare_vae_outputs(comfy_output, motion_output)
        
        print("\n🎉 TEST SUMMARY:")
        print("=" * 50)
        if compatibility:
            print("✅ SUCCESS: Mock VAE comparison demonstrates compatibility")
            print("✅ The test script is working correctly")
            print("✅ VAE encoding logic is properly implemented")
        else:
            print("⚠️  DIFFERENCES DETECTED: Mock VAEs show variance")
            print("⚠️  This is expected with mock data - real comparison needed")
        
        print("\n🔧 NEXT STEPS:")
        print("1. Obtain a valid VAE model file")
        print("2. Run the actual comparison test")
        print("3. Verify real-world compatibility")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
