#!/usr/bin/env python3
"""
Fast VAE Encode Comparison Script

This script creates a smaller video tensor, loads both VAE implementations,
performs .encode on both, and compares the range and mean of the output tensors.
"""

import torch
import sys
from pathlib import Path
import safetensors
import traceback

# Add motion pipeline to path


def load_vae_model():
    """Load the VAE model state dict"""
    model_path = 'models/vaes/wan_2.1_vae.safetensors'
    print(f"📁 Loading VAE model: {model_path}")
    
    with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"✅ Model loaded: {len(sd)} parameters")
    return sd

def create_sample_video_tensor():
    """Create a sample video tensor with smaller dimensions"""
    print(f"🎬 Creating sample video tensor...")
    
    # Create a smaller video tensor: [8, 128, 96, 3] float32
    torch.manual_seed(42)  # Ensure reproducible results
    video_tensor = torch.randn(8, 128, 96, 3, dtype=torch.float32)
    
    print(f"✅ Sample video created:")
    print(f"   Shape: {video_tensor.shape} (T,H,W,C)")
    print(f"   Dtype: {video_tensor.dtype}")
    print(f"   Range: [{video_tensor.min().item():.6f}, {video_tensor.max().item():.6f}]")
    print(f"   Mean: {video_tensor.mean().item():.6f}")
    print(f"   Memory: {video_tensor.numel() * 4 / (1024*1024):.2f} MB")
    
    return video_tensor

def prepare_video_tensor(video_tensor):
    """Prepare video tensor for VAE encoding"""
    print(f"\n🔧 Preparing video tensor for VAE encoding...")
    
    print(f"   Input range: [{video_tensor.min().item():.6f}, {video_tensor.max().item():.6f}]")
    print(f"   Input mean: {video_tensor.mean().item():.6f}")
    
    # Convert to WAN VAE format: [batch, channels, frames, height, width]
    # From [frames, height, width, channels] to [1, channels, frames, height, width]
    vae_input = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    
    print(f"✅ Converted to VAE format: {vae_input.shape}")
    print(f"   VAE input range: [{vae_input.min().item():.6f}, {vae_input.max().item():.6f}]")
    print(f"   VAE input mean: {vae_input.mean().item():.6f}")
    
    return vae_input

def test_motion_pipeline_vae(sd, vae_input):
    """Test motion pipeline VAE encoding"""
    print(f"\n🔧 TESTING MOTION PIPELINE VAE")
    print("-" * 50)
    
    try:
        from standalone_vae import VAE
        motion_vae = VAE(sd=sd)
        print(f"✅ Motion VAE initialized: {type(motion_vae.first_stage_model).__name__}")
        
        # Test process_input
        motion_processed = motion_vae.process_input(vae_input)
        print(f"✅ Motion process_input: {motion_processed.shape}")
        print(f"   Range: [{motion_processed.min().item():.6f}, {motion_processed.max().item():.6f}]")
        print(f"   Mean: {motion_processed.mean().item():.6f}")
        
        # Test encoding
        print(f"\n🔧 Motion VAE Encoding...")
        with torch.no_grad():
            motion_wan_vae = motion_vae.first_stage_model
            input_tensor = motion_processed.to(torch.float16)  # Match motion VAE dtype
            
            motion_encoded = motion_wan_vae.encode(input_tensor)
            print(f"✅ Motion encode successful!")
            print(f"   Encoded shape: {motion_encoded.shape}")
            print(f"   Encoded range: [{motion_encoded.min().item():.6f}, {motion_encoded.max().item(): VAE ENCODE COMPARISON TEST - FAST VERSION")
print("=" * 60)

try:
    # Load model and create sample video
    sd = load_vae_model()
    video_tensor = create_sample_video_tensor()
    
    # Prepare video tensor for VAE
    vae_input = prepare_video_tensor(video_tensor)
    
    # Test both VAE implementations
    motion_result = test_motion_pipeline_vae(sd, vae_input)
    comfy_result = test_comfyui_vae(sd, vae_input)
    
    # Compare results
    compare_results(motion_result, comfy_result)
    
    print("\n🎉 FAST VAE ENCODE COMPARISON TEST COMPLETED!")
    
except Exception as e:
    print(f"❌ Test failed with error: {e}")
    traceback.print_exc()

if __name__ == "__main__":
    main()
