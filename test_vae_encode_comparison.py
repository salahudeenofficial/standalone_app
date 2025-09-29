#!/usr/bin/env python3
"""
VAE Encode Comparison Script

This script loads the control_video, initializes both standalone_vae and ComfyUI vae,
performs .encode on both, and compares the range and mean of the output tensors.
"""

import torch
import sys
from pathlib import Path
import safetensors
import torchvision
import traceback

# Add motion pipeline to path
sys.path.insert(0, str(Path('.').absolute() / 'motion'))

def load_vae_model():
    """Load the VAE model state dict"""
    model_path = 'models/vaes/wan_2.1_vae.safetensors'
    print(f"📁 Loading VAE model: {model_path}")
    
    with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"✅ Model loaded: {len(sd)} parameters")
    return sd

def load_control_video():
    """Load control video from safu.mp4"""
    video_path = 'safu.mp4'
    print(f"📹 Loading control video: {video_path}")
    
    try:
        # Load video using torchvision
        video_tensor, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        
        print(f"✅ Video loaded:")
        print(f"   Shape: {video_tensor.shape} (T,H,W,C)")
        print(f"   Dtype: {video_tensor.dtype}")
        print(f"   Range: [{video_tensor.min().item()}, {video_tensor.max().item()}]")
        print(f"   Mean: {video_tensor.float().mean().item():.6f}")
        
        return video_tensor
        
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        print(f"📊 Creating dummy control video instead...")
        
        # Create dummy video tensor with similar dimensions
        dummy_video = torch.randint(0, 256, (37, 768, 576, 3), dtype=torch.uint8)
        print(f"✅ Dummy video created:")
        print(f"   Shape: {dummy_video.shape} (T,H,W,C)")
        print(f"   Dtype: {dummy_video.dtype}")
        print(f"   Range: [{dummy_video.min().item()}, {dummy_video.max().item()}]")
        
        return dummy_video

def prepare_video_tensor(video_tensor):
    """Prepare video tensor for VAE encoding"""
    print(f"\n🔧 Preparing video tensor for VAE encoding...")
    
    # Convert uint8 to float32 if needed
    if video_tensor.dtype == torch.uint8:
        video_float = video_tensor.float() / 255.0
        print(f"✅ Converted from uint8 to float32 [0,1] range")
    else:
        video_float = video_tensor.float()
        print(f"✅ Using float32 tensor")
    
    print(f"   Prepared range: [{video_float.min().item():.6f}, {video_float.max().item():.6f}]")
    print(f"   Prepared mean: {video_float.mean().item():.6f}")
    
    # Convert to WAN VAE format: [batch, channels, frames, height, width]
    # From [frames, height, width, channels] to [1, channels, frames, height, width]
    vae_input = video_float.permute(3, 0, 1, 2).unsqueeze(0)
    
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
            print(f"   Encoded range: [{motion_encoded.min().item():.6f}, {motion_encoded.max().item():.6f}]")
            print(f"   Encoded mean: {motion_encoded.mean().item():.6f}")
            print(f"   Encoded std: {motion_encoded.std().item():.6f}")
            
            # Convert to float32 for comparison
            motion_result = motion_encoded.float()
            print(f"✅ Converted to float32 for comparison")
            
            return motion_result
            
    except Exception as e:
        print(f"❌ Motion VAE failed: {e}")
        traceback.print_exc()
        return None

def test_comfyui_vae(sd, vae_input):
    """Test ComfyUI VAE encoding"""
    print(f"\n🔧 TESTING COMFYUI VAE")
    print("-" * 50)
    
    try:
        # Add ComfyUI to path
        sys.path.insert(0, '/home/fashionx/comfy/ComfyUI')
        from comfy.sd import VAE as ComfyVAE
        
        comfy_vae = ComfyVAE(sd=sd)
        print(f"✅ ComfyUI VAE initialized: {type(comfy_vae.first_stage_model).__name__}")
        
        # Test process_input
        comfy_processed = comfy_vae.process_input(vae_input)
        print(f"✅ ComfyUI process_input: {comfy_processed.shape}")
        print(f"   Range: [{comfy_processed.min().item():.6f}, {comfy_processed.max().item():.6f}]")
        print(f"   Mean: {comfy_processed.mean().item():.6f}")
        
        # Test encoding
        print(f"\n🔧 ComfyUI VAE Encoding...")
        with torch.no_grad():
            comfy_wan_vae = comfy_vae.first_stage_model
            input_tensor = comfy_processed.to(torch.bfloat16)  # Match ComfyUI VAE dtype
            
            comfy_encoded = comfy_wan_vae.encode(input_tensor)
            print(f"✅ ComfyUI encode successful!")
            print(f"   Encoded shape: {comfy_encoded.shape}")
            print(f"   Encoded range: [{comfy_encoded.min().item():.6f}, {comfy_encoded.max().item():.6f}]")
            print(f"   Encoded mean: {comfy_encoded.mean().item():.6f}")
            print(f"   Encoded std: {comfy_encoded.std().item():.6f}")
            
            # Convert to float32 for comparison
            comfy_result = comfy_encoded.float()
            print(f"✅ Converted to float32 for comparison")
            
            return comfy_result
            
    except Exception as e:
        print(f"❌ ComfyUI VAE failed: {e}")
        traceback.print_exc()
        return None

def compare_results(motion_result, comfy_result):
    """Compare the encoded results from both VAE implementations"""
    print(f"\n🔍 COMPARING VAE ENCODE RESULTS")
    print("=" * 60)
    
    if motion_result is None or comfy_result is None:
        print("❌ Cannot compare - one or both tests failed")
        if motion_result is None:
            print("   Motion pipeline VAE failed")
        if comfy_result is None:
            print("   ComfyUI VAE failed")
        return
    
    print("📊 ENCODED OUTPUT COMPARISON:")
    print(f"   Motion shape: {motion_result.shape}")
    print(f"   ComfyUI shape: {comfy_result.shape}")
    print(f"   Shapes match: {motion_result.shape == comfy_result.shape}")
    
    if motion_result.shape == comfy_result.shape:
        # Compare tensor values
        diff = torch.abs(motion_result - comfy_result)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        std_diff = diff.std().item()
        
        print(f"\n📊 DETAILED COMPARISON:")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Std difference: {std_diff:.6f}")
        
        print(f"\n🎯 COMPATIBILITY TOLERANCE CHECK:")
        print(f"   Identical (1e-6): {torch.allclose(motion_result, comfy_result, atol=1e-6)}")
        print(f"   Very close (1e-3): {torch.allclose(motion_result, comfy_result, atol=1e-3)}")
        print(f"   Close (1e-2): {torch.allclose(motion_result, comfy_result, atol=1e-2)}")
        print(f"   Similar (1e-1): {torch.allclose(motion_result, comfy_result, atol=1e-1)}")
        
        print(f"\n🎯 FINAL COMPATIBILITY ASSESSMENT:")
        if torch.allclose(motion_result, comfy_result, atol=1e-6):
            print("   ✅ PERFECT COMPATIBILITY - Identical outputs!")
        elif torch.allclose(motion_result, comfy_result, atol=1e-3):
            print("   ✅ EXCELLENT COMPATIBILITY - Very close outputs!")
        elif torch.allclose(motion_result, comfy_result, atol=1e-2):
            print("   ✅ GOOD COMPATIBILITY - Close outputs!")
        elif torch.allclose(motion_result, comfy_result, atol=1e-1):
            print("   ✅ ACCEPTABLE COMPATIBILITY - Similar outputs!")
        else:
            print("   ⚠️  LIMITED COMPATIBILITY - Significant differences")
            
        # Additional statistics
        print(f"\n📊 STATISTICAL SUMMARY:")
        print(f"   Motion result - Range: [{motion_result.min().item():.6f}, {motion_result.max().item():.6f}], Mean: {motion_result.mean().item():.6f}")
        print(f"   ComfyUI result - Range: [{comfy_result.min().item():.6f}, {comfy_result.max().item():.6f}], Mean: {comfy_result.mean().item():.6f}")
        
    else:
        print("   ❌ INCOMPATIBLE - Different output shapes")

def main():
    """Main test function"""
    print("🚀 VAE ENCODE COMPARISON TEST")
    print("=" * 60)
    print("This script loads control_video, initializes both VAE implementations,")
    print("performs .encode on both, and compares the output tensors.")
    print()
    
    try:
        # Load model and video
        sd = load_vae_model()
        video_tensor = load_control_video()
        
        # Prepare video tensor for VAE
        vae_input = prepare_video_tensor(video_tensor)
        
        # Test both VAE implementations
        motion_result = test_motion_pipeline_vae(sd, vae_input)
        comfy_result = test_comfyui_vae(sd, vae_input)
        
        # Compare results
        compare_results(motion_result, comfy_result)
        
        print("\n🎉 VAE ENCODE COMPARISON TEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
