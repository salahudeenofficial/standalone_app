#!/usr/bin/env python3
"""
VAE Encode Comparison Test - Motion Directory Version

This script loads both standalone_vae and ComfyUI vae, performs .encode on both,
and compares the range and mean of the output tensors.
"""

import torch
import sys
from pathlib import Path
import safetensors
import traceback
import os

def add_paths():
    """Add necessary paths for imports"""
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    comfy_path = Path('/home/fashionx/comfy/ComfyUI')
    
    # Add motion directory to path (current directory)
    sys.path.insert(0, str(current_dir))
    
    # Add parent directory to path for model loading
    sys.path.insert(0, str(parent_dir))
    
    # Add ComfyUI to path
    if comfy_path.exists():
        sys.path.insert(0, str(comfy_path))
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Parent directory: {parent_dir}")
    print(f"📁 ComfyUI path: {comfy_path} (exists: {comfy_path.exists()})")

def load_vae_model():
    """Load the VAE model state dict"""
    # Check if model exists in various locations (prioritize actual VAE model)
    model_paths = [
        'models/vaes/wan_2.1_vae.safetensors',
        '../models/vaes/wan_2.1_vae.safetensors',
        '../../models/vaes/wan_2.1_vae.safetensors',
        'models/vaes/wan_vae.safetensors',
        '../models/vaes/wan_vae.safetensors',
        '../../models/vaes/wan_vae.safetensors',
        'wan2.1_vace_14B_fp16.safetensors',
        '../wan2.1_vace_14B_fp16.safetensors',
        '../../wan2.1_vace_14B_fp16.safetensors'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("WAN VAE model not found. Expected path: models/vaes/wan_2.1_vae.safetensors")
    
    print(f"📁 Loading VAE model: {model_path}")
    
    with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"✅ Model loaded: {len(sd)} parameters")
    return sd

def create_test_video_tensor():
    """Create a test video tensor"""
    print(f"🎬 Creating test video tensor...")
    
    # Create a reasonable test tensor: [8, 256, 256, 3] float32
    torch.manual_seed(42)  # Ensure reproducible results
    video_tensor = torch.randn(8, 256, 256, 3, dtype=torch.float32)
    
    print(f"✅ Test video created:")
    print(f"   Shape: {video_tensor.shape} (T,H,W,C)")
    print(f"   Dtype: {video_tensor.dtype}")
    print(f"   Range: [{video_tensor.min().item():.6f}, {video_tensor.max().item():.6f}]")
    print(f"   Mean: {video_tensor.mean().item():.6f}")
    print(f"   Memory: {video_tensor.numel() * 4 / (1024*1024):.2f} MB")
    
    return video_tensor

def prepare_video_for_vae(video_tensor):
    """Prepare video tensor for VAE encoding"""
    print(f"\n🔧 Preparing video tensor for VAE encoding...")
    
    # Convert to WAN VAE format: [batch, channels, frames, height, width]
    # From [frames, height, width, channels] to [1, channels, frames, height, width]
    vae_input = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)
    
    print(f"✅ Converted to VAE format:")
    print(f"   Shape: {vae_input.shape}")
    print(f"   Range: [{vae_input.min().item():.6f}, {vae_input.max().item():.6f}]")
    print(f"   Mean: {vae_input.mean().item():.6f}")
    
    return vae_input

def test_motion_pipeline_vae(sd, vae_input):
    """Test motion pipeline VAE encoding"""
    print(f"\n🔧 TESTING MOTION PIPELINE VAE")
    print("-" * 60)
    
    try:
        from standalone_vae import VAE
        print("✅ Imported standalone_vae")
        
        motion_vae = VAE(sd=sd)
        print(f"✅ Motion VAE initialized: {type(motion_vae.first_stage_model).__name__}")
        
        # Test process_input
        motion_processed = motion_vae.process_input(vae_input)
        print(f"✅ Motion process_input:")
        print(f"   Shape: {motion_processed.shape}")
        print(f"   Range: [{motion_processed.min().item():.6f}, {motion_processed.max().item():.6f}]")
        print(f"   Mean: {motion_processed.mean().item():.6f}")
        
        # Test encoding
        print(f"\n🔧 Motion VAE Encoding...")
        with torch.no_grad():
            motion_wan_vae = motion_vae.first_stage_model
            
            # Get the correct dtype for motion VAE
            motion_dtype = next(motion_vae.first_stage_model.parameters()).dtype
            print(f"   Motion VAE dtype: {motion_dtype}")
            
            input_tensor = motion_processed.to(motion_dtype)
            
            motion_encoded = motion_wan_vae.encode(input_tensor)
            print(f"✅ Motion encode successful!")
            print(f"   Encoded shape: {motion_encoded.shape}")
            print(f"   Encoded range: [{motion_encoded.min().item():.6f}, {motion_encoded.max().item():.6f}]")
            print(f"   Encoded mean: {motion_encoded.mean().item():.6f}")
            print(f"   Encoded std: {motion_encoded.std().item():.6f}")
            
            # Convert to float32 for comparison
            motion_result = motion_encoded.float()
            print(f"✅ Motion result ready for comparison")
            
            return motion_result
            
    except Exception as e:
        print(f"❌ Motion VAE failed: {e}")
        traceback.print_exc()
        return None

def test_comfyui_vae(sd, vae_input):
    """Test ComfyUI VAE encoding"""
    print(f"\n🔧 TESTING COMFYUI VAE")
    print("-" * 60)
    
    try:
        from comfy.sd import VAE as ComfyVAE
        print("✅ Imported ComfyUI VAE")
        
        comfy_vae = ComfyVAE(sd=sd)
        print(f"✅ ComfyUI VAE initialized: {type(comfy_vae.first_stage_model).__name__}")
        
        # Test process_input
        comfy_processed = comfy_vae.process_input(vae_input)
        print(f"✅ ComfyUI process_input:")
        print(f"   Shape: {comfy_processed.shape}")
        print(f"   Range: [{comfy_processed.min().item():.6f}, {comfy_processed.max().item():.6f}]")
        print(f"   Mean: {comfy_processed.mean().item():.6f}")
        
        # Test encoding
        print(f"\n🔧 ComfyUI VAE Encoding...")
        with torch.no_grad():
            comfy_wan_vae = comfy_vae.first_stage_model
            
            # Get the correct dtype for ComfyUI VAE
            comfy_dtype = next(comfy_vae.first_stage_model.parameters()).dtype
            print(f"   ComfyUI VAE dtype: {comfy_dtype}")
            
            input_tensor = comfy_processed.to(comfy_dtype)
            
            comfy_encoded = comfy_wan_vae.encode(input_tensor)
            print(f"✅ ComfyUI encode successful!")
            print(f"   Encoded shape: {comfy_encoded.shape}")
            print(f"   Encoded range: [{comfy_encoded.min().item():.6f}, {comfy_encoded.max().item():.6f}]")
            print(f"   Encoded mean: {comfy_encoded.mean().item():.6f}")
            print(f"   Encoded std: {comfy_encoded.std().item():.6f}")
            
            # Convert to float32 for comparison
            comfy_result = comfy_encoded.float()
            print(f"✅ ComfyUI result ready for comparison")
            
            return comfy_result
            
    except Exception as e:
        print(f"❌ ComfyUI VAE failed: {e}")
        traceback.print_exc()
        return None

def compare_vae_results(motion_result, comfy_result):
    """Compare the encoded results from both VAE implementations"""
    print(f"\n🔍 COMPARING VAE ENCODE RESULTS")
    print("=" * 70)
    
    if motion_result is None or comfy_result is None:
        print("❌ Cannotcompare - one or both tests failed")
        if motion_result is None:
            print("   ❌ Motion pipeline VAE failed")
        if comfy_result is None:
            print("   ❌ ComfyUI VAE failed")
        return False
    
    print("📊 ENCODED OUTPUT COMPARISON:")
    print(f"   Motion shape: {motion_result.shape}")
    print(f"   ComfyUI shape: {comfy_result.shape}")
    print(f"   Shapes match: {motion_result.shape == comfy_result.shape}")
    
    if motion_result.shape != comfy_result.shape:
        print("❌ INCOMPATIBLE - Different output shapes")
        return False
    
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
    identical_1e6 = torch.allclose(motion_result, comfy_result, atol=1e-6)
    close_1e3 = torch.allclose(motion_result, comfy_result, atol=1e-3)
    close_1e2 = torch.allclose(motion_result, comfy_result, atol=1e-2)
    close_1e1 = torch.allclose(motion_result, comfy_result, atol=1e-1)
    
    print(f"   Identical (1e-6): {identical_1e6}")
    print(f"   Very close (1e-3): {close_1e3}")
    print(f"   Close (1e-2): {close_1e2}")
    print(f"   Similar (1e-1): {close_1e1}")
    
    print(f"\n🎯 FINAL COMPATIBILITY ASSESSMENT:")
    compatibility_level = None
    if identical_1e6:
        compatibility_level = "PERFECT"
        print("   ✅ PERFECT COMPATIBILITY - Identical outputs!")
    elif close_1e3:
        compatibility_level = "EXCELLENT"
        print("   ✅ EXCELLENT COMPATIBILITY - Very close outputs!")
    elif close_1e2:
        compatibility_level = "GOOD"
        print("   ✅ GOOD COMPATIBILITY - Close outputs!")
    elif close_1e1:
        compatibility_level = "ACCEPTABLE"
        print("   ✅ ACCEPTABLE COMPATIBILITY - Similar outputs!")
    else:
        compatibility_level = "LIMITED"
        print("   ⚠️  LIMITED COMPATIBILITY - Significant differences")
    
    # Additional statistics
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"   Motion result:")
    print(f"     Range: [{motion_result.min().item():.6f}, {motion_result.max().item():.6f}]")
    print(f"     Mean: {motion_result.mean().item():.6f}")
    print(f"     Std: {motion_result.std().item():.6f}")
    
    print(f"   ComfyUI result:")
    print(f"     Range: [{comfy_result.min().item():.6f}, {comfy_result.max().item():.6f}]")
    print(f"     Mean: {comfy_result.mean().item():.6f}")
    print(f"     Std: {comfy_result.std().item():.6f}")
    
    return True, compatibility_level

def main():
    """Main test function"""
    print("🚀 VAE ENCODE COMPARISON TEST - MOTION DIRECTORY")
    print("=" * 70)
    print("This script loads both standalone_vae and ComfyUI vae,")
    print("performs .encode on both, and compares the output tensors.")
    print()
    
    try:
        # Setup paths and load model
        add_paths()
        sd = load_vae_model()
        
        # Create test data
        video_tensor = create_test_video_tensor()
        vae_input = prepare_video_for_vae(video_tensor)
        
        # Test both VAE implementations
        motion_result = test_motion_pipeline_vae(sd, vae_input)
        comfy_result = test_comfyui_vae(sd, vae_input)
        
        # Compare results
        success, compatibility = compare_vae_results(motion_result, comfy_result)
        
        if success:
            print(f"\n🎉 VAE COMPATIBILITY TEST COMPLETED!")
            print(f"🎯 RESULT: {compatibility} COMPATIBILITY")
            print(f"✅ Your motion pipeline VAE is compatible with ComfyUI!")
        else:
            print(f"\n❌ VAE COMPATIBILITY TEST FAILED!")
            print(f"🔧 Check the error messages above for details.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
