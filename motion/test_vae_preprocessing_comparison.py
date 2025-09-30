#!/usr/bin/env python3
"""
VAE Preprocessing Comparison Test
================================

This script compares preprocessing between:
1. Motion pipeline preprocessing (current test script)
2. ComfyUI workflow_api_2.py preprocessing

It tests both VAEs with the same preprocessing to identify differences
that cause varying outputs.
"""

import os
import sys
import torch
import traceback
import safetensors

# Add paths for imports
def add_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, current_dir)
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'comfy'))

def load_vae_model():
    """Load VAE model from various possible locations"""
    print("🔍 Searching for VAE model files...")
    
    # Search paths
    search_paths = [
        "models/vaes/wan_vae.safetensors",
        "./models/vaes/wan_vae.safetensors",
        "../models/vaes/wan_vae.safetensors",
        "../../models/vaes/wan_vae.safetensors",
        "models/vaes/wan_2.1_vae.safetensors",
        "./models/vaes/wan_2.1_vae.safetensors",
        "models/vaes/wan2.1_vae.safetensors",
        "./models/vaes/wan2.1_vae.safetensors",
        "wan2.1_vace_14B_fp16.safetensors",
        "../wan2.1_vace_14B_fp16.safetensors"
    ]
    
    model_path = None
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            model_path = path
            break
        else:
            print(f"❌ Not found: {path}")
    
    if not model_path:
        raise FileNotFoundError("WAN VAE model not found in any expected location")
    
    print(f"📁 Loading VAE model: {model_path}")
    
    # Try safetensors first
    try:
        with safetensors.safe_open(model_path, framework='pt', device='cpu') as f:
            sd = {}
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        print(f"✅ Model loaded successfully with safetensors: {len(sd)} parameters")
        return sd
    except Exception as e:
        print(f"❌ Error loading with safetensors: {e}")
        # Try torch.load as fallback
        try:
            print("🔄 Trying alternative loading method...")
            sd = torch.load(model_path, map_location='cpu', weights_only=False)
            print(f"✅ Model loaded successfully with torch: {len(sd)} parameters")
            return sd
        except Exception as e2:
            print(f"❌ Error loading with torch: {e2}")
            raise e  # Re-raise the original safetensors error

def create_test_video():
    """Create a small test video tensor"""
    print(f"\n🎬 Creating test video tensor...")
    
    # Create a small test video: 8 frames, 256x256, 3 channels
    video_tensor = torch.randn(8, 256, 256, 3, dtype=torch.float32)
    
    print(f"✅ Test video created:")
    print(f"   Shape: {video_tensor.shape} (T,H,W,C)")
    print(f"   Dtype: {video_tensor.dtype}")
    print(f"   Range: [{video_tensor.min().item():.6f}, {video_tensor.max().item():.6f}]")
    print(f"   Mean: {video_tensor.mean().item():.6f}")
    print(f"   Memory: {video_tensor.numel() * 4 / (1024*1024):.2f} MB")
    
    return video_tensor

def motion_pipeline_preprocessing(video_tensor):
    """Motion pipeline preprocessing (current test script)"""
    print(f"\n🔧 MOTION PIPELINE PREPROCESSING:")
    print(f"   Using EXACT same preprocessing as motion pipeline")
    
    length, height, width, channels = video_tensor.shape
    
    # Create control mask using ComfyUI-compatible method (exact match to WanVaceToVideo)
    mask = torch.ones((length, height, width, 1), device=video_tensor.device)
    
    # Split control video by mask (EXACT pipeline logic)
    control_video = video_tensor - 0.5  # Center around 0
    inactive = (control_video * (1 - mask)) + 0.5  # Inactive regions
    reactive = (control_video * mask) + 0.5        # Active/controlled regions
    
    print(f"✅ Motion pipeline preprocessing completed:")
    print(f"   Control video (after -0.5):")
    print(f"     Shape: {control_video.shape}")
    print(f"     Range: [{control_video.min().item():.6f}, {control_video.max().item():.6f}]")
    print(f"     Mean: {control_video.mean().item():.6f}")
    print(f"   Reactive tensor:")
    print(f"     Shape: {reactive.shape}")
    print(f"     Range: [{reactive.min().item():.6f}, {reactive.max().item():.6f}]")
    print(f"     Mean: {reactive.mean().item():.6f}")
    print(f"   Inactive tensor:")
    print(f"     Shape: {inactive.shape}")
    print(f"     Range: [{inactive.min().item():.6f}, {inactive.max().item():.6f}]")
    print(f"     Mean: {inactive.mean().item():.6f}")
    
    return reactive, inactive

def comfyui_workflow_preprocessing(video_tensor):
    """ComfyUI workflow_api_2.py preprocessing"""
    print(f"\n🔧 COMFYUI WORKFLOW PREPROCESSING:")
    print(f"   Using preprocessing from workflow_api_2.py")
    
    # From workflow_api_2.py lines 3059-3074:
    # Normalize control video to [0,1] range if needed
    if video_tensor.dtype == torch.uint8:
        print(f"   🔧 Normalizing control video from uint8 to [0,1] range...")
        control_video_tensor = video_tensor.float() / 255.0
    elif video_tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        print(f"   🔧 Normalizing control video from {video_tensor.dtype} to [0,1] range...")
        # For integer types, assume they're in [0,255] range
        control_video_tensor = video_tensor.float() / 255.0
    elif video_tensor.dtype != torch.float32:
        print(f"   🔧 Converting control video to float32...")
        control_video_tensor = video_tensor.float()
    else:
        control_video_tensor = video_tensor.clone()
    
    # Verify normalization
    min_val = torch.min(control_video_tensor).item()
    max_val = torch.max(control_video_tensor).item()
    print(f"   ✅ Control video normalized: range [{min_val:.6f}, {max_val:.6f}], dtype: {control_video_tensor.dtype}")
    
    # Note: workflow_api_2.py doesn't do the mask-based splitting
    # It just normalizes and passes the video directly to WanVaceToVideo
    print(f"✅ ComfyUI workflow preprocessing completed:")
    print(f"   Control video tensor:")
    print(f"     Shape: {control_video_tensor.shape}")
    print(f"     Range: [{control_video_tensor.min().item():.6f}, {control_video_tensor.max().item():.6f}]")
    print(f"     Mean: {control_video_tensor.mean().item():.6f}")
    
    return control_video_tensor

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
    """Test Motion Pipeline VAE encoding"""
    print(f"\n🔧 TESTING MOTION PIPELINE VAE")
    print("-" * 60)
    
    try:
        from standalone_vae import VAE
        print("✅ Imported standalone_vae")
        
        motion_vae = VAE(sd=sd)
        print(f"✅ Motion VAE initialized: {type(motion_vae.first_stage_model).__name__}")
        
        print(f"\n🔧 Motion VAE Encoding...")
        print(f"   Motion VAE dtype: {motion_vae.vae_dtype}")
        
        motion_encoded = motion_vae.encode(vae_input)
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
        
        print(f"\n🔧 ComfyUI VAE Encoding...")
        print(f"   ComfyUI VAE dtype: {comfy_vae.vae_dtype}")
        
        comfy_encoded = comfy_vae.encode(vae_input)
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

def compare_results(motion_result, comfy_result, test_name):
    """Compare VAE encoding results"""
    print(f"\n🔍 COMPARING {test_name.upper()} RESULTS")
    print("=" * 60)
    
    if motion_result is None or comfy_result is None:
        print(f"❌ Cannot compare - one or both tests failed")
        return False, "FAILED"
    
    # Shape comparison
    print(f"📊 ENCODED OUTPUT COMPARISON:")
    print(f"   Motion shape: {motion_result.shape}")
    print(f"   ComfyUI shape: {comfy_result.shape}")
    print(f"   Shapes match: {motion_result.shape == comfy_result.shape}")
    
    if motion_result.shape != comfy_result.shape:
        print(f"❌ Shape mismatch - cannot compare")
        return False, "INCOMPATIBLE"
    
    # Statistical comparison
    print(f"\n📊 DETAILED COMPARISON:")
    diff = torch.abs(motion_result - comfy_result)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    
    print(f"   Max difference: {max_diff:.6f}")
    print(f"   Mean difference: {mean_diff:.6f}")
    print(f"   Std difference: {std_diff:.6f}")
    
    # Compatibility assessment
    print(f"\n🎯 COMPATIBILITY TOLERANCE CHECK:")
    identical_1e6 = max_diff < 1e-6
    close_1e3 = max_diff < 1e-3
    close_1e2 = max_diff < 1e-2
    close_1e1 = max_diff < 1e-1
    
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
    print("🚀 VAE PREPROCESSING COMPARISON TEST")
    print("=" * 70)
    print("This script compares preprocessing between motion pipeline")
    print("and ComfyUI workflow_api_2.py to identify differences.")
    print()
    
    try:
        # Setup paths and load model
        add_paths()
        sd = load_vae_model()
        
        # Create test video
        video_tensor = create_test_video()
        
        # Test 1: Motion Pipeline Preprocessing
        print(f"\n" + "="*70)
        print(f"TEST 1: MOTION PIPELINE PREPROCESSING")
        print(f"="*70)
        
        reactive, inactive = motion_pipeline_preprocessing(video_tensor)
        
        # Test reactive tensor
        print(f"\n🔧 TESTING REACTIVE TENSOR (Motion Pipeline Preprocessing)")
        reactive_vae_input = prepare_video_for_vae(reactive)
        motion_result_reactive = test_motion_pipeline_vae(sd, reactive_vae_input)
        comfy_result_reactive = test_comfyui_vae(sd, reactive_vae_input)
        
        if motion_result_reactive is not None and comfy_result_reactive is not None:
            success_reactive, compatibility_reactive = compare_results(
                motion_result_reactive, comfy_result_reactive, "Reactive Tensor (Motion Pipeline)"
            )
        else:
            print("❌ Reactive tensor test failed")
        
        # Test inactive tensor
        print(f"\n🔧 TESTING INACTIVE TENSOR (Motion Pipeline Preprocessing)")
        inactive_vae_input = prepare_video_for_vae(inactive)
        motion_result_inactive = test_motion_pipeline_vae(sd, inactive_vae_input)
        comfy_result_inactive = test_comfyui_vae(sd, inactive_vae_input)
        
        if motion_result_inactive is not None and comfy_result_inactive is not None:
            success_inactive, compatibility_inactive = compare_results(
                motion_result_inactive, comfy_result_inactive, "Inactive Tensor (Motion Pipeline)"
            )
        else:
            print("❌ Inactive tensor test failed")
        
        # Test 2: ComfyUI Workflow Preprocessing
        print(f"\n" + "="*70)
        print(f"TEST 2: COMFYUI WORKFLOW PREPROCESSING")
        print(f"="*70)
        
        comfyui_video = comfyui_workflow_preprocessing(video_tensor)
        
        # Test ComfyUI workflow preprocessing
        print(f"\n🔧 TESTING COMFYUI WORKFLOW PREPROCESSING")
        comfyui_vae_input = prepare_video_for_vae(comfyui_video)
        motion_result_comfyui = test_motion_pipeline_vae(sd, comfyui_vae_input)
        comfy_result_comfyui = test_comfyui_vae(sd, comfyui_vae_input)
        
        if motion_result_comfyui is not None and comfy_result_comfyui is not None:
            success_comfyui, compatibility_comfyui = compare_results(
                motion_result_comfyui, comfy_result_comfyui, "ComfyUI Workflow Preprocessing"
            )
        else:
            print("❌ ComfyUI workflow preprocessing test failed")
        
        # Summary
        print(f"\n" + "="*70)
        print(f"PREPROCESSING COMPARISON SUMMARY")
        print(f"="*70)
        
        print(f"\n🎯 RESULTS:")
        if 'compatibility_reactive' in locals():
            print(f"   Reactive Tensor (Motion Pipeline): {compatibility_reactive}")
        if 'compatibility_inactive' in locals():
            print(f"   Inactive Tensor (Motion Pipeline): {compatibility_inactive}")
        if 'compatibility_comfyui' in locals():
            print(f"   ComfyUI Workflow Preprocessing: {compatibility_comfyui}")
        
        print(f"\n💡 KEY DIFFERENCES IDENTIFIED:")
        print(f"   1. Motion Pipeline: Uses mask-based splitting (reactive/inactive)")
        print(f"   2. ComfyUI Workflow: Simple normalization, no mask splitting")
        print(f"   3. This difference likely causes varying VAE outputs")
        
        print(f"\n🎉 PREPROCESSING COMPARISON TEST COMPLETED!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
