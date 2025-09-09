"""
Test script for standalone VAE implementation using WAN VAE model
Focused on video tensor encoding test
"""

import torch
import torch.nn as nn
from standalone_vae import VAE, create_vae
import safetensors


def load_torch_file(file_path, device=None):
    """Load torch file using safetensors"""
    if device is None:
        device = torch.device("cpu")
    if file_path.lower().endswith(".safetensors"):
        try:
            with safetensors.safe_open(file_path, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    tensor = tensor.to(device)
                    sd[k] = tensor
        except Exception as e:
            raise ValueError(f"Error loading safetensors file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")
    
    return sd


def load_wan_vae_model(vae_path):
    """Load WAN VAE model from safetensors file"""
    print(f"Loading WAN VAE model from: {vae_path}")
    
    try:
        # Load the safetensors file
        state_dict = load_torch_file(vae_path)
        print(f"✅ Successfully loaded VAE model")
        print(f"Number of parameters: {len(state_dict)}")
        
        # Show some key parameters
        key_params = [
            "decoder.conv_in.weight",
            "decoder.conv_out.weight", 
            "encoder.conv_in.weight",
            "encoder.conv_out.weight",
            "decoder.middle.0.residual.0.gamma",
            "encoder.middle.0.residual.0.gamma"
        ]
        
        for key in key_params:
            if key in state_dict:
                shape = state_dict[key].shape
                print(f"  {key}: {shape}")
        
        # Show first 10 keys to understand structure
        print("First 10 keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}: {state_dict[key].shape}")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Failed to load VAE model: {e}")
        return None


def test_video_tensor_encoding(vae_path):
    """Test WAN VAE encoding with video tensor (37, 768, 576, 3)"""
    print("=== Testing WAN VAE Video Tensor Encoding ===")
    
    # Load the VAE model
    state_dict = load_wan_vae_model(vae_path)
    if state_dict is None:
        return
    
    # Create VAE with the loaded state dict
    try:
        vae = create_vae(state_dict=state_dict)
        print(f"✅ VAE created successfully")
        print(f"VAE model type: {type(vae.first_stage_model).__name__}")
        print(f"Latent channels: {vae.latent_channels}")
        print(f"Downscale ratio: {vae.downscale_ratio}")
        print(f"Upscale ratio: {vae.upscale_ratio}")
        print(f"Latent dim: {vae.latent_dim}")
        print(f"Device: {vae.device}")
        print(f"Dtype: {vae.vae_dtype}")
        
    except Exception as e:
        print(f"❌ Failed to create VAE: {e}")
        return
    
    # Test with video tensor (37, 768, 576, 3)
    video_shape = (37, 768, 576, 3)
    print(f"\n--- Testing Video Tensor: {video_shape} ---")
    
    # Create video tensor
    video_tensor = torch.randn(*video_shape)
    print(f"Input shape: {video_tensor.shape}")
    print(f"Input range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")
    print(f"Input dtype: {video_tensor.dtype}")
    
    # Test pixel cropping
    print(f"\n--- Testing Pixel Cropping ---")
    cropped_tensor = vae.vae_encode_crop_pixels(video_tensor)
    print(f"Cropped shape: {cropped_tensor.shape}")
    print(f"Cropped range: [{cropped_tensor.min():.3f}, {cropped_tensor.max():.3f}]")
    
    # Test spatial compression
    print(f"\n--- Testing Spatial Compression ---")
    spatial_compression = vae.spacial_compression_encode()
    print(f"Spatial compression ratio: {spatial_compression}")
    
    # Calculate expected output dimensions
    expected_height = cropped_tensor.shape[1] // spatial_compression
    expected_width = cropped_tensor.shape[2] // spatial_compression
    expected_channels = vae.latent_channels
    expected_frames = cropped_tensor.shape[0]
    
    print(f"Expected output shape: ({expected_frames}, {expected_channels}, {expected_height}, {expected_width})")
    
    # Test encoding
    print(f"\n--- Testing Video Encoding ---")
    with torch.no_grad():
        try:
            # Encode the video tensor
            latent = vae.encode(video_tensor)
            print(f"✅ Video encoding successful")
            print(f"  Latent shape: {latent.shape}")
            print(f"  Latent range: [{latent.min():.3f}, {latent.max():.3f}]")
            print(f"  Latent dtype: {latent.dtype}")
            
            # Verify dimensions
            if latent.shape[0] == expected_frames:
                print(f"  ✅ Frame count correct: {latent.shape[0]}")
            else:
                print(f"  ❌ Frame count mismatch: expected {expected_frames}, got {latent.shape[0]}")
            
            if latent.shape[1] == expected_channels:
                print(f"  ✅ Channel count correct: {latent.shape[1]}")
            else:
                print(f"  ❌ Channel count mismatch: expected {expected_channels}, got {latent.shape[1]}")
            
            if latent.shape[2] == expected_height:
                print(f"  ✅ Height correct: {latent.shape[2]}")
            else:
                print(f"  ❌ Height mismatch: expected {expected_height}, got {latent.shape[2]}")
            
            if latent.shape[3] == expected_width:
                print(f"  ✅ Width correct: {latent.shape[3]}")
            else:
                print(f"  ❌ Width mismatch: expected {expected_width}, got {latent.shape[3]}")
            
            # Calculate compression ratio
            input_size = video_tensor.numel()
            latent_size = latent.numel()
            compression_ratio = input_size / latent_size
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            
            # Calculate memory usage
            input_memory = input_size * video_tensor.element_size() / (1024*1024)
            latent_memory = latent_size * latent.element_size() / (1024*1024)
            print(f"  Input memory: {input_memory:.2f} MB")
            print(f"  Latent memory: {latent_memory:.2f} MB")
            print(f"  Memory reduction: {(input_memory - latent_memory) / input_memory * 100:.1f}%")
            
        except Exception as e:
            print(f"❌ Video encoding failed: {e}")
            import traceback
            traceback.print_exc()


def test_memory_usage(vae_path):
    """Test memory usage calculations for video tensor"""
    print("\n=== Testing Memory Usage Calculations ===")
    
    # Load the VAE model
    state_dict = load_wan_vae_model(vae_path)
    if state_dict is None:
        return
    
    # Create VAE
    try:
        vae = create_vae(state_dict=state_dict)
        print(f"✅ VAE created for memory testing")
        
    except Exception as e:
        print(f"❌ Failed to create VAE: {e}")
        return
    
    # Test video tensor shape
    video_shape = (37, 768, 576, 3)
    print(f"\n--- Video Shape: {video_shape} ---")
    
    # Calculate theoretical memory usage
    encode_memory = vae.memory_used_encode(video_shape, torch.float16)
    decode_memory = vae.memory_used_decode(video_shape, torch.float16)
    
    print(f"Theoretical encode memory: {encode_memory / (1024*1024):.2f} MB")
    print(f"Theoretical decode memory: {decode_memory / (1024*1024):.2f} MB")
    print(f"Total theoretical memory: {(encode_memory + decode_memory) / (1024*1024):.2f} MB")
    
    # Test actual memory usage
    video_tensor = torch.randn(*video_shape)
    input_memory = video_tensor.numel() * video_tensor.element_size() / (1024*1024)
    print(f"Input tensor memory: {input_memory:.2f} MB")
    
    with torch.no_grad():
        try:
            latent = vae.encode(video_tensor)
            latent_memory = latent.numel() * latent.element_size() / (1024*1024)
            print(f"Latent tensor memory: {latent_memory:.2f} MB")
            
            # Calculate actual compression
            compression_ratio = input_memory / latent_memory
            print(f"Actual compression ratio: {compression_ratio:.2f}x")
            
        except Exception as e:
            print(f"❌ Memory test failed: {e}")


def test_model_patcher_integration(vae_path):
    """Test WAN VAE with ModelPatcher"""
    print("\n=== Testing WAN VAE with ModelPatcher ===")
    
    # Load the VAE model
    state_dict = load_wan_vae_model(vae_path)
    if state_dict is None:
        return
    
    # Create VAE
    try:
        vae = create_vae(state_dict=state_dict)
        print(f"✅ VAE created with ModelPatcher")
        
    except Exception as e:
        print(f"❌ Failed to create VAE: {e}")
        return
    
    if vae.patcher is not None:
        print(f"ModelPatcher type: {type(vae.patcher).__name__}")
        print(f"Load device: {vae.patcher.load_device}")
        print(f"Offload device: {vae.patcher.offload_device}")
        print(f"Model size: {vae.patcher.model_size() / (1024*1024):.2f} MB")
        
        # Test encode with ModelPatcher
        video_tensor = torch.randn(37, 768, 576, 3)
        print(f"Input shape: {video_tensor.shape}")
        
        with torch.no_grad():
            try:
                latent = vae.encode(video_tensor)
                print(f"✅ Encode with ModelPatcher successful")
                print(f"  Latent shape: {latent.shape}")
                
                # Test memory management
                print(f"ModelPatcher memory tracking: {vae.patcher.model_size() / (1024*1024):.2f} MB")
                
            except Exception as e:
                print(f"❌ ModelPatcher test failed: {e}")
    else:
        print("⚠️  ModelPatcher not created")


def main():
    """Run WAN VAE tests focused on video tensor encoding"""
    print("WAN VAE Test Suite - Video Tensor Encoding Focus")
    print("=" * 60)
    
    # Path to the downloaded VAE model
    vae_path = "./models/vaes/wan_vae.safetensors"
    
    print(f"Testing VAE model at: {vae_path}")
    print(f"Target video tensor: (37, 768, 576, 3)")
    
    # Run tests
    test_video_tensor_encoding(vae_path)
    test_memory_usage(vae_path)
    test_model_patcher_integration(vae_path)
    
    print("\n" + "=" * 60)
    print("WAN VAE video tensor test suite completed!")


if __name__ == "__main__":
    main()
