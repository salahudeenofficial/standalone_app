"""
Test script for standalone VAE implementation using WAN VAE model
"""

import torch
import torch.nn as nn
from standalone_vae import VAE, create_vae
import utils

def load_wan_vae_model(vae_path):
    """Load WAN VAE model from safetensors file"""
    print(f"Loading WAN VAE model from: {vae_path}")
    
    try:
        # Load the safetensors file
        state_dict = utils.load_torch_file(vae_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Number of parameters: {len(state_dict)}")
        
        # Show some key parameters
        key_params = [
            "decoder.conv_in.weight",
            "decoder.conv_out.weight", 
            "encoder.conv_in.weight",
            "encoder.conv_out.weight"
        ]
        
        for key in key_params:
            if key in state_dict:
                shape = state_dict[key].shape
                print(f"  {key}: {shape}")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Failed to load VAE model: {e}")
        return None


def test_wan_vae_encode_decode(vae_path):
    """Test WAN VAE encoding and decoding"""
    print("=== Testing WAN VAE Encode/Decode ===")
    
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
        print(f"Device: {vae.device}")
        print(f"Dtype: {vae.vae_dtype}")
        
    except Exception as e:
        print(f"❌ Failed to create VAE: {e}")
        return
    
    # Test with different input sizes
    test_sizes = [
        (1, 3, 64, 64),    # Small image
        # (1, 3, 128, 128),  # Medium image
        # (1, 3, 256, 256),  # Large image
        (37, 832, 480, 3),  # Batch of 2
    ]
    
    for i, size in enumerate(test_sizes):
        print(f"\n--- Test {i+1}: Input size {size} ---")
        dummy_input = torch.randn(*size)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
        
        with torch.no_grad():
            try:
                # Encode
                z, mean, logvar = vae.encode(dummy_input)
                print(f"✅ Encoded successfully")
                print(f"  Latent shape: {z.shape}")
                print(f"  Latent range: [{z.min():.3f}, {z.max():.3f}]")
                
                if mean is not None:
                    print(f"  Mean shape: {mean.shape}")
                    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
                
                if logvar is not None:
                    print(f"  Logvar shape: {logvar.shape}")
                    print(f"  Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
                
                # Decode
                reconstructed = vae.decode(z)
                print(f"✅ Decoded successfully")
                print(f"  Reconstructed shape: {reconstructed.shape}")
                print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
                
                # Calculate reconstruction error
                mse = torch.mean((dummy_input - reconstructed) ** 2)
                print(f"  Reconstruction MSE: {mse:.6f}")
                
                # Calculate compression ratio
                input_size = dummy_input.numel()
                latent_size = z.numel()
                compression_ratio = input_size / latent_size
                print(f"  Compression ratio: {compression_ratio:.2f}x")
                
            except Exception as e:
                print(f"❌ Encode/decode failed: {e}")
                import traceback
                traceback.print_exc()


def test_wan_vae_memory_usage(vae_path):
    """Test WAN VAE memory usage"""
    print("\n=== Testing WAN VAE Memory Usage ===")
    
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
    
    # Test different input shapes
    test_shapes = [
        (1, 3, 64, 64),    # Small
        # (1, 3, 128, 128),  # Medium
        # (1, 3, 256, 256),  # Large
        # (1, 3, 512, 512),  # Very large
        (37, 832, 480, 3),  # Batch
    ]
    
    for shape in test_shapes:
        print(f"\n--- Shape {shape} ---")
        
        # Calculate theoretical memory usage
        encode_memory = vae.memory_used_encode(shape, torch.float16)
        decode_memory = vae.memory_used_decode(shape, torch.float16)
        
        print(f"Theoretical encode memory: {encode_memory / (1024*1024):.2f} MB")
        print(f"Theoretical decode memory: {decode_memory / (1024*1024):.2f} MB")
        print(f"Total theoretical memory: {(encode_memory + decode_memory) / (1024*1024):.2f} MB")
        
        # Test actual memory usage
        dummy_input = torch.randn(*shape)
        input_memory = dummy_input.numel() * dummy_input.element_size() / (1024*1024)
        print(f"Input tensor memory: {input_memory:.2f} MB")
        
        with torch.no_grad():
            try:
                z, mean, logvar = vae.encode(dummy_input)
                latent_memory = z.numel() * z.element_size() / (1024*1024)
                print(f"Latent tensor memory: {latent_memory:.2f} MB")
                
                reconstructed = vae.decode(z)
                output_memory = reconstructed.numel() * reconstructed.element_size() / (1024*1024)
                print(f"Output tensor memory: {output_memory:.2f} MB")
                
                # Calculate actual compression
                compression_ratio = input_memory / latent_memory
                print(f"Actual compression ratio: {compression_ratio:.2f}x")
                
            except Exception as e:
                print(f"❌ Memory test failed: {e}")


def test_wan_vae_model_patcher(vae_path):
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
        print(f"Memory usage factor: {vae.patcher.memory_usage_factor}")
        
        # Test encode/decode with ModelPatcher
        dummy_input = torch.randn(1, 3, 128, 128)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            try:
                z, mean, logvar = vae.encode(dummy_input)
                print(f"✅ Encode with ModelPatcher successful")
                print(f"  Latent shape: {z.shape}")
                
                reconstructed = vae.decode(z)
                print(f"✅ Decode with ModelPatcher successful")
                print(f"  Reconstructed shape: {reconstructed.shape}")
                
                # Test memory management
                print(f"ModelPatcher memory tracking: {vae.patcher.model_size() / (1024*1024):.2f} MB")
                
            except Exception as e:
                print(f"❌ ModelPatcher test failed: {e}")
    else:
        print("⚠️  ModelPatcher not created")


def main():
    """Run WAN VAE tests"""
    print("WAN VAE Test Suite")
    print("=" * 50)
    
    # Path to the downloaded VAE model
    vae_path = "./models/vaes/wan_vae.safetensors"
    
    print(f"Testing VAE model at: {vae_path}")
    
    # Run tests
    test_wan_vae_encode_decode(vae_path)
    test_wan_vae_memory_usage(vae_path)
    test_wan_vae_model_patcher(vae_path)
    
    print("\n" + "=" * 50)
    print("WAN VAE test suite completed!")


if __name__ == "__main__":
    main()
