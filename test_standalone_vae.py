"""
Test script for standalone VAE implementation
Focused on encoding and decoding functionality
"""

import torch
import torch.nn as nn
from standalone_vae import VAE, create_vae


def test_encode_decode():
    """Test VAE encoding and decoding with dummy data"""
    print("=== Testing VAE Encode/Decode ===")
    
    # Create VAE with empty state dict (will use default)
    vae = create_vae()
    
    print(f"VAE created successfully")
    print(f"Latent channels: {vae.latent_channels}")
    print(f"Downscale ratio: {vae.downscale_ratio}")
    print(f"Upscale ratio: {vae.upscale_ratio}")
    print(f"Device: {vae.device}")
    print(f"Dtype: {vae.vae_dtype}")
    
    if vae.first_stage_model is not None:
        print(f"VAE model type: {type(vae.first_stage_model).__name__}")
        
        # Test with different input sizes
        test_sizes = [(1, 3, 64, 64), (2, 3, 128, 128), (1, 3, 256, 256)]
        
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
                    
                except Exception as e:
                    print(f"❌ Encode/decode failed: {e}")
    else:
        print("⚠️  VAE model not initialized (no state dict provided)")


def test_vae_with_state_dict():
    """Test VAE with a simple state dict"""
    print("\n=== Testing VAE with State Dict ===")
    
    # Create a simple state dict for testing
    test_sd = {
        "decoder.conv_in.weight": torch.randn(4, 4, 3, 3),
        "decoder.conv_in.bias": torch.randn(4),
        "decoder.conv_out.weight": torch.randn(3, 128, 3, 3),
        "decoder.conv_out.bias": torch.randn(3),
        "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
        "encoder.conv_in.bias": torch.randn(128),
        "encoder.conv_out.weight": torch.randn(8, 128, 3, 3),
        "encoder.conv_out.bias": torch.randn(8),
    }
    
    try:
        vae = create_vae(state_dict=test_sd)
        print(f"✅ VAE created with state dict")
        print(f"VAE model type: {type(vae.first_stage_model).__name__}")
        
        # Test encode/decode with different data
        test_cases = [
            ("Random data", torch.randn(1, 3, 64, 64)),
            ("Normalized data", torch.randn(1, 3, 64, 64) * 0.5 + 0.5),
            ("Zero data", torch.zeros(1, 3, 64, 64)),
            ("Ones data", torch.ones(1, 3, 64, 64)),
        ]
        
        for name, dummy_input in test_cases:
            print(f"\n--- Testing {name} ---")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
            
            with torch.no_grad():
                try:
                    # Encode
                    z, mean, logvar = vae.encode(dummy_input)
                    print(f"✅ Encoded successfully")
                    print(f"  Latent shape: {z.shape}")
                    print(f"  Latent range: [{z.min():.3f}, {z.max():.3f}]")
                    
                    # Decode
                    reconstructed = vae.decode(z)
                    print(f"✅ Decoded successfully")
                    print(f"  Reconstructed shape: {reconstructed.shape}")
                    print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
                    
                    # Calculate reconstruction error
                    mse = torch.mean((dummy_input - reconstructed) ** 2)
                    print(f"  Reconstruction MSE: {mse:.6f}")
                    
                except Exception as e:
                    print(f"❌ Encode/decode failed: {e}")
        
    except Exception as e:
        print(f"❌ VAE with state dict failed: {e}")


def test_different_vae_types():
    """Test different VAE type detection and encode/decode"""
    print("\n=== Testing VAE Type Detection ===")
    
    # Test TAESD detection
    taesd_sd = {
        "taesd_decoder.1.weight": torch.randn(4, 64, 3, 3),
        "taesd_encoder.1.weight": torch.randn(64, 3, 3, 3),
    }
    
    try:
        vae = VAE(sd=taesd_sd)
        print(f"✅ TAESD VAE detected: {type(vae.first_stage_model).__name__}")
        print(f"TAESD latent channels: {vae.latent_channels}")
        
        # Test encode/decode
        dummy_input = torch.randn(1, 3, 64, 64)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            z, mean, logvar = vae.encode(dummy_input)
            print(f"✅ TAESD encoded successfully")
            print(f"  Latent shape: {z.shape}")
            
            reconstructed = vae.decode(z)
            print(f"✅ TAESD decoded successfully")
            print(f"  Reconstructed shape: {reconstructed.shape}")
            
    except Exception as e:
        print(f"❌ TAESD detection failed: {e}")
    
    # Test Stage A detection
    stage_a_sd = {
        "vquantizer.codebook.weight": torch.randn(1024, 16),
        "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
    }
    
    try:
        vae = VAE(sd=stage_a_sd)
        print(f"✅ Stage A VAE detected: {type(vae.first_stage_model).__name__}")
        print(f"Stage A downscale ratio: {vae.downscale_ratio}")
        
        # Test encode/decode
        dummy_input = torch.randn(1, 3, 64, 64)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            z, mean, logvar = vae.encode(dummy_input)
            print(f"✅ Stage A encoded successfully")
            print(f"  Latent shape: {z.shape}")
            
            reconstructed = vae.decode(z)
            print(f"✅ Stage A decoded successfully")
            print(f"  Reconstructed shape: {reconstructed.shape}")
            
    except Exception as e:
        print(f"❌ Stage A detection failed: {e}")


def test_memory_usage():
    """Test memory usage calculations"""
    print("\n=== Testing Memory Usage Calculations ===")
    
    vae = create_vae()
    
    # Test different input shapes
    test_shapes = [(1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256), (2, 3, 512, 512)]
    
    for shape in test_shapes:
        print(f"\n--- Shape {shape} ---")
        
        # Calculate memory usage
        encode_memory = vae.memory_used_encode(shape, torch.float16)
        decode_memory = vae.memory_used_decode(shape, torch.float16)
        
        print(f"Encode memory: {encode_memory / (1024*1024):.2f} MB")
        print(f"Decode memory: {decode_memory / (1024*1024):.2f} MB")
        print(f"Total memory: {(encode_memory + decode_memory) / (1024*1024):.2f} MB")
        
        # Test actual memory usage
        dummy_input = torch.randn(*shape)
        print(f"Input tensor memory: {dummy_input.numel() * dummy_input.element_size() / (1024*1024):.2f} MB")
        
        with torch.no_grad():
            try:
                z, mean, logvar = vae.encode(dummy_input)
                latent_memory = z.numel() * z.element_size() / (1024*1024)
                print(f"Latent tensor memory: {latent_memory:.2f} MB")
                
                reconstructed = vae.decode(z)
                output_memory = reconstructed.numel() * reconstructed.element_size() / (1024*1024)
                print(f"Output tensor memory: {output_memory:.2f} MB")
                
            except Exception as e:
                print(f"❌ Memory test failed: {e}")


def test_model_patcher_integration():
    """Test ModelPatcher integration"""
    print("\n=== Testing ModelPatcher Integration ===")
    
    try:
        vae = create_vae()
        
        if vae.patcher is not None:
            print(f"✅ ModelPatcher created successfully")
            print(f"ModelPatcher type: {type(vae.patcher).__name__}")
            print(f"Load device: {vae.patcher.load_device}")
            print(f"Offload device: {vae.patcher.offload_device}")
            print(f"Model size: {vae.patcher.model_size() / (1024*1024):.2f} MB")
            
            # Test memory management
            print(f"Memory usage factor: {vae.patcher.memory_usage_factor}")
            
            # Test with dummy data
            dummy_input = torch.randn(1, 3, 64, 64)
            print(f"Input shape: {dummy_input.shape}")
            
            with torch.no_grad():
                z, mean, logvar = vae.encode(dummy_input)
                print(f"✅ Encode with ModelPatcher successful")
                print(f"  Latent shape: {z.shape}")
                
                reconstructed = vae.decode(z)
                print(f"✅ Decode with ModelPatcher successful")
                print(f"  Reconstructed shape: {reconstructed.shape}")
            
        else:
            print("⚠️  ModelPatcher not created (VAE model not initialized)")
            
    except Exception as e:
        print(f"❌ ModelPatcher integration failed: {e}")


def main():
    """Run all tests"""
    print("Standalone VAE Test Suite - Encode/Decode Focus")
    print("=" * 60)
    
    test_encode_decode()
    test_vae_with_state_dict()
    test_different_vae_types()
    test_memory_usage()
    test_model_patcher_integration()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")


if __name__ == "__main__":
    main()
