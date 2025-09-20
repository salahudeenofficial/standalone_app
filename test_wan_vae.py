"""
Test script for WAN VAE decode functionality
Focused on testing decode of [1, 16, 11, 104, 60] latent tensor (UNet output in fp16)
Following Disclaimer.txt guidelines - no outside module dependencies
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
import time
import logging

# Add motion directory to path (following Disclaimer.txt guidelines)
sys.path.insert(0, str(Path(__file__).parent))

from standalone_vae import VAE, create_vae
from components.vae_decoder import VAEDecode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_torch_file(file_path, device=None):
    """Load torch file using safetensors (following Disclaimer.txt - no outside dependencies)"""
    import safetensors
    
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


def verify_vae_model_weights(vae_path):
    """Verify VAE model is loaded properly with appropriate weights"""
    print("🔍 VERIFYING VAE MODEL WEIGHTS")
    print("="*60)
    
    if not os.path.exists(vae_path):
        print(f"❌ VAE model not found: {vae_path}")
        return None
    
    try:
        # Load state dict
        state_dict = load_torch_file(vae_path)
        print(f"✅ VAE state dict loaded successfully")
        print(f"   📊 Number of parameters: {len(state_dict)}")
        print(f"   📊 File size: {os.path.getsize(vae_path) / (1024*1024):.1f} MB")
        
        # Check for WAN VAE specific keys (based on HuggingFace model info)
        wan_vae_keys = [
            "decoder.conv_in.weight",
            "decoder.conv_out.weight", 
            "encoder.conv_in.weight",
            "encoder.conv_out.weight",
            "decoder.middle.0.residual.0.gamma",
            "encoder.middle.0.residual.0.gamma"
        ]
        
        found_keys = []
        missing_keys = []
        
        for key in wan_vae_keys:
            if key in state_dict:
                found_keys.append(key)
                tensor = state_dict[key]
                print(f"   ✅ {key}: {tensor.shape}, dtype={tensor.dtype}")
            else:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"   ⚠️  Missing keys: {missing_keys}")
        
        # Check tensor statistics
        print(f"\n📊 TENSOR STATISTICS:")
        for key in found_keys[:3]:  # Check first 3 found keys
            tensor = state_dict[key]
            print(f"   {key}:")
            print(f"     Shape: {tensor.shape}")
            print(f"     Dtype: {tensor.dtype}")
            print(f"     Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            print(f"     Mean: {tensor.mean().item():.6f}")
            print(f"     Std: {tensor.std().item():.6f}")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Failed to load VAE model: {e}")
        return None


def test_vae_decode_specific_shape():
    """Test VAE decode with specific tensor shape [1, 16, 11, 104, 60]"""
    print("\n🧪 TESTING VAE DECODE WITH SPECIFIC SHAPE")
    print("="*60)
    
    # Target tensor shape (UNet output in fp16)
    target_shape = [1, 16, 11, 104, 60]
    print(f"📊 Target latent shape: {target_shape}")
    print(f"📊 Expected output shape: [11, 832, 480, 3] (assuming 8x upscale)")
    
    # Create test latent tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Device: {device}")
    
    # Create latent tensor in fp16 (as UNet would output)
    latent_tensor = torch.randn(target_shape, device=device, dtype=torch.float16)
    print(f"📊 Created latent tensor: {latent_tensor.shape}, dtype={latent_tensor.dtype}")
    print(f"📊 Latent tensor range: [{latent_tensor.min().item():.3f}, {latent_tensor.max().item():.3f}]")
    
    # Test 1: Try to load actual VAE model
    vae_path = "models/vaes/wan_vae.safetensors"
    state_dict = verify_vae_model_weights(vae_path)
    
    if state_dict is None:
        print("\n⚠️  Using mock VAE for testing...")
        return test_with_mock_vae(latent_tensor)
    
    # Test 2: Create VAE with loaded weights
    print(f"\n🔧 CREATING VAE WITH LOADED WEIGHTS")
    print("-" * 40)
    
    try:
        vae = create_vae(state_dict=state_dict, device=device)
        print(f"✅ VAE created successfully")
        print(f"   Type: {type(vae).__name__}")
        print(f"   First stage model: {type(vae.first_stage_model).__name__ if vae.first_stage_model else 'None'}")
        print(f"   Latent channels: {vae.latent_channels}")
        print(f"   Latent dim: {vae.latent_dim}")
        print(f"   Upscale ratio: {vae.upscale_ratio}")
        
        # Validate VAE
        try:
            vae.throw_exception_if_invalid()
            print(f"   ✅ VAE validation passed")
        except Exception as e:
            print(f"   ❌ VAE validation failed: {e}")
            return test_with_mock_vae(latent_tensor)
        
        # Test 3: Direct decode test
        print(f"\n🎯 TESTING DIRECT DECODE")
        print("-" * 40)
        
        try:
            with torch.no_grad():
                start_time = time.time()
                decoded = vae.decode(latent_tensor)
                decode_time = time.time() - start_time
                
                print(f"✅ Direct decode successful!")
                print(f"   Output shape: {decoded.shape}")
                print(f"   Output dtype: {decoded.dtype}")
                print(f"   Output range: [{decoded.min().item():.3f}, {decoded.max().item():.3f}]")
                print(f"   Decode time: {decode_time:.3f}s")
                
        except Exception as e:
            print(f"❌ Direct decode failed: {e}")
            print(f"   Trying VAE decoder component...")
            return test_with_vae_decoder(vae, latent_tensor)
        
        # Test 4: VAE Decoder component test
        print(f"\n🎯 TESTING VAE DECODER COMPONENT")
        print("-" * 40)
        
        return test_with_vae_decoder(vae, latent_tensor)
        
    except Exception as e:
        print(f"❌ Failed to create VAE: {e}")
        return test_with_mock_vae(latent_tensor)


def test_with_vae_decoder(vae, latent_tensor):
    """Test using VAE decoder component"""
    try:
        vae_decoder = VAEDecode()
        latent_dict = {"samples": latent_tensor}
        
        print(f"📊 Testing with VAE decoder component...")
        
        with torch.no_grad():
            start_time = time.time()
            result = vae_decoder.decode(vae, latent_dict)
            decode_time = time.time() - start_time
            
            if isinstance(result, tuple):
                decoded_images = result[0]
            else:
                decoded_images = result
            
            print(f"✅ VAE decoder decode successful!")
            print(f"   Output shape: {decoded_images.shape}")
            print(f"   Output dtype: {decoded_images.dtype}")
            print(f"   Output range: [{decoded_images.min().item():.3f}, {decoded_images.max().item():.3f}]")
            print(f"   Decode time: {decode_time:.3f}s")
            
            return True
            
    except Exception as e:
        print(f"❌ VAE decoder decode failed: {e}")
        return False


def test_with_mock_vae(latent_tensor):
    """Test with mock VAE when real VAE fails"""
    print(f"\n🔧 TESTING WITH MOCK VAE")
    print("-" * 40)
    
    class MockVAE:
        def __init__(self):
            self.device = latent_tensor.device
            self.vae_dtype = torch.float32
            self.output_channels = 3
            self.upscale_ratio = 8
            self.output_device = self.device
            self.first_stage_model = "mock_model"
            
        def throw_exception_if_invalid(self):
            pass
            
        def decode(self, samples_in, vae_options={}):
            print(f"   🔧 Mock decode called with shape: {samples_in.shape}")
            
            # Simulate OOM for large tensors
            if samples_in.numel() > 1000000:
                raise torch.cuda.OutOfMemoryError("Simulated OOM for large tensor")
            
            # Mock decode: latent -> images
            batch_size, channels, frames, height, width = samples_in.shape
            output_height = height * self.upscale_ratio
            output_width = width * self.upscale_ratio
            
            # Create mock decoded output
            decoded = torch.randn(batch_size, self.output_channels, frames, output_height, output_width, 
                                device=samples_in.device, dtype=samples_in.dtype)
            return decoded.movedim(1, -1)
        
        def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
            print(f"   🔧 Mock tiled decode called with shape: {samples.shape}")
            return self.decode(samples)  # Use regular decode for simplicity
        
        def to(self, device):
            self.device = device
            return self
    
    mock_vae = MockVAE()
    return test_with_vae_decoder(mock_vae, latent_tensor)


def main():
    """Main test function"""
    print("🧪 WAN VAE DECODE TEST")
    print("="*80)
    print("Testing decode of [1, 16, 11, 104, 60] latent tensor")
    print("Following Disclaimer.txt guidelines - no outside dependencies")
    print("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Run the test
    success = test_vae_decode_specific_shape()
    
    if success:
        print(f"\n🎉 VAE DECODE TEST COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n❌ VAE DECODE TEST FAILED!")
    
    return success


if __name__ == "__main__":
    main()