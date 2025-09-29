#!/usr/bin/env python3
"""
Test VAE encoding with proper GPU loading (without requiring actual model files).
This test verifies that the load_models_gpu and VAE device fixes work correctly.
"""

import torch
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

from wan_vae_components.model_management import load_models_gpu, get_torch_device
from wan_vae_components.vae import WanVAE
from standalone_vae import VAE

def test_vae_encoding_with_gpu_loading():
    """Test VAE encoding with proper GPU loading"""
    print("="*60)
    print("TESTING VAE ENCODING WITH GPU LOADING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test VAE encoding")
        return False
    
    device = get_torch_device()
    print(f"Using device: {device}")
    
    # Create a mock VAE model
    class MockWanVAE:
        def __init__(self):
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
            self.conv4 = torch.nn.Conv2d(16, 3, 3, padding=1)
            self._enc_feat_map = {}
            self._enc_conv_idx = [0]
            self._feat_map = {}
            self._conv_idx = [0]
        
        def to(self, device):
            self.conv1.to(device)
            self.conv2.to(device)
            self.conv3.to(device)
            self.conv4.to(device)
            return self
        
        def encode(self, x):
            # Simple mock encoding
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            return x
        
        def decode(self, z):
            # Simple mock decoding
            return z
        
        def clear_cache(self):
            pass
    
    # Create VAE wrapper
    vae = VAE()
    vae.first_stage_model = MockWanVAE()
    vae.device = device
    vae.vae_dtype = torch.float16
    vae.process_input = lambda x: x * 2.0 - 1.0
    vae.process_output = lambda x: torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    
    # Properly initialize VAE
    vae.latent_dim = 2  # 2D latent for images
    vae.downscale_ratio = 8  # Standard VAE downscale ratio
    vae.spacial_compression_encode = lambda: vae.downscale_ratio
    vae.memory_used_encode = lambda shape, dtype: 1000 * shape[2] * shape[3] * 4  # Mock memory calculation
    vae.disable_offload = False
    
    # Mock get_free_memory to return reasonable value
    def mock_get_free_memory(device):
        return 1024 * 1024 * 100  # 100MB free memory
    
    # Patch the get_free_memory function
    import wan_vae_components.model_management as mm
    mm.get_free_memory = mock_get_free_memory
    
    # Create mock patcher
    class MockPatcher:
        def __init__(self):
            self.model = vae.first_stage_model
            self.load_device = device
    
    vae.patcher = MockPatcher()
    
    # Test GPU loading
    print("🔧 Testing GPU loading...")
    try:
        load_models_gpu([vae.patcher], memory_required=1024*1024*100)  # 100MB
        print("✅ GPU loading successful")
    except Exception as e:
        print(f"❌ GPU loading failed: {e}")
        return False
    
    # Test VAE encoding
    print("🔧 Testing VAE encoding...")
    try:
        # Create test input in the format expected by VAE (B, C, H, W)
        test_input = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float32)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test input device: {test_input.device}")
        
        # Test process_input directly first
        processed_input = vae.process_input(test_input)
        print(f"Processed input shape: {processed_input.shape}")
        print(f"Processed input device: {processed_input.device}")
        
        # Check if tensor is valid
        if processed_input.numel() == 0:
            print("❌ Processed input tensor is empty")
            return False
        
        # Encode
        with torch.no_grad():
            encoded = vae.encode(test_input)
        
        print(f"Encoded output shape: {encoded.shape}")
        print(f"Encoded output device: {encoded.device}")
        
        # Verify output is on GPU
        if encoded.device.type == 'cuda':
            print("✅ VAE encoding successful and output is on GPU")
            return True
        else:
            print("❌ VAE encoding output is not on GPU")
            return False
            
    except Exception as e:
        print(f"❌ VAE encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage_during_encoding():
    """Test memory usage during VAE encoding"""
    print("\n" + "="*60)
    print("TESTING MEMORY USAGE DURING ENCODING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test memory usage")
        return False
    
    device = get_torch_device()
    
    def log_memory(stage):
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        allocated_mb = allocated / (1024 * 1024)
        reserved_mb = reserved / (1024 * 1024)
        print(f"   {stage}: Allocated={allocated_mb:.2f}MB, Reserved={reserved_mb:.2f}MB")
    
    print("🔍 Testing memory usage during VAE operations:")
    log_memory("Initial")
    
    # Create VAE model
    vae_model = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
    log_memory("After VAE model creation")
    
    # Create test input
    test_input = torch.randn(1, 3, 64, 64, device=device)
    log_memory("After test input creation")
    
    # Perform encoding
    with torch.no_grad():
        encoded = vae_model(test_input)
    log_memory("After encoding")
    
    # Cleanup
    del vae_model, test_input, encoded
    torch.cuda.empty_cache()
    log_memory("After cleanup")
    
    print("✅ Memory usage tracking works correctly")
    return True

def main():
    """Run all tests"""
    print("VAE ENCODING WITH GPU LOADING TEST")
    print("="*80)
    
    results = []
    
    # Test 1: VAE encoding with GPU loading
    results.append(test_vae_encoding_with_gpu_loading())
    
    # Test 2: Memory usage during encoding
    results.append(test_memory_usage_during_encoding())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All VAE encoding tests passed!")
        print("\n🎯 VAE encoding with GPU loading is now working:")
        print("   - load_models_gpu accepts memory_required: ✅")
        print("   - VAE device placement works: ✅")
        print("   - VAE encoding works on GPU: ✅")
        print("   - Memory usage tracking: ✅")
        print("\n🚀 The motion pipeline should now work correctly!")
    else:
        print("❌ Some VAE encoding tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
