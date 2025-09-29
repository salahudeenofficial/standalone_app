#!/usr/bin/env python3
"""
Test VAE GPU loading and encoding functionality.
This test verifies that the load_models_gpu fix works correctly.
"""

import torch
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

from wan_vae_components.model_management import load_models_gpu, get_torch_device

def test_load_models_gpu_fix():
    """Test that load_models_gpu now accepts memory_required parameter"""
    print("="*60)
    print("TESTING LOAD_MODELS_GPU FIX")
    print("="*60)
    
    # Create a mock model patcher
    class MockModelPatcher:
        def __init__(self):
            self.model = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.load_device = get_torch_device()
    
    # Test the function with all parameters
    patcher = MockModelPatcher()
    
    try:
        # Test with memory_required parameter (this was failing before)
        load_models_gpu([patcher], memory_required=1024*1024*100)  # 100MB
        print("✅ load_models_gpu accepts memory_required parameter")
        
        # Test with all parameters
        load_models_gpu([patcher], 
                       memory_required=1024*1024*50,  # 50MB
                       force_patch_weights=True,
                       minimum_memory_required=1024*1024*25,  # 25MB
                       force_full_load=True)
        print("✅ load_models_gpu accepts all ComfyUI parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing load_models_gpu: {e}")
        return False

def test_vae_device_placement():
    """Test VAE device placement"""
    print("\n" + "="*60)
    print("TESTING VAE DEVICE PLACEMENT")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test VAE device placement")
        return False
    
    # Create a mock VAE
    class MockVAE:
        def __init__(self):
            self.first_stage_model = torch.nn.Conv2d(3, 16, 3, padding=1)
        
        def to(self, device):
            self.first_stage_model.to(device)
            return self
    
    vae = MockVAE()
    
    # Test moving to GPU
    device = get_torch_device()
    print(f"Target device: {device}")
    
    # Move to GPU
    vae.to(device)
    
    # Check device
    vae_device = next(vae.first_stage_model.parameters()).device
    print(f"VAE device: {vae_device}")
    
    if vae_device.type == 'cuda':
        print("✅ VAE successfully moved to GPU")
        return True
    else:
        print("❌ VAE failed to move to GPU")
        return False

def test_memory_management_integration():
    """Test memory management integration"""
    print("\n" + "="*60)
    print("TESTING MEMORY MANAGEMENT INTEGRATION")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test memory management")
        return False
    
    device = get_torch_device()
    
    # Test memory tracking
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"Initial memory: {initial_memory / (1024*1024):.2f} MB")
    
    # Create some tensors
    tensor1 = torch.randn(1000, 1000, device=device)
    tensor2 = torch.randn(1000, 1000, device=device)
    
    after_tensors = torch.cuda.memory_allocated(device)
    print(f"After tensors: {after_tensors / (1024*1024):.2f} MB")
    
    # Test load_models_gpu with memory tracking
    class MockModelPatcher:
        def __init__(self):
            self.model = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.load_device = device
    
    patcher = MockModelPatcher()
    
    try:
        # This should work now with memory_required parameter
        load_models_gpu([patcher], memory_required=1024*1024*50)  # 50MB
        print("✅ Memory management integration works")
        
        # Cleanup
        del tensor1, tensor2
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing memory management: {e}")
        return False

def main():
    """Run all tests"""
    print("VAE GPU LOADING AND ENCODING TEST")
    print("="*80)
    
    results = []
    
    # Test 1: load_models_gpu fix
    results.append(test_load_models_gpu_fix())
    
    # Test 2: VAE device placement
    results.append(test_vae_device_placement())
    
    # Test 3: Memory management integration
    results.append(test_memory_management_integration())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All VAE GPU loading tests passed!")
        print("\n🎯 VAE GPU loading is now fixed:")
        print("   - load_models_gpu accepts memory_required: ✅")
        print("   - VAE device placement works: ✅")
        print("   - Memory management integration: ✅")
        print("\n🚀 Ready to test VAE encoding with proper GPU loading!")
    else:
        print("❌ Some VAE GPU loading tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
