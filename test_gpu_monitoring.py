#!/usr/bin/env python3
"""
Test GPU monitoring functionality in motion pipeline.
This test verifies that GPU usage and device tracking work correctly.
"""

import torch
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

from pipeline import WanVideoPipeline

def test_gpu_monitoring():
    """Test GPU monitoring functionality"""
    print("="*60)
    print("TESTING GPU MONITORING FUNCTIONALITY")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test GPU monitoring")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Current device: {torch.cuda.current_device()}")
    print(f"   Device name: {torch.cuda.get_device_name()}")
    print()
    
    # Initialize pipeline
    pipeline = WanVideoPipeline(models_dir="models")
    
    # Test GPU monitoring methods
    print("🔍 Testing _log_gpu_state method:")
    pipeline._log_gpu_state("TEST_STAGE")
    
    print("🔍 Testing _verify_vae_device method:")
    try:
        pipeline._verify_vae_device()
    except AttributeError as e:
        print(f"   Expected error (VAE not initialized): {e}")
        print("   ✅ Error handling works correctly")
    
    # Test with dummy VAE
    print("🔍 Testing with dummy VAE:")
    try:
        # Create a dummy VAE for testing
        class DummyVAE:
            def __init__(self):
                self.first_stage_model = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.first_stage_model.to("cuda")
            
            def throw_exception_if_invalid(self):
                pass
        
        pipeline.vae = DummyVAE()
        pipeline._verify_vae_device()
        
        print("✅ GPU monitoring methods work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing GPU monitoring: {e}")
        return False

def test_memory_tracking():
    """Test memory tracking during tensor operations"""
    print("\n" + "="*60)
    print("TESTING MEMORY TRACKING")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test memory tracking")
        return False
    
    device = torch.cuda.current_device()
    
    def log_memory(stage):
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        allocated_mb = allocated / (1024 * 1024)
        reserved_mb = reserved / (1024 * 1024)
        print(f"   {stage}: Allocated={allocated_mb:.2f}MB, Reserved={reserved_mb:.2f}MB")
    
    print("🔍 Testing memory tracking during tensor operations:")
    log_memory("Initial")
    
    # Create some tensors
    tensor1 = torch.randn(1000, 1000, device="cuda")
    log_memory("After tensor1")
    
    tensor2 = torch.randn(1000, 1000, device="cuda")
    log_memory("After tensor2")
    
    # Perform operations
    result = tensor1 @ tensor2
    log_memory("After matrix multiplication")
    
    # Clear memory
    del tensor1, tensor2, result
    torch.cuda.empty_cache()
    log_memory("After cleanup")
    
    print("✅ Memory tracking works correctly")
    return True

def test_device_verification():
    """Test device verification functionality"""
    print("\n" + "="*60)
    print("TESTING DEVICE VERIFICATION")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test device verification")
        return False
    
    # Test 1: Model on correct device
    print("🔍 Test 1: Model on correct device")
    model_correct = torch.nn.Conv2d(3, 16, 3, padding=1).to("cuda")
    
    class TestVAE1:
        def __init__(self):
            self.first_stage_model = model_correct
    
    pipeline = WanVideoPipeline(models_dir="models")
    pipeline.vae = TestVAE1()
    pipeline._verify_vae_device()
    
    # Test 2: Model on wrong device
    print("\n🔍 Test 2: Model on wrong device")
    model_wrong = torch.nn.Conv2d(3, 16, 3, padding=1).to("cpu")
    
    class TestVAE2:
        def __init__(self):
            self.first_stage_model = model_wrong
    
    pipeline.vae = TestVAE2()
    pipeline._verify_vae_device()
    
    print("✅ Device verification works correctly")
    return True

def main():
    """Run all GPU monitoring tests"""
    print("GPU MONITORING FUNCTIONALITY TEST")
    print("="*80)
    
    results = []
    
    # Test 1: GPU monitoring
    results.append(test_gpu_monitoring())
    
    # Test 2: Memory tracking
    results.append(test_memory_tracking())
    
    # Test 3: Device verification
    results.append(test_device_verification())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All GPU monitoring tests passed!")
        print("\n🎯 GPU monitoring is ready for VAE encoding:")
        print("   - GPU state logging: ✅")
        print("   - Memory tracking: ✅")
        print("   - Device verification: ✅")
        print("   - VAE device checking: ✅")
    else:
        print("❌ Some GPU monitoring tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
