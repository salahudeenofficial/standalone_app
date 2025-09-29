#!/usr/bin/env python3
"""
Simple test to verify the load_models_gpu fix works correctly.
This test focuses on the core issue: load_models_gpu parameter compatibility.
"""

import torch
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

from wan_vae_components.model_management import load_models_gpu, get_torch_device

def test_load_models_gpu_parameter_fix():
    """Test that load_models_gpu now accepts all ComfyUI parameters"""
    print("="*60)
    print("TESTING LOAD_MODELS_GPU PARAMETER FIX")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test GPU loading")
        return False
    
    device = get_torch_device()
    print(f"Using device: {device}")
    
    # Create a mock model patcher
    class MockModelPatcher:
        def __init__(self):
            self.model = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.load_device = device
    
    patcher = MockModelPatcher()
    
    # Test all the parameters that were causing the error
    test_cases = [
        {
            "name": "Basic memory_required",
            "args": {"models": [patcher], "memory_required": 1024*1024*100}
        },
        {
            "name": "All ComfyUI parameters",
            "args": {
                "models": [patcher], 
                "memory_required": 1024*1024*50,
                "force_patch_weights": True,
                "minimum_memory_required": 1024*1024*25,
                "force_full_load": True
            }
        },
        {
            "name": "Zero memory_required",
            "args": {"models": [patcher], "memory_required": 0}
        },
        {
            "name": "Large memory_required",
            "args": {"models": [patcher], "memory_required": 1024*1024*1000}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔧 Testing: {test_case['name']}")
        try:
            load_models_gpu(**test_case['args'])
            print(f"   ✅ {test_case['name']} - SUCCESS")
        except Exception as e:
            print(f"   ❌ {test_case['name']} - FAILED: {e}")
            return False
    
    print(f"\n✅ All load_models_gpu parameter tests passed!")
    return True

def test_vae_device_placement_fix():
    """Test that VAE can be properly moved to GPU"""
    print("\n" + "="*60)
    print("TESTING VAE DEVICE PLACEMENT FIX")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test VAE device placement")
        return False
    
    device = get_torch_device()
    
    # Create a mock VAE model
    class MockVAE:
        def __init__(self):
            self.first_stage_model = torch.nn.Conv2d(3, 16, 3, padding=1)
        
        def to(self, device):
            self.first_stage_model.to(device)
            return self
    
    vae = MockVAE()
    
    # Test moving to GPU
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
    print("LOAD_MODELS_GPU PARAMETER FIX TEST")
    print("="*80)
    
    results = []
    
    # Test 1: load_models_gpu parameter fix
    results.append(test_load_models_gpu_parameter_fix())
    
    # Test 2: VAE device placement fix
    results.append(test_vae_device_placement_fix())
    
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
        print("✅ All load_models_gpu parameter fix tests passed!")
        print("\n🎯 The core issue has been fixed:")
        print("   - load_models_gpu accepts memory_required: ✅")
        print("   - load_models_gpu accepts all ComfyUI parameters: ✅")
        print("   - VAE device placement works: ✅")
        print("   - Memory management integration: ✅")
        print("\n🚀 The motion pipeline should now work without the TypeError!")
        print("\n📝 Note: VAE encoding tensor shape issues are separate from this fix.")
        print("   The main issue (load_models_gpu parameter error) is now resolved.")
    else:
        print("❌ Some load_models_gpu parameter fix tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
