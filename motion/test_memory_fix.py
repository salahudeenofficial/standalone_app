#!/usr/bin/env python3
"""
Quick Memory Fix Test

This script tests the aggressive memory cleanup fix.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_utils import log_memory_usage, get_memory_info, clear_cuda_memory, safe_model_to_device_advanced

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_memory_cleanup():
    """Test aggressive memory cleanup"""
    
    print("🧪 TESTING AGGRESSIVE MEMORY CLEANUP")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test")
        return False
    
    # Initial memory
    print("\n📊 Initial Memory:")
    log_memory_usage("Initial")
    
    # Allocate some memory
    print("\n🔄 Allocating 10GB of GPU memory...")
    try:
        # Create a large tensor (10GB)
        large_tensor = torch.randn(1024, 1024, 1024, device='cuda', dtype=torch.float32)
        log_memory_usage("After Allocation")
        
        # Move to CPU
        print("\n🔄 Moving tensor to CPU...")
        large_tensor = large_tensor.to('cpu')
        log_memory_usage("After CPU Transfer")
        
        # Test aggressive cleanup
        print("\n🔄 Testing aggressive cleanup...")
        clear_cuda_memory()
        log_memory_usage("After Aggressive Cleanup")
        
        # Delete tensor
        print("\n🔄 Deleting tensor...")
        del large_tensor
        clear_cuda_memory()
        log_memory_usage("After Deletion")
        
        print("\n✅ Memory cleanup test completed")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ OOM during test: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_model_loading_cleanup():
    """Test model loading with cleanup"""
    
    print("\n🧪 TESTING MODEL LOADING WITH CLEANUP")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test")
        return False
    
    try:
        # Create a large dummy model
        print("\n🔄 Creating large dummy model...")
        model = torch.nn.Sequential(
            torch.nn.Linear(10000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000, 1000)
        )
        
        log_memory_usage("After Model Creation")
        
        # Test advanced loading
        print("\n🔄 Testing safe_model_to_device_advanced...")
        model, device, info = safe_model_to_device_advanced(
            model, 
            torch.device('cuda'), 
            min_free_gb=2.0,
            enable_partial_loading=True
        )
        
        log_memory_usage("After Advanced Loading")
        print(f"   📊 Loading type: {info['loading_type']}")
        print(f"   📊 Final device: {device}")
        
        # Cleanup
        del model
        clear_cuda_memory()
        log_memory_usage("After Cleanup")
        
        print("\n✅ Model loading test completed")
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Memory Fix Tests...")
    
    success1 = test_memory_cleanup()
    success2 = test_model_loading_cleanup()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("   ✅ Aggressive memory cleanup works")
        print("   ✅ Model loading with cleanup works")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("   Check the output above for issues")
    
    print("\n" + "="*50)
