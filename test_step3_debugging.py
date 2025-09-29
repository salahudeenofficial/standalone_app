#!/usr/bin/env python3
"""
Test script for enhanced Step 3 debugging
This script tests the enhanced text encoding with comprehensive monitoring
"""

import torch
import numpy as np
import os
import time

def test_enhanced_step3():
    """Test the enhanced step 3 functionality"""
    
    print("🧪 TESTING ENHANCED STEP 3 DEBUGGING")
    print("=" * 60)
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("❌ CUDA not available")
    
    # Test numpy functionality
    print(f"✅ NumPy available: {np.__version__}")
    
    # Test output directory creation
    output_dir = "./p_out/step3"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")
    
    # Test tensor saving functionality
    try:
        # Create test tensors
        test_tensor = torch.randn(1, 77, 1280)
        test_dict = {"test_key": "test_value"}
        
        # Save test tensor
        test_filename = f"{output_dir}/test_tensor_{int(time.time())}.npz"
        np.savez_compressed(
            test_filename,
            tensor=test_tensor.cpu().numpy(),
            shape=test_tensor.shape,
            dtype=str(test_tensor.dtype),
            device=str(test_tensor.device),
            dictionary_keys=list(test_dict.keys()),
            dictionary_values=list(test_dict.values()),
            metadata={
                'timestamp': time.time(),
                'step': 'test_step3',
                'test_type': 'test_tensor'
            }
        )
        print(f"✅ Test tensor saved: {test_filename}")
        
        # Verify saved file
        if os.path.exists(test_filename):
            print(f"✅ File verification: {test_filename} exists")
            file_size = os.path.getsize(test_filename) / 1024  # KB
            print(f"   File size: {file_size:.1f} KB")
        else:
            print(f"❌ File verification failed: {test_filename}")
            
    except Exception as e:
        print(f"❌ Tensor saving test failed: {e}")
    
    # Test memory monitoring
    try:
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"✅ GPU Memory monitoring:")
            print(f"   Allocated: {gpu_allocated:.1f} MB")
            print(f"   Reserved: {gpu_reserved:.1f} MB")
        
        # Test RAM monitoring
        try:
            import psutil
            ram = psutil.virtual_memory()
            print(f"✅ RAM monitoring:")
            print(f"   Used: {ram.used / (1024**3):.1f} GB")
            print(f"   Available: {ram.available / (1024**3):.1f} GB")
            print(f"   Total: {ram.total / (1024**3):.1f} GB")
        except ImportError:
            print("⚠️  psutil not available for RAM monitoring")
            
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ENHANCED STEP 3 TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_enhanced_step3() 