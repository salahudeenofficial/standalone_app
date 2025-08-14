#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        from components.memory_manager import MemoryManager
        print("✓ MemoryManager imported successfully")
    except Exception as e:
        print(f"✗ MemoryManager import failed: {e}")
        return False
    
    try:
        from components.model_manager import ModelManager
        print("✓ ModelManager imported successfully")
    except Exception as e:
        print(f"✗ ModelManager import failed: {e}")
        return False
    
    try:
        from components.chunked_processor import ChunkedProcessor
        print("✓ ChunkedProcessor imported successfully")
    except Exception as e:
        print(f"✗ ChunkedProcessor import failed: {e}")
        return False
    
    return True

def test_memory_manager():
    """Test memory manager basic functionality"""
    print("\nTesting MemoryManager basic functionality...")
    
    try:
        from components.memory_manager import MemoryManager
        memory_manager = MemoryManager()
        print("✓ MemoryManager created successfully")
        
        # Test adding cleanup point
        memory_manager.add_cleanup_point("test", "Test cleanup point")
        print("✓ Cleanup point added successfully")
        
        # Test marking cleanup point complete
        memory_manager.mark_cleanup_point_complete("test")
        print("✓ Cleanup point marked complete successfully")
        
        return True
    except Exception as e:
        print(f"✗ MemoryManager test failed: {e}")
        return False

def test_model_manager():
    """Test model manager basic functionality"""
    print("\nTesting ModelManager basic functionality...")
    
    try:
        from components.model_manager import ModelManager
        model_manager = ModelManager()
        print("✓ ModelManager created successfully")
        
        # Test status methods
        status = model_manager.get_status()
        print(f"✓ ModelManager status: {status}")
        
        return True
    except Exception as e:
        print(f"✗ ModelManager test failed: {e}")
        return False

def test_chunked_processor():
    """Test chunked processor basic functionality"""
    print("\nTesting ChunkedProcessor basic functionality...")
    
    try:
        from components.chunked_processor import ChunkedProcessor
        chunked_processor = ChunkedProcessor()
        print("✓ ChunkedProcessor created successfully")
        
        # Test VRAM status
        vram_status = chunked_processor.get_vram_status()
        print(f"✓ VRAM status: {vram_status}")
        
        return True
    except Exception as e:
        print(f"✗ ChunkedProcessor test failed: {e}")
        return False

def test_cuda_availability():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
        
        try:
            # Test basic CUDA operations
            x = torch.randn(100, 100).cuda()
            y = x * 2
            del x, y
            torch.cuda.empty_cache()
            print("✓ Basic CUDA operations successful")
        except Exception as e:
            print(f"✗ Basic CUDA operations failed: {e}")
            return False
    else:
        print("ℹ CUDA not available - using CPU only")
    
    return True

if __name__ == "__main__":
    print("=== Simple Functionality Test ===\n")
    
    tests = [
        test_basic_imports,
        test_memory_manager,
        test_model_manager,
        test_chunked_processor,
        test_cuda_availability,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Basic functionality is working.")
    else:
        print("⚠ Some tests failed. Check the errors above.")
    
    print("===============================") 