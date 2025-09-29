#!/usr/bin/env python3
"""
Test script for Memory Management features
Demonstrates tensor tracking, cleanup, and memory optimization
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from components.memory_manager import MemoryManager

def test_memory_manager():
    """Test the memory manager functionality"""
    print("Testing Memory Manager...")
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Create some test tensors
    print("\n1. Creating test tensors...")
    tensor1 = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor2 = torch.randn(500, 500, device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor3 = torch.randn(2000, 2000, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track tensors
    memory_manager.track_tensor(tensor1, "test_tensor_1", cleanup_priority=1)
    memory_manager.track_tensor(tensor2, "test_tensor_2", cleanup_priority=2)
    memory_manager.track_tensor(tensor3, "test_tensor_3", cleanup_priority=1)
    
    # Print memory stats
    print("\n2. Memory stats after creating tensors:")
    memory_manager.print_memory_stats()
    
    # Clean up tensor by priority
    print("\n3. Cleaning up tensors with priority <= 1...")
    memory_manager.cleanup_tensors_by_priority(max_priority=1)
    
    print("\n4. Memory stats after priority cleanup:")
    memory_manager.print_memory_stats()
    
    # Clean up specific tensor
    print("\n5. Cleaning up specific tensor...")
    memory_manager.cleanup_tensor(tensor2, "test_tensor_2")
    
    print("\n6. Final memory stats:")
    memory_manager.print_memory_stats()
    
    # Clean up all remaining tensors
    print("\n7. Cleaning up all remaining tensors...")
    memory_manager.cleanup_all_tracked_tensors()
    
    print("\n8. Final memory stats after cleanup:")
    memory_manager.print_memory_stats()
    
    print("\nMemory Manager test completed!")

def test_tensor_cleanup():
    """Test explicit tensor cleanup"""
    print("\n\nTesting Explicit Tensor Cleanup...")
    
    memory_manager = MemoryManager()
    
    # Create a large tensor
    print("Creating large test tensor...")
    large_tensor = torch.randn(5000, 5000, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track it
    memory_manager.track_tensor(large_tensor, "large_test_tensor", cleanup_priority=1)
    
    print("Memory stats before cleanup:")
    memory_manager.print_memory_stats()
    
    # Clean it up
    print("Cleaning up large tensor...")
    memory_manager.cleanup_tensor(large_tensor, "large_test_tensor")
    
    print("Memory stats after cleanup:")
    memory_manager.print_memory_stats()
    
    print("Explicit tensor cleanup test completed!")

if __name__ == "__main__":
    print("=== Memory Management Test Suite ===\n")
    
    try:
        test_memory_manager()
        test_tensor_cleanup()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 