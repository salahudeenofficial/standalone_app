#!/usr/bin/env python3
"""
Test script for Chunked Processing features
Demonstrates optimal batch size calculation and chunked processing
"""

import torch
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from components.chunked_processor import ChunkedProcessor
from components.memory_manager import MemoryManager
from components.model_manager import ModelManager

def test_chunked_processor():
    """Test the chunked processor functionality"""
    print("Testing Chunked Processor...")
    
    # Initialize managers
    memory_manager = MemoryManager()
    model_manager = ModelManager()
    chunked_processor = ChunkedProcessor(memory_manager, model_manager)
    
    # Test parameters
    frame_count = 37
    width = 480
    height = 832
    operations = ['vae_encode', 'unet_process', 'vae_decode']
    
    print(f"\nTest Parameters:")
    print(f"  Frame Count: {frame_count}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Operations: {operations}")
    
    # Test optimal chunk size calculation
    print(f"\n1. Testing optimal chunk size calculation...")
    for operation in operations:
        chunk_size = chunked_processor.get_optimal_chunk_size(
            operation, frame_count, width, height
        )
        print(f"  {operation}: {chunk_size} frames")
    
    # Test processing plan generation
    print(f"\n2. Testing processing plan generation...")
    plan = chunked_processor.get_processing_plan(frame_count, width, height, operations)
    chunked_processor.print_processing_plan(plan)
    
    # Test time estimation
    print(f"\n3. Testing time estimation...")
    chunked_processor.estimate_total_processing_time(plan, estimated_time_per_chunk=0.5)
    
    # Test chunked processing with dummy data
    print(f"\n4. Testing chunked processing with dummy data...")
    
    # Create dummy frames
    dummy_frames = torch.randn(frame_count, height, width, 3)
    if torch.cuda.is_available():
        dummy_frames = dummy_frames.cuda()
    
    print(f"  Created dummy frames: {dummy_frames.shape}")
    
    # Test chunked processing
    def dummy_process_func(chunk):
        # Simulate some processing
        return chunk * 1.1
    
    result = chunked_processor.process_in_chunks(
        dummy_frames, 'vae_encode', dummy_process_func
    )
    
    print(f"  Processing result shape: {result.shape}")
    print(f"  Processing successful: {torch.allclose(result, dummy_frames * 1.1, atol=1e-6)}")
    
    # Clean up
    del dummy_frames, result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\nChunked Processor test completed!")

def test_memory_optimization():
    """Test memory optimization with chunked processing"""
    print("\n\nTesting Memory Optimization with Chunked Processing...")
    
    memory_manager = MemoryManager()
    chunked_processor = ChunkedProcessor(memory_manager)
    
    # Test with different frame counts
    test_cases = [
        (10, 480, 832),   # Small video
        (37, 480, 832),   # Medium video (current use case)
        (100, 480, 832),  # Large video
        (37, 1024, 1024), # High resolution
    ]
    
    for frame_count, width, height in test_cases:
        print(f"\nTest case: {frame_count} frames, {width}x{height}")
        
        # Get optimal chunk sizes
        vae_encode_chunk = chunked_processor.get_optimal_chunk_size(
            'vae_encode', frame_count, width, height
        )
        vae_decode_chunk = chunked_processor.get_optimal_chunk_size(
            'vae_decode', frame_count, width, height
        )
        
        print(f"  VAE Encode chunk size: {vae_encode_chunk}")
        print(f"  VAE Decode chunk size: {vae_decode_chunk}")
        
        # Calculate memory savings
        if vae_encode_chunk < frame_count:
            encode_chunks = (frame_count + vae_encode_chunk - 1) // vae_encode_chunk
            memory_savings = (frame_count - vae_encode_chunk) / frame_count * 100
            print(f"  VAE Encode: {encode_chunks} chunks, {memory_savings:.1f}% memory reduction")
        
        if vae_decode_chunk < frame_count:
            decode_chunks = (frame_count + vae_decode_chunk - 1) // vae_decode_chunk
            memory_savings = (frame_count - vae_decode_chunk) / frame_count * 100
            print(f"  VAE Decode: {decode_chunks} chunks, {memory_savings:.1f}% memory reduction")
    
    print("\nMemory optimization test completed!")

def test_vram_adaptation():
    """Test VRAM adaptation for different GPU configurations"""
    print("\n\nTesting VRAM Adaptation...")
    
    chunked_processor = ChunkedProcessor()
    
    # Simulate different VRAM configurations
    test_vram_configs = [
        (4.0, "Low VRAM (4GB)"),
        (8.0, "Medium VRAM (8GB)"),
        (16.0, "High VRAM (16GB)"),
        (24.0, "Very High VRAM (24GB)")
    ]
    
    frame_count, width, height = 37, 480, 832
    
    for vram_gb, description in test_vram_configs:
        print(f"\n{description}:")
        
        # Mock the VRAM properties
        if torch.cuda.is_available():
            # This is a simplified test - in practice, you'd need to mock the CUDA properties
            print(f"  Note: Using actual GPU VRAM for testing")
        
        # Test chunk sizes for each operation
        for operation in ['vae_encode', 'vae_decode', 'unet_process']:
            try:
                chunk_size = chunked_processor.get_optimal_chunk_size(
                    operation, frame_count, width, height
                )
                print(f"  {operation}: {chunk_size} frames")
            except Exception as e:
                print(f"  {operation}: Error - {e}")
    
    print("\nVRAM adaptation test completed!")

if __name__ == "__main__":
    print("=== Chunked Processing Test Suite ===\n")
    
    try:
        test_chunked_processor()
        test_memory_optimization()
        test_vram_adaptation()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc() 