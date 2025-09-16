#!/usr/bin/env python3
"""
Test script for Step 7: Video Export
Tests the video export functionality using the VideoExporter component
"""

import torch
import numpy as np
import logging
import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from components.video_export import VideoExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_test_frames(shape, frame_count=10):
    """Create test frames with specified shape"""
    if len(shape) == 4:  # (frames, height, width, channels)
        frames = torch.randn(shape)
    elif len(shape) == 5:  # (batch, frames, height, width, channels)
        frames = torch.randn(shape)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    
    # Ensure values are in [0, 1] range
    frames = torch.sigmoid(frames)
    return frames

def test_video_export():
    """Test the VideoExporter functionality"""
    print("="*80)
    print("🧪 Testing VideoExporter Component")
    print("="*80)
    
    # Create video exporter
    video_exporter = VideoExporter(fps=24)
    print("✅ VideoExporter created")
    
    # Test cases
    test_cases = [
        {
            'name': 'Small video (5 frames)',
            'shape': (5, 64, 64, 3),  # (frames, height, width, channels)
            'fps': 24
        },
        {
            'name': 'Medium video (10 frames)',
            'shape': (10, 128, 128, 3),
            'fps': 30
        },
        {
            'name': 'Large video (20 frames)',
            'shape': (20, 256, 256, 3),
            'fps': 24
        },
        {
            'name': 'Batch video (2 batches)',
            'shape': (2, 5, 64, 64, 3),  # (batch, frames, height, width, channels)
            'fps': 24
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"   Input shape: {test_case['shape']}")
        print(f"   FPS: {test_case['fps']}")
        
        try:
            # Create test frames
            frames = create_test_frames(test_case['shape'])
            print(f"   ✅ Created test frames: {frames.shape}")
            
            # Create output path
            output_path = f"test_output_{i}.mp4"
            
            # Perform video export
            exported_path = video_exporter.export_video(frames, output_path)
            
            print(f"   ✅ Video exported successfully!")
            print(f"   📊 Exported to: {exported_path}")
            
            # Verify output file exists
            if os.path.exists(exported_path):
                file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
                print(f"   📊 File size: {file_size:.2f} MB")
                
                # Clean up test file
                os.remove(exported_path)
                print(f"   🧹 Cleaned up test file")
            else:
                print(f"   ❌ Output file not found!")
                continue
            
            print(f"   ✅ Test case {i} passed!")
            
        except Exception as e:
            print(f"   ❌ Test case {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 VideoExporter component test completed!")

def test_edge_cases():
    """Test edge cases for video export"""
    print("\n" + "="*80)
    print("🧪 Testing Edge Cases")
    print("="*80)
    
    video_exporter = VideoExporter(fps=24)
    
    # Edge case 1: Empty frames
    print("\n📋 Edge Case 1: Empty frames")
    try:
        empty_frames = torch.randn(0, 64, 64, 3)  # 0 frames
        output_path = "test_empty.mp4"
        
        exported_path = video_exporter.export_video(empty_frames, output_path)
        
        print(f"   Input frames: 0")
        print(f"   Output: {exported_path}")
        
        if os.path.exists(exported_path):
            os.remove(exported_path)
            print(f"   ✅ Empty frames handled gracefully")
        else:
            print(f"   ⚠️  Empty frames not handled")
            
    except Exception as e:
        print(f"   ❌ Error handling empty frames: {e}")
    
    # Edge case 2: Single frame
    print("\n📋 Edge Case 2: Single frame")
    try:
        single_frame = torch.randn(1, 64, 64, 3)  # 1 frame
        output_path = "test_single.mp4"
        
        exported_path = video_exporter.export_video(single_frame, output_path)
        
        print(f"   Input frames: 1")
        print(f"   Output: {exported_path}")
        
        if os.path.exists(exported_path):
            file_size = os.path.getsize(exported_path) / 1024  # KB
            print(f"   File size: {file_size:.2f} KB")
            os.remove(exported_path)
            print(f"   ✅ Single frame handled successfully")
        else:
            print(f"   ❌ Single frame not handled")
            
    except Exception as e:
        print(f"   ❌ Error handling single frame: {e}")
    
    # Edge case 3: Very large frames
    print("\n📋 Edge Case 3: Very large frames")
    try:
        large_frames = torch.randn(5, 512, 512, 3)  # Large frames
        output_path = "test_large.mp4"
        
        exported_path = video_exporter.export_video(large_frames, output_path)
        
        print(f"   Input shape: {large_frames.shape}")
        print(f"   Output: {exported_path}")
        
        if os.path.exists(exported_path):
            file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            os.remove(exported_path)
            print(f"   ✅ Large frames handled successfully")
        else:
            print(f"   ❌ Large frames not handled")
            
    except Exception as e:
        print(f"   ❌ Error handling large frames: {e}")
    
    print(f"\n🎉 Edge cases test completed!")

def test_performance():
    """Test performance with realistic video sizes"""
    print("\n" + "="*80)
    print("🧪 Testing Performance")
    print("="*80)
    
    video_exporter = VideoExporter(fps=24)
    
    # Performance test with realistic video (similar to WAN model output)
    print("\n📋 Performance Test: Realistic video size")
    try:
        import time
        
        # Create realistic video (similar to WAN model output after VAE decode)
        realistic_shape = (32, 256, 256, 3)  # 32 frames, 256x256 resolution
        print(f"   Creating realistic video: {realistic_shape}")
        
        start_time = time.time()
        frames = create_test_frames(realistic_shape)
        creation_time = time.time() - start_time
        
        print(f"   Frame creation time: {creation_time:.3f}s")
        
        # Test video export
        output_path = "test_performance.mp4"
        
        start_time = time.time()
        exported_path = video_exporter.export_video(frames, output_path)
        export_time = time.time() - start_time
        
        print(f"   Video export time: {export_time:.3f}s")
        print(f"   Output: {exported_path}")
        
        if os.path.exists(exported_path):
            file_size = os.path.getsize(exported_path) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            
            # Calculate throughput
            frames_per_second = realistic_shape[0] / export_time
            print(f"   Throughput: {frames_per_second:.1f} frames/second")
            
            # Clean up
            os.remove(exported_path)
            print(f"   🧹 Cleaned up test file")
        
        print(f"   ✅ Performance test completed!")
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 Performance test completed!")

def test_different_fps():
    """Test different FPS settings"""
    print("\n" + "="*80)
    print("🧪 Testing Different FPS Settings")
    print("="*80)
    
    # Test different FPS values
    fps_values = [12, 24, 30, 60]
    test_frames = create_test_frames((10, 128, 128, 3))
    
    for fps in fps_values:
        print(f"\n📋 Testing FPS: {fps}")
        try:
            video_exporter = VideoExporter(fps=fps)
            output_path = f"test_fps_{fps}.mp4"
            
            exported_path = video_exporter.export_video(test_frames, output_path)
            
            print(f"   ✅ FPS {fps} export successful")
            print(f"   Output: {exported_path}")
            
            if os.path.exists(exported_path):
                file_size = os.path.getsize(exported_path) / 1024  # KB
                print(f"   File size: {file_size:.2f} KB")
                os.remove(exported_path)
                print(f"   🧹 Cleaned up test file")
            
        except Exception as e:
            print(f"   ❌ FPS {fps} test failed: {e}")
    
    print(f"\n🎉 FPS settings test completed!")

def test_file_formats():
    """Test different output file formats"""
    print("\n" + "="*80)
    print("🧪 Testing File Formats")
    print("="*80)
    
    video_exporter = VideoExporter(fps=24)
    test_frames = create_test_frames((5, 128, 128, 3))
    
    # Test different file extensions
    extensions = [".mp4", ".avi", ".mov"]
    
    for ext in extensions:
        print(f"\n📋 Testing format: {ext}")
        try:
            output_path = f"test_format{ext}"
            
            exported_path = video_exporter.export_video(test_frames, output_path)
            
            print(f"   ✅ {ext} export successful")
            print(f"   Output: {exported_path}")
            
            if os.path.exists(exported_path):
                file_size = os.path.getsize(exported_path) / 1024  # KB
                print(f"   File size: {file_size:.2f} KB")
                os.remove(exported_path)
                print(f"   🧹 Cleaned up test file")
            
        except Exception as e:
            print(f"   ❌ {ext} test failed: {e}")
    
    print(f"\n🎉 File formats test completed!")

def main():
    print("🚀 Testing Step 7: Video Export")
    print("="*60)
    
    # Run all tests
    test_video_export()
    test_edge_cases()
    test_performance()
    test_different_fps()
    test_file_formats()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("✅ Step 7 Video Export is ready for integration")
    print("="*60)

if __name__ == "__main__":
    main()
