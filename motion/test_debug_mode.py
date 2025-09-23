#!/usr/bin/env python3
"""
Test script to demonstrate debug mode functionality
"""

import os
import sys
import subprocess

def test_debug_mode():
    """Test the debug mode functionality"""
    print("🔍 Testing Debug Mode (Steps 1-4 only)")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("pipeline.py"):
        print("❌ pipeline.py not found in current directory")
        return False
    
    # Run debug mode
    print("🚀 Running: python3.10 pipeline.py --debug")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ["python3.10", "pipeline.py", "--debug"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        # Check if debug output file was created
        if os.path.exists("debug_step4_output.npy"):
            file_size = os.path.getsize("debug_step4_output.npy") / (1024*1024)
            print(f"✅ Debug output file created: debug_step4_output.npy ({file_size:.2f} MB)")
            return True
        else:
            print("❌ Debug output file not found")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running debug mode: {e}")
        return False

def test_complete_mode():
    """Test the complete pipeline mode"""
    print("\n🚀 Testing Complete Pipeline Mode (All 7 steps)")
    print("="*50)
    
    print("🚀 Running: python3.10 pipeline.py")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ["python3.10", "pipeline.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        # Check if video output was created
        if os.path.exists("output_video.mp4") or os.path.exists("output_video_frames"):
            print("✅ Video output created")
            return True
        else:
            print("❌ Video output not found")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running complete mode: {e}")
        return False

if __name__ == "__main__":
    print("🧪 WAN Pipeline Debug Mode Test")
    print("="*60)
    
    # Test debug mode
    debug_success = test_debug_mode()
    
    # Ask user if they want to test complete mode
    if debug_success:
        print("\n" + "="*60)
        response = input("✅ Debug mode successful! Test complete pipeline? (y/n): ")
        if response.lower() in ['y', 'yes']:
            test_complete_mode()
    
    print("\n🎉 Test completed!")
