#!/usr/bin/env python3
"""
Test script for multithreaded memory tracking system
"""

import torch
import time
import sys
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_tracker import create_memory_tracker, track_memory_during_operation

def test_memory_tracking():
    """Test the memory tracking system"""
    print("🧪 Testing Multithreaded Memory Tracking System")
    print("="*60)
    
    # Create memory tracker
    tracker = create_memory_tracker(interval=0.1, log_file="test_memory_tracking.log")
    
    try:
        # Start tracking
        print("\n🔍 Starting memory tracking...")
        tracker.start_tracking()
        
        # Simulate some operations
        print("\n📊 Simulating memory operations...")
        
        # Track a simple operation
        @track_memory_during_operation(tracker, "test_operation", "test_step", {"description": "Testing memory tracking"})
        def simulate_operation():
            print("   🔄 Simulating operation...")
            time.sleep(1)
            
            # Simulate GPU memory allocation if available
            if torch.cuda.is_available():
                print("   🎮 Allocating GPU memory...")
                test_tensor = torch.randn(1000, 1000).cuda()
                time.sleep(0.5)
                
                print("   🧹 Freeing GPU memory...")
                del test_tensor
                torch.cuda.empty_cache()
                time.sleep(0.5)
            else:
                print("   💻 GPU not available, simulating CPU operations...")
                test_tensor = torch.randn(1000, 1000)
                time.sleep(0.5)
                del test_tensor
                time.sleep(0.5)
            
            print("   ✅ Operation completed")
        
        # Execute the operation
        simulate_operation()
        
        # Wait a bit more
        print("\n⏱️  Waiting for more samples...")
        time.sleep(2)
        
        # Stop tracking
        tracker.stop_tracking()
        
        # Print summary
        tracker.print_memory_summary()
        
        print("\n✅ Memory tracking test completed!")
        print("📄 Memory data saved to test_memory_tracking.log")
        
    except Exception as e:
        print(f"\n❌ Memory tracking test failed: {e}")
        tracker.stop_tracking()
        import traceback
        traceback.print_exc()

def test_step_tracking():
    """Test tracking during step execution"""
    print("\n🧪 Testing Step Tracking")
    print("="*40)
    
    # Create memory tracker
    tracker = create_memory_tracker(interval=0.05, log_file="test_step_tracking.log")
    
    try:
        # Start tracking
        tracker.start_tracking()
        
        # Simulate Step 1
        print("\n📋 Simulating Step 1...")
        @track_memory_during_operation(tracker, "step1_vae_latent", "step1", {
            "width": 256, "height": 256, "length": 16
        })
        def simulate_step1():
            print("   🔄 Loading VAE...")
            time.sleep(0.5)
            
            if torch.cuda.is_available():
                print("   🎮 Allocating VAE memory...")
                vae_tensor = torch.randn(4, 32, 32, 32).cuda()
                time.sleep(0.3)
                del vae_tensor
                torch.cuda.empty_cache()
            
            print("   ✅ Step 1 completed")
        
        simulate_step1()
        
        # Simulate Step 2
        print("\n📋 Simulating Step 2...")
        @track_memory_during_operation(tracker, "step2_model_loading", "step2", {
            "unet_path": "models/unet.safetensors",
            "clip_path": "models/clip.safetensors"
        })
        def simulate_step2():
            print("   🔄 Loading UNet...")
            time.sleep(0.8)
            
            if torch.cuda.is_available():
                print("   🎮 Allocating UNet memory...")
                unet_tensor = torch.randn(4, 64, 64, 64).cuda()
                time.sleep(0.5)
                
                print("   🔄 Loading CLIP...")
                time.sleep(0.3)
                clip_tensor = torch.randn(4, 16, 16, 16).cuda()
                time.sleep(0.2)
                
                print("   🧹 Freeing memory...")
                del unet_tensor, clip_tensor
                torch.cuda.empty_cache()
            
            print("   ✅ Step 2 completed")
        
        simulate_step2()
        
        # Stop tracking
        tracker.stop_tracking()
        
        # Print summary
        tracker.print_memory_summary()
        
        print("\n✅ Step tracking test completed!")
        print("📄 Step data saved to test_step_tracking.log")
        
    except Exception as e:
        print(f"\n❌ Step tracking test failed: {e}")
        tracker.stop_tracking()
        import traceback
        traceback.print_exc()

def main():
    """Run all memory tracking tests"""
    print("🚀 Memory Tracking System Tests")
    print("="*60)
    
    # Test basic memory tracking
    test_memory_tracking()
    
    # Test step tracking
    test_step_tracking()
    
    print("\n🎉 All memory tracking tests completed!")
    print("📊 Check the generated log files for detailed memory data")

if __name__ == "__main__":
    main()
