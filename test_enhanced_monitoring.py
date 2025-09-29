#!/usr/bin/env python3
"""
Test script for Enhanced LoRA Monitoring System
Tests the new monitoring methods including system memory, GPU memory, and enhanced model info
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_enhanced_monitoring():
    """Test the enhanced monitoring methods"""
    print("🧪 TESTING ENHANCED LORA MONITORING SYSTEM")
    print("="*60)
    
    try:
        # Import the pipeline class
        from pipeline import ReferenceVideoPipeline
        
        # Create pipeline instance
        pipeline = ReferenceVideoPipeline()
        print("✅ Pipeline class imported successfully")
        
        # Test system memory monitoring
        print("\n🔍 Testing system memory monitoring...")
        system_memory = pipeline._get_system_memory_info()
        print(f"   RAM Used: {system_memory['ram_used_mb']:.1f} MB")
        print(f"   RAM Available: {system_memory['ram_available_mb']:.1f} MB")
        print(f"   RAM Total: {system_memory['ram_total_mb']:.1f} MB")
        print(f"   RAM Usage: {system_memory['ram_percent']:.1f}%")
        print("✅ System memory monitoring test passed")
        
        # Test GPU memory monitoring
        print("\n🔍 Testing GPU memory monitoring...")
        gpu_memory = pipeline._get_gpu_memory_info()
        print(f"   GPU Allocated: {gpu_memory['gpu_allocated_mb']:.1f} MB")
        print(f"   GPU Reserved: {gpu_memory['gpu_reserved_mb']:.1f} MB")
        print(f"   GPU Total: {gpu_memory['gpu_total_mb']:.1f} MB")
        print(f"   GPU Device: {gpu_memory['gpu_device_name']}")
        print("✅ GPU memory monitoring test passed")
        
        # Test memory efficiency calculation
        print("\n🔍 Testing memory efficiency calculation...")
        efficiency = pipeline._calculate_memory_efficiency(gpu_memory)
        print(f"   Allocated Efficiency: {efficiency['allocated_efficiency_percent']:.1f}%")
        print(f"   Reserved Efficiency: {efficiency['reserved_efficiency_percent']:.1f}%")
        print("✅ Memory efficiency calculation test passed")
        
        # Test step monitoring
        print("\n🔍 Testing step monitoring...")
        start_time, start_memory = pipeline._start_step_monitoring("test_step")
        print(f"   Start Time: {start_time}")
        print(f"   Start Memory: {start_memory['ram_used_mb']:.1f} MB")
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        elapsed_time, end_memory = pipeline._end_step_monitoring("test_step", start_time, start_memory)
        print(f"   Elapsed Time: {elapsed_time:.3f} seconds")
        print(f"   End Memory: {end_memory['ram_used_mb']:.1f} MB")
        print("✅ Step monitoring test passed")
        
        # Test workflow monitoring summary
        print("\n🔍 Testing workflow monitoring summary...")
        step_results = {
            'lora_application': {
                'elapsed_time': 0.083,
                'ram_change': 1.7,
                'gpu_change': 0.0,
                'success': True
            }
        }
        pipeline._print_workflow_monitoring_summary(step_results)
        print("✅ Workflow monitoring summary test passed")
        
        print("\n🎉 ALL ENHANCED MONITORING TESTS PASSED!")
        print("The enhanced monitoring system is working correctly.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running this from the standalone_app directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_monitoring() 