#!/usr/bin/env python3
"""
Simple test to verify monitoring system methods exist and work
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_monitoring_methods_exist():
    """Test if monitoring methods exist"""
    print("🧪 TESTING MONITORING METHODS EXISTENCE")
    print("="*50)
    
    try:
        # Import the pipeline class
        from pipeline import ReferenceVideoPipeline
        
        # Create pipeline instance
        pipeline = ReferenceVideoPipeline()
        print("✅ Pipeline class imported successfully")
        
        # Check if monitoring methods exist
        methods_to_check = [
            '_start_step_monitoring',
            '_end_step_monitoring', 
            '_capture_lora_baseline',
            '_get_system_memory_info',
            '_get_gpu_memory_info',
            '_print_enhanced_model_summary',
            '_print_comprehensive_memory_breakdown',
            '_print_peak_memory_summary',
            '_print_workflow_monitoring_summary',
            '_print_final_workflow_summary'
        ]
        
        for method_name in methods_to_check:
            if hasattr(pipeline, method_name):
                print(f"✅ {method_name} exists")
            else:
                print(f"❌ {method_name} MISSING")
        
        print("\n🎉 Monitoring methods check complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monitoring_methods_exist() 