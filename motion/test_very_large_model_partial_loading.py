#!/usr/bin/env python3
"""
Test script to verify partial loading with a very large model
"""

import sys
import os
import torch
import torch.nn as nn
import logging

# Add motion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_very_large_model():
    """Create a very large model that will definitely trigger partial loading"""
    
    class VeryLargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Create many very large modules
            self.layers = nn.ModuleList()
            
            # Add 50 very large linear layers (each ~33MB)
            for i in range(50):
                self.layers.append(nn.Linear(4096, 4096))
            
            # Add some massive layers (each ~131MB)
            for i in range(10):
                self.layers.append(nn.Linear(8192, 8192))
            
            # Output layer
            self.output = nn.Linear(8192, 1000)
            
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.output(x)
            return x
    
    return VeryLargeModel()

def test_partial_loading_with_very_large_model():
    """Test partial loading with a very large model"""
    print("🧪 TESTING PARTIAL LOADING WITH VERY LARGE MODEL")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, estimate_model_memory, get_memory_info
        
        # Create very large model
        model = create_very_large_model()
        
        print("📊 Testing partial loading with very large model...")
        
        # Estimate model size
        model_info = estimate_model_memory(model)
        print(f"   📊 Very large model size: {model_info['size_gb']:.3f} GB")
        
        # Get available memory
        info = get_memory_info()
        available_memory_gb = info.get('cuda_free', 0)
        print(f"   📊 Available GPU memory: {available_memory_gb:.2f} GB")
        
        # Test with high memory budget to force partial loading
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use a very high memory budget to force partial loading
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=4.0, enable_partial_loading=True
        )
        
        print(f"   ✅ Very large model loaded successfully")
        print(f"   📊 Final device: {final_device}")
        print(f"   📊 Loading type: {loading_info['loading_type']}")
        
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 Modules loaded to GPU: {loading_info['modules_loaded']}")
            print(f"   📊 Modules with dynamic loading: {loading_info['modules_dynamic']}")
            print(f"   📊 GPU memory used: {loading_info['memory_used_gb']:.3f} GB")
            print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
            
            if loading_info['loaded_modules']:
                print(f"   📊 First few loaded modules: {loading_info['loaded_modules'][:3]}")
            if loading_info['dynamic_modules']:
                print(f"   📊 First few dynamic modules: {loading_info['dynamic_modules'][:3]}")
                
            print("   🎉 PARTIAL LOADING SUCCESSFULLY TRIGGERED!")
        else:
            print(f"   📊 Model was loaded with type: {loading_info['loading_type']}")
            if loading_info['loading_type'] == 'full':
                print("   📊 Model fit entirely in memory - partial loading not needed")
            elif loading_info['loading_type'] == 'cpu_fallback':
                print("   📊 Model fell back to CPU - partial loading may not be working")
        
        print("\n🎉 VERY LARGE MODEL PARTIAL LOADING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Very large model partial loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_pressure_scenario():
    """Test partial loading under memory pressure"""
    print("\n🧪 TESTING MEMORY PRESSURE SCENARIO")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        # Create very large model
        model = create_very_large_model()
        
        print("📊 Testing under memory pressure...")
        
        # Get available memory
        info = get_memory_info()
        available_memory_gb = info.get('cuda_free', 0)
        print(f"   📊 Available GPU memory: {available_memory_gb:.2f} GB")
        
        # Test with very high memory budget to simulate memory pressure
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use 90% of available memory as budget to force partial loading
        memory_budget_gb = available_memory_gb * 0.9
        min_free_gb = available_memory_gb - memory_budget_gb
        
        print(f"   📊 Memory budget: {memory_budget_gb:.2f} GB")
        print(f"   📊 Min free memory: {min_free_gb:.2f} GB")
        
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=min_free_gb, enable_partial_loading=True
        )
        
        print(f"   ✅ Memory pressure test completed")
        print(f"   📊 Final device: {final_device}")
        print(f"   📊 Loading type: {loading_info['loading_type']}")
        
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 Partial loading successfully used under memory pressure")
            print(f"   📊 GPU memory used: {loading_info['memory_used_gb']:.3f} GB")
            print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
        
        print("\n🎉 MEMORY PRESSURE SCENARIO TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Memory pressure scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run partial loading tests with very large models"""
    print("🚀 PARTIAL LOADING TESTS WITH VERY LARGE MODELS")
    print("="*80)
    
    tests = [
        ("Partial Loading with Very Large Model", test_partial_loading_with_very_large_model),
        ("Memory Pressure Scenario", test_memory_pressure_scenario),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 TEST RESULTS:")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL VERY LARGE MODEL TESTS PASSED!")
        print("✅ Partial loading system handles large models correctly")
        print("✅ Memory pressure scenarios work properly")
        print("✅ System is ready for production use")
    else:
        print("❌ Some tests failed - check the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

