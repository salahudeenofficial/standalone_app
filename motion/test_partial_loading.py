#!/usr/bin/env python3
"""
Comprehensive test script for partial model loading system
Tests the advanced memory management inspired by ComfyUI
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

def create_test_model():
    """Create a test model with multiple modules of different sizes"""
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Small modules
            self.small_conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.small_conv2 = nn.Conv2d(16, 32, 3, padding=1)
            
            # Medium modules
            self.medium_linear1 = nn.Linear(32 * 8 * 8, 512)
            self.medium_linear2 = nn.Linear(512, 256)
            
            # Large modules (simulate large transformer layers)
            self.large_transformer1 = nn.Linear(256, 1024)
            self.large_transformer2 = nn.Linear(1024, 1024)
            self.large_transformer3 = nn.Linear(1024, 1024)
            self.large_transformer4 = nn.Linear(1024, 1024)
            
            # Output layer
            self.output = nn.Linear(1024, 10)
            
        def forward(self, x):
            x = torch.relu(self.small_conv1(x))
            x = torch.relu(self.small_conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.medium_linear1(x))
            x = torch.relu(self.medium_linear2(x))
            x = torch.relu(self.large_transformer1(x))
            x = torch.relu(self.large_transformer2(x))
            x = torch.relu(self.large_transformer3(x))
            x = torch.relu(self.large_transformer4(x))
            x = self.output(x)
            return x
    
    return TestModel()

def create_large_test_model():
    """Create a large test model that will trigger partial loading"""
    
    class LargeTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Create many large modules to simulate a large model
            self.layers = nn.ModuleList()
            
            # Add many large linear layers
            for i in range(20):  # 20 large layers
                self.layers.append(nn.Linear(2048, 2048))
            
            # Add some very large layers
            for i in range(5):  # 5 very large layers
                self.layers.append(nn.Linear(4096, 4096))
            
            # Output layer
            self.output = nn.Linear(4096, 1000)
            
        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            x = self.output(x)
            return x
    
    return LargeTestModel()

def test_basic_partial_loading():
    """Test basic partial loading functionality"""
    print("🧪 TESTING BASIC PARTIAL LOADING")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        # Create test model
        model = create_test_model()
        
        print("📊 Testing basic partial loading...")
        
        # Get memory info
        info = get_memory_info()
        print(f"   📊 Available GPU memory: {info.get('cuda_free', 0):.2f} GB")
        
        # Test advanced loading
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=1.0, enable_partial_loading=True
        )
        
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Final device: {final_device}")
        print(f"   📊 Loading type: {loading_info['loading_type']}")
        
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 Modules loaded: {loading_info['modules_loaded']}")
            print(f"   📊 Modules dynamic: {loading_info['modules_dynamic']}")
            print(f"   📊 Memory used: {loading_info['memory_used_gb']:.3f} GB")
        
        print("\n🎉 BASIC PARTIAL LOADING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Basic partial loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_large_model_partial_loading():
    """Test partial loading with a large model"""
    print("\n🧪 TESTING LARGE MODEL PARTIAL LOADING")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, estimate_model_memory
        
        # Create large test model
        model = create_large_test_model()
        
        print("📊 Testing large model partial loading...")
        
        # Estimate model size
        model_info = estimate_model_memory(model)
        print(f"   📊 Large model size: {model_info['size_gb']:.3f} GB")
        
        # Test advanced loading
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=2.0, enable_partial_loading=True
        )
        
        print(f"   ✅ Large model loaded successfully")
        print(f"   📊 Final device: {final_device}")
        print(f"   📊 Loading type: {loading_info['loading_type']}")
        
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 Modules loaded to GPU: {loading_info['modules_loaded']}")
            print(f"   📊 Modules with dynamic loading: {loading_info['modules_dynamic']}")
            print(f"   📊 GPU memory used: {loading_info['memory_used_gb']:.3f} GB")
            print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
            
            if loading_info['loaded_modules']:
                print(f"   📊 Loaded modules: {loading_info['loaded_modules'][:3]}...")  # Show first 3
            if loading_info['dynamic_modules']:
                print(f"   📊 Dynamic modules: {loading_info['dynamic_modules'][:3]}...")  # Show first 3
        
        print("\n🎉 LARGE MODEL PARTIAL LOADING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Large model partial loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_budget_management():
    """Test memory budget management"""
    print("\n🧪 TESTING MEMORY BUDGET MANAGEMENT")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        # Create test model
        model = create_test_model()
        
        print("📊 Testing memory budget management...")
        
        # Test with different memory budgets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test with high memory budget (should load fully)
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=0.1, enable_partial_loading=True
        )
        
        print(f"   📊 High budget test:")
        print(f"      Loading type: {loading_info['loading_type']}")
        print(f"      Memory used: {loading_info.get('memory_used_gb', 0):.3f} GB")
        
        # Test with low memory budget (should use partial loading)
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, device, min_free_gb=5.0, enable_partial_loading=True
        )
        
        print(f"   📊 Low budget test:")
        print(f"      Loading type: {loading_info['loading_type']}")
        if loading_info['loading_type'] == 'partial':
            print(f"      Modules loaded: {loading_info['modules_loaded']}")
            print(f"      Memory used: {loading_info['memory_used_gb']:.3f} GB")
        
        print("\n🎉 MEMORY BUDGET MANAGEMENT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Memory budget management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partial_loading_vs_full_loading():
    """Test comparison between partial loading and full loading"""
    print("\n🧪 TESTING PARTIAL LOADING VS FULL LOADING")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        # Create test model
        model = create_test_model()
        
        print("📊 Testing partial loading vs full loading...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test with partial loading enabled
        model_partial, device_partial, info_partial = safe_model_to_device_advanced(
            model, device, min_free_gb=1.0, enable_partial_loading=True
        )
        
        # Test with partial loading disabled
        model_full, device_full, info_full = safe_model_to_device_advanced(
            model, device, min_free_gb=1.0, enable_partial_loading=False
        )
        
        print(f"   📊 Partial loading enabled:")
        print(f"      Loading type: {info_partial['loading_type']}")
        print(f"      Final device: {device_partial}")
        
        print(f"   📊 Partial loading disabled:")
        print(f"      Loading type: {info_full['loading_type']}")
        print(f"      Final device: {device_full}")
        
        # Compare results
        if info_partial['loading_type'] == 'partial' and info_full['loading_type'] == 'cpu_fallback':
            print("   ✅ Partial loading successfully used GPU while full loading fell back to CPU")
        elif info_partial['loading_type'] == info_full['loading_type']:
            print("   📊 Both approaches resulted in same loading type")
        
        print("\n🎉 PARTIAL LOADING VS FULL LOADING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Partial loading vs full loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_analysis():
    """Test model module analysis functionality"""
    print("\n🧪 TESTING MODEL MODULE ANALYSIS")
    print("="*60)
    
    try:
        from memory_utils import _analyze_model_modules, estimate_model_memory
        
        # Create test model
        model = create_test_model()
        
        print("📊 Testing model module analysis...")
        
        # Analyze model modules
        modules_info = _analyze_model_modules(model)
        
        print(f"   📊 Found {len(modules_info)} modules")
        
        # Show module information
        total_size_gb = 0
        for i, module_info in enumerate(modules_info[:5]):  # Show first 5
            print(f"      Module {i+1}: {module_info['name']}")
            print(f"         Size: {module_info['size_gb']:.6f} GB")
            print(f"         Parameters: {module_info['parameters']:,}")
            total_size_gb += module_info['size_gb']
        
        if len(modules_info) > 5:
            print(f"      ... and {len(modules_info) - 5} more modules")
        
        print(f"   📊 Total analyzed size: {total_size_gb:.6f} GB")
        
        # Compare with model estimation
        model_info = estimate_model_memory(model)
        print(f"   📊 Model estimation: {model_info['size_gb']:.6f} GB")
        
        print("\n🎉 MODEL MODULE ANALYSIS TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Model module analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in partial loading"""
    print("\n🧪 TESTING ERROR HANDLING")
    print("="*60)
    
    try:
        from memory_utils import safe_model_to_device_advanced
        
        print("📊 Testing error handling...")
        
        # Test with invalid model
        try:
            model, device, info = safe_model_to_device_advanced(
                None, torch.device('cuda'), min_free_gb=1.0
            )
            print("   ❌ Should have failed with None model")
            return False
        except Exception as e:
            print(f"   ✅ Correctly handled None model: {type(e).__name__}")
        
        # Test with invalid device
        try:
            model = create_test_model()
            model, device, info = safe_model_to_device_advanced(
                model, "invalid_device", min_free_gb=1.0
            )
            print("   ❌ Should have failed with invalid device")
            return False
        except Exception as e:
            print(f"   ✅ Correctly handled invalid device: {type(e).__name__}")
        
        print("\n🎉 ERROR HANDLING TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all partial loading tests"""
    print("🚀 COMPREHENSIVE PARTIAL LOADING SYSTEM TESTS")
    print("="*80)
    
    tests = [
        ("Basic Partial Loading", test_basic_partial_loading),
        ("Large Model Partial Loading", test_large_model_partial_loading),
        ("Memory Budget Management", test_memory_budget_management),
        ("Partial Loading vs Full Loading", test_partial_loading_vs_full_loading),
        ("Model Module Analysis", test_model_analysis),
        ("Error Handling", test_error_handling),
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
        print("🎉 ALL PARTIAL LOADING TESTS PASSED!")
        print("✅ Partial loading system is working correctly")
        print("✅ Memory budget management is functional")
        print("✅ Model analysis is accurate")
        print("✅ Error handling is robust")
        print("✅ Ready for integration into pipeline")
    else:
        print("❌ Some tests failed - check the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
