#!/usr/bin/env python3
"""
Test Enhanced Model Integration
Tests the improved model patcher integration and error handling
"""

import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

def test_enhanced_model_patcher_integration():
    """Test enhanced model patcher integration"""
    print("🧪 Testing Enhanced Model Patcher Integration")
    print("=" * 60)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider, WrapperExecutor
        
        # Create enhanced mock model patcher
        class EnhancedMockModelPatcher:
            def __init__(self):
                self.model_options = {
                    'hook_mode': 'normal',
                    'registered_hooks': None,
                    'transformer_options': {}
                }
                self.model = EnhancedMockModel()
                self.current_patcher = None
                self.hook_patches = {}
                self.hook_patches_backup = None
                self.hook_backup = {}
                self.cached_hook_patches = {}
                self.current_hooks = None
                self.forced_hooks = None
                self.is_clip = False
                self.hook_mode = 'normal'
            
            def pre_run(self):
                print("   🔧 Enhanced model patcher pre_run called")
                self.current_patcher = self
            
            def cleanup(self):
                print("   🔧 Enhanced model patcher cleanup called")
                self.current_patcher = None
            
            def restore_hook_patches(self):
                print("   🔧 Enhanced model patcher restore_hook_patches called")
                if self.hook_patches_backup is not None:
                    self.hook_patches = self.hook_patches_backup
                    self.hook_patches_backup = None
            
            def clean_hooks(self):
                print("   🔧 Enhanced model patcher clean_hooks called")
                self.unpatch_hooks()
                self.clear_cached_hook_weights()
            
            def unpatch_hooks(self, whitelist_keys_set=None):
                print("   🔧 Enhanced model patcher unpatch_hooks called")
                if len(self.hook_backup) == 0:
                    self.current_hooks = None
                    return
                self.hook_backup.clear()
                self.current_hooks = None
            
            def clear_cached_hook_weights(self):
                print("   🔧 Enhanced model patcher clear_cached_hook_weights called")
                self.cached_hook_patches.clear()
            
            def set_hook_mode(self, hook_mode):
                print(f"   🔧 Enhanced model patcher set_hook_mode called: {hook_mode}")
                self.hook_mode = hook_mode
            
            def prepare_hook_patches_current_keyframe(self, timestep, hook_group, model_options):
                print(f"   🔧 Enhanced model patcher prepare_hook_patches_current_keyframe called")
                print(f"      Timestep: {timestep}")
                print(f"      Hook group: {hook_group}")
                print(f"      Model options: {model_options}")
        
        class EnhancedMockModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.current_patcher = None
            
            def forward(self, x, timestep, conditioning=None):
                # Return dummy prediction
                return torch.randn_like(x)
        
        # Test enhanced CFGGuider initialization
        print("1. Testing enhanced CFGGuider initialization...")
        model_patcher = EnhancedMockModelPatcher()
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        print(f"   ✅ CFGGuider initialized with wrapper executor")
        print(f"   ✅ Wrapper executor type: {type(cfg_guider.wrapper_executor)}")
        
        # Test wrapper executor functionality
        print("\n2. Testing wrapper executor functionality...")
        
        # Add test callbacks
        def test_pre_run_callback(model_patcher):
            print("   🔧 Test pre_run callback executed")
        
        def test_cleanup_callback(model_patcher):
            print("   🔧 Test cleanup callback executed")
        
        cfg_guider.wrapper_executor.add_callback('pre_run', test_pre_run_callback)
        cfg_guider.wrapper_executor.add_callback('cleanup', test_cleanup_callback)
        
        print("   ✅ Callbacks added to wrapper executor")
        
        # Test enhanced model patcher preparation
        print("\n3. Testing enhanced model patcher preparation...")
        cfg_guider._prepare_model_patcher()
        print("   ✅ Enhanced model patcher preparation completed")
        
        # Test hook mode determination
        print("\n4. Testing hook mode determination...")
        hook_mode = cfg_guider._get_hook_mode()
        print(f"   ✅ Hook mode determined: {hook_mode}")
        
        # Test hook mode application
        print("\n5. Testing hook mode application...")
        cfg_guider._apply_hook_mode(hook_mode)
        print("   ✅ Hook mode applied successfully")
        
        # Test hook patch preparation
        print("\n6. Testing hook patch preparation...")
        timestep = torch.tensor([100.0])
        cfg_guider._prepare_hook_patches_for_timestep(timestep)
        print("   ✅ Hook patch preparation completed")
        
        # Test enhanced cleanup
        print("\n7. Testing enhanced cleanup...")
        cfg_guider._cleanup_model_patcher()
        print("   ✅ Enhanced cleanup completed")
        
        # Test hook patch restoration
        print("\n8. Testing hook patch restoration...")
        cfg_guider._restore_hook_patches()
        print("   ✅ Hook patch restoration completed")
        
        # Test wrapper executor cleanup
        print("\n9. Testing wrapper executor cleanup...")
        cfg_guider.wrapper_executor.cleanup()
        print("   ✅ Wrapper executor cleanup completed")
        
        print("\n✅ Enhanced model patcher integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced model patcher integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_error_handling():
    """Test enhanced error handling and recovery"""
    print("\n🧪 Testing Enhanced Error Handling")
    print("=" * 60)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider
        
        # Create mock model patcher with error scenarios
        class ErrorMockModelPatcher:
            def __init__(self, error_scenario='none'):
                self.model_options = {}
                self.model = ErrorMockModel(error_scenario)
                self.error_scenario = error_scenario
            
            def pre_run(self):
                if self.error_scenario == 'pre_run_error':
                    raise RuntimeError("Simulated pre_run error")
                print("   🔧 Mock model patcher pre_run called")
            
            def cleanup(self):
                if self.error_scenario == 'cleanup_error':
                    raise RuntimeError("Simulated cleanup error")
                print("   🔧 Mock model patcher cleanup called")
            
            def restore_hook_patches(self):
                if self.error_scenario == 'restore_error':
                    raise RuntimeError("Simulated restore error")
                print("   🔧 Mock model patcher restore_hook_patches called")
        
        class ErrorMockModel:
            def __init__(self, error_scenario='none'):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.error_scenario = error_scenario
            
            def forward(self, x, timestep, conditioning=None):
                if self.error_scenario == 'forward_error':
                    raise RuntimeError("Simulated forward error")
                return torch.randn_like(x)
        
        # Test 1: Normal operation
        print("1. Testing normal operation...")
        model_patcher = ErrorMockModelPatcher('none')
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        # Set up conditioning
        positive_cond = torch.randn(1, 77, 4096)
        negative_cond = torch.randn(1, 77, 4096)
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        # Test noise prediction
        x = torch.randn(1, 16, 11, 104, 60)
        timestep = torch.tensor([100.0])
        noise_pred = cfg_guider.predict_noise(x, timestep)
        
        print(f"   ✅ Normal operation: noise_pred shape {noise_pred.shape}")
        
        # Test 2: Pre-run error handling
        print("\n2. Testing pre-run error handling...")
        model_patcher = ErrorMockModelPatcher('pre_run_error')
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        try:
            cfg_guider._prepare_model_patcher()
            print("   ✅ Pre-run error handled gracefully")
        except Exception as e:
            print(f"   ❌ Pre-run error not handled: {e}")
            return False
        
        # Test 3: Cleanup error handling
        print("\n3. Testing cleanup error handling...")
        model_patcher = ErrorMockModelPatcher('cleanup_error')
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        try:
            cfg_guider._cleanup_model_patcher()
            print("   ✅ Cleanup error handled gracefully")
        except Exception as e:
            print(f"   ❌ Cleanup error not handled: {e}")
            return False
        
        # Test 4: Restore error handling
        print("\n4. Testing restore error handling...")
        model_patcher = ErrorMockModelPatcher('restore_error')
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        try:
            cfg_guider._restore_hook_patches()
            print("   ✅ Restore error handled gracefully")
        except Exception as e:
            print(f"   ❌ Restore error not handled: {e}")
            return False
        
        # Test 5: Forward error handling
        print("\n5. Testing forward error handling...")
        model_patcher = ErrorMockModelPatcher('forward_error')
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        # Set up conditioning
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        try:
            noise_pred = cfg_guider.predict_noise(x, timestep)
            print(f"   ✅ Forward error handled gracefully: noise_pred shape {noise_pred.shape}")
        except Exception as e:
            print(f"   ❌ Forward error not handled: {e}")
            return False
        
        print("\n✅ Enhanced error handling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_wrapper_executor():
    """Test wrapper executor functionality"""
    print("\n🧪 Testing Wrapper Executor")
    print("=" * 60)
    
    try:
        from standalone_ksampler import WrapperExecutor
        
        # Create mock model patcher
        class MockModelPatcher:
            def __init__(self):
                self.model_options = {}
        
        model_patcher = MockModelPatcher()
        executor = WrapperExecutor(model_patcher)
        
        # Test callback functionality
        print("1. Testing callback functionality...")
        
        callback_called = False
        def test_callback(model_patcher):
            nonlocal callback_called
            callback_called = True
            print("   🔧 Test callback executed")
        
        executor.add_callback('test_type', test_callback)
        executor.execute_callbacks('test_type', model_patcher)
        
        if callback_called:
            print("   ✅ Callback functionality working")
        else:
            print("   ❌ Callback functionality failed")
            return False
        
        # Test wrapper functionality
        print("\n2. Testing wrapper functionality...")
        
        wrapper_called = False
        def test_wrapper(*args, **kwargs):
            nonlocal wrapper_called
            wrapper_called = True
            print("   🔧 Test wrapper executed")
            return args, kwargs
        
        executor.add_wrapper('test_type', test_wrapper)
        args, kwargs = executor.execute_wrappers('test_type', 'test_arg', test_kwarg='test_value')
        
        if wrapper_called:
            print("   ✅ Wrapper functionality working")
        else:
            print("   ❌ Wrapper functionality failed")
            return False
        
        # Test cleanup
        print("\n3. Testing cleanup...")
        executor.cleanup()
        print("   ✅ Wrapper executor cleanup completed")
        
        print("\n✅ Wrapper executor test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Wrapper executor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all enhanced model integration tests"""
    print("🚀 Enhanced Model Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("Enhanced Model Patcher Integration", test_enhanced_model_patcher_integration),
        ("Enhanced Error Handling", test_enhanced_error_handling),
        ("Wrapper Executor", test_wrapper_executor)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*25} {test_name} {'='*25}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 ENHANCED MODEL INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All enhanced model integration tests passed! Model integration is ready.")
    else:
        print("⚠️  Some enhanced model integration tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
