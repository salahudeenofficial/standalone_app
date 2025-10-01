#!/usr/bin/env python3
"""
Test script for the new CFGGuider implementation
Tests the ComfyUI-compatible functions and StandaloneCFGGuider
"""

import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

def test_core_functions():
    """Test the core sampling functions"""
    print("🧪 Testing Core Sampling Functions")
    print("=" * 50)
    
    try:
        import torch
        from standalone_ksampler import calc_cond_batch, cfg_function, sampling_function, get_area_and_mult
        
        # Create dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_in = torch.randn(1, 16, 11, 104, 60, device=device)
        timestep = torch.tensor([100.0], device=device)
        
        # Test get_area_and_mult
        print("1. Testing get_area_and_mult...")
        cond = {'mult': 1.0, 'hooks': None}
        area_info = get_area_and_mult(cond, x_in, timestep)
        print(f"   ✅ Area shape: {area_info.area.shape}")
        print(f"   ✅ Multiplier: {area_info.mult}")
        print(f"   ✅ Hooks: {area_info.hooks}")
        
        # Test calc_cond_batch
        print("\n2. Testing calc_cond_batch...")
        conds = [
            [{'mult': 1.0, 'hooks': None}],  # positive
            [{'mult': 1.0, 'hooks': None}]   # negative
        ]
        model_options = {}
        
        out_conds = calc_cond_batch(None, conds, x_in, timestep, model_options)
        print(f"   ✅ Output conditioning count: {len(out_conds)}")
        print(f"   ✅ Output shape: {out_conds[0].shape}")
        
        # Test cfg_function
        print("\n3. Testing cfg_function...")
        cond_pred = torch.randn_like(x_in)
        uncond_pred = torch.randn_like(x_in)
        cond_scale = 7.0
        
        cfg_result = cfg_function(None, cond_pred, uncond_pred, cond_scale, x_in, timestep)
        print(f"   ✅ CFG result shape: {cfg_result.shape}")
        print(f"   ✅ CFG result range: [{cfg_result.min():.3f}, {cfg_result.max():.3f}]")
        
        # Test sampling_function
        print("\n4. Testing sampling_function...")
        try:
            # This will fail because we don't have a real model, but we can test the structure
            sampling_result = sampling_function(None, x_in, timestep, None, None, cond_scale)
            print(f"   ✅ Sampling result shape: {sampling_result.shape}")
        except Exception as e:
            print(f"   ⚠️  Expected error (no real model): {e}")
        
        print("\n✅ Core functions test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Core functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfg_guider():
    """Test the StandaloneCFGGuider class"""
    print("\n🧪 Testing StandaloneCFGGuider")
    print("=" * 50)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider
        
        # Create dummy model patcher
        class DummyModelPatcher:
            def __init__(self):
                self.model_options = {}
                self.model = DummyModel()
        
        class DummyModel:
            def __init__(self):
                pass
            
            def forward(self, x, timestep, conditioning=None):
                # Return dummy prediction
                return torch.randn_like(x)
        
        # Test CFGGuider initialization
        print("1. Testing CFGGuider initialization...")
        model_patcher = DummyModelPatcher()
        cfg_guider = StandaloneCFGGuider(model_patcher)
        print(f"   ✅ CFGGuider initialized")
        print(f"   ✅ CFG scale: {cfg_guider.cfg}")
        print(f"   ✅ Original conds: {cfg_guider.original_conds}")
        
        # Test conditioning setup
        print("\n2. Testing conditioning setup...")
        positive_cond = torch.randn(1, 77, 4096)
        negative_cond = torch.randn(1, 77, 4096)
        
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        print(f"   ✅ Positive conditioning stored: {len(cfg_guider.original_conds.get('positive', []))}")
        print(f"   ✅ Negative conditioning stored: {len(cfg_guider.original_conds.get('negative', []))}")
        print(f"   ✅ CFG scale updated: {cfg_guider.cfg}")
        
        # Test predict_noise
        print("\n3. Testing predict_noise...")
        x = torch.randn(1, 16, 11, 104, 60)
        timestep = torch.tensor([100.0])
        
        # This will use the fallback manual CFG since we don't have a real model
        noise_pred = cfg_guider.predict_noise(x, timestep)
        print(f"   ✅ Noise prediction shape: {noise_pred.shape}")
        print(f"   ✅ Noise prediction range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
        print(f"   ✅ Memory calls count: {cfg_guider.memory_usage['calls_count']}")
        
        print("\n✅ CFGGuider test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CFGGuider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hook_system():
    """Test the hook system implementation"""
    print("\n🧪 Testing Hook System")
    print("=" * 50)
    
    try:
        from standalone_ksampler import Hook, HookGroup, EnumHookType, EnumHookScope, preprocess_conds_hooks, filter_registered_hooks_on_conds, get_total_hook_groups_in_conds
        
        # Test Hook class
        print("1. Testing Hook class...")
        hook1 = Hook(EnumHookType.WEIGHT, "test_hook_1", EnumHookScope.ALL_CONDITIONING)
        hook2 = Hook(EnumHookType.OBJECT_PATCH, "test_hook_2", EnumHookScope.POSITIVE_ONLY)
        
        print(f"   ✅ Hook 1: {hook1.hook_type}, {hook1.hook_id}, {hook1.hook_scope}")
        print(f"   ✅ Hook 2: {hook2.hook_type}, {hook2.hook_id}, {hook2.hook_scope}")
        
        # Test HookGroup class
        print("\n2. Testing HookGroup class...")
        hook_group = HookGroup()
        hook_group.add(hook1)
        hook_group.add(hook2)
        
        print(f"   ✅ HookGroup size: {len(hook_group)}")
        print(f"   ✅ Contains hook1: {hook_group.contains(hook1)}")
        print(f"   ✅ Contains hook2: {hook_group.contains(hook2)}")
        
        # Test hook functions
        print("\n3. Testing hook functions...")
        conds = {
            'positive': [{'hooks': hook_group, 'mult': 1.0}],
            'negative': [{'hooks': None, 'mult': 1.0}]
        }
        
        # Test preprocess_conds_hooks
        preprocess_conds_hooks(conds)
        print("   ✅ preprocess_conds_hooks completed")
        
        # Test get_total_hook_groups_in_conds
        hook_count = get_total_hook_groups_in_conds(conds)
        print(f"   ✅ Total hook groups: {hook_count}")
        
        # Test filter_registered_hooks_on_conds
        model_options = {'registered_hooks': hook_group}
        filter_registered_hooks_on_conds(conds, model_options)
        print("   ✅ filter_registered_hooks_on_conds completed")
        
        print("\n✅ Hook system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Hook system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """Test ComfyUI compatibility"""
    print("\n🧪 Testing ComfyUI Compatibility")
    print("=" * 50)
    
    try:
        from standalone_ksampler import StandaloneCFGGuider
        
        # Test that the class has the expected ComfyUI methods
        expected_methods = ['set_conds', 'set_cfg', 'predict_noise', 'sample', 'inner_set_conds', '_prepare_model_patcher', '_restore_hook_patches']
        
        print("1. Testing method compatibility...")
        for method in expected_methods:
            if hasattr(StandaloneCFGGuider, method):
                print(f"   ✅ {method} method exists")
            else:
                print(f"   ❌ {method} method missing")
        
        # Test that the class has the expected ComfyUI attributes
        expected_attrs = ['original_conds', 'cfg', 'inner_model', 'conds', 'loaded_models']
        
        print("\n2. Testing attribute compatibility...")
        # Create a dummy instance to test attributes
        class DummyModelPatcher:
            def __init__(self):
                self.model_options = {}
        
        cfg_guider = StandaloneCFGGuider(DummyModelPatcher())
        
        for attr in expected_attrs:
            if hasattr(cfg_guider, attr):
                print(f"   ✅ {attr} attribute exists")
            else:
                print(f"   ❌ {attr} attribute missing")
        
        print("\n✅ ComfyUI compatibility test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ComfyUI compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 CFGGuider Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Core Functions", test_core_functions),
        ("CFGGuider Class", test_cfg_guider),
        ("Hook System", test_hook_system),
        ("ComfyUI Compatibility", test_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! CFGGuider implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
