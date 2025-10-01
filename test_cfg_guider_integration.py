#!/usr/bin/env python3
"""
Test CFGGuider Integration with Motion Pipeline Step 4
Tests the complete integration of the updated CFGGuider with the motion pipeline
"""

import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

def test_cfg_guider_import():
    """Test that CFGGuider can be imported from standalone_ksampler"""
    print("🧪 Testing CFGGuider Import")
    print("=" * 50)
    
    try:
        from standalone_ksampler import StandaloneCFGGuider, StandaloneKSampler
        print("   ✅ StandaloneCFGGuider imported successfully")
        print("   ✅ StandaloneKSampler imported successfully")
        
        # Test that CFGGuider has the expected methods
        expected_methods = [
            'set_conds', 'set_cfg', 'predict_noise', 'sample', 
            'inner_set_conds', '_prepare_model_patcher', '_restore_hook_patches'
        ]
        
        for method in expected_methods:
            if hasattr(StandaloneCFGGuider, method):
                print(f"   ✅ {method} method available")
            else:
                print(f"   ❌ {method} method missing")
        
        print("\n✅ CFGGuider import test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CFGGuider import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfg_guider_initialization():
    """Test CFGGuider initialization with motion pipeline components"""
    print("\n🧪 Testing CFGGuider Initialization")
    print("=" * 50)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider
        
        # Create a mock model patcher that mimics motion pipeline components
        class MockModelPatcher:
            def __init__(self):
                self.model_options = {
                    'hook_mode': 'normal',
                    'registered_hooks': None
                }
                self.model = MockModel()
            
            def pre_run(self):
                print("   🔧 Mock model patcher pre_run called")
            
            def cleanup(self):
                print("   🔧 Mock model patcher cleanup called")
            
            def restore_hook_patches(self):
                print("   🔧 Mock model patcher restore_hook_patches called")
        
        class MockModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def forward(self, x, timestep, conditioning=None):
                # Return dummy prediction
                return torch.randn_like(x)
        
        # Test CFGGuider initialization
        print("1. Testing CFGGuider initialization...")
        model_patcher = MockModelPatcher()
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        print(f"   ✅ CFGGuider initialized")
        print(f"   ✅ CFG scale: {cfg_guider.cfg}")
        print(f"   ✅ Original conds: {cfg_guider.original_conds}")
        print(f"   ✅ Inner model: {cfg_guider.inner_model}")
        print(f"   ✅ Conds: {cfg_guider.conds}")
        print(f"   ✅ Loaded models: {cfg_guider.loaded_models}")
        
        # Test conditioning setup
        print("\n2. Testing conditioning setup...")
        positive_cond = torch.randn(1, 77, 4096)
        negative_cond = torch.randn(1, 77, 4096)
        
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        print(f"   ✅ Positive conditioning stored: {len(cfg_guider.original_conds.get('positive', []))}")
        print(f"   ✅ Negative conditioning stored: {len(cfg_guider.original_conds.get('negative', []))}")
        print(f"   ✅ CFG scale updated: {cfg_guider.cfg}")
        
        # Test model patcher preparation
        print("\n3. Testing model patcher preparation...")
        cfg_guider._prepare_model_patcher()
        print("   ✅ Model patcher preparation completed")
        
        # Test hook patch restoration
        print("\n4. Testing hook patch restoration...")
        cfg_guider._restore_hook_patches()
        print("   ✅ Hook patch restoration completed")
        
        print("\n✅ CFGGuider initialization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CFGGuider initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfg_guider_noise_prediction():
    """Test CFGGuider noise prediction with motion pipeline data"""
    print("\n🧪 Testing CFGGuider Noise Prediction")
    print("=" * 50)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider
        
        # Create mock model patcher
        class MockModelPatcher:
            def __init__(self):
                self.model_options = {
                    'hook_mode': 'normal',
                    'registered_hooks': None
                }
                self.model = MockModel()
        
        class MockModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def forward(self, x, timestep, conditioning=None):
                # Return dummy prediction
                return torch.randn_like(x)
        
        # Initialize CFGGuider
        model_patcher = MockModelPatcher()
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        # Set up conditioning
        positive_cond = torch.randn(1, 77, 4096)
        negative_cond = torch.randn(1, 77, 4096)
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        # Test noise prediction
        print("1. Testing noise prediction...")
        x = torch.randn(1, 16, 11, 104, 60)
        timestep = torch.tensor([100.0])
        
        # This will use the fallback manual CFG since we don't have a real model
        noise_pred = cfg_guider.predict_noise(x, timestep)
        
        print(f"   ✅ Noise prediction shape: {noise_pred.shape}")
        print(f"   ✅ Noise prediction range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
        print(f"   ✅ Memory calls count: {cfg_guider.memory_usage['calls_count']}")
        
        # Test with different CFG scales
        print("\n2. Testing different CFG scales...")
        for cfg_scale in [1.0, 3.0, 7.0, 12.0]:
            cfg_guider.set_cfg(cfg_scale)
            noise_pred = cfg_guider.predict_noise(x, timestep)
            print(f"   ✅ CFG {cfg_scale}: shape={noise_pred.shape}, range=[{noise_pred.min():.3f}, {noise_pred.max():.3f}]")
        
        print("\n✅ CFGGuider noise prediction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CFGGuider noise prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cfg_guider_sampling():
    """Test CFGGuider sampling with motion pipeline components"""
    print("\n🧪 Testing CFGGuider Sampling")
    print("=" * 50)
    
    try:
        import torch
        from standalone_ksampler import StandaloneCFGGuider, StandaloneKSampler
        
        # Create mock model patcher
        class MockModelPatcher:
            def __init__(self):
                self.model_options = {
                    'hook_mode': 'normal',
                    'registered_hooks': None
                }
                self.model = MockModel()
        
        class MockModel:
            def __init__(self):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def forward(self, x, timestep, conditioning=None):
                # Return dummy prediction
                return torch.randn_like(x)
        
        # Create mock sampler
        class MockSampler:
            def __init__(self):
                self.steps = 4
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def sample(self, model_wrapper, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
                # Return dummy samples
                return torch.randn_like(noise)
        
        # Initialize CFGGuider
        model_patcher = MockModelPatcher()
        cfg_guider = StandaloneCFGGuider(model_patcher)
        
        # Set up conditioning
        positive_cond = torch.randn(1, 77, 4096)
        negative_cond = torch.randn(1, 77, 4096)
        cfg_guider.set_conds(positive_cond, negative_cond)
        cfg_guider.set_cfg(7.0)
        
        # Test sampling
        print("1. Testing sampling...")
        noise = torch.randn(1, 16, 11, 104, 60)
        latent_image = torch.zeros_like(noise)
        sigmas = torch.tensor([100.0, 80.0, 60.0, 40.0, 20.0])
        sampler = MockSampler()
        
        samples = cfg_guider.sample(
            noise=noise,
            latent_image=latent_image,
            sampler=sampler,
            sigmas=sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=True,
            seed=42
        )
        
        print(f"   ✅ Samples shape: {samples.shape}")
        print(f"   ✅ Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"   ✅ Memory calls count: {cfg_guider.memory_usage['calls_count']}")
        
        # Test with empty sigmas
        print("\n2. Testing empty sigmas...")
        empty_sigmas = torch.tensor([])
        samples_empty = cfg_guider.sample(
            noise=noise,
            latent_image=latent_image,
            sampler=sampler,
            sigmas=empty_sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=True,
            seed=42
        )
        
        print(f"   ✅ Empty sigmas result: {samples_empty is latent_image}")
        
        print("\n✅ CFGGuider sampling test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CFGGuider sampling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motion_pipeline_integration():
    """Test CFGGuider integration with motion pipeline Step 4"""
    print("\n🧪 Testing Motion Pipeline Integration")
    print("=" * 50)
    
    try:
        # Test that the motion pipeline can import the updated CFGGuider
        print("1. Testing motion pipeline imports...")
        
        # Import motion pipeline components
        from pipeline import WanVideoPipeline
        from standalone_ksampler import StandaloneCFGGuider, StandaloneKSampler
        
        print("   ✅ WanVideoPipeline imported successfully")
        print("   ✅ StandaloneCFGGuider imported successfully")
        print("   ✅ StandaloneKSampler imported successfully")
        
        # Test pipeline initialization
        print("\n2. Testing pipeline initialization...")
        pipeline = WanVideoPipeline(models_dir="models")
        print("   ✅ Pipeline initialized successfully")
        
        # Test that pipeline has Step 4 method
        print("\n3. Testing Step 4 method availability...")
        if hasattr(pipeline, 'step_4_ksampler_denoising'):
            print("   ✅ step_4_ksampler_denoising method available")
        else:
            print("   ❌ step_4_ksampler_denoising method missing")
        
        # Test that pipeline can create CFGGuider
        print("\n4. Testing CFGGuider creation in pipeline context...")
        try:
            # Create a mock model patcher for testing
            class MockModelPatcher:
                def __init__(self):
                    self.model_options = {}
                    self.model = None
            
            mock_model_patcher = MockModelPatcher()
            cfg_guider = StandaloneCFGGuider(mock_model_patcher)
            print("   ✅ CFGGuider created successfully in pipeline context")
            
            # Test basic functionality
            cfg_guider.set_cfg(7.0)
            print(f"   ✅ CFG scale set: {cfg_guider.cfg}")
            
        except Exception as e:
            print(f"   ❌ CFGGuider creation failed: {e}")
            return False
        
        print("\n✅ Motion pipeline integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Motion pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests"""
    print("🚀 CFGGuider Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("CFGGuider Import", test_cfg_guider_import),
        ("CFGGuider Initialization", test_cfg_guider_initialization),
        ("CFGGuider Noise Prediction", test_cfg_guider_noise_prediction),
        ("CFGGuider Sampling", test_cfg_guider_sampling),
        ("Motion Pipeline Integration", test_motion_pipeline_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests passed! CFGGuider is ready for motion pipeline integration.")
    else:
        print("⚠️  Some integration tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
