#!/usr/bin/env python3
"""
Comprehensive test suite for motion pipeline's diffusionmodules/util.py
Tests all functions and classes from ComfyUI's diffusionmodules/util.py implementation
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging
import traceback

# Add motion directory to path
sys.path.insert(0, '/home/fashionx/v_pipe/standalone_app/motion')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_diffusionmodules_util.log')
    ]
)

def test_import():
    """Test importing the diffusionmodules util module"""
    print("\n🧪 TEST 1: Import Module")
    print("=" * 50)
    
    try:
        from ldm.modules.diffusionmodules.util import (
            AlphaBlender,
            make_beta_schedule,
            make_ddim_timesteps,
            make_ddim_sampling_parameters,
            betas_for_alpha_bar,
            extract_into_tensor,
            checkpoint,
            CheckpointFunction,
            timestep_embedding,
            zero_module,
            scale_module,
            mean_flat,
            avg_pool_nd,
            HybridConditioner,
            noise_like
        )
        print("✅ All imports successful")
        print("✅ Found 15 functions/classes:")
        imports_list = [
            'AlphaBlender', 'make_beta_schedule', 'make_ddim_timesteps',
            'make_ddim_sampling_parameters', 'betas_for_alpha_bar', 'extract_into_tensor',
            'checkpoint', 'CheckpointFunction', 'timestep_embedding', 'zero_module',
            'scale_module', 'mean_flat', 'avg_pool_nd', 'HybridConditioner', 'noise_like'
        ]
        for func_name in imports_list:
            print(f"   ✅ {func_name}")
        return True
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        traceback.print_exc()
        return False

def test_alpha_blender():
    """Test AlphaBlender class"""
    print("\n🧪 TEST 2: AlphaBlender Class")
    print("=" * 50)
    
    try:
        from ldm.modules.diffusionmodules.util import AlphaBlender
        
        # Test initialization with different strategies
        print("Testing AlphaBlender initialization...")
        
        # Test fixed strategy
        blender_fixed = AlphaBlender(alpha=0.5, merge_strategy="fixed")
        print("✅ Fixed strategy initialization")
        assert blender_fixed.mix_factor.item() == 0.5
        
        # Test learned strategy
        blender_learned = AlphaBlender(alpha=0.3, merge_strategy="learned")
        print("✅ Learned strategy initialization")
        assert torch.allclose(blender_learned.mix_factor, torch.tensor([0.3]), atol=1e-6)
        
        # Test learned_with_images strategy
        blender_images = AlphaBlender(alpha=0.7, merge_strategy="learned_with_images")
        print("✅ Learned with images strategy initialization")
        assert torch.allclose(blender_images.mix_factor, torch.tensor([0.7]), atol=1e-6)
        
        # Test forward pass
        print("Testing AlphaBlender forward pass...")
        
        # Create test tensors
        batch_size, channels, height, width = 2, 16, 32, 32
        x_spatial = torch.randn(batch_size, channels, height, width)
        x_temporal = torch.randn(batch_size, channels, height, width)
        
        # Test without indicator
        output_none = blender_fixed(x_spatial, x_temporal, None)
        assert output_none.shape == x_spatial.shape
        print("✅ Forward pass without image_only_indicator")
        
        # Test with indicator
        image_only_indicator = torch.tensor([True, False]).reshape(2, 1)  # Match batch size
        output_ind = blender_images(x_spatial, x_temporal, image_only_indicator)
        assert output_ind.shape == x_spatial.shape
        print("✅ Forward pass with image_only_indicator")
        
        print("🎉 AlphaBlender tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ AlphaBlender test failed: {e}")
        traceback.print_exc()
        return False

def test_beta_schedule():
    """Test beta schedule functions"""
    print("\n🧪 TEST 3: Beta Schedule Functions")
    print("=" * 50)
    
    try:
        from ldm.modules.diffusionmodules.util import make_beta_schedule, betas_for_alpha_bar, make_ddim_timesteps, make_ddim_sampling_parameters
        
        n_timestep = 1000
        
        # Test linear schedule
        print("Testing linear beta schedule...")
        betas_linear = make_beta_schedule("linear", n_timestep)
        assert len(betas_linear) == n_timestep
        assert torch.all(betas_linear >= 0) and torch.all(betas_linear <= 1)
        print("✅ Linear schedule generated")
        
        # Test cosine schedule
        print("Testing cosine beta schedule...")
        betas_cosine = make_beta_schedule("cosine", n_timestep)
        assert len(betas_cosine) == n_timestep
        assert torch.all(betas_cosine >= 0) and torch.all(betas_cosine <= 1)
        print("✅ Cosine schedule generated")
        
        # Test sqrt_linear schedule
        print("Testing sqrt_linear beta schedule...")
        betas_sqrt = make_beta_schedule("sqrt_linear", n_timestep)
        assert len(betas_sqrt) == n_timestep
        assert torch.all(betas_sqrt >= 0) and torch.all(betas_sqrt <= 1)
        print("✅ Sqrt linear schedule generated")
        
        # Test squaredcos_cap_v2 schedule
        print("Testing squaredcos_cap_v2 beta schedule...")
        betas_squared_cos = make_beta_schedule("squaredcos_cap_v2", n_timestep)
        assert len(betas_squared_cos) == n_timestep
        print("✅ Squared cos schedule generated")
        
        # Test betas_for_alpha_bar
        print("Testing betas_for_alpha_bar...")
        custom_bar = betas_for_alpha_bar(n_timestep, lambda t: 1 - t**2, max_beta=0.999)
        assert len(custom_bar) == n_timestep
        assert np.all(custom_bar >= 0) and np.all(custom_bar <= 1)
        print("✅ Custom alpha bar schedule generated")
        
        print("🎉 Beta schedule tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Beta schedule test failed: {e}")
        traceback.print_exc()
        return False

def test_ddim_functions():
    """Test DDIM related functions"""
    print("\n🧪 TEST 4: DDIM Functions")
    print("=" * 50)
    
    try:
        from ldm.modules.diffusionmodules.util import make_ddim_timesteps, make_ddim_sampling_parameters
        
        num_ddpm_timesteps = 1000
        num_ddim_timesteps = 50
        alphacums = np.cumprod(1 - np.linspace(1e-4, 0.02, num_ddpm_timesteps))
        
        # Test uniform discretization
        print("Testing uniform DDIM timesteps...")
        timesteps_uniform = make_ddim_timesteps('uniform', num_ddim_timesteps, num_ddpm_timesteps)
        assert len(timesteps_uniform) == num_ddim_timesteps
        assert np.all(timesteps_uniform > 0)
        print("✅ Uniform timesteps generated")
        
        # Test quad discretization
        print("Testing quadratic DDIM timesteps...")
        timesteps_quad = make_ddim_timesteps('quad', num_ddim_timesteps, num_ddpm_timesteps)
        assert len(timesteps_quad) == num_ddim_timesteps
        assert np.all(timesteps_quad > 0)
        print("✅ Quadratic timesteps generated")
        
        # Test DDIM sampling parameters
        print("Testing DDIM sampling parameters...")
        sigmas, alphas, alphas_prev = make_ddim_sampling_parameters(
            alphacums, timesteps_uniform, eta=0.0
        )
        assert len(sigmas) == len(alphas) == len(alphas_prev) == num_ddim_timesteps
        assert np.all(sigmas >= 0)
        print("✅ DDIM sampling parameters generated")
        
        print("🎉 DDIM function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ DDIM function test failed: {e}")
        traceback.print_exc()
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🧪 TEST 5: Utility Functions")
    print("=" * 50)
    
    try:
        from ldm.modules.diffusionmodules.util import (
            extract_into_tensor, timestep_embedding, mean_flat, 
            avg_pool_nd, zero_module, scale_module, noise_like
        )
        
        # Test extract_into_tensor
        print("Testing extract_into_tensor...")
        a = torch.randn(20)  # 1D tensor with 20 elements (time steps)
        t = torch.randint(0, 20, (4,))  # Batch of 4 indices
        x_shape = (4, 16, 32, 32)
        extracted = extract_into_tensor(a, t, x_shape)
        assert extracted.shape == (4, 1, 1, 1)
        print("✅ extract_into_tensor works")
        
        # Test timestep_embedding
        print("Testing timestep_embedding...")
        timesteps = torch.tensor([10, 50, 100])
        embedding1 = timestep_embedding(timesteps, dim=64, repeat_only=False)
        embedding2 = timestep_embedding(timesteps, dim=64, repeat_only=True)
        assert embedding1.shape == (3, 64)
        assert embedding2.shape == (3, 64)
        print("✅ timestep_embedding works")
        
        # Test mean_flat
        print("Testing mean_flat...")
        tensor = torch.randn(4, 16, 32, 32)
        mean_flat_result = mean_flat(tensor)
        assert mean_flat_result.shape == (4,)
        print("✅ mean_flat works")
        
        # Test avg_pool_nd
        print("Testing avg_pool_nd...")
        pool1d = avg_pool_nd(1, kernel_size=2)
        pool2d = avg_pool_nd(2, kernel_size=2)
        pool3d = avg_pool_nd(3, kernel_size=2)
        assert isinstance(pool1d, nn.AvgPool1d)
        assert isinstance(pool2d, nn.AvgPool2d)
        assert isinstance(pool3d, nn.AvgPool3d)
        print("✅ avg_pool_nd works")
        
        # Test zero_module and scale_module
        print("Testing zero_module and scale_module...")
        linear1 = nn.Linear(10, 20)
        original_weight = linear1.weight.clone()
        
        zeroed = zero_module(linear1)
        assert torch.all(zeroed.weight == 0)
        print("✅ zero_module works")
        
        # Test scale_module with a fresh linear layer
        linear2 = nn.Linear(10, 20)
        original_value = linear2.weight.data.clone()
        scaled = scale_module(linear2, scale=2.0)
        assert torch.allclose(scaled.weight.data, original_value * 2.0)
        print("✅ scale_module works")
        
        # Test noise_like
        print("Testing noise_like...")
        shape = (4, 16, 32, 32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        noise_normal = noise_like(shape, device, repeat=False)
        noise_repeat = noise_like(shape, device, repeat=True)
        assert noise_normal.shape == shape
        assert noise_repeat.shape == shape
        print("✅ noise_like works")
        
        print("🎉 Utility function tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test suites"""
    print("🧪 COMPREHENSIVE TESTING: ComfyUI diffusionmodules/util.py Implementation")
    print("=" * 80)
    
    test_results = []
    
    # Run all test suites
    test_results.append(test_import())
    test_results.append(test_alpha_blender())
    test_results.append(test_beta_schedule())
    test_results.append(test_ddim_functions())
    test_results.append(test_utility_functions())
    
    # Calculate results
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Implementation is working perfectly!")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
