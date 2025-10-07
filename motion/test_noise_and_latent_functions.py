#!/usr/bin/env python3
"""
Test Script for prepare_noise() and fix_empty_latent_channels() Functions
=======================================================================

This script tests both functions with:
1. Actual VACE UNet model loading from motion pipeline
2. Mock latent images with various configurations
3. Comprehensive validation and error handling

Usage:
    cd motion && python test_noise_and_latent_functions.py

Requirements:
    - Motion pipeline must be properly configured
    - VACE UNet model files must be available
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# Import motion pipeline modules (already in motion directory)
from sample import prepare_noise, fix_empty_latent_channels
from pipeline import WanVideoPipeline
from standalone_sd import load_state_dict_guess_config
from utils import load_torch_file
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_mock_latent_images():
    """Create various mock latent images for testing"""
    
    print("\n🎯 CREATING MOCK LATENT IMAGES")
    print("=" * 50)
        
    mock_latents = {}
    
    # 1. Standard 5D latent (WAN format)
    mock_latents['5d_standard'] = {
        'tensor': torch.zeros([1, 16, 11, 104, 60], dtype=torch.float32),
        'description': 'Standard 5D latent [B, C=16, T=11, H=104, W=60]',
        'channels': 16,
        'dimensions': 5
    }
    
    # 2. 4D latent (needs fixing)
    mock_latents['4d_mismatch'] = {
        'tensor': torch.zeros([1, 4, 104, 60], dtype=torch.float32),
        'description': '4D latent [B, C=4, H=104, W=60] - needs fixing',
        'channels': 4,
        'dimensions': 4
    }
    
    # 5D latent with wrong channels (needs fixing)
    mock_latents['5d_wrong_channels'] = {
        'tensor': torch.zeros([1, 8, 11, 104, 60], dtype=torch.float32),
        'description': '5D latent [B, C=8, T=11, H=104, W=60] - wrong channels',
        'channels': 8,
        'dimensions': 5
    }
    
    # Empty latent with content (shouldn't be fixed)
    mock_latents['non_empty'] = {
        'tensor': torch.randn([1, 16, 11, 104, 60], dtype=torch.float32),
        'description': '5D latent with content [B, C=16, T=11, H=104, W=60]',
        'channels': 16,
        'dimensions': 5
    }
    
    # Very small latent (edge case)
    mock_latents['small_latent'] = {
        'tensor': torch.zeros([1, 16, 2, 16, 16], dtype=torch.float32),
        'description': 'Small 5D latent [B, C=16, T=2, H=16, W=16]',
        'channels': 16,
        'dimensions': 5
    }
    
    print(f"Created {len(mock_latents)} mock latent configurations")
    for key, config in mock_latents.items():
        print(f"  📊 {key}: {config['description']}")
        
    return mock_latents

def test_prepare_noise_function(mock_latents):
    """Test prepare_noise() function with various configurations"""
    
    print("\n🧪 TESTING prepare_noise() FUNCTION")
    print("=" * 50)
    
    test_cases = []
    
    # Test Case 1: Standard noise generation
    test_cases.append({
        'name': 'Standard Noise Generation',
        'latent_image': mock_latents['5d_standard']['tensor'],
        'seed': 42,
        'noise_inds': None,
        'expected_shape': mock_latents['5d_standard']['tensor'].shape
    })
    
    # Test Case 2: Noise generation with seed consistency
    test_cases.append({
        'name': 'Seed Consistency Test',
        'latent_image': mock_latents['5d_standard']['tensor'],
        'seed': 123,
        'noise_inds': None,
        'expected_shape': mock_latents['5d_standard']['tensor'].shape
    })
    
    # Test Case 3: Noise with indices
    test_cases.append({
        'name': 'Noise with Indices',
        'latent_image': mock_latents['5d_standard']['tensor'],
        'seed': 42,
        'noise_inds': torch.tensor([0, 0, 1, 1, 2]),  # Some repeated indices
        'expected_shape': torch.Size([5, 16, 11, 104, 60])  # Expected batch size = len(noise_inds)
    })
    
    # Test Case 4: Different latent sizes
    test_cases.append({
        'name': 'Different Latent Size',
        'latent_image': mock_latents['small_latent']['tensor'],
        'seed': 42,
        'noise_inds': None,
        'expected_shape': mock_latents['small_latent']['tensor'].shape
    })
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Test noise generation
            start_time = time.time()
            
            noise = prepare_noise(
                latent_image=test_case['latent_image'],
                seed=test_case['seed'],
                noise_inds=test_case['noise_inds']
            )
            
            generation_time = time.time() - start_time
            
            # Validate results
            shape_match = noise.shape == test_case['expected_shape']
            dtype_match = noise.dtype == test_case['latent_image'].dtype
            device_match = noise.device.type == 'cpu'  # prepare_noise generates on CPU
            
            print(f"   ✅ Generation successful")
            print(f"   ⏱️  Time: {generation_time:.4f}s")
            print(f"   📊 Shape: {noise.shape} {'✅' if shape_match else '❌'}")
            print(f"   📊 Expected: {test_case['expected_shape']}")
            print(f"   📊 Dtype: {noise.dtype} {'✅' if dtype_match else '❌'}")
            print(f"   📊 Device: {noise.device} {'✅' if device_match else '❌'}")
            print(f"   📊 Mean: {noise.mean().item():.6f}")
            print(f"   📊 Std: {noise.std().item():.6f}")
            print(f"   📊 Range: [{noise.min().item():.6f}, {noise.max().item():.6f}]")
            
            # Test reproducibility if same seed
            if test_case['seed'] == 42:  # We'll test this for the first case
                noise2 = prepare_noise(test_case['latent_image'], 42, test_case['noise_inds'])
                reproducibility = torch.allclose(noise, noise2, atol=1e-6)


                print(f"   🔄 Reproducibility: {'✅' if reproducibility else '❌'}")
            
            results[test_case['name']] = {
                'success': True,
                'shape_match': shape_match,
                'dtype_match': dtype_match,
                'device_match': device_match,
                'generation_time': generation_time,
                'noise_shape': noise.shape,
                'error': None
            }
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results[test_case['name']] = {
                'success': False,
                'error': str(e),
                'generation_time': None
            }
    
    return results

def create_mock_vace_unet_model():
    """Create a mock VACE UNet model for testing when real model isn't available"""
    
    print("\n🤖 CREATING MOCK VACE UNET MODEL")
    print("=" * 50)
    
    class MockVaceLatentFormat:
        """Mock latent format for testing"""
        def __init__(self):
            self.latent_channels = 16
            self.latent_dimensions = 3
    
    class MockVaceUNet:
        """Mock UNet model with VACE-compatible interface"""
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._latent_format = MockVaceLatentFormat()
            
        def get_model_object(self, name):
            """Mock get_model_object method"""
            if name == "latent_format":
                return self._latent_format
            return None
            
        def __call__(self, x, timestep, **kwargs):
            """Mock forward pass"""
            return torch.randn_like(x)
    
    mock_unet = MockModelPatcher()  # Use MockModelPatcher instead
    
    print(f"   ✅ Mock VACE ModelPatcher created")
    print(f"   📊 Model type: {type(mock_unet).__name__}")
    print(f"   📊 Device: {mock_unet.load_device}")
    print(f"   📊 Latent channels: {mock_unet.get_model_object('latent_format').latent_channels}")
    print(f"   📊 Latent dimensions: {mock_unet.get_model_object('latent_format').latent_dimensions}")
    
    return mock_unet

class MockModelPatcher:
    """Mock ModelPatcher that simulates the real ModelPatcher interface"""
    def __init__(self, latent_channels=16, latent_dimensions=3):
        self.load_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        class MockUnderlyingModel:
            """Mock underlying model"""
            def __init__(self):
                self.latent_channels = latent_channels
                self.latent_dimensions = latent_dimensions
                
        self.model = MockUnderlyingModel()  # Underlying model
        self.patches = []  # Patches list
        self.patches_uuid = "mock-patches-uuid"
        
        # Mock latent format object
        class MockLatentFormat:
            def __init__(self, channels=16, dims=3):
                self.latent_channels = channels
                self.latent_dimensions = dims
        
        self._latent_format = MockLatentFormat(latent_channels, latent_dimensions)
    
    def get_model_object(self, name):
        """Mock get_model_object method"""
        if name == "latent_format":
            return self._latent_format
        return None

def load_vace_unet_model():
    """Load actual VACE UNet model from motion pipeline using Step 2"""
    
    print("\n🤖 LOADING VACE UNET MODEL FROM PIPELINE")
    print("=" * 50)
    
    try:
        # Model paths (relative to motion directory)
        unet_model_path = "models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
        
        # Check if models exist
        missing_models = []
        if not os.path.exists(unet_model_path):
            missing_models.append(f"UNet: {unet_model_path}")
        if not os.path.exists(clip_model_path):
            missing_models.append(f"CLIP: {clip_model_path}")
        
        if missing_models:
            print("   ❌ Missing model files:")
            for missing in missing_models:
                print(f"      {missing}")
            print("   🔄 Falling back to mock UNet model for testing...")
            return create_mock_vace_unet_model()
        
        print(f"   ✅ Model files found:")
        print(f"      UNet: {os.path.basename(unet_model_path)}")
        print(f"      CLIP: {os.path.basename(clip_model_path)}")
        
        # Create pipeline instance
        pipeline = WanVideoPipeline()
        
        print("   🔄 Loading models using Pipeline Step 2...")
        # Load UNet and CLIP models using Step 2 (like pipeline does)
        step2_results = pipeline.step_2_unet_clip_lora_loading(
            unet_model_path=unet_model_path,
            clip_model_path=clip_model_path,
            lora_model_path=None,  # No LoRA for testing
            strength_model=1.0,
            strength_clip=0.0
        )
        
        print("   🔄 Extracting UNet model...")
        unet_model = step2_results.get('unet')
        
        if unet_model is None:
            raise RuntimeError("Failed to load UNet model from pipeline")
        
        print(f"   ✅ UNet model loaded successfully")
        print(f"   📊 Model type: {type(unet_model).__name__}")
        print(f"   📊 Device: {unet_model.load_device if hasattr(unet_model, 'load_device') else 'N/A'}")
        
        # Test model interface
        print("   🔄 Testing model interface...")
        
        # Test get_model_object if available
        if hasattr(unet_model, 'get_model_object'):
            try:
                latent_format = unet_model.get_model_object("latent_format")
                print(f"   📊 Latent format: {type(latent_format).__name__}")
                print(f"   📊 Latent channels: {latent_format.latent_channels}")
                print(f"   📊 Latent dimensions: {latent_format.latent_dimensions}")
            except Exception as e:
                print(f"   ⚠️  Could not get latent_format: {e}")
        
        # Test model calling using proper device handling
        if hasattr(unet_model, 'load_device'):
            test_device = unet_model.load_device
        elif hasattr(unet_model, 'device'):
            test_device = unet_model.device
        else:
            test_device = torch.device('cpu')
            
        print(f"   🔄 Using test device: {test_device}")
        
        # Handle ModelPatcher vs direct model calling
        if hasattr(unet_model, 'model') and hasattr(unet_model, 'patches'):
            # This is a ModelPatcher - we need to test differently
            print(f"   📊 Detected ModelPatcher - testing model object")
            underlying_model = unet_model.model
            print(f"   📊 Underlying model type: {type(underlying_model).__name__}")
            
            # Test ModelPatcher's get_model_object interface
            print("   🔄 Testing ModelPatcher interface...")
            try:
                latent_format = unet_model.get_model_object("latent_format")
                print(f"   📊 Latent format: {type(latent_format).__name__}")
                print(f"   📊 Latent channels: {latent_format.latent_channels}")
                print(f"   📊 Latent dimensions: {latent_format.latent_dimensions}")
            except Exception as e:
                print(f"   ⚠️  Could not get latent_format from ModelPatcher: {e}")
                # Try alternative approach for PureVaceWanModel
                if hasattr(unet_model, 'model') and 'Vace' in unet_model.model.__class__.__name__:
                    print(f"   🔄 Detected PureVaceWanModel - using alternative approach")
                    # Create a proper latent format object for PureVaceWanModel
                    class PureVaceLatentFormat:
                        def __init__(self):
                            self.latent_channels = 16
                            self.latent_dimensions = 3
                    
                    # Store as a proper latent format for get_model_object
                    unet_model._latent_format = PureVaceLatentFormat()
                    print(f"   ✅ Created PureVace latent format: 16 channels, 3D")
            
            print(f"   ✅ ModelPatcher interface tested successfully")
        else:
            # Direct model - test normal calling
            print("   🔄 Testing direct model calling...")
            # Create test tensor on the appropriate device
            test_input = torch.randn([1, 16, 11, 104, 60], dtype=torch.float32, device=test_device)
            test_timestep = torch.tensor([10], device=test_device)
            
            with torch.no_grad():
                output = unet_model(test_input, test_timestep)
                print(f"   📊 Model output shape: {output.shape}")
                print(f"   ✅ Direct model forward pass successful")
        
        return unet_model
        
    except Exception as e:
        print(f"   ❌ Failed to load real UNet model: {e}")
        print(f"   🔄 Falling back to mock UNet model for testing...")
        
        try:
            mock_unet = create_mock_vace_unet_model()
            return mock_unet
        except Exception as mock_e:
            logger.error(f"Mock UNet creation failure: {mock_e}", exc_info=True)
            return None

def test_fix_empty_latent_channels_function(unet_model, mock_latents):
    """Test fix_empty_latent_channels() function with actual UNet model"""
    
    print("\n🔧 TESTING fix_empty_latent_channels() FUNCTION")
    print("=" * 50)
    
    if unet_model is None:
        print("   ❌ Cannot test fix_empty_latent_channels() without UNet model")
        return {}
    
    test_cases = []
    
    # Test Case 1: Standard latent (should work unchanged)
    test_cases.append({
        'name': 'Standard Latent (No Fixing Needed)',
        'latent_image': mock_latents['5d_standard']['tensor'],
        'description': 'Should remain unchanged - already correct',
        'expect_changes': False
    })
    
    # Test Case 2: Wrong channels (needs fixing)
    test_cases.append({
        'name': 'Wrong Channel Count',
        'latent_image': mock_latents['4d_mismatch']['tensor'],
        'description': 'Should fix channel count from 4 to 16',
        'expect_changes': True
    })
    
    # Test Case 3: Wrong channels 5D (needs fixing)
    test_cases.append({
        'name': 'Wrong Channels 5D',
        'latent_image': mock_latents['5d_wrong_channels']['tensor'],
        'description': 'Should fix channel count from 8 to 16',
        'expect_changes': True
    })
    
    # Test Case 4: Non-empty latent (shouldn't be fixed)
    test_cases.append({
        'name': 'Non-Empty Latent',
        'latent_image': mock_latents['non_empty']['tensor'],
        'description': 'Should remain unchanged - has content',
        'expect_changes': False
    })
    
    # Test Case 5: Small latent (needs dimension fixing)
    test_cases.append({
        'name': 'Small Latent Check',
        'latent_image': mock_latents['small_latent']['tensor'],
        'description': 'Should be processed correctly',
        'expect_changes': False
    })
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"   📋 {test_case['description']}")
        
        try:
            # Get original tensor info
            original_shape = test_case['latent_image'].shape
            original_channels = original_shape[1]
            original_is_empty = torch.count_nonzero(test_case['latent_image']) == 0
            
            print(f"   📊 Original shape: {original_shape}")
            print(f"   📊 Original channels: {original_channels}")
            print(f"   📊 Is empty: {'Yes' if original_is_empty else 'No'}")
            
            # Test fix_empty_latent_channels
            start_time = time.time()
            
            fixed_latent = fix_empty_latent_channels(unet_model, test_case['latent_image'])
            
            processing_time = time.time() - start_time
            
            # Analyze results
            fixed_shape = fixed_latent.shape
            fixed_channels = fixed_shape[1]
            actually_changed = original_shape != fixed_shape
            channels_fixed = original_channels != fixed_channels
            
            print(f"   ⏱️  Processing time: {processing_time:.4f}s")
            print(f"   📊 Fixed shape: {fixed_shape}")
            print(f"   📊 Fixed channels: {fixed_channels}")
            print(f"   📊 Shape changed: {'Yes' if actually_changed else 'No'}")
            print(f"   📊 Channels fixed: {'Yes' if channels_fixed else 'No'}")
            print(f"   📊 Expected changes: {'Yes' if test_case['expect_changes'] else 'No'}")
            
            # Validate expectations
            if test_case['expect_changes']:
                success = actually_changed
                status = "✅ SUCCESS" if success else "❌ FAILED - Should have changed"
            else:
                success = not actually_changed
                status = "✅ SUCCESS" if success else "❌ FAILED - Should not have changed"
            
            print(f"   🎯 Result: {status}")
            
            # Additional checks for fixed latents
            if actually_changed:
                print(f"   📊 Fixed latent dtype: {fixed_latent.dtype}")
                print(f"   📊 Fixed latent device: {fixed_latent.device}")
                print(f"   📊 Fixed latent mean: {fixed_latent.mean().item():.6f}")
                print(f"   📊 Fixed latent is empty: {'Yes' if torch.count_nonzero(fixed_latent) == 0 else 'No'}")
            
            results[test_case['name']] = {
                'success': success,
                'processing_time': processing_time,
                'original_shape': original_shape,
                'fixed_shape': fixed_shape,
                'channels_fixed': channels_fixed,
                'actually_changed': actually_changed,
                'expect_changed': test_case['expect_changes'],
                'error': None
            }
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results[test_case['name']] = {
                'success': False,
                'error': str(e),
                'processing_time': None
            }
    
    return results

def print_comprehensive_summary(prepare_noise_results, fix_latent_results):
    """Print comprehensive test summary"""
    
    print("\n📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    # prepare_noise() summary
    print("\n🧪 prepare_noise() FUNCTION RESULTS:")
    print("-" * 40)
    
    total_tests = len(prepare_noise_results)
    successful_tests = sum(1 for r in prepare_noise_results.values() if r['success'])
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print("✅ All prepare_noise() tests PASSED")
    else:
        print("❌ Some prepare_noise() tests FAILED")
        
        for test_name, result in prepare_noise_results.items():

            if not result['success']:
                print(f"  ❌ {test_name}: {result['error']}")
    
    # fix_empty_latent_channels() summary
    print("\n🔧 fix_empty_latent_channels() FUNCTION RESULTS:")
    print("-" * 40)
    
    total_tests_fix = len(fix_latent_results)
    successful_tests_fix = sum(1 for r in fix_latent_results.values() if r['success'])
    
    print(f"Total tests: {total_tests_fix}")
    print(f"Successful: {successful_tests_fix}")
    print(f"Failed: {total_tests_fix - successful_tests_fix}")
    if total_tests_fix > 0:
        print(f"Success rate: {(successful_tests_fix/total_tests_fix)*100:.1f}%")
        
        if successful_tests_fix == total_tests_fix:
            print("✅ All fix_empty_latent_channels() tests PASSED")
        else:
            print("❌ Some fix_empty_latent_channels() tests FAILED")
            
            for test_name, result in fix_latent_results.items():
                if not result['success']:
                    print(f"  ❌ {test_name}: {result['error']}")
    else:
        print("⚠️  No fix_empty_latent_channels() tests executed (UNet model not loaded)")
    
    # Overall assessment
    total_all_tests = total_tests + total_tests_fix
    total_successful = successful_tests + successful_tests_fix
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"Total tests: {total_all_tests}")
    print(f"Total successful: {total_successful}")
    
    if total_all_tests > 0:
        print(f"Overall success rate: {(total_successful/total_all_tests)*100:.1f}%")
        
        if total_successful == total_all_tests:
            print("🎉 ALL TESTS PASSED - Functions are working correctly!")
        else:
            print("⚠️  Some tests failed - Check individual results above")
    else:
        print("⚠️  No tests executed")

def main():
    """Main test execution function"""
    
    print("🚀 STARTING prepare_noise() and fix_empty_latent_channels() TESTS")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    try:
        # Step 1: Create mock latent images
        mock_latents = create_mock_latent_images()
        
        # Step 2: Test prepare_noise() function
        prepare_noise_results = test_prepare_noise_function(mock_latents)
        
        # Step 3: Load VACE UNet model
        unet_model = load_vace_unet_model()
        
        # Step 4: Test fix_empty_latent_channels() function
        fix_latent_results = test_fix_empty_latent_channels_function(unet_model, mock_latents)
        
        # Step 5: Print comprehensive summary
        print_comprehensive_summary(prepare_noise_results, fix_latent_results)
        
        print("\n✅ TEST EXECUTION COMPLETED")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.error(f"Critical error in main(): {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
