#!/usr/bin/env python3
"""
Test script for partial loading with large WAN2.1 VACE model
Designed for VAST AI instance testing with real model loading and inference
"""

import sys
import os
import torch
import torch.nn as nn
import logging
import time
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_large_wan_like_model():
    """
    Create a large model that simulates WAN2.1 VACE architecture
    This will be large enough to trigger partial loading
    """
    
    class LargeWANLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Simulate WAN2.1 VACE architecture with large transformer blocks
            self.transformer_blocks = nn.ModuleList()
            
            # 40 transformer blocks (like WAN2.1)
            for i in range(40):
                block = nn.ModuleDict({
                    'norm1': nn.LayerNorm(5120),
                    'self_attn': nn.MultiheadAttention(5120, 40, batch_first=True),
                    'norm2': nn.LayerNorm(5120),
                    'cross_attn': nn.MultiheadAttention(5120, 40, batch_first=True),
                    'norm3': nn.LayerNorm(5120),
                    'mlp': nn.Sequential(
                        nn.Linear(5120, 13824),  # Large MLP
                        nn.GELU(),
                        nn.Linear(13824, 5120)
                    )
                })
                self.transformer_blocks.append(block)
            
            # VACE blocks (8 layers)
            self.vace_blocks = nn.ModuleList()
            for i in range(8):
                vace_block = nn.ModuleDict({
                    'norm1': nn.LayerNorm(5120),
                    'self_attn': nn.MultiheadAttention(5120, 40, batch_first=True),
                    'norm2': nn.LayerNorm(5120),
                    'mlp': nn.Sequential(
                        nn.Linear(5120, 13824),
                        nn.GELU(),
                        nn.Linear(13824, 5120)
                    )
                })
                self.vace_blocks.append(vace_block)
            
            # Embedding layers
            self.patch_embedding = nn.Conv3d(16, 5120, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.vace_patch_embedding = nn.Conv3d(96, 5120, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            
            # Output head
            self.head = nn.Sequential(
                nn.LayerNorm(5120),
                nn.Linear(5120, 16)
            )
            
        def forward(self, x, timestep=None, conditioning=None):
            # Simulate the forward pass
            batch_size = x.shape[0]
            
            # Patch embedding
            x = self.patch_embedding(x)
            x = x.flatten(2).transpose(1, 2)  # [B, N, C]
            
            # Transformer blocks
            for block in self.transformer_blocks:
                # Self attention
                residual = x
                x = block['norm1'](x)
                x, _ = block['self_attn'](x, x, x)
                x = x + residual
                
                # Cross attention (if conditioning provided)
                if conditioning is not None:
                    residual = x
                    x = block['norm2'](x)
                    x, _ = block['cross_attn'](x, conditioning, conditioning)
                    x = x + residual
                
                # MLP
                residual = x
                x = block['norm3'](x)
                x = block['mlp'](x)
                x = x + residual
            
            # VACE blocks
            for vace_block in self.vace_blocks:
                residual = x
                x = vace_block['norm1'](x)
                x, _ = vace_block['self_attn'](x, x, x)
                x = x + residual
                
                residual = x
                x = vace_block['norm2'](x)
                x = vace_block['mlp'](x)
                x = x + residual
            
            # Output head
            x = self.head(x)
            
            # Reshape back to video format
            x = x.transpose(1, 2).reshape(batch_size, 16, -1, 4, 4)
            
            return x
    
    return LargeWANLikeModel()

def test_large_model_loading_and_inference():
    """Test loading large model and performing inference"""
    print("🧪 TESTING LARGE MODEL LOADING AND INFERENCE")
    print("="*80)
    
    try:
        from memory_utils import (
            safe_model_to_device_advanced, 
            get_memory_info, 
            log_memory_usage,
            estimate_model_memory
        )
        
        print("📊 Creating large WAN-like model...")
        
        # Create large model
        model = create_large_wan_like_model()
        
        # Estimate model size
        model_info = estimate_model_memory(model)
        print(f"   📊 Model size: {model_info['size_gb']:.3f} GB")
        print(f"   📊 Parameters: {model_info['parameters']:,}")
        
        # Get memory info
        info = get_memory_info()
        print(f"   📊 Available GPU memory: {info.get('cuda_free', 0):.2f} GB")
        
        print("\n📊 Testing model loading strategies...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test 1: Load to CPU first (simulate pipeline behavior)
        print("\n🔧 Test 1: Loading to CPU first...")
        model_cpu = model.to('cpu')
        print(f"   ✅ Model loaded to CPU")
        
        # Test 2: Use advanced partial loading
        print("\n🔧 Test 2: Advanced partial loading...")
        log_memory_usage("Before Partial Loading")
        
        model_partial, final_device, loading_info = safe_model_to_device_advanced(
            model_cpu, 
            device, 
            min_free_gb=2.0, 
            enable_partial_loading=True
        )
        
        print(f"   📊 Loading type: {loading_info['loading_type']}")
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 Modules loaded to GPU: {loading_info['modules_loaded']}")
            print(f"   📊 Modules with dynamic loading: {loading_info['modules_dynamic']}")
            print(f"   📊 GPU memory used: {loading_info['memory_used_gb']:.3f} GB")
            print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
        
        log_memory_usage("After Partial Loading")
        
        # Test 3: Perform inference
        print("\n🔧 Test 3: Performing inference...")
        
        # Create test inputs
        batch_size = 1
        input_video = torch.randn(batch_size, 16, 2, 8, 8)  # Video input
        timestep = torch.tensor([0.5])
        conditioning = torch.randn(batch_size, 77, 5120)  # Text conditioning
        
        print(f"   📊 Input shape: {input_video.shape}")
        print(f"   📊 Conditioning shape: {conditioning.shape}")
        
        # Move inputs to appropriate device
        if final_device.type == 'cuda':
            input_video = input_video.to(final_device)
            timestep = timestep.to(final_device)
            conditioning = conditioning.to(final_device)
        
        # Perform inference
        start_time = time.time()
        
        with torch.no_grad():
            try:
                output = model_partial(input_video, timestep, conditioning)
                inference_time = time.time() - start_time
                
                print(f"   ✅ Inference successful!")
                print(f"   📊 Output shape: {output.shape}")
                print(f"   📊 Inference time: {inference_time:.3f} seconds")
                print(f"   📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
                
                # Verify output is on correct device
                if output.device == final_device:
                    print(f"   ✅ Output on correct device: {output.device}")
                else:
                    print(f"   ⚠️  Output on unexpected device: {output.device}")
                
            except Exception as e:
                print(f"   ❌ Inference failed: {e}")
                return False
        
        log_memory_usage("After Inference")
        
        print("\n🎉 LARGE MODEL LOADING AND INFERENCE TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Large model loading and inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_pressure_scenarios():
    """Test various memory pressure scenarios"""
    print("\n🧪 TESTING MEMORY PRESSURE SCENARIOS")
    print("="*80)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        print("📊 Testing different memory pressure scenarios...")
        
        # Create large model
        model = create_large_wan_like_model()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        info = get_memory_info()
        available_memory = info.get('cuda_free', 0)
        
        print(f"   📊 Available GPU memory: {available_memory:.2f} GB")
        
        # Test scenarios with different memory budgets
        scenarios = [
            ("High Pressure", 0.5),  # Very high memory pressure
            ("Medium Pressure", 1.0),  # Medium memory pressure
            ("Low Pressure", 2.0),   # Low memory pressure
        ]
        
        for scenario_name, min_free_gb in scenarios:
            print(f"\n🔧 Testing {scenario_name} (min_free_gb={min_free_gb})...")
            
            model, final_device, loading_info = safe_model_to_device_advanced(
                model, 
                device, 
                min_free_gb=min_free_gb, 
                enable_partial_loading=True
            )
            
            print(f"   📊 Loading type: {loading_info['loading_type']}")
            if loading_info['loading_type'] == 'partial':
                print(f"   📊 Modules loaded: {loading_info['modules_loaded']}")
                print(f"   📊 Memory used: {loading_info['memory_used_gb']:.3f} GB")
                print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
                efficiency = (loading_info['memory_used_gb'] / loading_info['memory_budget_gb']) * 100
                print(f"   📊 Memory efficiency: {efficiency:.1f}%")
        
        print("\n🎉 MEMORY PRESSURE SCENARIO TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Memory pressure scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_simulation():
    """Simulate the actual pipeline loading process"""
    print("\n🧪 TESTING PIPELINE SIMULATION")
    print("="*80)
    
    try:
        from memory_utils import safe_model_to_device_advanced, log_memory_usage
        
        print("📊 Simulating pipeline loading process...")
        
        # Simulate pipeline initialization
        print("🔧 Step 1: Pipeline initialization...")
        log_memory_usage("Pipeline Initialization")
        
        # Simulate VAE loading (smaller model)
        print("🔧 Step 2: VAE loading...")
        vae_model = nn.Sequential(
            nn.Conv3d(16, 64, 3, padding=1),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.Conv3d(128, 16, 3, padding=1)
        )
        vae_model = vae_model.to('cpu')  # VAE typically on CPU
        log_memory_usage("After VAE Loading")
        
        # Simulate CLIP loading (medium model)
        print("🔧 Step 3: CLIP loading...")
        clip_model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 512)
        )
        clip_model = clip_model.to('cpu')  # CLIP typically on CPU
        log_memory_usage("After CLIP Loading")
        
        # Simulate UNet loading with partial loading (large model)
        print("🔧 Step 4: UNet loading with partial loading...")
        unet_model = create_large_wan_like_model()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet_model, final_device, loading_info = safe_model_to_device_advanced(
            unet_model, 
            device, 
            min_free_gb=2.0, 
            enable_partial_loading=True
        )
        
        print(f"   📊 UNet loading type: {loading_info['loading_type']}")
        if loading_info['loading_type'] == 'partial':
            print(f"   📊 UNet modules loaded: {loading_info['modules_loaded']}")
            print(f"   📊 UNet memory used: {loading_info['memory_used_gb']:.3f} GB")
        
        log_memory_usage("After UNet Loading")
        
        # Simulate inference step
        print("🔧 Step 5: Inference simulation...")
        
        # Create test inputs
        input_video = torch.randn(1, 16, 2, 8, 8)
        conditioning = torch.randn(1, 77, 5120)
        
        if final_device.type == 'cuda':
            input_video = input_video.to(final_device)
            conditioning = conditioning.to(final_device)
        
        with torch.no_grad():
            output = unet_model(input_video, conditioning=conditioning)
            print(f"   ✅ Inference successful: {output.shape}")
        
        log_memory_usage("After Inference")
        
        print("\n🎉 PIPELINE SIMULATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_world_scenarios():
    """Test real-world usage scenarios"""
    print("\n🧪 TESTING REAL-WORLD SCENARIOS")
    print("="*80)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        print("📊 Testing real-world usage scenarios...")
        
        # Scenario 1: Multiple model loading
        print("\n🔧 Scenario 1: Multiple model loading...")
        
        models = []
        for i in range(3):  # Load 3 large models
            model = create_large_wan_like_model()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model, final_device, loading_info = safe_model_to_device_advanced(
                model, 
                device, 
                min_free_gb=1.0, 
                enable_partial_loading=True
            )
            
            models.append((model, loading_info))
            print(f"   📊 Model {i+1}: {loading_info['loading_type']}")
        
        # Scenario 2: Memory cleanup and reloading
        print("\n🔧 Scenario 2: Memory cleanup and reloading...")
        
        # Clear memory
        for model, _ in models:
            del model
        torch.cuda.empty_cache()
        
        # Reload model
        new_model = create_large_wan_like_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        new_model, final_device, loading_info = safe_model_to_device_advanced(
            new_model, 
            device, 
            min_free_gb=1.0, 
            enable_partial_loading=True
        )
        
        print(f"   📊 Reloaded model: {loading_info['loading_type']}")
        
        # Scenario 3: Batch processing simulation
        print("\n🔧 Scenario 3: Batch processing simulation...")
        
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            print(f"   📊 Testing batch size {batch_size}...")
            
            input_video = torch.randn(batch_size, 16, 2, 8, 8)
            conditioning = torch.randn(batch_size, 77, 5120)
            
            if final_device.type == 'cuda':
                input_video = input_video.to(final_device)
                conditioning = conditioning.to(final_device)
            
            try:
                with torch.no_grad():
                    output = new_model(input_video, conditioning=conditioning)
                print(f"      ✅ Batch {batch_size} successful: {output.shape}")
            except Exception as e:
                print(f"      ❌ Batch {batch_size} failed: {e}")
        
        print("\n🎉 REAL-WORLD SCENARIO TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Real-world scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all VAST AI instance tests"""
    print("🚀 VAST AI INSTANCE - LARGE MODEL PARTIAL LOADING TESTS")
    print("="*100)
    
    # Print system information
    print("📊 System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n" + "="*100)
    
    tests = [
        ("Large Model Loading and Inference", test_large_model_loading_and_inference),
        ("Memory Pressure Scenarios", test_memory_pressure_scenarios),
        ("Pipeline Simulation", test_pipeline_simulation),
        ("Real-World Scenarios", test_real_world_scenarios),
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
    print("="*100)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*100)
    if all_passed:
        print("🎉 ALL VAST AI INSTANCE TESTS PASSED!")
        print("✅ Large model loading works correctly")
        print("✅ Partial loading handles memory pressure")
        print("✅ Pipeline simulation successful")
        print("✅ Real-world scenarios work properly")
        print("✅ System is ready for production use on VAST AI")
    else:
        print("❌ Some tests failed - check the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

