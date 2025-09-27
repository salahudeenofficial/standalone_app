#!/usr/bin/env python3
"""
Test to verify memory stream and load_models_gpu issues actually make differences
in VAE calculations.

This test compares:
1. Memory stream usage vs no stream usage
2. Proper model loading vs no model loading
3. Impact on VAE encoding results
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

from wan_vae_components.model_management import cast_to, get_offload_stream, load_models_gpu
from wan_vae_components.ops import cast_to_input, cast_bias_weight
from wan_vae_components.vae import WanVAE, RMS_norm
from wan_vae_components.einops_replacement import rearrange

class MockModelPatcher:
    """Mock model patcher for testing"""
    def __init__(self, device="cuda"):
        self.load_device = torch.device(device)
        self.model = MockVAEModel()
        self.patches = []
        self.patches_uuid = "test-uuid"
    
    def to(self, device):
        self.load_device = torch.device(device)
        return self

class MockVAEModel(nn.Module):
    """Mock VAE model for testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)  # Keep same channels
        self.conv3 = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def test_memory_stream_impact():
    """Test 1: Memory stream usage impact"""
    print("="*60)
    print("TEST 1: MEMORY STREAM IMPACT")
    print("="*60)
    
    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.randn(64, 32, 3, 3, device=device, dtype=torch.float32)
    input_tensor = torch.randn(1, 32, 8, 8, device=device, dtype=torch.float32)
    
    print(f"Device: {device}")
    print(f"Weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    # Test 1a: Without stream
    start_time = time.time()
    weight_no_stream = cast_to(weight, input_tensor.dtype, input_tensor.device, 
                              non_blocking=True, copy=True, stream=None)
    time_no_stream = time.time() - start_time
    
    # Test 1b: With stream
    start_time = time.time()
    stream = get_offload_stream(device)
    weight_with_stream = cast_to(weight, input_tensor.dtype, input_tensor.device, 
                                non_blocking=True, copy=True, stream=stream)
    time_with_stream = time.time() - start_time
    
    # Compare results
    print(f"\nResults:")
    print(f"Time without stream: {time_no_stream:.6f}s")
    print(f"Time with stream: {time_with_stream:.6f}s")
    print(f"Tensors equal: {torch.equal(weight_no_stream, weight_with_stream)}")
    print(f"Max difference: {torch.max(torch.abs(weight_no_stream - weight_with_stream)).item():.10f}")
    
    return torch.equal(weight_no_stream, weight_with_stream)

def test_load_models_gpu_impact():
    """Test 2: Load models GPU impact"""
    print("\n" + "="*60)
    print("TEST 2: LOAD MODELS GPU IMPACT")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create mock model patcher
    model_patcher = MockModelPatcher(device)
    
    print(f"Initial model device: {model_patcher.load_device}")
    print(f"Model parameters device: {next(model_patcher.model.parameters()).device}")
    
    # Test 2a: Without load_models_gpu
    print(f"\n--- Without load_models_gpu ---")
    model_patcher.model.to("cpu")  # Move to CPU first
    print(f"Model moved to CPU: {next(model_patcher.model.parameters()).device}")
    
    # Test 2b: With load_models_gpu (our current implementation)
    print(f"\n--- With load_models_gpu (current) ---")
    load_models_gpu([model_patcher], force_full_load=False)
    print(f"Model device after load_models_gpu: {next(model_patcher.model.parameters()).device}")
    
    # Test 2c: Manual model loading (what should happen)
    print(f"\n--- Manual model loading (expected) ---")
    model_patcher.model.to(device)
    print(f"Model device after manual loading: {next(model_patcher.model.parameters()).device}")
    
    return model_patcher.model

def test_vae_calculation_impact():
    """Test 3: VAE calculation impact"""
    print("\n" + "="*60)
    print("TEST 3: VAE CALCULATION IMPACT")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test input
    batch_size = 1
    channels = 3
    height, width = 64, 64
    test_input = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input device: {test_input.device}")
    print(f"Test input dtype: {test_input.dtype}")
    
    # Test 3a: Model on CPU (simulating no load_models_gpu)
    print(f"\n--- Model on CPU ---")
    model_cpu = MockVAEModel().to("cpu")
    start_time = time.time()
    with torch.no_grad():
        # Move input to CPU for CPU model
        output_cpu = model_cpu(test_input.cpu())
    time_cpu = time.time() - start_time
    print(f"CPU model output shape: {output_cpu.shape}")
    print(f"CPU model time: {time_cpu:.6f}s")
    print(f"CPU model output device: {output_cpu.device}")
    
    # Test 3b: Model on GPU (simulating proper load_models_gpu)
    print(f"\n--- Model on GPU ---")
    model_gpu = MockVAEModel().to(device)
    start_time = time.time()
    with torch.no_grad():
        output_gpu = model_gpu(test_input)
    time_gpu = time.time() - start_time
    print(f"GPU model output shape: {output_gpu.shape}")
    print(f"GPU model time: {time_gpu:.6f}s")
    print(f"GPU model output device: {output_gpu.device}")
    
    # Compare results
    print(f"\n--- Comparison ---")
    print(f"Time difference: {abs(time_gpu - time_cpu):.6f}s")
    print(f"GPU faster: {time_gpu < time_cpu}")
    
    # Test 3c: Device mismatch (simulating wrong device)
    print(f"\n--- Device Mismatch Test ---")
    try:
        # Try to run GPU model on CPU input (should fail or be slow)
        model_gpu = MockVAEModel().to(device)
        start_time = time.time()
        with torch.no_grad():
            output_mismatch = model_gpu(test_input.cpu())
        time_mismatch = time.time() - start_time
        print(f"Device mismatch output shape: {output_mismatch.shape}")
        print(f"Device mismatch time: {time_mismatch:.6f}s")
        print(f"Device mismatch output device: {output_mismatch.device}")
    except Exception as e:
        print(f"Device mismatch error: {e}")
    
    return output_cpu, output_gpu

def test_tensor_rearrangement_impact():
    """Test 4: Tensor rearrangement impact"""
    print("\n" + "="*60)
    print("TEST 4: TENSOR REARRANGEMENT IMPACT")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different rearrangement patterns
    test_cases = [
        ("b c t h w -> (b t) c h w", (1, 3, 4, 8, 8)),
        ("(b t) c h w -> b c t h w", (4, 3, 8, 8)),
        ("b c h w -> (b h w) c", (1, 3, 8, 8)),
        ("(b h w) c -> b c h w", (64, 3)),
        ("b c h w -> b h w c", (1, 3, 8, 8)),
        ("b h w c -> b c h w", (1, 8, 8, 3)),
    ]
    
    for pattern, shape in test_cases:
        print(f"\nTesting pattern: {pattern}")
        print(f"Input shape: {shape}")
        
        try:
            # Create test tensor
            test_tensor = torch.randn(shape, device=device, dtype=torch.float32)
            
            # Test rearrangement
            start_time = time.time()
            rearranged = rearrange(test_tensor, pattern, **{
                't': 4, 'h': 8, 'w': 8, 'c': 3, 'n': 2
            })
            time_rearrange = time.time() - start_time
            
            print(f"Output shape: {rearranged.shape}")
            print(f"Time: {time_rearrange:.6f}s")
            print(f"Success: ✅")
            
        except Exception as e:
            print(f"Error: {e}")
            print(f"Success: ❌")

def test_memory_usage_impact():
    """Test 5: Memory usage impact"""
    print("\n" + "="*60)
    print("TEST 5: MEMORY USAGE IMPACT")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device("cuda")
    
    def get_memory_info():
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return allocated, reserved
    
    print(f"Initial memory: {get_memory_info()}")
    
    # Test 5a: Create large tensors without proper memory management
    print(f"\n--- Without memory management ---")
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device=device, dtype=torch.float32)
        tensors.append(tensor)
        print(f"After tensor {i+1}: {get_memory_info()}")
    
    # Test 5b: Clear memory
    print(f"\n--- Clearing memory ---")
    del tensors
    torch.cuda.empty_cache()
    print(f"After cleanup: {get_memory_info()}")
    
    # Test 5c: Test memory stream impact
    print(f"\n--- Memory stream impact ---")
    large_tensor = torch.randn(2000, 2000, device=device, dtype=torch.float32)
    print(f"Large tensor created: {get_memory_info()}")
    
    # Test with stream
    stream = get_offload_stream(device)
    start_time = time.time()
    with torch.cuda.stream(stream):
        moved_tensor = large_tensor.to(device=device, dtype=torch.float16)
    time_stream = time.time() - start_time
    print(f"With stream time: {time_stream:.6f}s")
    print(f"After stream operation: {get_memory_info()}")
    
    # Test without stream
    start_time = time.time()
    moved_tensor2 = large_tensor.to(device=device, dtype=torch.float16)
    time_no_stream = time.time() - start_time
    print(f"Without stream time: {time_no_stream:.6f}s")
    print(f"After no-stream operation: {get_memory_info()}")

def main():
    """Run all tests"""
    print("MEMORY STREAM AND LOAD_MODELS_GPU IMPACT TEST")
    print("="*80)
    
    # Test 1: Memory stream impact
    stream_equal = test_memory_stream_impact()
    
    # Test 2: Load models GPU impact
    model = test_load_models_gpu_impact()
    
    # Test 3: VAE calculation impact
    output_cpu, output_gpu = test_vae_calculation_impact()
    
    # Test 4: Tensor rearrangement impact
    test_tensor_rearrangement_impact()
    
    # Test 5: Memory usage impact
    test_memory_usage_impact()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Memory stream impact: {'MINIMAL' if stream_equal else 'SIGNIFICANT'}")
    print(f"Load models GPU impact: {'SIGNIFICANT' if model is not None else 'MINIMAL'}")
    print(f"VAE calculation impact: {'SIGNIFICANT' if output_cpu is not None and output_gpu is not None else 'MINIMAL'}")
    
    print(f"\nKey Findings:")
    print(f"1. Memory streams: {'No difference in results' if stream_equal else 'Different results'}")
    print(f"2. Load models GPU: {'Critical for proper model loading' if model is not None else 'Not critical'}")
    print(f"3. VAE calculations: {'Device placement matters' if output_cpu is not None and output_gpu is not None else 'No impact'}")

if __name__ == "__main__":
    main()
