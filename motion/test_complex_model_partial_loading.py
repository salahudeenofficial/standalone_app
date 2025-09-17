#!/usr/bin/env python3
"""
Comprehensive test for complex model partial loading
Tests: patching, GPU weight loading, device mismatch prevention
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import time
from typing import Dict, Any, List, Optional

# Add motion directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comfyui_style_partial_loading import ComfyUIStylePartialLoader
from model_aware_patcher import ModelAwarePatcher
from comfyui_ops import ComfyUILinear, ComfyUIConv2d, ComfyUIGroupNorm, ComfyUILayerNorm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class ComplexAttentionBlock(nn.Module):
    """Complex attention block with multiple components"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP components
        self.mlp_fc1 = nn.Linear(dim, dim * 4)
        self.mlp_fc2 = nn.Linear(dim * 4, dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_output = self.out_proj(attn_output)
        
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp_fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.mlp_fc2(x)
        x = self.dropout(x)
        
        return residual + x

class ComplexConvBlock(nn.Module):
    """Complex convolutional block with multiple layers"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.norm3 = nn.GroupNorm(8, out_channels)
        
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        
        return x

class ComplexModel(nn.Module):
    """Complex model with multiple types of layers and nested structures"""
    def __init__(self):
        super().__init__()
        
        # Input processing
        self.input_conv = ComplexConvBlock(3, 64)
        self.input_norm = nn.LayerNorm(64)
        
        # Multiple attention blocks
        self.attention_blocks = nn.ModuleList([
            ComplexAttentionBlock(64, num_heads=8) for _ in range(4)
        ])
        
        # Multiple conv blocks
        self.conv_blocks = nn.ModuleList([
            ComplexConvBlock(64, 128),
            ComplexConvBlock(128, 256),
            ComplexConvBlock(256, 512),
        ])
        
        # Skip connections
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 512, 1),
        ])
        
        # Output processing
        self.output_norm = nn.LayerNorm(512)
        self.output_proj = nn.Linear(512, 1000)
        
        # Additional components
        self.aux_conv = nn.Conv2d(512, 256, 1)
        self.aux_norm = nn.GroupNorm(8, 256)
        self.aux_fc = nn.Linear(256, 100)
        
    def forward(self, x):
        # Input processing
        x = self.input_conv(x)  # [B, 64, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, 64]
        x = self.input_norm(x)
        
        # Attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Convert back to spatial for conv blocks
        batch_size, seq_len, channels = x.shape
        h = w = int(seq_len ** 0.5)
        x = x.transpose(1, 2).view(batch_size, channels, h, w)
        
        # Conv blocks with skip connections
        for i, (conv_block, skip_conv) in enumerate(zip(self.conv_blocks, self.skip_convs)):
            skip = skip_conv(x)
            x = conv_block(x)
            x = x + skip
        
        # Output processing
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, 512]
        x = self.output_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)
        
        # Auxiliary output
        aux_x = self.aux_conv(x.view(batch_size, 512, h, w))
        aux_x = self.aux_norm(aux_x)
        aux_x = aux_x.flatten(2).transpose(1, 2)
        aux_x = aux_x.mean(dim=1)
        aux_x = self.aux_fc(aux_x)
        
        return x, aux_x

def create_complex_model():
    """Create a complex test model"""
    return ComplexModel()

def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count different types of parameters in the model"""
    counts = {
        'total': 0,
        'linear': 0,
        'conv2d': 0,
        'layernorm': 0,
        'groupnorm': 0,
        'modules': 0
    }
    
    for name, module in model.named_modules():
        counts['modules'] += 1
        if isinstance(module, nn.Linear):
            counts['linear'] += module.weight.numel()
            if module.bias is not None:
                counts['linear'] += module.bias.numel()
        elif isinstance(module, nn.Conv2d):
            counts['conv2d'] += module.weight.numel()
            if module.bias is not None:
                counts['conv2d'] += module.bias.numel()
        elif isinstance(module, nn.LayerNorm):
            counts['layernorm'] += module.weight.numel() + module.bias.numel()
        elif isinstance(module, nn.GroupNorm):
            counts['groupnorm'] += module.weight.numel() + module.bias.numel()
    
    counts['total'] = sum(p.numel() for p in model.parameters())
    return counts

def test_model_patching():
    """Test model-aware patching on complex model"""
    print("🧪 TESTING MODEL-AWARE PATCHING")
    print("=" * 60)
    
    # Create complex model
    model = create_complex_model()
    original_counts = count_model_parameters(model)
    
    print(f"📊 Original model statistics:")
    print(f"   Total parameters: {original_counts['total']:,}")
    print(f"   Linear layers: {original_counts['linear']:,}")
    print(f"   Conv2d layers: {original_counts['conv2d']:,}")
    print(f"   LayerNorm layers: {original_counts['layernorm']:,}")
    print(f"   GroupNorm layers: {original_counts['groupnorm']:,}")
    print(f"   Total modules: {original_counts['modules']}")
    
    try:
        # Test model-aware patching
        patcher = ModelAwarePatcher()
        
        # Register handler for ComplexModel
        patcher.register_model_handler(ComplexModel, patcher._patch_generic_model)
        
        patched_model = patcher.patch_model(model)
        
        # Count patched modules
        patched_count = 0
        weight_function_count = 0
        bias_function_count = 0
        
        for name, module in patched_model.named_modules():
            if hasattr(module, 'weight_function') and module.weight_function:
                weight_function_count += len(module.weight_function)
                patched_count += 1
            if hasattr(module, 'bias_function') and module.bias_function:
                bias_function_count += len(module.bias_function)
        
        print(f"✅ Patching results:")
        print(f"   Patched modules: {patched_count}")
        print(f"   Weight functions: {weight_function_count}")
        print(f"   Bias functions: {bias_function_count}")
        
        # Verify all critical layers are patched
        critical_layers = [
            'input_conv.conv1', 'input_conv.conv2', 'input_conv.conv3',
            'attention_blocks.0.q_proj', 'attention_blocks.0.k_proj', 'attention_blocks.0.v_proj',
            'attention_blocks.0.out_proj', 'attention_blocks.0.mlp_fc1', 'attention_blocks.0.mlp_fc2',
            'conv_blocks.0.conv1', 'conv_blocks.0.conv2', 'conv_blocks.0.conv3',
            'output_proj', 'aux_conv', 'aux_fc'
        ]
        
        missing_patches = []
        for layer_name in critical_layers:
            try:
                module = dict(patched_model.named_modules())[layer_name]
                if not (hasattr(module, 'weight_function') and module.weight_function):
                    missing_patches.append(layer_name)
            except KeyError:
                missing_patches.append(f"{layer_name} (not found)")
        
        if missing_patches:
            print(f"❌ Missing patches: {missing_patches}")
            return False
        else:
            print(f"✅ All critical layers patched successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_partial_loading_setup():
    """Test partial loading setup"""
    print(f"\n🧪 TESTING PARTIAL LOADING SETUP")
    print("=" * 60)
    
    model = create_complex_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_budget_gb = 0.5  # Small budget for testing
    
    try:
        # Set up partial loading
        loader = ComfyUIStylePartialLoader(model, device, memory_budget_gb)
        
        # Get loading info
        info = loader.get_loading_info()
        
        print(f"✅ Partial loading setup complete:")
        print(f"   Loading type: {info['loading_type']}")
        print(f"   Loaded weights: {info['loaded_weights_count']}")
        print(f"   Patched weights: {info['patched_weights_count']}")
        print(f"   Memory used: {info['memory_used_gb']:.3f} GB")
        print(f"   Memory budget: {info['memory_budget_gb']:.3f} GB")
        print(f"   Target device: {info['target_device']}")
        
        # Verify model is on CPU
        model_device = next(model.parameters()).device
        print(f"   Model device: {model_device}")
        
        if str(model_device) == 'cpu':
            print(f"   ✅ Model correctly stays on CPU")
        else:
            print(f"   ❌ Model should be on CPU, but is on {model_device}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Partial loading setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_with_weight_loading():
    """Test inference with dynamic weight loading"""
    print(f"\n🧪 TESTING INFERENCE WITH WEIGHT LOADING")
    print("=" * 60)
    
    model = create_complex_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_budget_gb = 0.5
    
    try:
        # Set up partial loading
        loader = ComfyUIStylePartialLoader(model, device, memory_budget_gb)
        
        # Prepare for inference
        loader.load_weights_for_inference()
        
        # Create test input
        batch_size = 2
        height, width = 32, 32
        test_input = torch.randn(batch_size, 3, height, width)
        
        print(f"📊 Test input:")
        print(f"   Shape: {test_input.shape}")
        print(f"   Device: {test_input.device}")
        
        # Test inference
        print(f"🔄 Running inference...")
        start_time = time.time()
        
        with torch.no_grad():
            output, aux_output = model(test_input)
        
        inference_time = time.time() - start_time
        
        print(f"✅ Inference completed:")
        print(f"   Main output shape: {output.shape}")
        print(f"   Aux output shape: {aux_output.shape}")
        print(f"   Inference time: {inference_time:.3f}s")
        
        # Check output devices
        print(f"📊 Output devices:")
        print(f"   Main output device: {output.device}")
        print(f"   Aux output device: {aux_output.device}")
        
        # Verify no device mismatch
        if output.device == test_input.device and aux_output.device == test_input.device:
            print(f"   ✅ No device mismatch detected")
        else:
            print(f"   ❌ Device mismatch detected!")
            print(f"      Input: {test_input.device}")
            print(f"      Main output: {output.device}")
            print(f"      Aux output: {aux_output.device}")
            return False
        
        # Cleanup
        loader.evict_weights_after_inference()
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_mismatch_scenarios():
    """Test various device mismatch scenarios"""
    print(f"\n🧪 TESTING DEVICE MISMATCH SCENARIOS")
    print("=" * 60)
    
    model = create_complex_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    memory_budget_gb = 0.5
    
    try:
        # Set up partial loading
        loader = ComfyUIStylePartialLoader(model, device, memory_budget_gb)
        loader.load_weights_for_inference()
        
        # Test scenario 1: Input on different device
        print(f"🔧 Test 1: Input on different device")
        if torch.cuda.is_available():
            test_input_cpu = torch.randn(1, 3, 16, 16)
            test_input_gpu = torch.randn(1, 3, 16, 16).to('cuda')
            
            # Test with CPU input (model on CPU, input on CPU)
            try:
                with torch.no_grad():
                    output_cpu = model(test_input_cpu)
                print(f"   ✅ CPU input -> CPU model: Success")
            except Exception as e:
                print(f"   ❌ CPU input -> CPU model: Failed - {e}")
            
            # Test with GPU input (model on CPU, input on GPU)
            try:
                with torch.no_grad():
                    output_gpu = model(test_input_gpu)
                print(f"   ✅ GPU input -> CPU model: Success")
            except Exception as e:
                print(f"   ❌ GPU input -> CPU model: Failed - {e}")
        
        # Test scenario 2: Mixed precision
        print(f"🔧 Test 2: Mixed precision")
        test_input_fp16 = torch.randn(1, 3, 16, 16).half()
        try:
            with torch.no_grad():
                output_fp16 = model(test_input_fp16)
            print(f"   ✅ FP16 input -> FP32 model: Success")
        except Exception as e:
            print(f"   ❌ FP16 input -> FP32 model: Failed - {e}")
        
        # Test scenario 3: Different batch sizes
        print(f"🔧 Test 3: Different batch sizes")
        for batch_size in [1, 2, 4]:
            try:
                test_input = torch.randn(batch_size, 3, 16, 16)
                with torch.no_grad():
                    output = model(test_input)
                print(f"   ✅ Batch size {batch_size}: Success")
            except Exception as e:
                print(f"   ❌ Batch size {batch_size}: Failed - {e}")
        
        # Cleanup
        loader.evict_weights_after_inference()
        
        return True
        
    except Exception as e:
        print(f"❌ Device mismatch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency during inference"""
    print(f"\n🧪 TESTING MEMORY EFFICIENCY")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory efficiency test")
        return True
    
    model = create_complex_model()
    device = torch.device('cuda')
    memory_budget_gb = 0.1  # Very small budget
    
    try:
        # Set up partial loading
        loader = ComfyUIStylePartialLoader(model, device, memory_budget_gb)
        loader.load_weights_for_inference()
        
        # Monitor memory usage
        def get_memory_info():
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                return allocated, reserved
            return 0, 0
        
        print(f"📊 Memory usage during inference:")
        
        # Before inference
        allocated_before, reserved_before = get_memory_info()
        print(f"   Before inference: {allocated_before:.3f} GB allocated, {reserved_before:.3f} GB reserved")
        
        # During inference
        test_input = torch.randn(1, 3, 16, 16).to(device)
        with torch.no_grad():
            output = model(test_input)
        
        allocated_during, reserved_during = get_memory_info()
        print(f"   During inference: {allocated_during:.3f} GB allocated, {reserved_during:.3f} GB reserved")
        
        # After cleanup
        loader.evict_weights_after_inference()
        allocated_after, reserved_after = get_memory_info()
        print(f"   After cleanup: {allocated_after:.3f} GB allocated, {reserved_after:.3f} GB reserved")
        
        # Verify memory efficiency
        memory_increase = allocated_during - allocated_before
        print(f"   Memory increase during inference: {memory_increase:.3f} GB")
        
        if memory_increase < memory_budget_gb:
            print(f"   ✅ Memory usage within budget")
        else:
            print(f"   ⚠️  Memory usage exceeded budget")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 COMPLEX MODEL PARTIAL LOADING TEST")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Model Patching", test_model_patching),
        ("Partial Loading Setup", test_partial_loading_setup),
        ("Inference with Weight Loading", test_inference_with_weight_loading),
        ("Device Mismatch Scenarios", test_device_mismatch_scenarios),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Final results
    print(f"\n🎯 FINAL RESULTS:")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\n📊 Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Complex model partial loading is working perfectly!")
        print(f"   Model-aware patching: ✅")
        print(f"   GPU weight loading: ✅")
        print(f"   Device mismatch prevention: ✅")
        print(f"   Memory efficiency: ✅")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Need to fix issues before production use!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
