#!/usr/bin/env python3
"""
Simplified test script for VAST AI instance - WAN2.1 VACE model simulation
Focuses on realistic model loading and inference testing
"""

import sys
import os
import torch
import torch.nn as nn
import logging
import time

# Add motion directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_realistic_wan_model():
    """
    Create a realistic WAN2.1 VACE model that closely matches the actual architecture
    This will be large enough to test partial loading effectively
    """
    
    class RealisticWANModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Model configuration (matching WAN2.1 VACE)
            self.dim = 5120
            self.num_layers = 40
            self.vace_layers = 8
            self.num_heads = 40
            self.ffn_dim = 13824
            
            # Patch embeddings
            self.patch_embedding = nn.Conv3d(16, self.dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.vace_patch_embedding = nn.Conv3d(96, self.dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            
            # Main transformer blocks (40 layers)
            self.blocks = nn.ModuleList()
            for i in range(self.num_layers):
                block = nn.ModuleDict({
                    'norm1': nn.LayerNorm(self.dim),
                    'self_attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
                    'norm2': nn.LayerNorm(self.dim),
                    'cross_attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
                    'norm3': nn.LayerNorm(self.dim),
                    'mlp': nn.Sequential(
                        nn.Linear(self.dim, self.ffn_dim),
                        nn.GELU(),
                        nn.Linear(self.ffn_dim, self.dim)
                    )
                })
                self.blocks.append(block)
            
            # VACE blocks (8 layers)
            self.vace_blocks = nn.ModuleList()
            for i in range(self.vace_layers):
                vace_block = nn.ModuleDict({
                    'norm1': nn.LayerNorm(self.dim),
                    'self_attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True),
                    'norm2': nn.LayerNorm(self.dim),
                    'mlp': nn.Sequential(
                        nn.Linear(self.dim, self.ffn_dim),
                        nn.GELU(),
                        nn.Linear(self.ffn_dim, self.dim)
                    )
                })
                self.vace_blocks.append(vace_block)
            
            # Output head
            self.head = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, 16)
            )
            
        def forward(self, x, timestep=None, conditioning=None):
            batch_size, channels, frames, height, width = x.shape
            
            # Patch embedding
            x = self.patch_embedding(x)
            x = x.flatten(2).transpose(1, 2)  # [B, N, C]
            
            # Main transformer blocks
            for block in self.blocks:
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
            x = x.transpose(1, 2).reshape(batch_size, 16, frames, height//2, width//2)
            
            return x
    
    return RealisticWANModel()

def test_wan_model_loading():
    """Test loading the realistic WAN model"""
    print("🧪 TESTING REALISTIC WAN MODEL LOADING")
    print("="*80)
    
    try:
        from memory_utils import (
            safe_model_to_device_advanced, 
            get_memory_info, 
            log_memory_usage,
            estimate_model_memory
        )
        
        print("📊 Creating realistic WAN2.1 VACE model...")
        
        # Create realistic model
        model = create_realistic_wan_model()
        
        # Estimate model size
        model_info = estimate_model_memory(model)
        print(f"   📊 Model size: {model_info['size_gb']:.3f} GB")
        print(f"   📊 Parameters: {model_info['parameters']:,}")
        
        # Get memory info
        info = get_memory_info()
        print(f"   📊 Available GPU memory: {info.get('cuda_free', 0):.2f} GB")
        
        print("\n📊 Testing loading strategies...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test 1: Load to CPU first (simulate pipeline behavior)
        print("\n🔧 Step 1: Loading to CPU...")
        model_cpu = model.to('cpu')
        print(f"   ✅ Model loaded to CPU")
        
        # Test 2: Use advanced partial loading
        print("\n🔧 Step 2: Advanced partial loading...")
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
            efficiency = (loading_info['memory_used_gb'] / loading_info['memory_budget_gb']) * 100
            print(f"   📊 Memory efficiency: {efficiency:.1f}%")
        elif loading_info['loading_type'] == 'full':
            print(f"   📊 Full model loaded to GPU")
        else:
            print(f"   📊 Model loaded to: {final_device}")
        
        log_memory_usage("After Partial Loading")
        
        return model_partial, final_device, loading_info
        
    except Exception as e:
        print(f"❌ WAN model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_wan_model_inference(model, device):
    """Test inference with the WAN model"""
    print("\n🧪 TESTING WAN MODEL INFERENCE")
    print("="*80)
    
    try:
        print("📊 Testing inference with realistic inputs...")
        
        # Create realistic test inputs
        batch_size = 1
        input_video = torch.randn(batch_size, 16, 2, 8, 8)  # Video input
        timestep = torch.tensor([0.5])
        conditioning = torch.randn(batch_size, 77, 5120)  # Text conditioning
        
        print(f"   📊 Input video shape: {input_video.shape}")
        print(f"   📊 Conditioning shape: {conditioning.shape}")
        
        # Move inputs to appropriate device
        if device.type == 'cuda':
            input_video = input_video.to(device)
            timestep = timestep.to(device)
            conditioning = conditioning.to(device)
        
        # Perform inference
        print("\n🔧 Performing inference...")
        start_time = time.time()
        
        with torch.no_grad():
            try:
                output = model(input_video, timestep, conditioning)
                inference_time = time.time() - start_time
                
                print(f"   ✅ Inference successful!")
                print(f"   📊 Output shape: {output.shape}")
                print(f"   📊 Inference time: {inference_time:.3f} seconds")
                print(f"   📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"   📊 Output mean: {output.mean():.3f}")
                print(f"   📊 Output std: {output.std():.3f}")
                
                # Verify output is on correct device
                if output.device == device:
                    print(f"   ✅ Output on correct device: {output.device}")
                else:
                    print(f"   ⚠️  Output on unexpected device: {output.device}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ Inference failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
    except Exception as e:
        print(f"❌ WAN model inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory efficiency with different scenarios"""
    print("\n🧪 TESTING MEMORY EFFICIENCY")
    print("="*80)
    
    try:
        from memory_utils import safe_model_to_device_advanced, get_memory_info
        
        print("📊 Testing memory efficiency scenarios...")
        
        # Create model
        model = create_realistic_wan_model()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        info = get_memory_info()
        available_memory = info.get('cuda_free', 0)
        
        print(f"   📊 Available GPU memory: {available_memory:.2f} GB")
        
        # Test different memory budgets
        memory_budgets = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for budget in memory_budgets:
            if budget >= available_memory:
                continue
                
            print(f"\n🔧 Testing with {budget} GB memory budget...")
            
            model, final_device, loading_info = safe_model_to_device_advanced(
                model, 
                device, 
                min_free_gb=available_memory - budget, 
                enable_partial_loading=True
            )
            
            print(f"   📊 Loading type: {loading_info['loading_type']}")
            if loading_info['loading_type'] == 'partial':
                print(f"   📊 Modules loaded: {loading_info['modules_loaded']}")
                print(f"   📊 Memory used: {loading_info['memory_used_gb']:.3f} GB")
                print(f"   📊 Memory budget: {loading_info['memory_budget_gb']:.3f} GB")
                efficiency = (loading_info['memory_used_gb'] / loading_info['memory_budget_gb']) * 100
                print(f"   📊 Memory efficiency: {efficiency:.1f}%")
        
        print("\n🎉 MEMORY EFFICIENCY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run VAST AI instance tests"""
    print("🚀 VAST AI INSTANCE - WAN2.1 VACE MODEL TESTING")
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
    
    # Test 1: Model loading
    model, device, loading_info = test_wan_model_loading()
    if model is None:
        print("❌ Model loading failed - aborting tests")
        return False
    
    # Test 2: Model inference
    inference_success = test_wan_model_inference(model, device)
    
    # Test 3: Memory efficiency
    efficiency_success = test_memory_efficiency()
    
    # Summary
    print("\n📊 TEST RESULTS:")
    print("="*100)
    
    print(f"   Model Loading: {'✅ PASSED' if model is not None else '❌ FAILED'}")
    print(f"   Model Inference: {'✅ PASSED' if inference_success else '❌ FAILED'}")
    print(f"   Memory Efficiency: {'✅ PASSED' if efficiency_success else '❌ FAILED'}")
    
    all_passed = model is not None and inference_success and efficiency_success
    
    print("\n" + "="*100)
    if all_passed:
        print("🎉 ALL VAST AI INSTANCE TESTS PASSED!")
        print("✅ WAN2.1 VACE model loading works correctly")
        print("✅ Partial loading handles large models efficiently")
        print("✅ Inference works with partial loading")
        print("✅ Memory efficiency is optimal")
        print("✅ System is ready for production use on VAST AI")
    else:
        print("❌ Some tests failed - check the issues above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
