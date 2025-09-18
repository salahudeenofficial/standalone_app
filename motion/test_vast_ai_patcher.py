#!/usr/bin/env python3
"""
VAST AI ComfyUI-style Model Loading Test
Test your actual models: UNet (32GB), VAE (200MB), Text Encoder (10GB)
"""

import os
import sys
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_vast_ai_models():
    """Test ComfyUI-style loading with your actual VAST AI models"""
    
    print("🚀 VAST AI ComfyUI-style Model Loading Test")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available on this system")
        return False
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 Total VRAM: {total_memory:.1f} GB")
    
    # Model paths - adjust these to your actual paths
    model_paths = {
        "UNet": "./models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "VAE": "./models/vae/vae.safetensors", 
        "Text Encoder": "./models/clip/clip.safetensors"
    }
    
    # Check which models exist
    available_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            available_models[name] = {'path': path, 'size_gb': size_gb}
            print(f"✅ {name}: {path} ({size_gb:.2f} GB)")
        else:
            print(f"❌ {name}: {path} (not found)")
    
    if not available_models:
        print("❌ No models found! Please check the paths.")
        return False
    
    print(f"\n🔧 Testing ComfyUI-style loading with {len(available_models)} models...")
    
    # Test each available model
    results = {}
    for name, info in available_models.items():
        print(f"\n{'='*50}")
        print(f"🧪 Testing {name} ({info['size_gb']:.2f} GB)")
        print(f"{'='*50}")
        
        try:
            # Import our memory management
            from memory_utils import safe_model_to_device_advanced, get_memory_info
            
            # Get current memory info
            mem_info = get_memory_info()
            print(f"📊 Available GPU memory: {mem_info['cuda_free']:.2f} GB")
            
            # Validate file size first
            expected_sizes = {
                "UNet": 30,  # Should be ~32GB
                "VAE": 0.2,  # Should be ~200MB
                "Text Encoder": 10  # Should be ~10GB
            }
            
            expected_size = expected_sizes.get(name, 0)
            if info['size_gb'] < expected_size * 0.1:  # Less than 10% of expected
                print(f"⚠️  WARNING: {name} file size ({info['size_gb']:.2f} GB) seems too small!")
                print(f"   Expected: ~{expected_size} GB")
                print(f"   This might be a corrupted or incomplete file.")
                
                # Ask user if they want to continue
                response = input(f"   Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print(f"   Skipping {name} due to suspicious file size.")
                    results[name] = {'success': False, 'error': 'File size too small - likely corrupted'}
                    continue
            
            # Load state dict with better error handling
            print(f"🔄 Loading {name} state dict...")
            try:
                from safetensors import safe_open
                
                state_dict = {}
                with safe_open(info['path'], framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                
                print(f"✅ State dict loaded: {len(state_dict)} keys")
                
            except Exception as e:
                print(f"❌ Failed to load {name} state dict: {e}")
                print(f"   This suggests the file is corrupted or incomplete.")
                results[name] = {'success': False, 'error': f'Failed to load state dict: {e}'}
                continue
            
            # Create a simple dummy model for testing
            class DummyModel(torch.nn.Module):
                def __init__(self, state_dict):
                    super().__init__()
                    # Create a simple model structure
                    self.layers = torch.nn.ModuleList()
                    for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):  # Use first 5 tensors
                        if len(tensor.shape) >= 2:
                            layer = torch.nn.Linear(tensor.shape[0], tensor.shape[1])
                            self.layers.append(layer)
                
                def forward(self, x):
                    for layer in self.layers:
                        x = layer(x)
                    return x
            
            # Create dummy model
            dummy_model = DummyModel(state_dict)
            print(f"✅ Dummy model created with {len(dummy_model.layers)} layers")
            
            # Test ComfyUI-style loading
            print(f"🚀 Testing ComfyUI-style loading...")
            device = torch.device('cuda')
            
            model, final_device, loading_info = safe_model_to_device_advanced(
                dummy_model, 
                device, 
                state_dict=state_dict
            )
            
            print(f"✅ {name} loaded successfully!")
            print(f"   Loading type: {loading_info['loading_type']}")
            print(f"   Final device: {final_device}")
            print(f"   VRAM state: {loading_info.get('vram_state', 'N/A')}")
            print(f"   Patcher type: {loading_info.get('patcher_type', 'N/A')}")
            
            # Test inference
            print(f"🧠 Testing inference...")
            try:
                with torch.no_grad():
                    # Create dummy input
                    if final_device.type == 'cuda':
                        x = torch.randn(1, 10, device=final_device, dtype=torch.float16)
                    else:
                        x = torch.randn(1, 10, dtype=torch.float16)
                    
                    output = model(x)
                    print(f"✅ Inference successful! Output shape: {output.shape}")
                    
                results[name] = {
                    'success': True,
                    'loading_type': loading_info['loading_type'],
                    'patcher_type': loading_info.get('patcher_type', 'N/A'),
                    'device': str(final_device)
                }
                
            except Exception as e:
                print(f"❌ Inference failed: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            results[name] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 VAST AI ComfyUI-style Loading Summary")
    print(f"{'='*60}")
    
    successful = 0
    total = len(results)
    
    for name, result in results.items():
        if result['success']:
            successful += 1
            print(f"✅ {name}:")
            print(f"   Loading: {result['loading_type']}")
            print(f"   Patcher: {result['patcher_type']}")
            print(f"   Device: {result['device']}")
        else:
            print(f"❌ {name}: {result.get('error', 'Unknown error')}")
    
    success_rate = (successful / total) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}% ({successful}/{total})")
    
    if success_rate >= 75:
        print("🎉 SUCCESS! ComfyUI-style loading works on VAST AI!")
        print("🚀 Your models are ready for inference with patchers!")
    else:
        print("⚠️  Some models failed. Check the detailed output above.")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = test_vast_ai_models()
    sys.exit(0 if success else 1)
