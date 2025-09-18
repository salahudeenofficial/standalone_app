#!/usr/bin/env python3
"""
ComfyUI-style model loading and verification with actual models on VAST AI
Tests UNet (32GB), VAE (200MB), Text Encoder (10GB) with patcher system
"""

import sys
import os
import torch
import logging
from memory_utils import safe_model_to_device_advanced, get_memory_info
from standalone_sd import load_state_dict_guess_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_loading_with_patcher(model_path, model_name):
    """
    Test ComfyUI-style model loading with patcher assignment
    
    Args:
        model_path: Path to model file
        model_name: Name of the model (UNet, VAE, Text Encoder)
    
    Returns:
        dict: Loading results and patcher info
    """
    print(f"\n🔧 Testing {model_name} Loading with ComfyUI-style Patcher")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found: {model_path}")
        return {'success': False, 'error': 'Model not found'}
    
    file_size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"📁 {model_name} file: {model_path}")
    print(f"📏 File size: {file_size_gb:.2f} GB")
    
    try:
        # Load state dict
        print(f"🔄 Loading {model_name} state dict...")
        state_dict = load_state_dict_guess_config(model_path)
        
        if state_dict is None:
            print(f"❌ Failed to load {model_name} state dict")
            return {'success': False, 'error': 'Failed to load state dict'}
        
        print(f"✅ {model_name} state dict loaded successfully")
        print(f"   Keys: {len(state_dict)}")
        
        # Create model from config
        print(f"🏗️  Creating {model_name} model...")
        model = create_model_from_config(state_dict)
        
        if model is None:
            print(f"❌ Failed to create {model_name} model")
            return {'success': False, 'error': 'Failed to create model'}
        
        print(f"✅ {model_name} model created successfully")
        
        # Test ComfyUI-style loading with patcher
        print(f"🚀 Testing ComfyUI-style loading with patcher...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model, final_device, loading_info = safe_model_to_device_advanced(
            model, 
            device, 
            state_dict=state_dict
        )
        
        print(f"✅ {model_name} loaded successfully!")
        print(f"   Loading type: {loading_info['loading_type']}")
        print(f"   Final device: {final_device}")
        print(f"   VRAM state: {loading_info.get('vram_state', 'N/A')}")
        print(f"   Patcher type: {loading_info.get('patcher_type', 'N/A')}")
        
        if loading_info['loading_type'] == 'full_gpu':
            print(f"   🎯 Complete patcher: Model fully loaded to GPU")
        elif loading_info['loading_type'] == 'cpu_first_partial':
            print(f"   🔧 Partial patcher: Model on CPU with dynamic loading")
            print(f"   Modules available: {loading_info.get('modules_available', 0)}")
            print(f"   Memory budget: {loading_info.get('memory_budget_gb', 0):.2f} GB")
        else:
            print(f"   📱 CPU-only patcher: Model loaded to CPU")
        
        return {
            'success': True,
            'model': model,
            'device': final_device,
            'loading_info': loading_info,
            'file_size_gb': file_size_gb,
            'state_dict_keys': len(state_dict)
        }
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return {'success': False, 'error': str(e)}

def create_model_from_config(state_dict):
    """
    Create model from state dict using standalone_sd
    """
    try:
        # Use standalone_sd to create model from state dict
        from model_detection import create_model_from_config as create_model
        
        # Detect model configuration
        from model_detection import detect_unet_config, model_config_from_unet_config
        
        # Detect UNet config
        unet_config = detect_unet_config(state_dict)
        if unet_config is None:
            logging.error("Failed to detect UNet config")
            return None
        
        # Create model config
        model_config = model_config_from_unet_config(unet_config)
        if model_config is None:
            logging.error("Failed to create model config")
            return None
        
        # Create model
        model = create_model(model_config, state_dict=state_dict)
        return model
        
    except Exception as e:
        logging.error(f"Failed to create model from config: {e}")
        return None

def test_inference_with_patcher(model, model_name, device):
    """
    Test inference with ComfyUI-style patcher
    """
    print(f"\n🧠 Testing {model_name} Inference with Patcher")
    print("=" * 50)
    
    try:
        # Test basic forward pass
        if model_name == "UNet":
            # Test UNet forward pass
            batch_size = 1
            height, width = 64, 64
            frames = 16
            
            # Create dummy inputs
            if device.type == 'cuda':
                x = torch.randn(batch_size, 4, frames, height, width, device=device, dtype=torch.float16)
                timestep = torch.tensor([100], device=device)
                context = torch.randn(batch_size, 77, 5120, device=device, dtype=torch.float16)
            else:
                x = torch.randn(batch_size, 4, frames, height, width, dtype=torch.float16)
                timestep = torch.tensor([100])
                context = torch.randn(batch_size, 77, 5120, dtype=torch.float16)
            
            print(f"   Input shape: {x.shape}")
            print(f"   Timestep: {timestep}")
            print(f"   Context shape: {context.shape}")
            
            # Run forward pass
            with torch.no_grad():
                output = model(x, timestep, context)
            
            print(f"✅ {model_name} forward pass successful!")
            print(f"   Output shape: {output.shape}")
            
        elif model_name == "VAE":
            # Test VAE forward pass
            batch_size = 1
            height, width = 64, 64
            frames = 16
            
            # Create dummy inputs
            if device.type == 'cuda':
                x = torch.randn(batch_size, 4, frames, height, width, device=device, dtype=torch.float16)
            else:
                x = torch.randn(batch_size, 4, frames, height, width, dtype=torch.float16)
            
            print(f"   Input shape: {x.shape}")
            
            # Run forward pass
            with torch.no_grad():
                output = model.decode(x)
            
            print(f"✅ {model_name} forward pass successful!")
            print(f"   Output shape: {output.shape}")
            
        elif model_name == "Text Encoder":
            # Test Text Encoder forward pass
            batch_size = 1
            seq_len = 77
            
            # Create dummy inputs
            if device.type == 'cuda':
                x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            else:
                x = torch.randint(0, 1000, (batch_size, seq_len))
            
            print(f"   Input shape: {x.shape}")
            
            # Run forward pass
            with torch.no_grad():
                output = model(x)
            
            print(f"✅ {model_name} forward pass successful!")
            print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} inference failed: {e}")
        return False

def main():
    """Run ComfyUI-style model loading and verification on VAST AI"""
    
    print("🚀 ComfyUI-style Model Loading and Verification on VAST AI")
    print("=" * 70)
    
    # Get system info
    info = get_memory_info()
    print(f"📊 VAST AI System Information:")
    print(f"   Available GPU memory: {info['cuda_free']:.2f} GB")
    print(f"   Total VRAM: {info['cuda_total']:.2f} GB")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Model paths (adjust these to your actual VAST AI paths)
    models = {
        "UNet": "../models/diffusion_models/wan_2.1_diffusion_model.safetensors",  # 32GB UNet
        "VAE": "../models/vae/vae.safetensors",                                    # 200MB VAE  
        "Text Encoder": "../models/clip/clip.safetensors"                          # 10GB Text Encoder
    }
    
    results = {}
    
    # Test each model with ComfyUI-style patcher
    for model_name, model_path in models.items():
        result = test_model_loading_with_patcher(model_path, model_name)
        results[model_name] = result
        
        if result['success']:
            # Test inference
            inference_success = test_inference_with_patcher(
                result['model'], 
                model_name, 
                result['device']
            )
            results[model_name]['inference_success'] = inference_success
    
    # Print summary
    print(f"\n📊 ComfyUI-style Loading Summary")
    print("=" * 50)
    
    successful_models = 0
    total_models = len(models)
    
    for model_name, result in results.items():
        if result['success']:
            successful_models += 1
            patcher_type = result['loading_info'].get('patcher_type', 'Unknown')
            loading_type = result['loading_info']['loading_type']
            inference_success = result.get('inference_success', False)
            
            print(f"✅ {model_name}:")
            print(f"   Loading: {loading_type}")
            print(f"   Patcher: {patcher_type}")
            print(f"   Inference: {'✅ Success' if inference_success else '❌ Failed'}")
            print(f"   Size: {result['file_size_gb']:.2f} GB")
        else:
            print(f"❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    success_rate = (successful_models / total_models) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({successful_models}/{total_models})")
    
    if success_rate >= 75:
        print("🎉 SUCCESS! ComfyUI-style loading and verification completed successfully!")
        print("🚀 Your models are ready for inference with ComfyUI-style patchers!")
    else:
        print("⚠️  Some models failed to load. Check the detailed output above.")
    
    return 0 if success_rate >= 75 else 1

if __name__ == "__main__":
    sys.exit(main())
