#!/usr/bin/env python3
"""
ComfyUI-style model loading and verification with actual models on VAST AI
Tests UNet (32GB), VAE (200MB), Text Encoder (10GB) with patcher system
Based on the working test_wan21_vace_16b_complete.py approach
"""

import sys
import os
import torch
import logging
from standalone_sd import load_state_dict_guess_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_loading_with_patcher(model_path, model_name):
    """
    Test ComfyUI-style model loading with patcher assignment using the working approach
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
        # Use the working approach from test_wan21_vace_16b_complete.py
        print(f"🔄 Loading {model_name} using working ModelPatcher pipeline...")
        
        # Try normal loading first (same as working script)
        try:
            print(f"🚀 Attempting normal loading mode...")
            model_patcher, clip, vae, clipvision = load_state_dict_guess_config(
                model_path, 
                output_vae=False, 
                output_clip=False, 
                output_clipvision=False,
                output_model=True
            )
            
            if model_patcher is None:
                print(f"❌ Normal loading failed - model_patcher is None")
                return {'success': False, 'error': 'Model patcher is None'}
            
            print(f"✅ Normal loading successful")
            
            # Get the actual model from the patcher
            model = model_patcher.model
            device = next(model.parameters()).device
            
            print(f"✅ {model_name} loaded successfully!")
            print(f"   Model type: {type(model)}")
            print(f"   Device: {device}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return {
                'success': True,
                'model': model,
                'model_patcher': model_patcher,
                'device': device,
                'loading_info': {
                    'loading_type': 'normal',
                    'patcher_type': 'complete',
                    'vram_state': 'HIGH_VRAM'
                },
                'file_size_gb': file_size_gb
            }
            
        except Exception as e:
            print(f"❌ Normal loading failed: {e}")
            return {'success': False, 'error': f'Normal loading failed: {e}'}
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return {'success': False, 'error': str(e)}

def test_inference_with_patcher(model, model_name, device):
    """
    Test inference with the loaded model using proper WAN model parameters
    """
    print(f"\n🧠 Testing {model_name} Inference with Patcher")
    print("=" * 50)
    
    try:
        # Get model dtype from first parameter
        model_dtype = next(model.parameters()).dtype
        print(f"📊 Model dtype: {model_dtype}")
        
        # Create proper inputs for WAN model
        batch_size = 1
        frames = 16
        height, width = 64, 64
        
        if device.type == 'cuda':
            # Create proper WAN model inputs
            x = torch.randn(batch_size, 4, frames, height, width, device=device, dtype=model_dtype)
            timestep = torch.tensor([100], device=device)
            context = torch.randn(batch_size, 77, 5120, device=device, dtype=model_dtype)
        else:
            x = torch.randn(batch_size, 4, frames, height, width, dtype=model_dtype)
            timestep = torch.tensor([100])
            context = torch.randn(batch_size, 77, 5120, dtype=model_dtype)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Timestep: {timestep}")
        print(f"   Context shape: {context.shape}")
        
        # Run forward pass
        with torch.no_grad():
            output = model(x, timestep, context)
        
        print(f"✅ {model_name} inference successful!")
        print(f"   Output shape: {output.shape}, dtype: {output.dtype}")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} inference failed: {e}")
        return False

def main():
    """Run ComfyUI-style model loading and verification on VAST AI"""
    
    print("🚀 ComfyUI-style Model Loading and Verification on VAST AI")
    print("=" * 70)
    
    # Get system info
    print(f"📊 VAST AI System Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Model paths (adjust these to your actual VAST AI paths)
    models = {
        "UNet": "./models/diffusion_models/wan_2.1_diffusion_model.safetensors",  # 32GB UNet
        "VAE": "./models/vaes/wan_vae.safetensors",                                # ~200MB VAE  
        "Text Encoder": "./models/text_encoders/wan_clip_model.safetensors"         # ~10GB Text Encoder
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
            print(f"❌ {model_name}: {result['error']}")
    
    print(f"\n🎯 Overall Success Rate: {successful_models/total_models*100:.1f}% ({successful_models}/{total_models})")
    
    if successful_models == total_models:
        print("🎉 All models loaded successfully with ComfyUI-style patcher!")
    else:
        print("⚠️  Some models failed. Check the detailed output above.")

if __name__ == "__main__":
    main()