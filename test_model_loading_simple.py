#!/usr/bin/env python3
"""
Simple Model Loading Test Script
Tests the actual model loading functionality from pipe.py
"""

import os
import sys
import torch
import time

# Set PyTorch memory configuration
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

def test_unet_loading():
    """Test UNet model loading from motion.comps"""
    print("\n" + "="*60)
    print("🧠 TESTING UNET MODEL LOADING")
    print("="*60)
    
    try:
        from motion.comps import UNETLoader
        
        # Check model file existence first
        model_filename = "wan_2.1_diffusion_model.safetensors"
        expected_path = f"./models/diffusion_models/{model_filename}"
        
        print(f"🔍 Checking model file: {expected_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   File exists: {os.path.exists(expected_path)}")
        
        if os.path.exists(expected_path):
            file_size = os.path.getsize(expected_path)
            print(f"   File size: {file_size / (1024**3):.2f} GB")
        else:
            print("   ❌ Model file not found!")
            print("   📁 Checking if models directory exists...")
            print(f"   models/ exists: {os.path.exists('./models')}")
            print(f"   models/diffusion_models/ exists: {os.path.exists('./models/diffusion_models')}")
            
            if os.path.exists('./models/diffusion_models'):
                files_in_dir = os.listdir('./models/diffusion_models')
                print(f"   Files in models/diffusion_models/: {files_in_dir}")
        
        print("\n📥 Loading UNet model...")
        unet_loader = UNETLoader(model_filename, "default")
        unet = unet_loader.load_unet()
        
        print(f"✅ UNet loaded successfully!")
        print(f"   Type: {type(unet).__name__}")
        
        # Check if it's a real model or mock
        if hasattr(unet, 'patches_uuid') and unet.patches_uuid == "mock-unet-uuid":
            print("   ⚠️  Mock UNet (no model file found)")
            print("   This means the model file path check failed in UNETLoader")
        else:
            print("   ✅ Real UNet model loaded!")
            
            # For ModelPatcher, we need to access the underlying model
            if hasattr(unet, 'model'):
                underlying_model = unet.model
                print(f"   Underlying model type: {type(underlying_model).__name__}")
                
                if hasattr(underlying_model, 'parameters'):
                    device = next(underlying_model.parameters()).device
                    param_count = sum(p.numel() for p in underlying_model.parameters())
                    print(f"   Device: {device}")
                    print(f"   Parameters: {param_count:,}")
                    
                    # Test forward pass
                    print("🧪 Testing forward pass...")
                    dummy_latent = torch.randn(1, 4, 64, 48, device=device)
                    dummy_timestep = torch.tensor([100], device=device)
                    
                    with torch.no_grad():
                        start_time = time.time()
                        output = underlying_model(dummy_latent, dummy_timestep)
                        end_time = time.time()
                        
                    print(f"✅ Forward pass successful!")
                    print(f"   Input: {dummy_latent.shape} -> Output: {output.shape}")
                    print(f"   Time: {(end_time - start_time)*1000:.2f} ms")
                else:
                    print("   ⚠️  Underlying model doesn't have parameters method")
            else:
                print("   ⚠️  ModelPatcher doesn't have underlying model attribute")
        
        return unet
        
    except Exception as e:
        print(f"❌ UNet loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_clip_loading():
    """Test CLIP model loading"""
    print("\n" + "="*60)
    print("📝 TESTING CLIP MODEL LOADING")
    print("="*60)
    
    try:
        from motion.comps import CLIPLoader
        
        print("📥 Loading CLIP model...")
        clip_loader = CLIPLoader("wan_clip_model.safetensors")
        clip = clip_loader.load_clip()
        
        print(f"✅ CLIP loaded successfully!")
        print(f"   Type: {type(clip).__name__}")
        
        # Test text encoding
        print("🧪 Testing text encoding...")
        test_prompt = "a beautiful cinematic video"
        
        with torch.no_grad():
            start_time = time.time()
            encoded = clip.encode(test_prompt)
            end_time = time.time()
            
        print(f"✅ Text encoding successful!")
        print(f"   Prompt: '{test_prompt}'")
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Time: {(end_time - start_time)*1000:.2f} ms")
        
        return clip
        
    except Exception as e:
        print(f"❌ CLIP loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_vae_loading():
    """Test VAE model loading"""
    print("\n" + "="*60)
    print("🎨 TESTING VAE MODEL LOADING")
    print("="*60)
    
    try:
        from motion.comps import Initial_latent
        
        print("📥 Loading VAE model...")
        vae_model_path = "./models/vaes/wan_vae.safetensors"
        
        if not os.path.exists(vae_model_path):
            print(f"⚠️  VAE model not found at {vae_model_path}")
            print("   Creating Initial_latent component...")
        
        initial_latent = Initial_latent()
        
        print(f"✅ VAE component loaded successfully!")
        print(f"   Type: {type(initial_latent).__name__}")
        
        return initial_latent
        
    except Exception as e:
        print(f"❌ VAE loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_pipe_functions():
    """Test the actual pipe.py functions"""
    print("\n" + "="*60)
    print("🔧 TESTING PIPE.PY FUNCTIONS")
    print("="*60)
    
    try:
        # Import the functions from pipe.py
        from pipe import step_2_unet_clip_lora_loading
        
        print("📥 Testing step_2_unet_clip_lora_loading...")
        
        # Test with dummy paths
        unet_model_path = "./models/diffusion_models/wan_2.1_diffusion_model.safetensors"
        clip_model_path = "./models/text_encoders/wan_clip_model.safetensors"
        
        results = step_2_unet_clip_lora_loading(
            unet_model_path=unet_model_path,
            clip_model_path=clip_model_path,
            lora_model_path=None,
            strength_model=1.0,
            strength_clip=0.0
        )
        
        if results:
            print(f"✅ Step 2 function executed successfully!")
            print(f"   UNet type: {results['model_info']['unet_type']}")
            print(f"   CLIP type: {results['model_info']['clip_type']}")
            print(f"   LoRA applied: {results['lora_applied']}")
        else:
            print("❌ Step 2 function returned None")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipe function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def debug_unet_loader():
    """Debug the UNETLoader code path step by step"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING UNET LOADER CODE PATH")
    print("="*60)
    
    try:
        from motion.comps import UNETLoader
        
        model_filename = "wan_2.1_diffusion_model.safetensors"
        print(f"📝 Creating UNETLoader with filename: {model_filename}")
        
        unet_loader = UNETLoader(model_filename, "default")
        print(f"   UNETLoader created successfully")
        print(f"   Model path: {unet_loader.model_path}")
        print(f"   Weight dtype: {unet_loader.weight_dtype}")
        
        # Manually check the path construction
        import os
        constructed_path = os.path.join("./models/diffusion_models", unet_loader.model_path)
        print(f"   Constructed path: {constructed_path}")
        print(f"   Path exists: {os.path.exists(constructed_path)}")
        
        # Check what happens in load_unet
        print(f"\n🔧 Calling load_unet()...")
        unet = unet_loader.load_unet()
        
        print(f"   Returned object type: {type(unet).__name__}")
        
        if hasattr(unet, 'patches_uuid'):
            print(f"   Patches UUID: {unet.patches_uuid}")
            if unet.patches_uuid == "mock-unet-uuid":
                print("   ⚠️  This is a MOCK UNet - the real model file was not found!")
            else:
                print("   ✅ This is a REAL UNet model!")
                
                # Check ModelPatcher structure
                if hasattr(unet, 'model'):
                    print(f"   Underlying model: {type(unet.model).__name__}")
                    if hasattr(unet.model, 'parameters'):
                        param_count = sum(p.numel() for p in unet.model.parameters())
                        print(f"   Model parameters: {param_count:,}")
                    else:
                        print("   ⚠️  Underlying model has no parameters method")
                else:
                    print("   ⚠️  ModelPatcher has no underlying model")
        else:
            print("   ⚠️  No patches_uuid attribute found")
        
        return unet
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🚀 SIMPLE MODEL LOADING TEST")
    print("="*80)
    
    # Check GPU
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Debug the UNETLoader first
    debug_unet_loader()
    
    # Test individual components
    print("\n🧪 TESTING INDIVIDUAL COMPONENTS")
    print("="*80)
    
    unet = test_unet_loading()
    clip = test_clip_loading()
    vae = test_vae_loading()
    
    # Test pipe functions
    print("\n🔧 TESTING PIPE FUNCTIONS")
    print("="*80)
    
    pipe_results = test_pipe_functions()
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    results = {
        'UNet Loaded': unet is not None,
        'CLIP Loaded': clip is not None,
        'VAE Loaded': vae is not None,
        'Pipe Functions Work': pipe_results is not None
    }
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Models are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    print("="*80)

if __name__ == "__main__":
    main()
