#!/usr/bin/env python3
"""
Test script to verify the standalone app setup
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all components can be imported"""
    print("Testing standalone app setup...")
    
    try:
        # Test component imports
        from components.model_loader import UNETLoader, CLIPLoader, VAELoader
        print("✓ Model loader components imported successfully")
        
        from components.lora_loader import LoraLoader
        print("✓ LoRA loader component imported successfully")
        
        from components.text_encoder import CLIPTextEncode
        print("✓ Text encoder component imported successfully")
        
        from components.model_sampling import ModelSamplingSD3
        print("✓ Model sampling component imported successfully")
        
        from components.video_generator import WanVaceToVideo
        print("✓ Video generator component imported successfully")
        
        from components.sampler import KSampler
        print("✓ Sampler component imported successfully")
        
        from components.video_processor import TrimVideoLatent
        print("✓ Video processor component imported successfully")
        
        from components.vae_decoder import VAEDecode
        print("✓ VAE decoder component imported successfully")
        
        from components.video_export import VideoExporter
        print("✓ Video exporter component imported successfully")
        
        # Test main pipeline
        from pipeline import ReferenceVideoPipeline
        print("✓ Main pipeline imported successfully")
        
        print("\n🎉 All components imported successfully!")
        print("Standalone app is ready to use.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_comfy_modules():
    """Test that ComfyUI modules are accessible"""
    print("\nTesting ComfyUI module access...")
    
    try:
        # Test core comfy modules
        import comfy.sd
        print("✓ comfy.sd module accessible")
        
        import comfy.utils
        print("✓ comfy.utils module accessible")
        
        import comfy.model_management
        print("✓ comfy.model_management module accessible")
        
        import comfy.samplers
        print("✓ comfy.samplers module accessible")
        
        print("✓ All ComfyUI modules accessible")
        
    except ImportError as e:
        print(f"❌ ComfyUI module error: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("STANDALONE APP SETUP TEST")
    print("=" * 50)
    
    success = True
    
    # Test component imports
    if not test_imports():
        success = False
    
    # Test ComfyUI modules
    if not test_comfy_modules():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED - Standalone app is ready!")
    else:
        print("❌ SOME TESTS FAILED - Check the errors above")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    main() 