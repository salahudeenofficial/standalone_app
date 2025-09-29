#!/usr/bin/env python3
"""
Restore models from trash and add validation
"""

import os
import shutil
from pathlib import Path

def restore_models():
    """Restore models from trash to proper location"""
    print("🔄 RESTORING MODELS FROM TRASH")
    print("="*60)
    
    # Create models directory structure
    models_dir = Path("/home/fashionx/v_pipe/standalone_app/models")
    subdirs = ["diffusion_models", "text_encoders", "vaes"]
    
    for subdir in subdirs:
        (models_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Model mappings
    models = {
        "wan_2.1_diffusion_model.safetensors": "diffusion_models",
        "wan_clip_model.safetensors": "text_encoders", 
        "wan_vae.safetensors": "vaes"
    }
    
    trash_dir = Path("/home/fashionx/.local/share/Trash/files")
    
    for filename, subdir in models.items():
        source = trash_dir / filename
        dest = models_dir / subdir / filename
        
        if source.exists():
            print(f"📄 Restoring {filename}...")
            shutil.copy2(source, dest)
            print(f"   ✅ Restored to {dest}")
        else:
            print(f"❌ {filename} not found in trash")
    
    print(f"\n✅ Model restoration completed!")

def validate_model_sizes():
    """Validate that restored models have reasonable sizes"""
    print(f"\n🔍 VALIDATING MODEL SIZES")
    print("="*60)
    
    models = {
        "UNet": "models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "CLIP": "models/text_encoders/wan_clip_model.safetensors",
        "VAE": "models/vaes/wan_vae.safetensors"
    }
    
    expected_sizes = {
        "UNet": 30 * 1024**3,  # 30GB minimum
        "CLIP": 500 * 1024**2,  # 500MB minimum  
        "VAE": 500 * 1024**2   # 500MB minimum
    }
    
    all_valid = True
    
    for name, path in models.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            expected = expected_sizes[name]
            
            print(f"📄 {name}:")
            print(f"   Size: {size/(1024**2):.2f} MB")
            print(f"   Expected: {expected/(1024**2):.2f} MB minimum")
            
            if size >= expected:
                print(f"   ✅ Valid size")
            else:
                print(f"   ❌ TOO SMALL - likely corrupted!")
                all_valid = False
        else:
            print(f"❌ {name}: {path} not found")
            all_valid = False
    
    if all_valid:
        print(f"\n✅ All models have valid sizes!")
    else:
        print(f"\n❌ Some models are corrupted - need to re-download!")
    
    return all_valid

def main():
    """Main function"""
    print("🚀 MODEL RESTORATION AND VALIDATION")
    print("="*80)
    
    # Restore models
    restore_models()
    
    # Validate sizes
    valid = validate_model_sizes()
    
    if valid:
        print(f"\n🎉 SUCCESS: Models restored and validated!")
        print(f"   You can now run the pipeline with proper models")
    else:
        print(f"\n⚠️  WARNING: Models are still corrupted!")
        print(f"   You need to re-download the models:")
        print(f"   - UNet: ~32GB")
        print(f"   - CLIP: ~1GB") 
        print(f"   - VAE: ~1GB")

if __name__ == "__main__":
    main()
