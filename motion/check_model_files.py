#!/usr/bin/env python3
"""
VAST AI Model File Checker
Check if your model files are complete and not corrupted
"""

import os
import sys

def check_model_files():
    """Check model files for completeness and corruption"""
    
    print("🔍 VAST AI Model File Checker")
    print("=" * 50)
    
    # Check current directory structure
    print("📁 Current directory structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            if file.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                file_path = os.path.join(root, file)
                size_gb = os.path.getsize(file_path) / (1024**3)
                print(f"{subindent}{file} ({size_gb:.2f} GB)")
    
    print("\n🎯 Looking for model files...")
    
    # Model paths to check
    model_paths = {
        "UNet": "./models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        "VAE": "./models/vaes/wan_vae.safetensors", 
        "Text Encoder": "./models/text_encoders/wan_clip_model.safetensors"
    }
    
    # Expected sizes
    expected_sizes = {
        "UNet": 30,  # Should be ~32GB
        "VAE": 0.2,  # Should be ~200MB
        "Text Encoder": 10  # Should be ~10GB
    }
    
    found_models = {}
    
    for name, path in model_paths.items():
        print(f"\n🔍 Checking {name}:")
        print(f"   Path: {path}")
        
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            expected_size = expected_sizes.get(name, 0)
            
            print(f"   ✅ File exists")
            print(f"   📏 Size: {size_gb:.2f} GB")
            print(f"   🎯 Expected: ~{expected_size} GB")
            
            if size_gb < expected_size * 0.1:  # Less than 10% of expected
                print(f"   ⚠️  WARNING: File size is suspiciously small!")
                print(f"   🚨 This file is likely corrupted or incomplete")
                
                # Try to get more info about the file
                try:
                    with open(path, 'rb') as f:
                        header = f.read(100)  # Read first 100 bytes
                        print(f"   🔍 First 100 bytes: {header[:50]}...")
                except Exception as e:
                    print(f"   ❌ Cannot read file: {e}")
            else:
                print(f"   ✅ File size looks reasonable")
                
                # Try to validate safetensors header
                try:
                    from safetensors import safe_open
                    with safe_open(path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        print(f"   ✅ Safetensors header valid")
                        print(f"   🔑 Keys: {len(keys)}")
                        if len(keys) > 0:
                            print(f"   📝 Sample keys: {keys[:3]}...")
                except Exception as e:
                    print(f"   ❌ Safetensors validation failed: {e}")
                    print(f"   🚨 File is corrupted!")
                
            found_models[name] = {
                'path': path,
                'size_gb': size_gb,
                'expected_size': expected_size,
                'valid': size_gb >= expected_size * 0.1
            }
        else:
            print(f"   ❌ File not found")
            print(f"   💡 Check if the path is correct")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"=" * 30)
    
    valid_models = 0
    total_models = len(model_paths)
    
    for name, info in found_models.items():
        if info['valid']:
            valid_models += 1
            print(f"✅ {name}: Valid ({info['size_gb']:.2f} GB)")
        else:
            print(f"❌ {name}: Invalid ({info['size_gb']:.2f} GB)")
    
    print(f"\n🎯 Valid models: {valid_models}/{total_models}")
    
    if valid_models == 0:
        print("\n🚨 No valid models found!")
        print("💡 Possible solutions:")
        print("   1. Check if model files are in the correct location")
        print("   2. Re-download corrupted model files")
        print("   3. Check if files are still downloading")
        print("   4. Verify file permissions")
    elif valid_models < total_models:
        print(f"\n⚠️  Some models are corrupted or incomplete")
        print("💡 Re-download the corrupted files")
    else:
        print(f"\n🎉 All models are valid and ready for testing!")
    
    return valid_models == total_models

if __name__ == "__main__":
    success = check_model_files()
    sys.exit(0 if success else 1)
