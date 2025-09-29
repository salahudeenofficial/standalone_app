#!/usr/bin/env python3
"""
Check actual file sizes of model files
"""

import os
from pathlib import Path

def check_model_sizes():
    """Check actual file sizes of model files"""
    print("📁 MODEL FILE SIZE ANALYSIS")
    print("="*60)
    
    models = {
        'UNet': "models/diffusion_models/wan_2.1_diffusion_model.safetensors",
        'CLIP': "models/text_encoders/wan_clip_model.safetensors", 
        'VAE': "models/vaes/wan_vae.safetensors",
        'LoRA': "models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
    }
    
    total_size = 0
    
    for name, path in models.items():
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024**2)
            size_gb = size_bytes / (1024**3)
            
            print(f"📄 {name}:")
            print(f"   Path: {path}")
            print(f"   Size: {size_bytes:,} bytes")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Size: {size_gb:.3f} GB")
            
            total_size += size_bytes
        else:
            print(f"❌ {name}: {path} (not found)")
    
    print(f"\n📊 TOTAL FILE SIZE:")
    print(f"   Total: {total_size:,} bytes")
    print(f"   Total: {total_size/(1024**2):.2f} MB")
    print(f"   Total: {total_size/(1024**3):.3f} GB")
    
    # Compare with memory usage
    print(f"\n🔍 ANALYSIS:")
    print(f"   File size: {total_size/(1024**3):.3f} GB")
    print(f"   Memory usage: ~70 GB (from your report)")
    print(f"   Ratio: {70 / (total_size/(1024**3)):.1f}x")
    print(f"   This suggests significant overhead during loading!")

if __name__ == "__main__":
    check_model_sizes()
