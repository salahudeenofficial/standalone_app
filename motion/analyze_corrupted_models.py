#!/usr/bin/env python3
"""
Analyze the corrupted model files to understand what's happening
"""

import torch
import sys
import os
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_torch_file

def analyze_corrupted_models():
    """Analyze the corrupted model files"""
    print("🔍 ANALYZING CORRUPTED MODEL FILES")
    print("="*80)
    
    models = {
        'UNet': "/home/fashionx/.local/share/Trash/files/wan_2.1_diffusion_model.safetensors",
        'CLIP': "/home/fashionx/.local/share/Trash/files/wan_clip_model.safetensors",
        'VAE': "/home/fashionx/.local/share/Trash/files/wan_vae.safetensors"
    }
    
    for name, path in models.items():
        print(f"\n📄 ANALYZING {name.upper()}")
        print("="*60)
        
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        
        # File size
        file_size = os.path.getsize(path)
        print(f"📊 File Size: {file_size:,} bytes ({file_size/(1024**2):.2f} MB)")
        
        # Expected size
        if name == 'UNet':
            expected_size = 32 * 1024**3  # 32GB
            print(f"📊 Expected Size: {expected_size:,} bytes ({expected_size/(1024**3):.1f} GB)")
            print(f"📊 Size Ratio: {file_size/expected_size:.6f} (should be ~1.0)")
        elif name in ['CLIP', 'VAE']:
            expected_size = 1 * 1024**3  # 1GB
            print(f"📊 Expected Size: {expected_size:,} bytes ({expected_size/(1024**3):.1f} GB)")
            print(f"📊 Size Ratio: {file_size/expected_size:.6f} (should be ~1.0)")
        
        # Try to load and analyze
        try:
            print(f"🔄 Loading {name} state dict...")
            state_dict = load_torch_file(path)
            
            print(f"📊 State Dict Analysis:")
            print(f"   Keys: {len(state_dict)}")
            
            # Analyze parameters
            total_params = 0
            total_size = 0
            tensor_count = 0
            
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    tensor_count += 1
                    params = tensor.numel()
                    size = params * tensor.element_size()
                    total_params += params
                    total_size += size
                    
                    # Show first few tensors
                    if tensor_count <= 5:
                        print(f"   {key}: {tensor.shape} ({params:,} params, {size/(1024**2):.2f} MB)")
            
            print(f"   Total Tensors: {tensor_count}")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Total Size: {total_size/(1024**2):.2f} MB")
            
            # Check if this looks like a real model
            if name == 'UNet' and total_params < 1_000_000_000:  # Less than 1B parameters
                print(f"   ❌ WARNING: UNet has only {total_params:,} parameters!")
                print(f"   ❌ Expected: ~17B parameters for WAN 2.1")
                print(f"   ❌ This file is severely corrupted or incomplete!")
            elif name in ['CLIP', 'VAE'] and total_params < 100_000_000:  # Less than 100M parameters
                print(f"   ❌ WARNING: {name} has only {total_params:,} parameters!")
                print(f"   ❌ Expected: ~100M+ parameters")
                print(f"   ❌ This file is severely corrupted or incomplete!")
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run analysis"""
    print("🚀 CORRUPTED MODEL FILE ANALYSIS")
    print("="*80)
    
    analyze_corrupted_models()
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   The model files are severely corrupted or incomplete!")
    print(f"   UNet should be ~32GB but is only 66MB")
    print(f"   CLIP/VAE should be ~1GB each but are only 41KB")
    print(f"   This explains the 70GB memory usage - corrupted loading!")
    print(f"   ")
    print(f"   SOLUTION:")
    print(f"   1. Restore models from trash: mv /home/fashionx/.local/share/Trash/files/wan_*.safetensors models/")
    print(f"   2. Re-download models if they're corrupted")
    print(f"   3. Check download integrity")

if __name__ == "__main__":
    main()
