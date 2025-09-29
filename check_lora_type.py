#!/usr/bin/env python3
"""
Script to analyze LoRA file type and show the difference between 
true LoRA and diff LoRA files
"""

import torch
import sys
import os
from pathlib import Path

def analyze_lora_file(lora_path):
    """Analyze a LoRA file to determine its type and contents"""
    
    if not os.path.exists(lora_path):
        print(f"❌ LoRA file not found: {lora_path}")
        return
    
    print(f"🔍 Analyzing LoRA file: {lora_path}")
    print("="*80)
    
    # Load the LoRA file
    try:
        lora = torch.load(lora_path, map_location='cpu')
    except:
        try:
            # Try safetensors
            from safetensors.torch import load_file
            lora = load_file(lora_path)
        except Exception as e:
            print(f"❌ Error loading LoRA file: {e}")
            return
    
    # Analyze keys
    keys = list(lora.keys())
    total_keys = len(keys)
    
    # Count different types
    lora_up_keys = [k for k in keys if '.lora_up.weight' in k]
    lora_down_keys = [k for k in keys if '.lora_down.weight' in k]
    diff_keys = [k for k in keys if '.diff' in k]
    alpha_keys = [k for k in keys if '.alpha' in k]
    
    print(f"📊 LORA FILE ANALYSIS:")
    print(f"   Total Keys: {total_keys}")
    print(f"   LoRA Up Keys: {len(lora_up_keys)}")
    print(f"   LoRA Down Keys: {len(lora_down_keys)}")
    print(f"   Diff Keys: {len(diff_keys)}")
    print(f"   Alpha Keys: {len(alpha_keys)}")
    
    # Determine LoRA type
    if len(diff_keys) > 0:
        lora_type = "DIFF LoRA (Full Weight Replacement)"
        expected_patches = len(diff_keys)
        print(f"\n🔧 LORA TYPE: {lora_type}")
        print(f"   This will patch {expected_patches} layers with full weight differences")
        print(f"   Expected behavior: Patches ALL specified model layers")
        
    elif len(lora_up_keys) > 0 and len(lora_down_keys) > 0:
        lora_type = "TRUE LoRA (Low-Rank Adaptation)"
        expected_patches = len(lora_up_keys)
        print(f"\n🔧 LORA TYPE: {lora_type}")
        print(f"   This will patch {expected_patches} layers with low-rank matrices")
        print(f"   Expected behavior: Patches only strategic attention/linear layers")
        
    else:
        lora_type = "UNKNOWN LoRA Type"
        expected_patches = 0
        print(f"\n🔧 LORA TYPE: {lora_type}")
        print(f"   Cannot determine LoRA type from key patterns")
    
    # Show sample keys
    print(f"\n📝 SAMPLE KEYS (first 10):")
    for i, key in enumerate(keys[:10]):
        tensor_shape = lora[key].shape if hasattr(lora[key], 'shape') else 'scalar'
        print(f"   {i+1:2d}. {key} -> {tensor_shape}")
    
    if total_keys > 10:
        print(f"   ... and {total_keys - 10} more keys")
    
    # File size analysis
    file_size_mb = os.path.getsize(lora_path) / (1024**2)
    print(f"\n💾 FILE INFO:")
    print(f"   File Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 50:
        size_category = "Small (likely True LoRA)"
    elif file_size_mb < 200:
        size_category = "Medium (could be either type)"
    else:
        size_category = "Large (likely Diff LoRA or Full Fine-tune)"
    
    print(f"   Size Category: {size_category}")
    
    print("="*80)
    
    return {
        'type': lora_type,
        'expected_patches': expected_patches,
        'total_keys': total_keys,
        'file_size_mb': file_size_mb
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_lora_type.py <lora_file_path>")
        print("Example: python check_lora_type.py models/loras/my_lora.safetensors")
        sys.exit(1)
    
    lora_path = sys.argv[1]
    result = analyze_lora_file(lora_path) 