#!/usr/bin/env python3
"""
Debug why we're only getting 647 patches instead of 1053
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet, load_lora
from utils import load_torch_file

def debug_patch_count():
    """Debug why patch count is low"""
    print("🔍 DEBUGGING PATCH COUNT ISSUE")
    print("="*50)
    
    # Load real models
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    print("📥 Loading UNet model...")
    unet_sd = load_torch_file(unet_path)
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    print("📥 Loading LoRA...")
    lora_sd = load_torch_file(lora_path)
    
    print(f"UNet state dict keys: {len(unet_sd)}")
    print(f"LoRA keys: {len(lora_sd)}")
    
    # Convert LoRA
    converted_lora = convert_lora(lora_sd)
    print(f"Converted LoRA keys: {len(converted_lora)}")
    
    # Generate key mappings
    key_map = model_lora_keys_unet(unet_model, {})
    print(f"Generated key mappings: {len(key_map)}")
    
    # Check what keys are being mapped
    print("\n🔍 FIRST 10 KEY MAPPINGS:")
    for i, (lora_key, model_key) in enumerate(list(key_map.items())[:10]):
        print(f"  {i+1}. {lora_key} -> {model_key}")
    
    # Check what LoRA keys are available
    print("\n🔍 FIRST 10 LORA KEYS:")
    for i, key in enumerate(list(converted_lora.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Check what model keys are available
    print("\n🔍 FIRST 10 MODEL KEYS:")
    for i, key in enumerate(list(unet_sd.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Try to load LoRA
    print("\n🔧 LOADING LORA...")
    loaded_patches = load_lora(converted_lora, key_map)
    print(f"Loaded patches: {len(loaded_patches)}")
    
    # Check for missing keys
    missing_keys = set(converted_lora.keys()) - set(key_map.keys())
    print(f"Missing key mappings: {len(missing_keys)}")
    if missing_keys:
        print("First 10 missing keys:")
        for i, key in enumerate(list(missing_keys)[:10]):
            print(f"  {i+1}. {key}")
    
    # Check for unused mappings
    unused_mappings = set(key_map.keys()) - set(converted_lora.keys())
    print(f"Unused key mappings: {len(unused_mappings)}")

if __name__ == "__main__":
    debug_patch_count()
