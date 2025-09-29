#!/usr/bin/env python3
"""
Comprehensive debug of the LoRA loading process
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet, load_lora
from utils import load_torch_file

def comprehensive_debug():
    """Comprehensive debug of LoRA loading"""
    print("🔍 COMPREHENSIVE LORA DEBUG")
    print("="*60)
    
    # Load models
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    print("📥 Loading models...")
    unet_sd = load_torch_file(unet_path)
    lora_sd = load_torch_file(lora_path)
    
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    print(f"📊 UNet keys: {len(unet_sd)}")
    print(f"📊 LoRA keys: {len(lora_sd)}")
    
    # Convert LoRA
    converted_lora = convert_lora(lora_sd)
    print(f"📊 Converted LoRA keys: {len(converted_lora)}")
    
    # Generate key mappings
    key_map = model_lora_keys_unet(unet_model, {})
    print(f"📊 Generated key mappings: {len(key_map)}")
    
    # Analyze the key mappings
    print(f"\n🔍 KEY MAPPING ANALYSIS:")
    
    # Check what types of mappings we have
    weight_mappings = {k: v for k, v in key_map.items() if v.endswith('.weight')}
    bias_mappings = {k: v for k, v in key_map.items() if v.endswith('.bias')}
    other_mappings = {k: v for k, v in key_map.items() if not v.endswith('.weight') and not v.endswith('.bias')}
    
    print(f"  Weight mappings: {len(weight_mappings)}")
    print(f"  Bias mappings: {len(bias_mappings)}")
    print(f"  Other mappings: {len(other_mappings)}")
    
    # Check what LoRA keys we have
    lora_weight_keys = [k for k in converted_lora.keys() if k.endswith('.weight')]
    lora_bias_keys = [k for k in converted_lora.keys() if k.endswith('.diff_b')]
    lora_other_keys = [k for k in converted_lora.keys() if not k.endswith('.weight') and not k.endswith('.diff_b')]
    
    print(f"\n🔍 LORA KEY ANALYSIS:")
    print(f"  LoRA weight keys: {len(lora_weight_keys)}")
    print(f"  LoRA diff_b keys: {len(lora_bias_keys)}")
    print(f"  LoRA other keys: {len(lora_other_keys)}")
    
    # Check which LoRA keys have mappings
    mapped_weight_keys = set(lora_weight_keys) & set(key_map.keys())
    mapped_bias_keys = set(lora_bias_keys) & set(key_map.keys())
    mapped_other_keys = set(lora_other_keys) & set(key_map.keys())
    
    print(f"\n🔍 MAPPING COVERAGE:")
    print(f"  Mapped weight keys: {len(mapped_weight_keys)}")
    print(f"  Mapped bias keys: {len(mapped_bias_keys)}")
    print(f"  Mapped other keys: {len(mapped_other_keys)}")
    
    # Check for missing mappings
    missing_weight_keys = set(lora_weight_keys) - set(key_map.keys())
    missing_bias_keys = set(lora_bias_keys) - set(key_map.keys())
    missing_other_keys = set(lora_other_keys) - set(key_map.keys())
    
    print(f"\n🔍 MISSING MAPPINGS:")
    print(f"  Missing weight keys: {len(missing_weight_keys)}")
    print(f"  Missing bias keys: {len(missing_bias_keys)}")
    print(f"  Missing other keys: {len(missing_other_keys)}")
    
    if missing_bias_keys:
        print(f"\n🔍 FIRST 10 MISSING BIAS KEYS:")
        for i, key in enumerate(list(missing_bias_keys)[:10]):
            print(f"  {i+1}. {key}")
    
    # Try to load LoRA and see what happens
    print(f"\n🔧 LOADING LORA...")
    loaded_patches = load_lora(converted_lora, key_map)
    print(f"📊 Loaded patches: {len(loaded_patches)}")
    
    # Check what types of patches were loaded
    weight_patches = {k: v for k, v in loaded_patches.items() if k.endswith('.weight')}
    bias_patches = {k: v for k, v in loaded_patches.items() if k.endswith('.bias')}
    other_patches = {k: v for k, v in loaded_patches.items() if not k.endswith('.weight') and not k.endswith('.bias')}
    
    print(f"\n🔍 LOADED PATCHES ANALYSIS:")
    print(f"  Weight patches: {len(weight_patches)}")
    print(f"  Bias patches: {len(bias_patches)}")
    print(f"  Other patches: {len(other_patches)}")
    
    # Check for unused mappings
    used_mappings = set(loaded_patches.keys())
    unused_mappings = set(key_map.values()) - used_mappings
    print(f"\n🔍 UNUSED MAPPINGS:")
    print(f"  Unused mappings: {len(unused_mappings)}")
    
    if unused_mappings:
        print(f"  First 10 unused mappings:")
        for i, key in enumerate(list(unused_mappings)[:10]):
            print(f"    {i+1}. {key}")

if __name__ == "__main__":
    comprehensive_debug()
