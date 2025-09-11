#!/usr/bin/env python3
"""
Debug key mapping mismatch between generated mappings and converted LoRA keys
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet
from utils import load_torch_file

def debug_key_mapping_mismatch():
    """Debug the mismatch between generated mappings and actual LoRA keys"""
    print("🔍 KEY MAPPING MISMATCH DEBUG")
    print("="*60)
    
    # Load models
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    unet_sd = load_torch_file(unet_path)
    lora_sd = load_torch_file(lora_path)
    
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    # Convert LoRA
    converted_lora = convert_lora(lora_sd)
    
    # Generate key mappings
    key_map = model_lora_keys_unet(unet_model, {})
    
    print(f"📊 Converted LoRA keys: {len(converted_lora)}")
    print(f"📊 Generated key mappings: {len(key_map)}")
    
    # Show first 10 converted LoRA keys
    print(f"\n🔍 FIRST 10 CONVERTED LORA KEYS:")
    for i, key in enumerate(list(converted_lora.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Show first 10 generated mapping keys (left side)
    print(f"\n🔍 FIRST 10 GENERATED MAPPING KEYS:")
    for i, key in enumerate(list(key_map.keys())[:10]):
        print(f"  {i+1}. {key} -> {key_map[key]}")
    
    # Check if any converted LoRA keys match generated mapping keys
    matching_keys = set(converted_lora.keys()) & set(key_map.keys())
    print(f"\n🔍 MATCHING KEYS:")
    print(f"  Matching keys: {len(matching_keys)}")
    
    if matching_keys:
        print(f"  First 10 matching keys:")
        for i, key in enumerate(list(matching_keys)[:10]):
            print(f"    {i+1}. {key}")
    
    # Check patterns
    print(f"\n🔍 PATTERN ANALYSIS:")
    
    # LoRA key patterns
    lora_patterns = {}
    for key in converted_lora.keys():
        if '.lora_down.weight' in key:
            pattern = 'lora_down'
        elif '.lora_up.weight' in key:
            pattern = 'lora_up'
        elif '.diff_b' in key:
            pattern = 'diff_b'
        elif '.diff' in key:
            pattern = 'diff'
        elif '.alpha' in key:
            pattern = 'alpha'
        else:
            pattern = 'other'
        lora_patterns[pattern] = lora_patterns.get(pattern, 0) + 1
    
    print(f"  LoRA key patterns:")
    for pattern, count in lora_patterns.items():
        print(f"    {pattern}: {count}")
    
    # Mapping key patterns
    mapping_patterns = {}
    for key in key_map.keys():
        if key.startswith('lora_unet_'):
            pattern = 'lora_unet_*'
        elif key.startswith('diffusion_model.'):
            pattern = 'diffusion_model.*'
        elif '.' in key and not key.startswith('lora_'):
            pattern = 'direct_key'
        else:
            pattern = 'other'
        mapping_patterns[pattern] = mapping_patterns.get(pattern, 0) + 1
    
    print(f"  Mapping key patterns:")
    for pattern, count in mapping_patterns.items():
        print(f"    {pattern}: {count}")
    
    # Show some specific examples
    print(f"\n🔍 SPECIFIC EXAMPLES:")
    
    # Look for a specific LoRA key and see what mappings might match
    example_lora_key = "blocks.24.self_attn.o.lora_down.weight"
    if example_lora_key in converted_lora:
        print(f"  Example LoRA key: {example_lora_key}")
        
        # Show what mappings we have that might be related
        related_mappings = []
        for mapping_key in key_map.keys():
            if "blocks" in mapping_key and "self_attn" in mapping_key and "24" in mapping_key:
                related_mappings.append(mapping_key)
        
        print(f"  Related mappings found: {len(related_mappings)}")
        for mapping in related_mappings[:5]:
            print(f"    {mapping} -> {key_map[mapping]}")
        
        # Check what the expected model key should be
        expected_model_key = "blocks.24.self_attn.o.weight"
        if expected_model_key in unet_sd:
            print(f"  Expected model key exists: {expected_model_key}")
        else:
            print(f"  Expected model key missing: {expected_model_key}")

if __name__ == "__main__":
    debug_key_mapping_mismatch()
