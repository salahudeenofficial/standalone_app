#!/usr/bin/env python3
"""
Comprehensive analysis of why we're getting 647 patches instead of 1053
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet, load_lora
from utils import load_torch_file

def analyze_missing_patches():
    """Analyze what patches are missing and why"""
    print("🔍 COMPREHENSIVE PATCH ANALYSIS")
    print("="*60)
    
    # Load models
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    unet_sd = load_torch_file(unet_path)
    lora_sd = load_torch_file(lora_path)
    
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    # Convert LoRA and create mappings
    converted_lora = convert_lora(lora_sd)
    key_map = model_lora_keys_unet(unet_model, {})
    
    print(f"📊 UNet parameters: {len(unet_sd)}")
    print(f"📊 LoRA keys: {len(converted_lora)}")
    print(f"📊 Key mappings: {len(key_map)}")
    
    # Analyze LoRA components
    lora_components = {
        'lora_down': [k for k in converted_lora.keys() if k.endswith('.lora_down.weight')],
        'lora_up': [k for k in converted_lora.keys() if k.endswith('.lora_up.weight')],
        'alpha': [k for k in converted_lora.keys() if k.endswith('.alpha')],
        'diff': [k for k in converted_lora.keys() if k.endswith('.diff')],
        'diff_b': [k for k in converted_lora.keys() if k.endswith('.diff_b')],
        'other': [k for k in converted_lora.keys() if not any(k.endswith(suffix) for suffix in ['.lora_down.weight', '.lora_up.weight', '.alpha', '.diff', '.diff_b'])]
    }
    
    print(f"\n🔍 LORA COMPONENT BREAKDOWN:")
    for component, keys in lora_components.items():
        print(f"  {component}: {len(keys)}")
    
    # Check which LoRA keys have mappings
    mapped_keys = set(converted_lora.keys()) & set(key_map.keys())
    unmapped_keys = set(converted_lora.keys()) - set(key_map.keys())
    
    print(f"\n🔍 MAPPING COVERAGE:")
    print(f"  Mapped LoRA keys: {len(mapped_keys)}")
    print(f"  Unmapped LoRA keys: {len(unmapped_keys)}")
    
    # Analyze unmapped keys by component
    unmapped_by_component = {}
    for component, keys in lora_components.items():
        unmapped_component_keys = set(keys) & unmapped_keys
        if unmapped_component_keys:
            unmapped_by_component[component] = unmapped_component_keys
    
    print(f"\n🔍 UNMAPPED KEYS BY COMPONENT:")
    for component, keys in unmapped_by_component.items():
        print(f"  {component}: {len(keys)}")
        if len(keys) <= 10:
            for key in list(keys)[:10]:
                print(f"    {key}")
        else:
            for key in list(keys)[:5]:
                print(f"    {key}")
            print(f"    ... and {len(keys)-5} more")
    
    # Check what model parameters exist
    weight_params = [k for k in unet_sd.keys() if k.endswith('.weight')]
    bias_params = [k for k in unet_sd.keys() if k.endswith('.bias')]
    
    print(f"\n🔍 MODEL PARAMETERS:")
    print(f"  Weight parameters: {len(weight_params)}")
    print(f"  Bias parameters: {len(bias_params)}")
    
    # For diff_b keys, check which ones have corresponding bias parameters
    if 'diff_b' in unmapped_by_component:
        print(f"\n🔍 DIFF_B KEY ANALYSIS:")
        missing_bias_params = []
        for diff_b_key in unmapped_by_component['diff_b']:
            expected_bias_key = diff_b_key.replace('.diff_b', '.bias')
            if expected_bias_key not in unet_sd:
                missing_bias_params.append((diff_b_key, expected_bias_key))
        
        print(f"  diff_b keys without corresponding bias: {len(missing_bias_params)}")
        for diff_b_key, expected_bias_key in missing_bias_params[:10]:
            print(f"    {diff_b_key} -> {expected_bias_key} (MISSING)")
    
    # Load patches and see what gets loaded
    loaded_patches = load_lora(converted_lora, key_map)
    print(f"\n🔍 ACTUAL LOADING RESULTS:")
    print(f"  Loaded patches: {len(loaded_patches)}")
    
    # Calculate theoretical maximum accounting for missing bias parameters
    complete_lora_sets = len(lora_components['lora_down'])  # Each down+up+alpha = 1 patch
    diff_patches = len(lora_components['diff'])
    valid_diff_b = len(lora_components['diff_b']) - len(missing_bias_params) if 'diff_b' in unmapped_by_component else len(lora_components['diff_b'])
    
    theoretical_max = complete_lora_sets + diff_patches + valid_diff_b
    print(f"  Theoretical max (accounting for missing bias): {theoretical_max}")
    print(f"  Loading efficiency: {len(loaded_patches)/theoretical_max*100:.1f}%")
    
    # Check if there are still issues with the loaded patches
    if len(loaded_patches) < theoretical_max:
        print(f"\n🔍 STILL MISSING PATCHES:")
        print(f"  Expected: {theoretical_max}")
        print(f"  Actual: {len(loaded_patches)}")
        print(f"  Missing: {theoretical_max - len(loaded_patches)}")

if __name__ == "__main__":
    analyze_missing_patches()
