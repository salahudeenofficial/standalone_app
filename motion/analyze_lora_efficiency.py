#!/usr/bin/env python3
"""
Simple analysis of LoRA loading efficiency
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet, load_lora
from utils import load_torch_file

def analyze_lora_efficiency():
    """Analyze LoRA loading efficiency"""
    print("🔍 LORA LOADING EFFICIENCY ANALYSIS")
    print("="*50)
    
    # Load models
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    
    unet_sd = load_torch_file(unet_path)
    lora_sd = load_torch_file(lora_path)
    
    result = load_state_dict_guess_config(unet_sd, output_model=True, output_clip=False)
    unet_model = result[0]
    
    print(f"📊 UNet parameters: {len(unet_sd)}")
    print(f"📊 LoRA keys: {len(lora_sd)}")
    
    # Convert LoRA
    converted_lora = convert_lora(lora_sd)
    print(f"📊 Converted LoRA keys: {len(converted_lora)}")
    
    # Analyze LoRA key types
    lora_down_keys = [k for k in converted_lora.keys() if k.endswith('.lora_down.weight')]
    lora_up_keys = [k for k in converted_lora.keys() if k.endswith('.lora_up.weight')]
    alpha_keys = [k for k in converted_lora.keys() if k.endswith('.alpha')]
    diff_b_keys = [k for k in converted_lora.keys() if k.endswith('.diff_b')]
    diff_keys = [k for k in converted_lora.keys() if k.endswith('.diff')]
    other_keys = [k for k in converted_lora.keys() if not any(k.endswith(suffix) for suffix in ['.lora_down.weight', '.lora_up.weight', '.alpha', '.diff_b', '.diff'])]
    
    print(f"\n🔍 LORA KEY BREAKDOWN:")
    print(f"  lora_down.weight: {len(lora_down_keys)}")
    print(f"  lora_up.weight: {len(lora_up_keys)}")
    print(f"  alpha: {len(alpha_keys)}")
    print(f"  diff_b: {len(diff_b_keys)}")
    print(f"  diff: {len(diff_keys)}")
    print(f"  other: {len(other_keys)}")
    
    # Check how many complete LoRA sets we have
    # Each complete LoRA should have lora_down + lora_up + alpha
    base_keys = set()
    for key in lora_down_keys:
        base_key = key[:-len('.lora_down.weight')]
        base_keys.add(base_key)
    
    complete_lora_sets = 0
    for base_key in base_keys:
        has_down = f"{base_key}.lora_down.weight" in converted_lora
        has_up = f"{base_key}.lora_up.weight" in converted_lora
        has_alpha = f"{base_key}.alpha" in converted_lora
        if has_down and has_up:
            complete_lora_sets += 1
    
    print(f"\n🔍 COMPLETE LORA SETS:")
    print(f"  Complete LoRA sets (down+up): {complete_lora_sets}")
    print(f"  Expected patches from LoRA: {complete_lora_sets}")
    print(f"  Additional diff patches: {len(diff_keys)}")
    print(f"  Additional diff_b patches: {len(diff_b_keys)}")
    print(f"  Theoretical max patches: {complete_lora_sets + len(diff_keys) + len(diff_b_keys)}")
    
    # Check how many bias parameters the model actually has
    model_bias_keys = [k for k in unet_sd.keys() if k.endswith('.bias')]
    print(f"\n🔍 MODEL STRUCTURE:")
    print(f"  Model bias parameters: {len(model_bias_keys)}")
    
    # Check how many diff_b keys have corresponding bias parameters
    valid_diff_b = 0
    for diff_b_key in diff_b_keys:
        expected_bias_key = diff_b_key.replace('.diff_b', '.bias')
        if expected_bias_key in unet_sd:
            valid_diff_b += 1
    
    print(f"  diff_b keys with valid bias targets: {valid_diff_b}")
    print(f"  diff_b keys without valid bias targets: {len(diff_b_keys) - valid_diff_b}")
    
    # Refined theoretical max
    refined_max = complete_lora_sets + len(diff_keys) + valid_diff_b
    print(f"  Refined theoretical max patches: {refined_max}")
    
    # Load LoRA and compare
    key_map = model_lora_keys_unet(unet_model, {})
    loaded_patches = load_lora(converted_lora, key_map)
    
    print(f"\n🔍 ACTUAL RESULTS:")
    print(f"  Actually loaded patches: {len(loaded_patches)}")
    print(f"  Loading efficiency: {len(loaded_patches)/refined_max*100:.1f}%")
    
    # Check why we might be missing patches
    if len(loaded_patches) < refined_max:
        print(f"\n🔍 MISSING PATCHES ANALYSIS:")
        print(f"  Expected but missing: {refined_max - len(loaded_patches)}")
        
        # This would require checking which specific LoRA components aren't being loaded
        mapped_lora_keys = set(key_map.keys()) & set(converted_lora.keys())
        unmapped_lora_keys = set(converted_lora.keys()) - set(key_map.keys())
        
        print(f"  LoRA keys with mappings: {len(mapped_lora_keys)}")
        print(f"  LoRA keys without mappings: {len(unmapped_lora_keys)}")
        
        if unmapped_lora_keys:
            print(f"  First 10 unmapped keys:")
            for i, key in enumerate(list(unmapped_lora_keys)[:10]):
                print(f"    {i+1}. {key}")

if __name__ == "__main__":
    analyze_lora_efficiency()
