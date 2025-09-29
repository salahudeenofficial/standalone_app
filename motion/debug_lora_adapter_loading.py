#!/usr/bin/env python3
"""
Debug LoRA adapter loading specifically
"""

import torch
from standalone_sd import load_state_dict_guess_config
from lora import convert_lora, model_lora_keys_unet, load_lora, LoRAAdapter
from utils import load_torch_file

def debug_lora_adapter_loading():
    """Debug LoRA adapter loading process"""
    print("🔍 DEBUGGING LORA ADAPTER LOADING")
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
    
    print(f"📊 Converted LoRA keys: {len(converted_lora)}")
    print(f"📊 Key mappings: {len(key_map)}")
    
    # Test LoRA adapter loading for a specific example
    test_base_key = "blocks.0.cross_attn.k"
    
    print(f"\n🔍 TESTING LORA ADAPTER FOR: {test_base_key}")
    
    # Check if the required LoRA components exist
    required_keys = [
        f"{test_base_key}.lora_down.weight",
        f"{test_base_key}.lora_up.weight",
        f"{test_base_key}.alpha"
    ]
    
    print(f"  Required LoRA components:")
    for key in required_keys:
        exists = key in converted_lora
        print(f"    {key}: {'✅' if exists else '❌'}")
    
    # Try to load the adapter directly
    alpha = None
    alpha_key = f"{test_base_key}.alpha"
    if alpha_key in converted_lora:
        alpha = converted_lora[alpha_key].item()
    
    print(f"  Alpha value: {alpha}")
    
    # Test LoRAAdapter.load directly
    loaded_keys = set()
    adapter = LoRAAdapter.load(test_base_key, converted_lora, alpha, None, loaded_keys)
    
    if adapter is not None:
        print(f"  ✅ LoRA adapter loaded successfully")
        print(f"  Loaded keys: {len(loaded_keys)}")
        for key in loaded_keys:
            print(f"    {key}")
    else:
        print(f"  ❌ LoRA adapter failed to load")
    
    # Check what keys are being processed in load_lora
    print(f"\n🔍 KEYS BEING PROCESSED IN load_lora:")
    
    # Get the first 20 keys from to_load
    to_load_keys = list(key_map.keys())[:20]
    processed_base_keys = set()
    
    for i, x in enumerate(to_load_keys):
        base_key = x
        is_lora_component = False
        
        if x.endswith('.lora_down.weight') or x.endswith('.lora_up.weight') or x.endswith('.alpha'):
            is_lora_component = True
            if x.endswith('.lora_down.weight'):
                base_key = x[:-len('.lora_down.weight')]
            elif x.endswith('.lora_up.weight'):
                base_key = x[:-len('.lora_up.weight')]
            elif x.endswith('.alpha'):
                base_key = x[:-len('.alpha')]
        
        skip_processing = base_key in processed_base_keys if is_lora_component else False
        if is_lora_component:
            processed_base_keys.add(base_key)
        
        print(f"  {i+1:2d}. {x}")
        print(f"      → base_key: {base_key}")
        print(f"      → is_lora_component: {is_lora_component}")
        print(f"      → skip_processing: {skip_processing}")
        print()
    
    # Test the actual load_lora function
    print(f"\n🔍 TESTING load_lora FUNCTION:")
    loaded_patches = load_lora(converted_lora, key_map)
    print(f"  Loaded patches: {len(loaded_patches)}")
    
    # Count different types of loaded patches
    lora_patches = 0
    diff_patches = 0
    diff_b_patches = 0
    other_patches = 0
    
    for model_key, patch in loaded_patches.items():
        if hasattr(patch, 'weights') and len(patch.weights) >= 2:  # LoRA adapter
            lora_patches += 1
        elif isinstance(patch, tuple) and patch[0] == "diff":
            if model_key.endswith('.bias'):
                diff_b_patches += 1
            else:
                diff_patches += 1
        else:
            other_patches += 1
    
    print(f"  LoRA adapters: {lora_patches}")
    print(f"  Diff patches: {diff_patches}")
    print(f"  Diff_b patches: {diff_b_patches}")
    print(f"  Other patches: {other_patches}")
    print(f"  Total: {lora_patches + diff_patches + diff_b_patches + other_patches}")

if __name__ == "__main__":
    debug_lora_adapter_loading()
