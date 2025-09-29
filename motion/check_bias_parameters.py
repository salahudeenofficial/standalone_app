#!/usr/bin/env python3
"""
Check if UNet model has bias parameters for the missing diff_b keys
"""

import torch
from standalone_sd import load_state_dict_guess_config
from utils import load_torch_file

def check_bias_parameters():
    """Check if UNet model has bias parameters"""
    print("🔍 CHECKING BIAS PARAMETERS IN UNET MODEL")
    print("="*60)
    
    # Load UNet model
    unet_path = './models/diffusion_models/wan_2.1_diffusion_model.safetensors'
    unet_sd = load_torch_file(unet_path)
    
    print(f"📊 Total UNet keys: {len(unet_sd)}")
    
    # Count different types of parameters
    weight_keys = [k for k in unet_sd.keys() if k.endswith('.weight')]
    bias_keys = [k for k in unet_sd.keys() if k.endswith('.bias')]
    other_keys = [k for k in unet_sd.keys() if not k.endswith('.weight') and not k.endswith('.bias')]
    
    print(f"📊 Weight parameters: {len(weight_keys)}")
    print(f"📊 Bias parameters: {len(bias_keys)}")
    print(f"📊 Other parameters: {len(other_keys)}")
    
    # Check specific missing keys from the warning
    missing_keys = [
        'blocks.6.self_attn.o.diff_b',
        'blocks.4.self_attn.o.diff_b', 
        'blocks.26.ffn.2.diff_b',
        'blocks.9.cross_attn.v.diff_b',
        'blocks.24.cross_attn.o.diff_b'
    ]
    
    print(f"\n🔍 CHECKING MISSING DIFF_B KEYS:")
    for missing_key in missing_keys:
        # Convert diff_b key to expected bias key
        expected_bias_key = missing_key.replace('.diff_b', '.bias')
        has_bias = expected_bias_key in unet_sd
        
        # Also check if there's a weight parameter
        expected_weight_key = missing_key.replace('.diff_b', '.weight')
        has_weight = expected_weight_key in unet_sd
        
        print(f"  {missing_key}")
        print(f"    Expected bias key: {expected_bias_key} - {'✅ EXISTS' if has_bias else '❌ MISSING'}")
        print(f"    Expected weight key: {expected_weight_key} - {'✅ EXISTS' if has_weight else '❌ MISSING'}")
        
        if not has_bias and not has_weight:
            print(f"    ⚠️  Neither weight nor bias parameter exists!")
        elif not has_bias:
            print(f"    💡 Weight exists but no bias - this is normal for some layers")
    
    # Show some examples of bias parameters that do exist
    print(f"\n🔍 EXAMPLES OF EXISTING BIAS PARAMETERS:")
    for i, bias_key in enumerate(bias_keys[:10]):
        print(f"  {i+1}. {bias_key}")
    
    # Check if there are any diff_b keys in the LoRA that should map to existing bias parameters
    print(f"\n🔍 CHECKING LORA STRUCTURE:")
    lora_path = './models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors'
    lora_sd = load_torch_file(lora_path)
    
    diff_b_keys = [k for k in lora_sd.keys() if k.endswith('.diff_b')]
    print(f"📊 Total diff_b keys in LoRA: {len(diff_b_keys)}")
    
    # Check how many diff_b keys have corresponding bias parameters in the model
    matching_bias = 0
    for diff_b_key in diff_b_keys:
        expected_bias_key = diff_b_key.replace('.diff_b', '.bias')
        if expected_bias_key in unet_sd:
            matching_bias += 1
    
    print(f"📊 diff_b keys with matching bias parameters: {matching_bias}")
    print(f"📊 diff_b keys without matching bias parameters: {len(diff_b_keys) - matching_bias}")

if __name__ == "__main__":
    check_bias_parameters()
