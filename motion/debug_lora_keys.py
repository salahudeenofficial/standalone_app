#!/usr/bin/env python3
"""
Debug LoRA Key Mapping Issues
Test the key mapping between UNet and LoRA without requiring actual model files
"""

import torch
import logging
from lora import model_lora_keys_unet, model_lora_keys_clip, convert_lora
from standalone_model_patcher import create_model_patcher
from standalone_sd import WANModel, T5CLIPModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_dummy_unet_state_dict():
    """Create a dummy UNet state dict with typical WAN structure (empty prefix)"""
    sd = {}
    
    # Add typical WAN UNet keys WITHOUT diffusion_model prefix
    sd["head.modulation"] = torch.randn(1, 2, 1024)
    sd["head.head.weight"] = torch.randn(4096, 1024)
    sd["blocks.0.ffn.0.weight"] = torch.randn(4096, 1024)
    sd["patch_embedding.weight"] = torch.randn(1024, 3, 1, 2, 2)
    
    # Add some typical layer weights that would have LoRA
    sd["blocks.0.self_attn.q.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.self_attn.k.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.self_attn.v.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.self_attn.out.weight"] = torch.randn(1024, 1024)
    
    # Add cross-attention weights if present
    sd["blocks.0.cross_attn.q.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.cross_attn.k.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.cross_attn.v.weight"] = torch.randn(1024, 1024)
    sd["blocks.0.cross_attn.out.weight"] = torch.randn(1024, 1024)
    
    return sd

def create_dummy_clip_state_dict():
    """Create a dummy CLIP state dict with T5-XXL structure"""
    sd = {}
    
    # Add typical T5-XXL keys
    sd["t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight"] = torch.randn(4096, 4096)
    sd["t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight"] = torch.randn(4096, 4096)
    sd["t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.v.weight"] = torch.randn(4096, 4096)
    sd["t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.o.weight"] = torch.randn(4096, 4096)
    
    return sd

def create_dummy_lora_state_dict():
    """Create a dummy LoRA state dict with typical WAN LoRA keys"""
    sd = {}
    
    # Add typical WAN LoRA keys (these are the problematic ones)
    sd["lora_unet__blocks_0_self_attn_q.lora_down.weight"] = torch.randn(32, 1024)
    sd["lora_unet__blocks_0_self_attn_q.lora_up.weight"] = torch.randn(1024, 32)
    sd["lora_unet__blocks_0_self_attn_q.alpha"] = torch.tensor(32.0)
    
    sd["lora_unet__blocks_0_self_attn_k.lora_down.weight"] = torch.randn(32, 1024)
    sd["lora_unet__blocks_0_self_attn_k.lora_up.weight"] = torch.randn(1024, 32)
    sd["lora_unet__blocks_0_self_attn_k.alpha"] = torch.tensor(32.0)
    
    sd["lora_unet__blocks_0_cross_attn_q.lora_down.weight"] = torch.randn(32, 1024)
    sd["lora_unet__blocks_0_cross_attn_q.lora_up.weight"] = torch.randn(1024, 32)
    sd["lora_unet__blocks_0_cross_attn_q.alpha"] = torch.tensor(32.0)
    
    # Add some CLIP LoRA keys
    sd["lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_down.weight"] = torch.randn(32, 4096)
    sd["lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_up.weight"] = torch.randn(4096, 32)
    sd["lora_te_encoder_block_0_layer_0_SelfAttention_q.alpha"] = torch.tensor(32.0)
    
    return sd

def test_key_mapping():
    """Test the key mapping between models and LoRA"""
    print("🔧 DEBUGGING LORA KEY MAPPING")
    print("="*60)
    
    # Create dummy models
    print("📦 Creating dummy models...")
    unet_sd = create_dummy_unet_state_dict()
    clip_sd = create_dummy_clip_state_dict()
    lora_sd = create_dummy_lora_state_dict()
    
    # Create model objects
    unet_model = WANModel(unet_sd)
    clip_model = T5CLIPModel(clip_sd)
    
    # Create ModelPatcher for UNet
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    print(f"✅ Created UNet with {len(unet_sd)} keys")
    print(f"✅ Created CLIP with {len(clip_sd)} keys")
    print(f"✅ Created LoRA with {len(lora_sd)} keys")
    
    # Test UNet key mapping
    print("\n🔍 TESTING UNET KEY MAPPING:")
    print("-" * 40)
    unet_key_map = model_lora_keys_unet(unet_patcher)
    
    print("UNet Key Mappings:")
    for lora_key, model_key in unet_key_map.items():
        if "lora_unet" in lora_key:
            print(f"  {lora_key} -> {model_key}")
    
    # Test CLIP key mapping
    print("\n🔍 TESTING CLIP KEY MAPPING:")
    print("-" * 40)
    clip_key_map = model_lora_keys_clip(clip_model)
    
    print("CLIP Key Mappings:")
    for lora_key, model_key in clip_key_map.items():
        if "lora_te" in lora_key:
            print(f"  {lora_key} -> {model_key}")
    
    # Test LoRA conversion
    print("\n🔍 TESTING LORA CONVERSION:")
    print("-" * 40)
    converted_lora = convert_lora(lora_sd)
    
    print("Original LoRA keys:")
    for key in lora_sd.keys():
        print(f"  {key}")
    
    print("\nConverted LoRA keys:")
    for key in converted_lora.keys():
        print(f"  {key}")
    
    # Check for key mismatches
    print("\n🔍 CHECKING KEY MISMATCHES:")
    print("-" * 40)
    
    all_model_keys = set(unet_key_map.values()) | set(clip_key_map.values())
    all_lora_keys = set(converted_lora.keys())
    
    print(f"Total model keys: {len(all_model_keys)}")
    print(f"Total LoRA keys: {len(all_lora_keys)}")
    
    # Find matching keys
    matching_keys = set()
    for lora_key in all_lora_keys:
        if lora_key in unet_key_map or lora_key in clip_key_map:
            matching_keys.add(lora_key)
    
    print(f"Matching keys: {len(matching_keys)}")
    
    # Find missing keys
    missing_keys = all_lora_keys - matching_keys
    if missing_keys:
        print(f"\n❌ MISSING KEYS ({len(missing_keys)}):")
        for key in missing_keys:
            print(f"  {key}")
    else:
        print("\n✅ All LoRA keys have matching model keys!")
    
    # Find unused model keys
    used_model_keys = set()
    for lora_key in all_lora_keys:
        if lora_key in unet_key_map:
            used_model_keys.add(unet_key_map[lora_key])
        if lora_key in clip_key_map:
            used_model_keys.add(clip_key_map[lora_key])
    
    unused_model_keys = all_model_keys - used_model_keys
    if unused_model_keys:
        print(f"\n⚠️  UNUSED MODEL KEYS ({len(unused_model_keys)}):")
        for key in list(unused_model_keys)[:10]:  # Show first 10
            print(f"  {key}")
        if len(unused_model_keys) > 10:
            print(f"  ... and {len(unused_model_keys) - 10} more")
    
    return matching_keys, missing_keys, unused_model_keys

def analyze_specific_issue():
    """Analyze the specific WAN LoRA key issue with empty prefix"""
    print("\n🔍 ANALYZING SPECIFIC WAN LORA ISSUE (EMPTY PREFIX):")
    print("="*60)
    
    # The issue: WAN LoRA uses double underscores in keys, but model has empty prefix
    wan_lora_key = "lora_unet__blocks_0_self_attn_q.lora_down.weight"
    expected_model_key = "blocks.0.self_attn.q.weight"  # No diffusion_model prefix!
    
    print(f"WAN LoRA Key: {wan_lora_key}")
    print(f"Expected Model Key: {expected_model_key}")
    
    # Test the conversion
    test_sd = {wan_lora_key: torch.randn(32, 1024)}
    converted = convert_lora(test_sd)
    
    print(f"\nAfter conversion:")
    for key in converted.keys():
        print(f"  {key}")
    
    # Test key mapping with empty prefix model
    unet_sd = {"blocks.0.self_attn.q.weight": torch.randn(1024, 1024)}  # No prefix!
    unet_model = WANModel(unet_sd)
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    key_map = model_lora_keys_unet(unet_patcher)
    
    print(f"\nGenerated key mappings:")
    for lora_key, model_key in key_map.items():
        if "blocks_0_self_attn_q" in lora_key:
            print(f"  {lora_key} -> {model_key}")
    
    # Check if the mapping works
    converted_lora_key = "lora_unet_blocks_0_self_attn_q.lora_down.weight"
    if converted_lora_key in key_map:
        print(f"\n✅ SUCCESS: {converted_lora_key} -> {key_map[converted_lora_key]}")
    else:
        print(f"\n❌ FAILED: {converted_lora_key} not found in key mappings")

if __name__ == "__main__":
    matching_keys, missing_keys, unused_model_keys = test_key_mapping()
    analyze_specific_issue()
    
    print("\n" + "="*60)
    print("🎯 SUMMARY:")
    print(f"✅ Matching keys: {len(matching_keys)}")
    print(f"❌ Missing keys: {len(missing_keys)}")
    print(f"⚠️  Unused model keys: {len(unused_model_keys)}")
    
    if missing_keys:
        print("\n💡 RECOMMENDATION:")
        print("The LoRA key mapping issue is likely due to:")
        print("1. Double underscore conversion in convert_lora_wan()")
        print("2. Key format mismatch between LoRA and model keys")
        print("3. Missing key mappings in model_lora_keys_unet()")
    else:
        print("\n✅ Key mapping appears to be working correctly!")
