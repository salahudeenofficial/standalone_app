#!/usr/bin/env python3
"""
Test the fix for proper key mapping separation
"""

import torch
from lora import load_lora_for_models, convert_lora, load_lora
from standalone_sd import WANModel, T5CLIPModel
from standalone_model_patcher import create_model_patcher

def test_key_mapping_separation():
    """Test that key mapping works correctly for UNet and CLIP separation"""
    print("🧪 TESTING KEY MAPPING SEPARATION")
    print("="*60)
    
    # Create UNet model
    unet_sd = {
        "blocks.0.self_attn.q.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.k.weight": torch.randn(1024, 1024),
        "blocks.0.cross_attn.q.weight": torch.randn(1024, 1024),
    }
    
    unet_model = WANModel(unet_sd)
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    # Create CLIP model
    clip_sd = {
        "t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(4096, 4096),
    }
    
    clip_model = T5CLIPModel(clip_sd)
    
    # Create LoRA with both UNet and CLIP keys
    lora_sd = {
        # UNet keys (should be applied)
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.randn(32, 1024),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.randn(1024, 32),
        "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(32.0),
        
        # CLIP keys (should NOT be applied with strength_clip=0.0)
        "lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_down.weight": torch.randn(32, 4096),
        "lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_up.weight": torch.randn(4096, 32),
        "lora_te_encoder_block_0_layer_0_SelfAttention_q.alpha": torch.tensor(32.0),
    }
    
    print(f"UNet model keys: {len(unet_sd)}")
    print(f"CLIP model keys: {len(clip_sd)}")
    print(f"LoRA keys: {len(lora_sd)}")
    
    # Test conversion
    print(f"\n🔧 TESTING CONVERSION:")
    converted_lora = convert_lora(lora_sd)
    print("Converted LoRA keys:")
    for key in converted_lora.keys():
        print(f"  {key}")
    
    # Test LoRA application
    print(f"\n🔧 TESTING LORA APPLICATION:")
    new_unet, new_clip = load_lora_for_models(unet_patcher, clip_model, lora_sd)
    
    if new_unet is not None:
        unet_patches = len(new_unet.patches)
        print(f"UNet patches: {unet_patches}")
        if unet_patches > 0:
            print("UNet patches:")
            for key, patches in new_unet.patches.items():
                print(f"  {key}: {len(patches)} patch(es)")
    else:
        print("UNet is None")
    
    if new_clip is not None:
        clip_patches = len(new_clip.patches)
        print(f"CLIP patches: {clip_patches}")
        if clip_patches > 0:
            print("CLIP patches:")
            for key, patches in new_clip.patches.items():
                print(f"  {key}: {len(patches)} patch(es)")
    else:
        print("CLIP is None")
    
    # Check results
    print(f"\n🔍 RESULTS:")
    if new_unet is not None and new_clip is not None:
        unet_patches = len(new_unet.patches)
        clip_patches = len(new_clip.patches)
        
        print(f"UNet patches: {unet_patches}")
        print(f"CLIP patches: {clip_patches}")
        
        if unet_patches > 0 and clip_patches == 0:
            print("🎉 SUCCESS: LoRA applied only to UNet!")
        elif unet_patches > 0 and clip_patches > 0:
            print("❌ FAILED: LoRA applied to both UNet and CLIP")
        elif unet_patches == 0 and clip_patches == 0:
            print("❌ FAILED: No patches applied to either model")
        else:
            print("❌ FAILED: Unexpected result")
    else:
        print("❌ FAILED: One or both models are None")

if __name__ == "__main__":
    test_key_mapping_separation()