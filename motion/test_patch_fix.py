#!/usr/bin/env python3
"""
Test the patch application fix
"""

import torch
from lora import load_lora_for_models, convert_lora
from standalone_sd import WANModel
from standalone_model_patcher import create_model_patcher

def test_patch_application():
    """Test that patches are actually applied to models"""
    print("🧪 TESTING PATCH APPLICATION FIX")
    print("="*50)
    
    # Create dummy model with some weights
    model_sd = {
        "blocks.0.self_attn.q.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.k.weight": torch.randn(1024, 1024),
        "blocks.0.self_attn.v.weight": torch.randn(1024, 1024),
    }
    
    # Create model and ModelPatcher
    unet_model = WANModel(model_sd)
    unet_patcher = create_model_patcher(unet_model, load_device="cpu")
    
    print(f"Original model has {len(model_sd)} weights")
    
    # Create dummy LoRA
    lora_sd = {
        "lora_unet__blocks_0_self_attn_q.lora_down.weight": torch.randn(32, 1024),
        "lora_unet__blocks_0_self_attn_q.lora_up.weight": torch.randn(1024, 32),
        "lora_unet__blocks_0_self_attn_q.alpha": torch.tensor(32.0),
    }
    
    print(f"LoRA has {len(lora_sd)} keys")
    
    # Apply LoRA
    new_unet, new_clip = load_lora_for_models(unet_patcher, None, lora_sd)
    
    if new_unet is not None:
        patches_count = len(new_unet.patches)
        print(f"\n✅ Patches applied: {patches_count}")
        
        if patches_count > 0:
            print("🎉 SUCCESS: Patches are being applied!")
            print("Patches:")
            for key, patches in new_unet.patches.items():
                print(f"  {key}: {len(patches)} patch(es)")
        else:
            print("❌ FAILED: No patches applied")
    else:
        print("❌ FAILED: Model is None")

if __name__ == "__main__":
    test_patch_application()
