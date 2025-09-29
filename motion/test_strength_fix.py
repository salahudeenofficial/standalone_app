#!/usr/bin/env python3
"""
Test the strength=0.0 fix for CLIP patches
"""

import torch
from standalone_sd import T5CLIPModel, StandaloneCLIP

def test_strength_fix():
    """Test that strength=0.0 doesn't add patches to CLIP"""
    print("🧪 TESTING STRENGTH=0.0 FIX")
    print("="*50)
    
    # Create a simple CLIP model
    clip_sd = {
        "t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(4096, 4096),
    }
    
    clip_model = StandaloneCLIP(T5CLIPModel(clip_sd))
    
    # Create some patches
    patches = {
        "lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_down.weight": torch.randn(32, 4096),
        "lora_te_encoder_block_0_layer_0_SelfAttention_q.lora_up.weight": torch.randn(4096, 32),
    }
    
    print(f"Initial patches count: {len(clip_model.patches)}")
    
    # Test with strength=0.0
    applied_keys = clip_model.add_patches(patches, strength=0.0)
    
    print(f"Applied keys with strength=0.0: {len(applied_keys)}")
    print(f"Final patches count: {len(clip_model.patches)}")
    
    if len(applied_keys) == 0 and len(clip_model.patches) == 0:
        print("✅ SUCCESS: No patches added when strength=0.0")
    else:
        print("❌ FAILED: Patches were added despite strength=0.0")
    
    # Test with strength=1.0
    applied_keys = clip_model.add_patches(patches, strength=1.0)
    
    print(f"Applied keys with strength=1.0: {len(applied_keys)}")
    print(f"Final patches count: {len(clip_model.patches)}")
    
    if len(applied_keys) > 0 and len(clip_model.patches) > 0:
        print("✅ SUCCESS: Patches added when strength=1.0")
    else:
        print("❌ FAILED: No patches added when strength=1.0")

if __name__ == "__main__":
    test_strength_fix()
