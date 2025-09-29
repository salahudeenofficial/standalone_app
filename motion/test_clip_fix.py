#!/usr/bin/env python3
"""
Simple test to verify the CLIP model fix
"""

def test_clip_model_fix():
    """Test that the CLIP model fix works"""
    print("🧪 TESTING CLIP MODEL FIX")
    print("="*50)
    
    # Test the fix by checking if the function call works
    try:
        from lora import model_lora_keys_clip
        from standalone_sd import T5CLIPModel
        import torch
        
        # Create a simple CLIP model
        clip_sd = {
            "t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight": torch.randn(4096, 4096),
        }
        
        clip_model = T5CLIPModel(clip_sd)
        
        # Test the function call that was failing
        key_map = model_lora_keys_clip(clip_model, {})
        
        print(f"✅ SUCCESS: model_lora_keys_clip works with T5CLIPModel")
        print(f"Generated {len(key_map)} key mappings")
        
        # Show some mappings
        for i, (lora_key, model_key) in enumerate(list(key_map.items())[:3]):
            print(f"  {lora_key} -> {model_key}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip_model_fix()
