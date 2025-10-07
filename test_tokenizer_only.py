#!/usr/bin/env python3
"""
Simple Tokenizer Test Script
Focuses specifically on the tokenizer loading issue
"""

import os
import sys
import torch

def test_tokenizer_only():
    """Test just the tokenizer loading part"""
    
    print("🔍 TOKENIZER ONLY TEST")
    print("=" * 50)
    
    # Add motion to path
    sys.path.insert(0, 'motion')
    
    # Check if CLIP file exists
    clip_path = "./models/text_encoders/wan_clip_model.safetensors"
    if not os.path.exists(clip_path):
        print(f"❌ CLIP file not found: {clip_path}")
        return
    
    print(f"✅ CLIP file found: {clip_path}")
    
    # Load CLIP data
    try:
        from motion.utils import load_torch_file
        clip_data = load_torch_file(clip_path, safe_load=True)
        print(f"✅ CLIP data loaded, {len(clip_data)} keys")
    except Exception as e:
        print(f"❌ Failed to load CLIP data: {e}")
        return
    
    # Check for spiece_model
    if 'spiece_model' not in clip_data:
        print("❌ spiece_model not found in CLIP data")
        print("Available keys:", list(clip_data.keys())[:10])
        return
    
    spiece_data = clip_data['spiece_model']
    print(f"✅ spiece_model found: {type(spiece_data)}")
    
    if torch.is_tensor(spiece_data):
        print(f"  Shape: {spiece_data.shape}")
        print(f"  Dtype: {spiece_data.dtype}")
    
    # Test SPieceTokenizer creation
    try:
        from motion.text_encoders.spiece_tokenizer import SPieceTokenizer
        
        print("\n🧪 Testing SPieceTokenizer creation...")
        
        # Method 1: Direct tensor
        print("Method 1: Direct tensor")
        tokenizer1 = SPieceTokenizer(spiece_data)
        print("✅ SPieceTokenizer created with tensor")
        
        # Test tokenization
        result1 = tokenizer1("hello world")
        print(f"✅ Tokenization result: {result1}")
        
    except Exception as e:
        print(f"❌ SPieceTokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test UMT5XXlTokenizer
    try:
        from motion.text_encoders.wan import UMT5XXlTokenizer
        
        print("\n🧪 Testing UMT5XXlTokenizer creation...")
        tokenizer2 = UMT5XXlTokenizer(tokenizer_data=clip_data)
        print("✅ UMT5XXlTokenizer created successfully")
        
    except Exception as e:
        print(f"❌ UMT5XXlTokenizer test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer_only()
