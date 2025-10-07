#!/usr/bin/env python3
"""
Comprehensive CLIP Loading Debug Script
This script will help diagnose CLIP loading issues on Vast AI instance
"""

import os
import sys
import torch
import traceback

def debug_clip_loading():
    """Debug CLIP loading step by step"""
    
    print("=" * 80)
    print("🔍 CLIP LOADING DEBUG SCRIPT")
    print("=" * 80)
    
    # Step 1: Check basic environment
    print("\n📋 STEP 1: Environment Check")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Step 2: Check file existence
    print("\n📋 STEP 2: File Existence Check")
    clip_path = "./models/text_encoders/wan_clip_model.safetensors"
    print(f"CLIP path: {clip_path}")
    print(f"File exists: {os.path.exists(clip_path)}")
    
    if os.path.exists(clip_path):
        file_size = os.path.getsize(clip_path)
        print(f"File size: {file_size} bytes")
    else:
        print("❌ CLIP file does not exist!")
        return
    
    # Step 3: Test motion imports
    print("\n📋 STEP 3: Motion Module Imports")
    try:
        sys.path.insert(0, 'motion')
        import motion.utils
        print("✅ motion.utils imported")
        
        import motion.text_encoders.spiece_tokenizer
        print("✅ motion.text_encoders.spiece_tokenizer imported")
        
        import motion.text_encoders.wan
        print("✅ motion.text_encoders.wan imported")
        
        import motion.standalone_sd
        print("✅ motion.standalone_sd imported")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return
    
    # Step 4: Test CLIP data loading
    print("\n📋 STEP 4: CLIP Data Loading")
    try:
        clip_data = motion.utils.load_torch_file(clip_path, safe_load=True)
        print(f"✅ CLIP data loaded successfully")
        print(f"Number of keys: {len(clip_data)}")
        
        # Check for tokenizer-related keys
        tokenizer_keys = [k for k in clip_data.keys() if 'tokenizer' in k.lower() or 'spiece' in k.lower()]
        print(f"Tokenizer keys found: {tokenizer_keys}")
        
        # Check for spiece_model key specifically
        if 'spiece_model' in clip_data:
            spiece_data = clip_data['spiece_model']
            print(f"✅ spiece_model found")
            print(f"spiece_model type: {type(spiece_data)}")
            if hasattr(spiece_data, 'shape'):
                print(f"spiece_model shape: {spiece_data.shape}")
            if hasattr(spiece_data, 'dtype'):
                print(f"spiece_model dtype: {spiece_data.dtype}")
        else:
            print("❌ spiece_model key not found in CLIP data")
            print("Available keys (first 20):", list(clip_data.keys())[:20])
        
    except Exception as e:
        print(f"❌ CLIP data loading failed: {e}")
        traceback.print_exc()
        return
    
    # Step 5: Test SPieceTokenizer directly
    print("\n📋 STEP 5: SPieceTokenizer Direct Test")
    try:
        if 'spiece_model' in clip_data:
            spiece_data = clip_data['spiece_model']
            print(f"Testing SPieceTokenizer with spiece_model data...")
            
            # Test different ways of passing the data
            print("Method 1: Direct tensor")
            try:
                tokenizer1 = motion.text_encoders.spiece_tokenizer.SPieceTokenizer(spiece_data)
                print("✅ SPieceTokenizer created with direct tensor")
                
                # Test tokenization
                test_text = "hello world"
                result1 = tokenizer1(test_text)
                print(f"✅ Tokenization test successful: {result1}")
                
            except Exception as e:
                print(f"❌ Method 1 failed: {e}")
            
            print("Method 2: Convert to bytes")
            try:
                if torch.is_tensor(spiece_data):
                    spiece_bytes = spiece_data.numpy().tobytes()
                    tokenizer2 = motion.text_encoders.spiece_tokenizer.SPieceTokenizer(spiece_bytes)
                    print("✅ SPieceTokenizer created with bytes")
                    
                    # Test tokenization
                    result2 = tokenizer2(test_text)
                    print(f"✅ Tokenization test successful: {result2}")
                
            except Exception as e:
                print(f"❌ Method 2 failed: {e}")
        
    except Exception as e:
        print(f"❌ SPieceTokenizer test failed: {e}")
        traceback.print_exc()
    
    # Step 6: Test UMT5XXlTokenizer
    print("\n📋 STEP 6: UMT5XXlTokenizer Test")
    try:
        from motion.text_encoders.wan import UMT5XXlTokenizer
        
        print("Creating UMT5XXlTokenizer...")
        tokenizer = UMT5XXlTokenizer(tokenizer_data=clip_data)
        print("✅ UMT5XXlTokenizer created successfully")
        
    except Exception as e:
        print(f"❌ UMT5XXlTokenizer creation failed: {e}")
        traceback.print_exc()
    
    # Step 7: Test full CLIP loading chain
    print("\n📋 STEP 7: Full CLIP Loading Chain Test")
    try:
        from motion.comps import CLIPLoader
        
        print("Creating CLIPLoader...")
        clip_loader = CLIPLoader("wan_clip_model.safetensors")
        print("✅ CLIPLoader created")
        
        print("Loading CLIP...")
        clip = clip_loader.load_clip()
        print("✅ CLIP loaded successfully!")
        
        # Test encoding
        from motion.comps import CLIPTextEncode
        clip_encode = CLIPTextEncode(clip)
        result = clip_encode.encode("test prompt")
        print(f"✅ CLIP encoding test successful: {result.shape if hasattr(result, 'shape') else type(result)}")
        
    except Exception as e:
        print(f"❌ Full CLIP loading failed: {e}")
        traceback.print_exc()
    
    # Step 8: Check tokenizer data format
    print("\n📋 STEP 8: Tokenizer Data Format Analysis")
    if 'spiece_model' in clip_data:
        spiece_data = clip_data['spiece_model']
        print(f"spiece_model analysis:")
        print(f"  Type: {type(spiece_data)}")
        print(f"  Is tensor: {torch.is_tensor(spiece_data)}")
        
        if torch.is_tensor(spiece_data):
            print(f"  Shape: {spiece_data.shape}")
            print(f"  Dtype: {spiece_data.dtype}")
            print(f"  Device: {spiece_data.device}")
            print(f"  First 10 values: {spiece_data.flatten()[:10].tolist()}")
            
            # Try to convert to bytes
            try:
                spiece_bytes = spiece_data.numpy().tobytes()
                print(f"  Bytes length: {len(spiece_bytes)}")
                print(f"  First 20 bytes: {list(spiece_bytes[:20])}")
            except Exception as e:
                print(f"  ❌ Failed to convert to bytes: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    debug_clip_loading()
