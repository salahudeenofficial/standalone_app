#!/usr/bin/env python3
"""
Side-by-Side Motion vs ComfyUI Debug Script
This script will compare Motion and ComfyUI implementations step by step
to pinpoint exactly where the error occurs.
"""

import os
import sys
import torch
import traceback
import json

def debug_side_by_side():
    """Debug Motion vs ComfyUI side by side"""
    
    print("=" * 100)
    print("🔍 SIDE-BY-SIDE MOTION vs COMFYUI DEBUG SCRIPT")
    print("=" * 100)
    
    # Step 1: Environment Setup
    print("\n📋 STEP 1: Environment Setup")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Add both paths
    sys.path.insert(0, 'motion')
    sys.path.insert(0, '/home/fashionx/comfy/ComfyUI')
    
    # Step 2: Import Test
    print("\n📋 STEP 2: Import Test")
    try:
        # Motion imports
        import motion.utils
        import motion.text_encoders.spiece_tokenizer as motion_spiece
        import motion.text_encoders.wan as motion_wan
        import motion.standalone_sd as motion_sd
        print("✅ Motion imports successful")
        
        # ComfyUI imports
        import comfy.utils
        import comfy.text_encoders.spiece_tokenizer as comfy_spiece
        import comfy.text_encoders.wan as comfy_wan
        import comfy.sd as comfy_sd
        print("✅ ComfyUI imports successful")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return
    
    # Step 3: CLIP File Loading Comparison
    print("\n📋 STEP 3: CLIP File Loading Comparison")
    clip_path = "./models/text_encoders/wan_clip_model.safetensors"
    
    try:
        # Motion loading
        print("Loading with Motion...")
        motion_clip_data = motion.utils.load_torch_file(clip_path, safe_load=True)
        print(f"✅ Motion loaded: {len(motion_clip_data)} keys")
        
        # ComfyUI loading
        print("Loading with ComfyUI...")
        comfy_clip_data = comfy.utils.load_torch_file(clip_path, safe_load=True)
        print(f"✅ ComfyUI loaded: {len(comfy_clip_data)} keys")
        
        # Compare keys
        motion_keys = set(motion_clip_data.keys())
        comfy_keys = set(comfy_clip_data.keys())
        
        print(f"Motion keys: {len(motion_keys)}")
        print(f"ComfyUI keys: {len(comfy_keys)}")
        print(f"Keys match: {motion_keys == comfy_keys}")
        
        if motion_keys != comfy_keys:
            print("Key differences:")
            print(f"Motion only: {motion_keys - comfy_keys}")
            print(f"ComfyUI only: {comfy_keys - motion_keys}")
        
        # Check spiece_model specifically
        print("\n🔍 SPiece Model Analysis:")
        print(f"Motion has 'spiece_model': {'spiece_model' in motion_clip_data}")
        print(f"ComfyUI has 'spiece_model': {'spiece_model' in comfy_clip_data}")
        
        if 'spiece_model' in motion_clip_data:
            motion_spiece = motion_clip_data['spiece_model']
            print(f"Motion spiece_model type: {type(motion_spiece)}")
            if torch.is_tensor(motion_spiece):
                print(f"Motion spiece_model shape: {motion_spiece.shape}")
                print(f"Motion spiece_model dtype: {motion_spiece.dtype}")
        
        if 'spiece_model' in comfy_clip_data:
            comfy_spiece = comfy_clip_data['spiece_model']
            print(f"ComfyUI spiece_model type: {type(comfy_spiece)}")
            if torch.is_tensor(comfy_spiece):
                print(f"ComfyUI spiece_model shape: {comfy_spiece.shape}")
                print(f"ComfyUI spiece_model dtype: {comfy_spiece.dtype}")
        
    except Exception as e:
        print(f"❌ CLIP loading failed: {e}")
        traceback.print_exc()
        return
    
    # Step 4: SPieceTokenizer Direct Comparison
    print("\n📋 STEP 4: SPieceTokenizer Direct Comparison")
    
    if 'spiece_model' in motion_clip_data:
        spiece_data = motion_clip_data['spiece_model']
        
        try:
            print("Testing Motion SPieceTokenizer...")
            motion_tokenizer = motion_spiece.SPieceTokenizer(spiece_data)
            print("✅ Motion SPieceTokenizer created")
            
            # Test tokenization
            test_text = "hello world"
            motion_result = motion_tokenizer(test_text)
            print(f"✅ Motion tokenization: {motion_result}")
            
        except Exception as e:
            print(f"❌ Motion SPieceTokenizer failed: {e}")
            traceback.print_exc()
        
        try:
            print("Testing ComfyUI SPieceTokenizer...")
            comfy_tokenizer = comfy_spiece.SPieceTokenizer(spiece_data)
            print("✅ ComfyUI SPieceTokenizer created")
            
            # Test tokenization
            comfy_result = comfy_tokenizer(test_text)
            print(f"✅ ComfyUI tokenization: {comfy_result}")
            
        except Exception as e:
            print(f"❌ ComfyUI SPieceTokenizer failed: {e}")
            traceback.print_exc()
    
    # Step 5: UMT5XXlTokenizer Comparison
    print("\n📋 STEP 5: UMT5XXlTokenizer Comparison")
    
    try:
        print("Testing Motion UMT5XXlTokenizer...")
        motion_tokenizer_data = {"spiece_model": motion_clip_data.get("spiece_model")}
        motion_umt5 = motion_wan.UMT5XXlTokenizer(tokenizer_data=motion_tokenizer_data)
        print("✅ Motion UMT5XXlTokenizer created")
        
    except Exception as e:
        print(f"❌ Motion UMT5XXlTokenizer failed: {e}")
        traceback.print_exc()
    
    try:
        print("Testing ComfyUI UMT5XXlTokenizer...")
        comfy_tokenizer_data = {"spiece_model": comfy_clip_data.get("spiece_model")}
        comfy_umt5 = comfy_wan.UMT5XXlTokenizer(tokenizer_data=comfy_tokenizer_data)
        print("✅ ComfyUI UMT5XXlTokenizer created")
        
    except Exception as e:
        print(f"❌ ComfyUI UMT5XXlTokenizer failed: {e}")
        traceback.print_exc()
    
    # Step 6: WanT5Tokenizer Comparison
    print("\n📋 STEP 6: WanT5Tokenizer Comparison")
    
    try:
        print("Testing Motion WanT5Tokenizer...")
        motion_wan_tokenizer = motion_wan.WanT5Tokenizer(tokenizer_data=motion_tokenizer_data)
        print("✅ Motion WanT5Tokenizer created")
        
    except Exception as e:
        print(f"❌ Motion WanT5Tokenizer failed: {e}")
        traceback.print_exc()
    
    try:
        print("Testing ComfyUI WanT5Tokenizer...")
        comfy_wan_tokenizer = comfy_wan.WanT5Tokenizer(tokenizer_data=comfy_tokenizer_data)
        print("✅ ComfyUI WanT5Tokenizer created")
        
    except Exception as e:
        print(f"❌ ComfyUI WanT5Tokenizer failed: {e}")
        traceback.print_exc()
    
    # Step 7: CLIP Loading Chain Comparison
    print("\n📋 STEP 7: CLIP Loading Chain Comparison")
    
    try:
        print("Testing Motion CLIP loading...")
        motion_clip = motion_sd.load_clip([clip_path], clip_type=motion_sd.CLIPType.WAN)
        print("✅ Motion CLIP loaded successfully")
        
    except Exception as e:
        print(f"❌ Motion CLIP loading failed: {e}")
        traceback.print_exc()
    
    try:
        print("Testing ComfyUI CLIP loading...")
        comfy_clip = comfy_sd.load_clip([clip_path], clip_type=comfy_sd.CLIPType.WAN)
        print("✅ ComfyUI CLIP loaded successfully")
        
    except Exception as e:
        print(f"❌ ComfyUI CLIP loading failed: {e}")
        traceback.print_exc()
    
    # Step 8: Detailed Error Analysis
    print("\n📋 STEP 8: Detailed Error Analysis")
    
    # Check if the error occurs in the same place
    print("Checking where the error occurs...")
    
    # Test the exact sequence that fails
    try:
        print("Testing Motion exact sequence...")
        
        # This is the exact sequence from the traceback
        clip_data = [motion_clip_data]
        tokenizer_data = {}
        
        # Extract spiece_model
        for c in clip_data:
            if 'spiece_model' in c:
                tokenizer_data['spiece_model'] = c['spiece_model']
        
        print(f"Tokenizer data keys: {list(tokenizer_data.keys())}")
        print(f"spiece_model in tokenizer_data: {'spiece_model' in tokenizer_data}")
        
        if 'spiece_model' in tokenizer_data:
            spiece_model = tokenizer_data['spiece_model']
            print(f"spiece_model type: {type(spiece_model)}")
            print(f"spiece_model is None: {spiece_model is None}")
            
            if spiece_model is not None:
                # Test the exact call that fails
                print("Testing UMT5XXlTokenizer creation...")
                umt5 = motion_wan.UMT5XXlTokenizer(tokenizer_data=tokenizer_data)
                print("✅ UMT5XXlTokenizer created successfully")
                
                # Test the exact call that fails in SD1Tokenizer
                print("Testing WanT5Tokenizer creation...")
                wan_tokenizer = motion_wan.WanT5Tokenizer(tokenizer_data=tokenizer_data)
                print("✅ WanT5Tokenizer created successfully")
        
    except Exception as e:
        print(f"❌ Motion exact sequence failed: {e}")
        traceback.print_exc()
    
    # Step 9: Code Comparison
    print("\n📋 STEP 9: Code Comparison")
    
    print("Comparing UMT5XXlTokenizer constructors...")
    
    # Read Motion implementation
    with open('motion/text_encoders/wan.py', 'r') as f:
        motion_code = f.read()
    
    # Read ComfyUI implementation
    with open('/home/fashionx/comfy/ComfyUI/comfy/text_encoders/wan.py', 'r') as f:
        comfy_code = f.read()
    
    print("Motion UMT5XXlTokenizer constructor:")
    motion_lines = motion_code.split('\n')
    for i, line in enumerate(motion_lines):
        if 'class UMT5XXlTokenizer' in line:
            print(f"Line {i+1}: {line}")
            for j in range(i+1, min(i+5, len(motion_lines))):
                print(f"Line {j+1}: {motion_lines[j]}")
            break
    
    print("\nComfyUI UMT5XXlTokenizer constructor:")
    comfy_lines = comfy_code.split('\n')
    for i, line in enumerate(comfy_lines):
        if 'class UMT5XXlTokenizer' in line:
            print(f"Line {i+1}: {line}")
            for j in range(i+1, min(i+5, len(comfy_lines))):
                print(f"Line {j+1}: {comfy_lines[j]}")
            break
    
    print("\n" + "=" * 100)
    print("🎯 SIDE-BY-SIDE DEBUG COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    debug_side_by_side()
