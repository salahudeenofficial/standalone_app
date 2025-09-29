#!/usr/bin/env python3
"""
Debug CLIP Model Structure
Simple script to understand the CLIP model structure when loaded
"""

import os
import sys
import torch
from pathlib import Path

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_clip_structure():
    """Debug the structure of loaded CLIP model"""
    print("🔍 Debugging CLIP Model Structure")
    print("="*50)
    
    try:
        from standalone_sd import load_state_dict_guess_config
        
        # Model path
        clip_model_path = "models/text_encoders/wan_clip_model.safetensors"
        
        if not os.path.exists(clip_model_path):
            print(f"❌ CLIP model not found: {clip_model_path}")
            return
        
        print(f"📁 Loading CLIP model: {clip_model_path}")
        
        # Load CLIP
        model_patcher, clip, vae, clipvision = load_state_dict_guess_config(
            clip_model_path,
            output_vae=False,
            output_clip=True,
            output_clipvision=False,
            output_model=False
        )
        
        if clip is None:
            print("❌ CLIP is None")
            return
        
        print(f"✅ CLIP loaded successfully")
        print(f"   Type: {type(clip).__name__}")
        
        # Debug CLIP structure
        print(f"\n🔍 CLIP Object Structure:")
        print(f"   Attributes: {dir(clip)}")
        
        # Check for model attribute
        if hasattr(clip, 'model'):
            print(f"   ✅ Has 'model' attribute: {type(clip.model).__name__}")
            if clip.model is not None:
                print(f"      Model type: {type(clip.model).__name__}")
                if hasattr(clip.model, 'parameters'):
                    params = list(clip.model.parameters())
                    print(f"      Model parameters: {len(params)}")
                    if params:
                        print(f"      First parameter shape: {params[0].shape}")
                        print(f"      First parameter dtype: {params[0].dtype}")
                        print(f"      First parameter device: {params[0].device}")
            else:
                print(f"      Model is None")
        else:
            print(f"   ❌ No 'model' attribute")
        
        # Check for cond_stage_model attribute
        if hasattr(clip, 'cond_stage_model'):
            print(f"   ✅ Has 'cond_stage_model' attribute: {type(clip.cond_stage_model).__name__}")
            if clip.cond_stage_model is not None:
                print(f"      cond_stage_model type: {type(clip.cond_stage_model).__name__}")
                if hasattr(clip.cond_stage_model, 'parameters'):
                    params = list(clip.cond_stage_model.parameters())
                    print(f"      cond_stage_model parameters: {len(params)}")
                    if params:
                        print(f"      First parameter shape: {params[0].shape}")
                        print(f"      First parameter dtype: {params[0].dtype}")
                        print(f"      First parameter device: {params[0].device}")
            else:
                print(f"      cond_stage_model is None")
        else:
            print(f"   ❌ No 'cond_stage_model' attribute")
        
        # Check for other important attributes
        important_attrs = ['load_device', 'offload_device', 'patches', 'uuid']
        for attr in important_attrs:
            if hasattr(clip, attr):
                value = getattr(clip, attr)
                print(f"   ✅ Has '{attr}': {value}")
            else:
                print(f"   ❌ No '{attr}' attribute")
        
        # Try to find the actual model
        print(f"\n🔍 Finding Actual Model:")
        actual_model = None
        
        if hasattr(clip, 'model') and clip.model is not None:
            actual_model = clip.model
            print(f"   ✅ Found model via clip.model: {type(actual_model).__name__}")
        elif hasattr(clip, 'cond_stage_model') and clip.cond_stage_model is not None:
            actual_model = clip.cond_stage_model
            print(f"   ✅ Found model via clip.cond_stage_model: {type(actual_model).__name__}")
        else:
            print(f"   ❌ No actual model found")
        
        if actual_model is not None:
            print(f"\n📊 Actual Model Details:")
            print(f"   Type: {type(actual_model).__name__}")
            print(f"   Attributes: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
            
            # Try to get parameters
            try:
                params = list(actual_model.parameters())
                print(f"   Parameters: {len(params)}")
                if params:
                    total_params = sum(p.numel() for p in params)
                    print(f"   Total parameters: {total_params:,}")
                    print(f"   First parameter shape: {params[0].shape}")
                    print(f"   First parameter dtype: {params[0].dtype}")
                    print(f"   First parameter device: {params[0].device}")
            except Exception as e:
                print(f"   ❌ Error getting parameters: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_clip_structure()
