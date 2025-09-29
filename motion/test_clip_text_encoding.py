#!/usr/bin/env python3
"""
Test Script 2: CLIP Text Encoding
Part 2 of Step 3 - Tests CLIPTextEncode functionality on loaded CLIP model
Based on pipeline_manual.py implementation
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import motion modules  
from standalone_sd import load_state_dict_guess_config
from lora import load_lora_for_models
from utils import load_torch_file, calculate_parameters
from wan_vae_components.model_management import get_torch_device, unet_offload_device

# Import CLIPTextEncode component (standalone)
from text_encoder import CLIPTextEncode

class CLIPTextEncodingTester:
    """Test CLIP Text Encoding functionality standalone"""
    
    def __init__(self):
        """Initialize the tester"""
        self.device = get_torch_device()
        self.offload_device = unet_offload_device()
        
        print("🔧 CLIP Text Encoding Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   Offload Device: {self.offload_device}")
    
    def test_clip_text_encoding(self, 
                               clip_model_path: str,
                               lora_model_path: Optional[str] = None,
                               positive_prompt: str = "very cinematic video",
                               negative_prompt: str = "bad quality, static, blurry",
                               strength_clip: float = 0.0) -> Dict[str, Any]:
        """
        Test CLIP Text Encoding on loaded model
        
        Steps:
        1. Load CLIP model using standalone_sd
        2. Optionally apply LoRA patches to CLIP
        3. Initialize CLIPTextEncode component
        4. Encode positive and negative prompts
        5. Analyze encoding results and memory usage
        
        Args:
            clip_model_path: Path to CLIP text encoder
            lora_model_path: Path to LoRA patches (optional)
            positive_prompt: Positive text prompt to encode
            negative_prompt: Negative text prompt to encode
            strength_clip: LoRA strength for CLIP (if applying LoRA)
            
        Returns:
            Test results with encoding information and timing
        """
        
        print("\n" + "="*80)
        print("🚀 CLIP TEXT ENCODING TEST")
        print("="*80)
        
        try:
            test_start = time.time()
            
            # ========================================================================
            # Step 1: Load CLIP Model
            # ========================================================================
            print("1. Loading CLIP text encoder...")
            clip_start = time.time()
            
            if not os.path.exists(clip_model_path):
                raise FileNotFoundError(f"CLIP model not found: {clip_model_path}")
            
            # Load CLIP state dict
            clip_state_dict = load_torch_file(clip_model_path)
            print(f"   📊 Loaded CLIP state dict with {len(clip_state_dict)} keys")
            
            # Load CLIP model
            result = load_state_dict_guess_config(
                clip_state_dict,
                output_vae=False,
                output_clip=True,
                output_clipvision=False,
                output_model=False
            )
            
            if result is None:
                raise RuntimeError("Failed to load CLIP - load_state_dict_guess_config returned None")
            
            _, clip_model, _, _ = result
            
            if clip_model is None:
                raise RuntimeError("CLIP model is None after loading")
            
            clip_time = time.time() - clip_start
            print(f"✅ CLIP loaded successfully in {clip_time:.2f}s")
            print(f"   Type: {type(clip_model).__name__}")
            print(f"   Device: {clip_model.load_device}")
            
            # Calculate CLIP model size
            clip_params = 0
            if hasattr(clip_model, 'cond_stage_model') and hasattr(clip_model.cond_stage_model, 'state_dict'):
                clip_state_dict_params = clip_model.cond_stage_model.state_dict()
                clip_params = calculate_parameters(clip_state_dict_params)
                print(f"   Parameters: {clip_params:,}")
                print(f"   Size: {clip_params * 4 / (1024*1024):.1f} MB")
            
            # ========================================================================
            # Step 2: Apply LoRA to CLIP (Optional)
            # ========================================================================
            lora_applied = False
            lora_time = 0.0
            
            if lora_model_path and os.path.exists(lora_model_path) and strength_clip > 0.0:
                print("\n2. Applying LoRA patches to CLIP...")
                lora_start = time.time()
                
                # Load LoRA state dict
                lora_state_dict = load_torch_file(lora_model_path)
                print(f"   📊 Loaded LoRA with {len(lora_state_dict)} keys")
                
                # Create dummy UNet for LoRA loading (required by load_lora_for_models)
                dummy_unet = None
                
                # Apply LoRA to CLIP only
                original_clip_patches = len(clip_model.patches) if hasattr(clip_model, 'patches') and clip_model.patches else 0
                
                _, new_clip = load_lora_for_models(
                    dummy_unet, clip_model, lora_state_dict,
                    strength_model=0.0,  # No UNet
                    strength_clip=strength_clip
                )
                
                if new_clip is not None:
                    clip_model = new_clip
                    lora_applied = True
                    
                    lora_time = time.time() - lora_start
                    print(f"✅ LoRA applied to CLIP successfully in {lora_time:.2f}s")
                    
                    # Report LoRA patch counts
                    new_clip_patches = len(clip_model.patches) if hasattr(clip_model, 'patches') and clip_model.patches else 0
                    
                    print(f"   🔧 CLIP Patches: {original_clip_patches} → {new_clip_patches} (+{new_clip_patches - original_clip_patches})")
                    print(f"   🔧 CLIP Strength: {strength_clip}")
                else:
                    print("❌ LoRA application to CLIP failed")
                    lora_applied = False
            else:
                print("\n2. ⚠️  No LoRA file specified, strength is 0.0, or file not found - skipping LoRA application")
                lora_applied = False
            
            # ========================================================================
            # Step 3: Initialize Text Encoder
            # ========================================================================
            print("\n3. Initializing CLIP text encoder...")
            encoder_start = time.time()
            
            text_encoder = CLIPTextEncode()
            
            encoder_init_time = time.time() - encoder_start
            print(f"✅ CLIPTextEncode initialized in {encoder_init_time:.3f}s")
            
            # ========================================================================
            # Step 4: Encode Text Prompts
            # ========================================================================
            print("\n4. Encoding text prompts...")
            encoding_start = time.time()
            
            print(f"   📝 Positive prompt: '{positive_prompt}'")
            print(f"   📝 Negative prompt: '{negative_prompt}'")
            
            # Memory before encoding
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"   💾 GPU memory before encoding: {mem_before:.1f} MB")
            
            # Encode positive prompt
            positive_encoding_start = time.time()
            positive_cond = text_encoder.encode(clip_model, positive_prompt)
            positive_encoding_time = time.time() - positive_encoding_start
            
            print(f"   ✅ Positive prompt encoded in {positive_encoding_time:.3f}s")
            
            # Encode negative prompt
            negative_encoding_start = time.time()
            negative_cond = text_encoder.encode(clip_model, negative_prompt)
            negative_encoding_time = time.time() - negative_encoding_start
            
            print(f"   ✅ Negative prompt encoded in {negative_encoding_time:.3f}s")
            
            total_encoding_time = time.time() - encoding_start
            
            # Memory after encoding
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_delta = mem_after - mem_before
                print(f"   💾 GPU memory after encoding: {mem_after:.1f} MB (+{mem_delta:.1f} MB)")
            
            # ========================================================================
            # Step 5: Analyze Encoding Results
            # ========================================================================
            print("\n5. Analyzing encoding results...")
            
            # Analyze positive conditioning
            print(f"   🔧 POSITIVE CONDITIONING ANALYSIS:")
            if isinstance(positive_cond, (tuple, list)) and len(positive_cond) > 0:
                pos_tensor = positive_cond[0]
                if hasattr(pos_tensor, 'shape'):
                    print(f"      Shape: {pos_tensor.shape}")
                    print(f"      Data Type: {pos_tensor.dtype}")
                    print(f"      Device: {pos_tensor.device}")
                    print(f"      Min Value: {pos_tensor.min().item():.6f}")
                    print(f"      Max Value: {pos_tensor.max().item():.6f}")
                    print(f"      Mean Value: {pos_tensor.mean().item():.6f}")
                    
                    # Check for valid embeddings (not all zeros)
                    non_zero_elements = torch.count_nonzero(pos_tensor).item()
                    total_elements = pos_tensor.numel()
                    non_zero_ratio = non_zero_elements / total_elements
                    print(f"      Non-zero ratio: {non_zero_ratio:.3f} ({non_zero_elements}/{total_elements})")
                    
                    if non_zero_ratio > 0.1:  # At least 10% non-zero
                        print(f"      ✅ Valid text embedding generated")
                    else:
                        print(f"      ⚠️  Warning: Text embedding mostly zeros")
                else:
                    print(f"      ❌ No shape information for positive conditioning")
            else:
                print(f"      ❌ Invalid positive conditioning format")
            
            # Analyze negative conditioning
            print(f"   🔧 NEGATIVE CONDITIONING ANALYSIS:")
            if isinstance(negative_cond, (tuple, list)) and len(negative_cond) > 0:
                neg_tensor = negative_cond[0]
                if hasattr(neg_tensor, 'shape'):
                    print(f"      Shape: {neg_tensor.shape}")
                    print(f"      Data Type: {neg_tensor.dtype}")
                    print(f"      Device: {neg_tensor.device}")
                    print(f"      Min Value: {neg_tensor.min().item():.6f}")
                    print(f"      Max Value: {neg_tensor.max().item():.6f}")
                    print(f"      Mean Value: {neg_tensor.mean().item():.6f}")
                    
                    # Check for valid embeddings (not all zeros)
                    non_zero_elements = torch.count_nonzero(neg_tensor).item()
                    total_elements = neg_tensor.numel()
                    non_zero_ratio = non_zero_elements / total_elements
                    print(f"      Non-zero ratio: {non_zero_ratio:.3f} ({non_zero_elements}/{total_elements})")
                    
                    if non_zero_ratio > 0.1:  # At least 10% non-zero
                        print(f"      ✅ Valid text embedding generated")
                    else:
                        print(f"      ⚠️  Warning: Text embedding mostly zeros")
                else:
                    print(f"      ❌ No shape information for negative conditioning")
            else:
                print(f"      ❌ Invalid negative conditioning format")
            
            # Test results
            test_results = {
                'success': True,
                'clip_model': clip_model,
                'positive_conditioning': positive_cond,
                'negative_conditioning': negative_cond,
                'lora_applied': lora_applied,
                'model_info': {
                    'clip_type': type(clip_model).__name__,
                    'clip_device': str(clip_model.load_device),
                    'clip_params': clip_params,
                    'lora_strength': strength_clip if lora_applied else 0.0
                },
                'encoding_info': {
                    'positive_prompt': positive_prompt,
                    'negative_prompt': negative_prompt,
                    'positive_shape': pos_tensor.shape if hasattr(pos_tensor, 'shape') else None,
                    'negative_shape': neg_tensor.shape if hasattr(neg_tensor, 'shape') else None,
                    'positive_dtype': str(pos_tensor.dtype) if hasattr(pos_tensor, 'dtype') else None,
                    'negative_dtype': str(neg_tensor.dtype) if hasattr(neg_tensor, 'dtype') else None
                },
                'timing': {
                    'clip_loading': clip_time,
                    'lora_application': lora_time,
                    'encoder_init': encoder_init_time,
                    'positive_encoding': positive_encoding_time,
                    'negative_encoding': negative_encoding_time,
                    'total_encoding': total_encoding_time,
                    'total_time': time.time() - test_start
                }
            }
            
            print(f"\n✅ CLIP TEXT ENCODING TEST COMPLETED SUCCESSFULLY in {time.time() - test_start:.2f}s")
            print("="*80)
            
            return test_results
            
        except Exception as e:
            print(f"❌ CLIP TEXT ENCODING TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Test CLIP Text Encoding with real models"""
    print("🧪 CLIP TEXT ENCODING TEST SCRIPT")
    print("="*60)
    
    # Initialize tester
    tester = CLIPTextEncodingTester()
    
    # Test parameters
    test_params = {
        'clip_model_path': 'models/text_encoders/wan_clip_model.safetensors',
        'lora_model_path': 'models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors',
        'positive_prompt': "very cinematic video",
        'negative_prompt': "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        'strength_clip': 0.0  # Default: no LoRA on CLIP
    }
    
    # Check if model files exist
    required_files = [test_params['clip_model_path']]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Required model files not found:")
        for missing in missing_files:
            print(f"   {missing}")
        print("\n💡 Run './download_models.sh' to download the required models")
        return False
    
    try:
        # Run the test
        results = tester.test_clip_text_encoding(**test_params)
        
        print("\n🎉 CLIP TEXT ENCODING TEST COMPLETED SUCCESSFULLY!")
        print(f"\n📋 TEST RESULTS SUMMARY:")
        print(f"   CLIP Model: {results['model_info']['clip_type']}")
        print(f"   CLIP Device: {results['model_info']['clip_device']}")
        print(f"   CLIP Parameters: {results['model_info']['clip_params']:,}")
        print(f"   LoRA Applied: {'✅' if results['lora_applied'] else '❌'}")
        if results['lora_applied']:
            print(f"   LoRA Strength: {results['model_info']['lora_strength']}")
        
        print(f"\n📝 ENCODING RESULTS:")
        print(f"   Positive Prompt: '{results['encoding_info']['positive_prompt']}'")
        print(f"   Negative Prompt: '{results['encoding_info']['negative_prompt']}'")
        print(f"   Positive Shape: {results['encoding_info']['positive_shape']}")
        print(f"   Negative Shape: {results['encoding_info']['negative_shape']}")
        print(f"   Data Type: {results['encoding_info']['positive_dtype']}")
        
        print(f"\n⏱️  TIMING BREAKDOWN:")
        print(f"   CLIP Loading: {results['timing']['clip_loading']:.2f}s")
        print(f"   LoRA Application: {results['timing']['lora_application']:.2f}s")
        print(f"   Positive Encoding: {results['timing']['positive_encoding']:.3f}s")
        print(f"   Negative Encoding: {results['timing']['negative_encoding']:.3f}s")
        print(f"   Total Processing: {results['timing']['total_time']:.2f}s")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"   Final GPU Memory: {allocated:.1f} MB allocated")
        
        print("\n✅ CLIP Text Encoding is working correctly!")
        print("✅ Ready for integration into pipeline Step 3")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CLIP TEXT ENCODING TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 CLIP TEXT ENCODING TEST COMPLETED SUCCESSFULLY!")
        print("✅ Ready for pipeline integration")
    else:
        print("\n💥 CLIP TEXT ENCODING TEST FAILED")
        print("❌ Check error messages above")
