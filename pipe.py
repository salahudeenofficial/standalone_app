from pydoc import cli
import sys
import os
import torch
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def step_1_vae_and_latent_creation(vae_model_path: str, positive_prompt: str = "", negative_prompt: str = "",
                                  control_video_path: Optional[str] = None, reference_image_path: Optional[str] = None,
                                  width: int = 480, height: int = 832, length: int = 37, batch_size: int = 1, 
                                  strength: float = 1.0) -> Dict[str, Any]:
    """Step 1: VAE Load + Reference Image/Control Video + Initial Latent Creation"""
    print("\n" + "="*80)
    print("🚀 STEP 1: VAE LOAD + REFERENCE IMAGE/CONTROL VIDEO + INITIAL LATENT CREATION")
    print("="*80)
    
    try:
        from motion.comps import Initial_latent
        
        initial_latent = Initial_latent()
        results = initial_latent.create_initial_latent(
            vae_model_path=vae_model_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            control_video_path=control_video_path,
            reference_image_path=reference_image_path,
            width=width, height=height, length=length, batch_size=batch_size, strength=strength
        )
        
        print(f"✅ STEP 1 COMPLETED SUCCESSFULLY!")
        return results
        
    except Exception as e:
        print(f"❌ STEP 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def step_2_unet_clip_lora_loading(unet_model_path: str, clip_model_path: str, 
                                 lora_model_path: Optional[str] = None, strength_model: float = 1.0, 
                                 strength_clip: float = 0.0) -> Dict[str, Any]:
    """Step 2: UNet + CLIP Load + LoRA Application"""
    print("\n" + "="*80)
    print("🚀 STEP 2: UNET + CLIP LOAD + LORA APPLICATION")
    print("="*80)
    
    try:
        from motion.comps import UNETLoader
        from motion.standalone_sd import load_wan_clip
        
        # Load UNet
        unet_loader = UNETLoader("wan_2.1_diffusion_model.safetensors", "default")
        unet = unet_loader.load_unet()
        
        # Load CLIP
        clip = load_wan_clip(clip_model_path)
        
        step_2_results = {
            'unet': unet,
            'clip': clip,
            'lora_applied': False,
            'model_info': {
                'unet_type': type(unet).__name__,
                'clip_type': type(clip).__name__,
                'unet_path': unet_model_path,
                'clip_path': clip_model_path
            }
        }
        
        print(f"✅ STEP 2 COMPLETED SUCCESSFULLY!")
        return step_2_results
        
    except Exception as e:
        print(f"❌ STEP 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def step_3_model_sampling_and_text_encoding(positive_prompt: str, negative_prompt: str, 
                                           vace_positive_conditioning=None, vace_negative_conditioning=None,
                                           shift: float = 8.0, multiplier: float = 1000.0) -> Dict[str, Any]:
    """Step 3: Model Sampling + Text Encoding"""
    print("\n" + "="*80)
    print("🚀 STEP 3: MODEL SAMPLING + TEXT ENCODING")
    print("="*80)
    
    try:
        from comps import CLIPTextEncode ,ModelSamplingSD3        
        # Get models from global state (set by step 2)
        global unet_model, clip_model
        if 'unet_model' not in globals() or 'clip_model' not in globals():
            raise RuntimeError("Step 2 must be run before Step 3")
        
        # Apply ModelSamplingSD3
        model_sampling = ModelSamplingSD3()
        patched_unet = model_sampling.patch(unet_model, shift=shift, multiplier=multiplier)
        
        # Text encoding
        text_encoder = CLIPTextEncode(clip_model)
        positive_conditioning = text_encoder.encode(positive_prompt)
        negative_conditioning = text_encoder.encode(negative_prompt)
        
        # Combine with VACE conditioning if provided
        if vace_positive_conditioning is not None and vace_negative_conditioning is not None:
            positive_conditioning = vace_positive_conditioning.copy()
            positive_conditioning[0] = positive_conditioning[0] if isinstance(positive_conditioning, list) else positive_conditioning
            
            negative_conditioning = vace_negative_conditioning.copy()
            negative_conditioning[0] = negative_conditioning[0] if isinstance(negative_conditioning, list) else negative_conditioning
        
        step_3_results = {
            'positive_conditioning': positive_conditioning,
            'negative_conditioning': negative_conditioning,
            'unet_patched': patched_unet,
            'clip_model': clip_model,
            'sampling_applied': True,
            'model_info': {
                'shift': shift,
                'multiplier': multiplier,
                'positive_prompt': positive_prompt,
                'negative_prompt': negative_prompt
            }
        }
        
        print(f"✅ STEP 3 COMPLETED SUCCESSFULLY!")
        return step_3_results
        
    except Exception as e:
        print(f"❌ STEP 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # """Main function to run Steps 1, 2, 3 and print Step 4 inputs"""
    # print("🎬 MOTION PIPELINE - STEPS 1, 2, 3 + STEP 4 INPUT ANALYSIS")
    # print("="*80)
    
    # # Model paths
    # vae_model_path = "./models/vaes/wan_vae.safetensors"
    # unet_model_path = "./models/diffusion_models/wan_2.1_diffusion_model.safetensors"
    # clip_model_path = "./models/text_encoders/wan_clip_model.safetensors"
    # # Prompts
    # positive_prompt = "very cinematic video"
    # negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量"
    
    # try:
    #     # Step 1: VAE Loading and Latent Creation
    #     print("🎬 STEP 1: VAE LOADING AND LATENT CREATION")
    #     step_1_results = step_1_vae_and_latent_creation(
    #         vae_model_path=vae_model_path,
    #         positive_prompt=positive_prompt,
    #         negative_prompt=negative_prompt,
    #         control_video_path="safu.mp4" if os.path.exists("safu.mp4") else None,
    #         reference_image_path="safu.jpg" if os.path.exists("safu.jpg") else None,
    #         width=480, height=832, length=37, batch_size=1, strength=1.0
    #     )
        
    #     if not step_1_results:
    #         print("❌ Step 1 failed, cannot continue")
    #         return None
        
    #     # Step 2: UNet + CLIP Loading
    #     print("\n🧠 STEP 2: UNET + CLIP LOADING")
    #     step_2_results = step_2_unet_clip_lora_loading(
    #         unet_model_path=unet_model_path,
    #         clip_model_path=clip_model_path,
    #         lora_model_path=None,
    #         strength_model=1.0, strength_clip=0.0
    #     )
        
    #     if not step_2_results:
    #         print("❌ Step 2 failed, cannot continue")
    #         return None
        
    #     # Set global models for step 3
    #     global unet_model, clip_model
    #     unet_model = step_2_results['unet']
    #     clip_model = step_2_results['clip']
        
    #     # Step 3: Model Sampling + Text Encoding
    #     print("\n📝 STEP 3: MODEL SAMPLING + TEXT ENCODING")
    #     step_3_results = step_3_model_sampling_and_text_encoding(
    #         positive_prompt=positive_prompt,
    #         negative_prompt=negative_prompt,
    #         vace_positive_conditioning=step_1_results['positive'],
    #         vace_negative_conditioning=step_1_results['negative'],
    #         shift=8.0, multiplier=1000
    #     )
        
    #     if not step_3_results:
    #         print("❌ Step 3 failed, cannot continue")
    #         return None
        
    #     # Prepare Step 4 inputs
    #     print("\n🎯 STEP 4 INPUT ANALYSIS")
    #     print("="*80)
        
    #     step_4_inputs = {
    #         'initial_latent': step_1_results['out_latent']['samples'],
    #         'positive_conditioning': step_3_results['positive_conditioning'],
    #         'negative_conditioning': step_3_results['negative_conditioning'],
    #         'seed': 42,
    #         'steps': 4,
    #         'cfg': 7.0,
    #         'sampler_name': 'euler',
    #         'scheduler': 'normal',
    #         'denoise': 1.0,
    #         'noise_inds': None
    #     }
        
    #     # Print Step 4 input analysis
    #     print("📊 STEP 4 INPUT TENSOR ANALYSIS:")
    #     print("="*60)
        
    #     # Initial latent analysis
    #     initial_latent = step_4_inputs['initial_latent']
    #     print(f"✅ Initial Latent:")
    #     print(f"   Shape: {initial_latent.shape}")
    #     print(f"   Dtype: {initial_latent.dtype}")
    #     print(f"   Device: {initial_latent.device}")
    #     print(f"   Mean: {initial_latent.mean().item():.6f}")
    #     print(f"   Range: [{initial_latent.min().item():.6f}, {initial_latent.max().item():.6f}]")
        
    #     # Positive conditioning analysis
    #     pos_cond = step_4_inputs['positive_conditioning']
    #     if isinstance(pos_cond, (list, tuple)) and len(pos_cond) > 0:
    #         pos_tensor = pos_cond[0]
    #         print(f"\n✅ Positive Conditioning:")
    #         print(f"   Shape: {pos_tensor.shape}")
    #         print(f"   Dtype: {pos_tensor.dtype}")
    #         print(f"   Device: {pos_tensor.device}")
    #         print(f"   Mean: {pos_tensor.mean().item():.6f}")
    #         print(f"   Range: [{pos_tensor.min().item():.6f}, {pos_tensor.max().item():.6f}]")
        
    #     # Negative conditioning analysis
    #     neg_cond = step_4_inputs['negative_conditioning']
    #     if isinstance(neg_cond, (list, tuple)) and len(neg_cond) > 0:
    #         neg_tensor = neg_cond[0]
    #         print(f"\n✅ Negative Conditioning:")
    #         print(f"   Shape: {neg_tensor.shape}")
    #         print(f"   Dtype: {neg_tensor.dtype}")
    #         print(f"   Device: {neg_tensor.device}")
    #         print(f"   Mean: {neg_tensor.mean().item():.6f}")
    #         print(f"   Range: [{neg_tensor.min().item():.6f}, {neg_tensor.max().item():.6f}]")
        
    #     # Step 4 parameters
    #     print(f"\n📋 STEP 4 PARAMETERS:")
    #     print(f"   Seed: {step_4_inputs['seed']}")
    #     print(f"   Steps: {step_4_inputs['steps']}")
    #     print(f"   CFG: {step_4_inputs['cfg']}")
    #     print(f"   Sampler: {step_4_inputs['sampler_name']}")
    #     print(f"   Scheduler: {step_4_inputs['scheduler']}")
    #     print(f"   Denoise: {step_4_inputs['denoise']}")
        
    #     print(f"\n🎉 STEPS 1, 2, 3 COMPLETED SUCCESSFULLY!")
    #     print(f"📊 Step 4 inputs prepared and analyzed")
    #     print("="*80)
        
    #     return {
    #         'step_1_results': step_1_results,
    #         'step_2_results': step_2_results,
    #         'step_3_results': step_3_results,
    #         'step_4_inputs': step_4_inputs
    #     }
        
    # except Exception as e:
    #     print(f"❌ PIPELINE FAILED: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     return None
    from motion.comps import CLIPLoader,CLIPTextEncode
    clip = CLIPLoader("wan_clip_model.safeetensors",type=13).load_clip()
    clip_encode = CLIPTextEncode(clip)
    clip_encode.encode("a beautiful woman")

if __name__ == "__main__":
    main()