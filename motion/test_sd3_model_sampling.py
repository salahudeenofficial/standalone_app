#!/usr/bin/env python3
"""
Test Script 1: SD3 Model Sampling 
Part 1 of Step 3 - Tests ModelSamplingSD3 application on loaded UNet model
Based on pipeline_manual.py implementation
"""

import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import motion modules  
from standalone_sd import load_state_dict_guess_config
from lora import load_lora_for_models
from utils import load_torch_file, calculate_parameters
from wan_vae_components.model_management import get_torch_device, unet_offload_device

# Import ModelSamplingSD3 component (standalone)
from model_sampling import ModelSamplingSD3

class SD3ModelSamplingTester:
    """Test SD3 Model Sampling functionality standalone"""
    
    def __init__(self):
        """Initialize the tester"""
        self.device = get_torch_device()
        self.offload_device = unet_offload_device()
        
        print("🔧 SD3 Model Sampling Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   Offload Device: {self.offload_device}")
    
    def test_sd3_model_sampling(self, 
                               unet_model_path: str,
                               clip_model_path: str,
                               lora_model_path: Optional[str] = None,
                               shift: float = 8.0,
                               multiplier: int = 1000,
                               strength_model: float = 1.0,
                               strength_clip: float = 0.0) -> Dict[str, Any]:
        """
        Test SD3 Model Sampling on loaded models
        
        Steps:
        1. Load UNet model using standalone_sd
        2. Load CLIP model using standalone_sd  
        3. Optionally apply LoRA patches
        4. Apply ModelSamplingSD3 with shift parameter
        5. Verify model patching and memory management
        
        Args:
            unet_model_path: Path to UNet diffusion model
            clip_model_path: Path to CLIP text encoder
            lora_model_path: Path to LoRA patches (optional)
            shift: SD3 shift parameter (default 8.0)
            multiplier: SD3 multiplier parameter (default 1000)
            strength_model: LoRA strength for UNet (if applying LoRA)
            strength_clip: LoRA strength for CLIP (if applying LoRA)
            
        Returns:
            Test results with model information and timing
        """
        
        print("\n" + "="*80)
        print("🚀 SD3 MODEL SAMPLING TEST")
        print("="*80)
        
        try:
            test_start = time.time()
            
            # ========================================================================
            # Step 1: Load UNet Model
            # ========================================================================
            print("1. Loading UNet diffusion model...")
            unet_start = time.time()
            
            if not os.path.exists(unet_model_path):
                raise FileNotFoundError(f"UNet model not found: {unet_model_path}")
            
            # Load UNet state dict
            unet_state_dict = load_torch_file(unet_model_path)
            print(f"   📊 Loaded UNet state dict with {len(unet_state_dict)} keys")
            
            # Load UNet model 
            result = load_state_dict_guess_config(
                unet_state_dict,
                output_vae=False,
                output_clip=False,
                output_clipvision=False,
                output_model=True
            )
            
            if result is None:
                raise RuntimeError("Failed to load UNet - load_state_dict_guess_config returned None")
            
            unet_model, _, _, _ = result
            
            if unet_model is None:
                raise RuntimeError("UNet model is None after loading")
            
            unet_time = time.time() - unet_start
            print(f"✅ UNet loaded successfully in {unet_time:.2f}s")
            print(f"   Type: {type(unet_model).__name__}")
            print(f"   Device: {unet_model.load_device}")
            
            # Calculate UNet model size
            unet_params = 0
            if hasattr(unet_model, 'model') and hasattr(unet_model.model, 'state_dict'):
                unet_state_dict_params = unet_model.model.state_dict()
                unet_params = calculate_parameters(unet_state_dict_params)
                print(f"   Parameters: {unet_params:,}")
                print(f"   Size: {unet_params * 4 / (1024*1024):.1f} MB")
            
            # ========================================================================
            # Step 2: Load CLIP Model
            # ========================================================================
            print("\n2. Loading CLIP text encoder...")
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
            
            # ========================================================================
            # Step 3: Apply LoRA (Optional)
            # ========================================================================
            lora_applied = False
            lora_time = 0.0
            
            if lora_model_path and os.path.exists(lora_model_path):
                print("\n3. Applying LoRA patches...")
                lora_start = time.time()
                
                # Load LoRA state dict
                lora_state_dict = load_torch_file(lora_model_path)
                print(f"   📊 Loaded LoRA with {len(lora_state_dict)} keys")
                
                # Apply LoRA to models
                original_unet_patches = len(unet_model.patches) if hasattr(unet_model, 'patches') and unet_model.patches else 0
                original_clip_patches = len(clip_model.patches) if hasattr(clip_model, 'patches') and clip_model.patches else 0
                
                new_unet, new_clip = load_lora_for_models(
                    unet_model, clip_model, lora_state_dict,
                    strength_model=strength_model,
                    strength_clip=strength_clip
                )
                
                if new_unet is not None and new_clip is not None:
                    unet_model = new_unet
                    clip_model = new_clip
                    lora_applied = True
                    
                    lora_time = time.time() - lora_start
                    print(f"✅ LoRA applied successfully in {lora_time:.2f}s")
                    
                    # Report LoRA patch counts
                    new_unet_patches = len(unet_model.patches) if hasattr(unet_model, 'patches') and unet_model.patches else 0
                    new_clip_patches = len(clip_model.patches) if hasattr(clip_model, 'patches') and clip_model.patches else 0
                    
                    print(f"   🔧 UNet Patches: {original_unet_patches} → {new_unet_patches} (+{new_unet_patches - original_unet_patches})")
                    print(f"   🔧 CLIP Patches: {original_clip_patches} → {new_clip_patches} (+{new_clip_patches - original_clip_patches})")
                    print(f"   🔧 Model Strength: {strength_model}")
                    print(f"   🔧 CLIP Strength: {strength_clip}")
                else:
                    print("❌ LoRA application failed - models are None")
                    lora_applied = False
            else:
                print("\n3. ⚠️  No LoRA file specified or file not found - skipping LoRA application")
                lora_applied = False
            
            # ========================================================================
            # Step 4: Apply ModelSamplingSD3
            # ========================================================================
            print("\n4. Applying ModelSamplingSD3...")
            sampling_start = time.time()
            
            # Store original model for comparison
            original_model = unet_model
            original_model_type = type(unet_model).__name__
            original_uuid = str(unet_model.patches_uuid) if hasattr(unet_model, 'patches_uuid') else None
            
            # Apply ModelSamplingSD3
            model_sampling = ModelSamplingSD3()
            patched_model = model_sampling.patch(unet_model, shift=shift, multiplier=multiplier)
            
            sampling_time = time.time() - sampling_start
            
            if patched_model is not None:
                print(f"✅ ModelSamplingSD3 applied successfully in {sampling_time:.2f}s")
                
                # Analyze the patched model
                print(f"\n🔧 MODEL SAMPLING ANALYSIS:")
                print(f"   Original Model Type: {original_model_type}")
                print(f"   Patched Model Type: {type(patched_model).__name__}")
                print(f"   Model Cloned: {'✅ YES' if patched_model != original_model else '❌ NO'}")
                print(f"   UUID Changed: {'✅ YES' if str(patched_model.patches_uuid) != original_uuid else '❌ NO'}")
                
                # Check for model_sampling patch
                has_sampling_patch = False
                if hasattr(patched_model, 'object_patches'):
                    if 'model_sampling' in patched_model.object_patches:
                        has_sampling_patch = True
                        sampling_obj = patched_model.object_patches['model_sampling']
                        print(f"   Model Sampling Patch: ✅ Applied ({type(sampling_obj).__name__})")
                        print(f"   Shift Parameter: {shift}")
                        print(f"   Multiplier Parameter: {multiplier}")
                    else:
                        print(f"   Model Sampling Patch: ❌ Not found in object_patches")
                else:
                    print(f"   Model Sampling Patch: ❌ No object_patches attribute")
                
                # Memory status
                print(f"\n🔧 MEMORY STATUS AFTER PATCHING:")
                print(f"   Original Model Device: {original_model.load_device}")
                print(f"   Patched Model Device: {patched_model.load_device}")
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"   GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
                
                # Test results
                test_results = {
                    'success': True,
                    'original_model': original_model,
                    'patched_model': patched_model,
                    'lora_applied': lora_applied,
                    'model_info': {
                        'original_type': original_model_type,
                        'patched_type': type(patched_model).__name__,
                        'model_cloned': patched_model != original_model,
                        'has_sampling_patch': has_sampling_patch,
                        'shift': shift,
                        'multiplier': multiplier,
                        'unet_params': unet_params
                    },
                    'timing': {
                        'unet_loading': unet_time,
                        'clip_loading': clip_time,
                        'lora_application': lora_time,
                        'model_sampling': sampling_time,
                        'total_time': time.time() - test_start
                    }
                }
                
                print(f"\n✅ SD3 MODEL SAMPLING TEST COMPLETED SUCCESSFULLY in {time.time() - test_start:.2f}s")
                print("="*80)
                
                return test_results
                
            else:
                raise RuntimeError("ModelSamplingSD3 returned None - patching failed")
            
        except Exception as e:
            print(f"❌ SD3 MODEL SAMPLING TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Test SD3 Model Sampling with real models"""
    print("🧪 SD3 MODEL SAMPLING TEST SCRIPT")
    print("="*60)
    
    # Initialize tester
    tester = SD3ModelSamplingTester()
    
    # Test parameters
    test_params = {
        'unet_model_path': 'models/diffusion_models/wan_2.1_diffusion_model.safetensors',
        'clip_model_path': 'models/text_encoders/wan_clip_model.safetensors',
        'lora_model_path': 'models/loras/Wan21_CausVid_14B_T2V_lora_rank32.safetensors',
        'shift': 8.0,
        'multiplier': 1000,
        'strength_model': 1.0,
        'strength_clip': 0.0
    }
    
    # Check if model files exist
    required_files = [
        test_params['unet_model_path'],
        test_params['clip_model_path']
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Required model files not found:")
        for missing in missing_files:
            print(f"   {missing}")
        print("\n💡 Run './download_models.sh' to download the required models")
        return False
    
    try:
        # Run the test
        results = tester.test_sd3_model_sampling(**test_params)
        
        print("\n🎉 SD3 MODEL SAMPLING TEST COMPLETED SUCCESSFULLY!")
        print(f"\n📋 TEST RESULTS SUMMARY:")
        print(f"   Original Model: {results['model_info']['original_type']}")
        print(f"   Patched Model: {results['model_info']['patched_type']}")
        print(f"   Model Cloned: {'✅' if results['model_info']['model_cloned'] else '❌'}")
        print(f"   Sampling Patch Applied: {'✅' if results['model_info']['has_sampling_patch'] else '❌'}")
        print(f"   LoRA Applied: {'✅' if results['lora_applied'] else '❌'}")
        print(f"   Shift Parameter: {results['model_info']['shift']}")
        print(f"   Multiplier Parameter: {results['model_info']['multiplier']}")
        print(f"   UNet Parameters: {results['model_info']['unet_params']:,}")
        print(f"   Total Processing Time: {results['timing']['total_time']:.2f}s")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"   Final GPU Memory: {allocated:.1f} MB allocated")
        
        print("\n✅ SD3 Model Sampling is working correctly!")
        print("✅ Ready for integration into pipeline Step 3")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SD3 MODEL SAMPLING TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 SD3 MODEL SAMPLING TEST COMPLETED SUCCESSFULLY!")
        print("✅ Ready for pipeline integration")
    else:
        print("\n💥 SD3 MODEL SAMPLING TEST FAILED")
        print("❌ Check error messages above")
