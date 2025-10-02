#!/usr/bin/env python3
"""
ComfyUI Step 4 Reference Data Collector
Collects essential data from ComfyUI for motion pipeline verification
"""

import sys
import os
import json
import torch
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

class ComfyUIStep4ReferenceCollector:
    """
    Collects reference data from ComfyUI for Step 4 verification
    """
    
    def __init__(self):
        self.reference_data = {}
        self.collection_start = time.time()
        
    def collect_section_4_2_reference(self, 
                                     initial_latent: torch.Tensor,
                                     unet_model,
                                     seed: int = 42,
                                     noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Collect Section 4.2 reference data: Latent Preparation
        """
        print("\n" + "="*80)
        print("📊 COLLECTING SECTION 4.2 REFERENCE: LATENT PREPARATION")
        print("="*80)
        
        section_start = time.time()
        reference = {
            'section': '4.2_latent_preparation',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'comfyui_version': 'reference'
        }
        
        try:
            # 1. Input validation
            print("1. Collecting input data...")
            input_data = {
                'initial_latent_shape': list(initial_latent.shape),
                'initial_latent_dtype': str(initial_latent.dtype),
                'initial_latent_device': str(initial_latent.device),
                'initial_latent_mean': initial_latent.mean().item(),
                'initial_latent_std': initial_latent.std().item(),
                'initial_latent_min': initial_latent.min().item(),
                'initial_latent_max': initial_latent.max().item(),
                'seed': seed,
                'noise_inds_provided': noise_inds is not None
            }
            
            if noise_inds is not None:
                input_data['noise_inds_shape'] = list(noise_inds.shape)
                input_data['noise_inds_dtype'] = str(noise_inds.dtype)
            
            reference['inputs'] = input_data
            print(f"   Initial latent shape: {initial_latent.shape}")
            print(f"   Initial latent mean: {initial_latent.mean().item():.6f}")
            print(f"   Initial latent std: {initial_latent.std().item():.6f}")
            
            # 2. Fix empty latent channels (ComfyUI reference)
            print("\n2. Fixing empty latent channels (ComfyUI reference)...")
            try:
                # Use ComfyUI's fix_empty_latent_channels
                from comfy.sample import fix_empty_latent_channels
                
                original_latent = initial_latent.clone()
                fixed_latent = fix_empty_latent_channels(unet_model, initial_latent)
                
                fix_data = {
                    'original_shape': list(original_latent.shape),
                    'fixed_shape': list(fixed_latent.shape),
                    'shape_changed': original_latent.shape != fixed_latent.shape,
                    'fixed_mean': fixed_latent.mean().item(),
                    'fixed_std': fixed_latent.std().item(),
                    'fixed_min': fixed_latent.min().item(),
                    'fixed_max': fixed_latent.max().item()
                }
                
                reference['outputs']['fix_empty_channels'] = fix_data
                print(f"   Original shape: {original_latent.shape}")
                print(f"   Fixed shape: {fixed_latent.shape}")
                print(f"   Shape changed: {fix_data['shape_changed']}")
                
            except Exception as e:
                print(f"   ⚠️  fix_empty_latent_channels failed: {e}")
                reference['outputs']['fix_empty_channels'] = {'error': str(e)}
            
            # 3. Prepare noise (ComfyUI reference)
            print("\n3. Preparing noise (ComfyUI reference)...")
            try:
                # Use ComfyUI's prepare_noise
                from comfy.sample import prepare_noise
                
                noise = prepare_noise(fixed_latent, seed, noise_inds)
                
                noise_data = {
                    'noise_shape': list(noise.shape),
                    'noise_dtype': str(noise.dtype),
                    'noise_device': str(noise.device),
                    'noise_mean': noise.mean().item(),
                    'noise_std': noise.std().item(),
                    'noise_min': noise.min().item(),
                    'noise_max': noise.max().item(),
                    'shape_match_fixed_latent': noise.shape == fixed_latent.shape,
                    'device_match_fixed_latent': noise.device == fixed_latent.device,
                    'dtype_match_fixed_latent': noise.dtype == fixed_latent.dtype
                }
                
                reference['outputs']['noise'] = noise_data
                print(f"   Noise shape: {noise.shape}")
                print(f"   Noise mean: {noise.mean().item():.6f}")
                print(f"   Noise std: {noise.std().item():.6f}")
                
            except Exception as e:
                print(f"   ⚠️  prepare_noise failed: {e}")
                reference['outputs']['noise'] = {'error': str(e)}
            
            # 4. Store tensors for next section
            reference['outputs']['fixed_latent_tensor'] = fixed_latent
            reference['outputs']['noise_tensor'] = noise
            
            reference['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.2 REFERENCE COLLECTED in {time.time() - section_start:.2f}s")
            return reference
            
        except Exception as e:
            print(f"❌ SECTION 4.2 REFERENCE COLLECTION FAILED: {str(e)}")
            reference['error'] = str(e)
            reference['timing']['section_time'] = time.time() - section_start
            return reference
    
    def collect_section_4_3_reference(self, 
                                     fixed_latent: torch.Tensor,
                                     unet_model,
                                     steps: int = 4,
                                     sampler_name: str = "euler",
                                     scheduler: str = "normal",
                                     denoise: float = 1.0) -> Dict[str, Any]:
        """
        Collect Section 4.3 reference data: KSampler Setup
        """
        print("\n" + "="*80)
        print("📊 COLLECTING SECTION 4.3 REFERENCE: KSAMPLER SETUP")
        print("="*80)
        
        section_start = time.time()
        reference = {
            'section': '4.3_ksampler_setup',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'comfyui_version': 'reference'
        }
        
        try:
            # 1. Input validation
            print("1. Collecting input data...")
            input_data = {
                'fixed_latent_shape': list(fixed_latent.shape),
                'fixed_latent_dtype': str(fixed_latent.dtype),
                'fixed_latent_device': str(fixed_latent.device),
                'steps': steps,
                'sampler_name': sampler_name,
                'scheduler': scheduler,
                'denoise': denoise
            }
            
            reference['inputs'] = input_data
            print(f"   Fixed latent shape: {fixed_latent.shape}")
            print(f"   Steps: {steps}")
            print(f"   Sampler: {sampler_name}")
            print(f"   Scheduler: {scheduler}")
            print(f"   Denoise: {denoise}")
            
            # 2. UNet model validation
            print("\n2. Validating UNet model...")
            unet_data = {
                'unet_loaded': unet_model is not None,
                'unet_type': str(type(unet_model)),
                'unet_has_forward': hasattr(unet_model, 'forward'),
                'unet_has_call': hasattr(unet_model, '__call__'),
                'unet_device': str(getattr(unet_model, 'device', 'unknown'))
            }
            
            reference['outputs']['unet_validation'] = unet_data
            print(f"   UNet type: {type(unet_model)}")
            print(f"   UNet device: {getattr(unet_model, 'device', 'unknown')}")
            
            # 3. KSampler creation (ComfyUI reference)
            print("\n3. Creating KSampler (ComfyUI reference)...")
            try:
                # Use ComfyUI's KSampler
                from comfy.samplers import KSampler
                
                ksampler = KSampler(
                    model=unet_model,
                    steps=steps,
                    device=unet_model.device,
                    sampler=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise
                )
                
                ksampler_data = {
                    'ksampler_created': ksampler is not None,
                    'ksampler_type': str(type(ksampler)),
                    'model_attached': ksampler.model is not None,
                    'steps_set': ksampler.steps == steps,
                    'device_set': str(ksampler.device),
                    'sampler_set': ksampler.sampler == sampler_name,
                    'scheduler_set': ksampler.scheduler == scheduler,
                    'denoise_set': ksampler.denoise == denoise,
                    'has_sample_method': hasattr(ksampler, 'sample')
                }
                
                reference['outputs']['ksampler'] = ksampler_data
                print(f"   KSampler created: {type(ksampler)}")
                print(f"   KSampler device: {ksampler.device}")
                
            except Exception as e:
                print(f"   ⚠️  KSampler creation failed: {e}")
                reference['outputs']['ksampler'] = {'error': str(e)}
            
            # 4. Store KSampler for next section
            reference['outputs']['ksampler_instance'] = ksampler
            
            reference['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.3 REFERENCE COLLECTED in {time.time() - section_start:.2f}s")
            return reference
            
        except Exception as e:
            print(f"❌ SECTION 4.3 REFERENCE COLLECTION FAILED: {str(e)}")
            reference['error'] = str(e)
            reference['timing']['section_time'] = time.time() - section_start
            return reference
    
    def collect_section_4_4_reference(self, 
                                     ksampler,
                                     noise: torch.Tensor,
                                     positive_conditioning: Any,
                                     negative_conditioning: Any,
                                     fixed_latent: torch.Tensor,
                                     cfg: float = 7.0,
                                     seed: int = 42) -> Dict[str, Any]:
        """
        Collect Section 4.4 reference data: Denoising Execution
        """
        print("\n" + "="*80)
        print("📊 COLLECTING SECTION 4.4 REFERENCE: DENOISING EXECUTION")
        print("="*80)
        
        section_start = time.time()
        reference = {
            'section': '4.4_denoising_execution',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'comfyui_version': 'reference'
        }
        
        try:
            # 1. Input validation
            print("1. Collecting input data...")
            input_data = {
                'noise_shape': list(noise.shape),
                'noise_dtype': str(noise.dtype),
                'noise_device': str(noise.device),
                'noise_mean': noise.mean().item(),
                'noise_std': noise.std().item(),
                'fixed_latent_shape': list(fixed_latent.shape),
                'fixed_latent_dtype': str(fixed_latent.dtype),
                'fixed_latent_device': str(fixed_latent.device),
                'fixed_latent_mean': fixed_latent.mean().item(),
                'fixed_latent_std': fixed_latent.std().item(),
                'cfg': cfg,
                'seed': seed
            }
            
            reference['inputs'] = input_data
            print(f"   Noise shape: {noise.shape}")
            print(f"   Fixed latent shape: {fixed_latent.shape}")
            print(f"   CFG: {cfg}")
            print(f"   Seed: {seed}")
            
            # 2. Conditioning analysis
            print("\n2. Analyzing conditioning...")
            conditioning_data = {
                'positive_conditioning_type': str(type(positive_conditioning)),
                'negative_conditioning_type': str(type(negative_conditioning)),
                'positive_conditioning_length': len(positive_conditioning) if isinstance(positive_conditioning, list) else None,
                'negative_conditioning_length': len(negative_conditioning) if isinstance(negative_conditioning, list) else None
            }
            
            # Analyze conditioning tensors
            if isinstance(positive_conditioning, list):
                for i, item in enumerate(positive_conditioning):
                    if hasattr(item, 'shape'):
                        conditioning_data[f'positive_tensor_{i}_shape'] = list(item.shape)
                        conditioning_data[f'positive_tensor_{i}_dtype'] = str(item.dtype)
                        conditioning_data[f'positive_tensor_{i}_device'] = str(item.device)
                        conditioning_data[f'positive_tensor_{i}_mean'] = item.mean().item()
                        conditioning_data[f'positive_tensor_{i}_std'] = item.std().item()
            
            if isinstance(negative_conditioning, list):
                for i, item in enumerate(negative_conditioning):
                    if hasattr(item, 'shape'):
                        conditioning_data[f'negative_tensor_{i}_shape'] = list(item.shape)
                        conditioning_data[f'negative_tensor_{i}_dtype'] = str(item.dtype)
                        conditioning_data[f'negative_tensor_{i}_device'] = str(item.device)
                        conditioning_data[f'negative_tensor_{i}_mean'] = item.mean().item()
                        conditioning_data[f'negative_tensor_{i}_std'] = item.std().item()
            
            reference['outputs']['conditioning_analysis'] = conditioning_data
            
            # 3. CRITICAL: Perform denoising (ComfyUI reference)
            print("\n3. 🎯 CRITICAL: Performing denoising (ComfyUI reference)...")
            denoising_start = time.time()
            
            try:
                # Use ComfyUI's KSampler sample method
                denoised_latent = ksampler.sample(
                    noise=noise,
                    positive=positive_conditioning,
                    negative=negative_conditioning,
                    cfg=cfg,
                    latent_image=fixed_latent,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False,
                    denoise_mask=None,
                    sigmas=None,
                    callback=None,
                    disable_pbar=False,
                    seed=seed
                )
                
                denoising_time = time.time() - denoising_start
                print(f"   ✅ ComfyUI denoising completed in {denoising_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ ComfyUI denoising failed: {e}")
                reference['outputs']['denoising'] = {'error': str(e)}
                reference['timing']['section_time'] = time.time() - section_start
                return reference
            
            # 4. CRITICAL: Analyze denoising results
            print("\n4. 🎯 CRITICAL: Analyzing denoising results...")
            denoising_data = {
                'denoised_latent_shape': list(denoised_latent.shape),
                'denoised_latent_dtype': str(denoised_latent.dtype),
                'denoised_latent_device': str(denoised_latent.device),
                'denoised_latent_mean': denoised_latent.mean().item(),
                'denoised_latent_std': denoised_latent.std().item(),
                'denoised_latent_min': denoised_latent.min().item(),
                'denoised_latent_max': denoised_latent.max().item(),
                'shape_preserved': denoised_latent.shape == fixed_latent.shape,
                'device_consistent': denoised_latent.device == fixed_latent.device,
                'dtype_consistent': denoised_latent.dtype == fixed_latent.dtype,
                'values_finite': torch.isfinite(denoised_latent).all().item(),
                'denoising_time': denoising_time
            }
            
            # Calculate differences
            if denoised_latent.shape == fixed_latent.shape:
                diff = torch.abs(denoised_latent - fixed_latent)
                denoising_data['mean_absolute_diff'] = diff.mean().item()
                denoising_data['max_absolute_diff'] = diff.max().item()
                denoising_data['std_absolute_diff'] = diff.std().item()
                denoising_data['significant_change'] = diff.mean().item() > 0.001
            
            reference['outputs']['denoising'] = denoising_data
            
            # 5. Store final output
            reference['outputs']['denoised_latent_tensor'] = denoised_latent
            
            reference['timing']['denoising_time'] = denoising_time
            reference['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.4 REFERENCE COLLECTED in {time.time() - section_start:.2f}s")
            return reference
            
        except Exception as e:
            print(f"❌ SECTION 4.4 REFERENCE COLLECTION FAILED: {str(e)}")
            reference['error'] = str(e)
            reference['timing']['section_time'] = time.time() - section_start
            return reference
    
    def collect_all_sections_reference(self, 
                                     initial_latent: torch.Tensor,
                                     positive_conditioning: Any,
                                     negative_conditioning: Any,
                                     unet_model,
                                     seed: int = 42,
                                     steps: int = 4,
                                     cfg: float = 7.0,
                                     sampler_name: str = "euler",
                                     scheduler: str = "normal",
                                     denoise: float = 1.0,
                                     noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Collect reference data for all critical sections
        """
        print("\n" + "="*80)
        print("🚀 COLLECTING ALL CRITICAL SECTIONS REFERENCE DATA")
        print("="*80)
        print("⚠️  Collecting ComfyUI reference data for motion pipeline verification")
        
        total_start = time.time()
        all_reference = {
            'sections': {},
            'overall_status': 'success',
            'timing': {},
            'comfyui_version': 'reference',
            'collection_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'purpose': 'motion_pipeline_verification'
            }
        }
        
        try:
            # Section 4.2: Latent Preparation
            print("\n" + "="*60)
            print("🔧 COLLECTING SECTION 4.2 REFERENCE")
            print("="*60)
            
            section_4_2 = self.collect_section_4_2_reference(
                initial_latent, unet_model, seed, noise_inds
            )
            all_reference['sections']['4.2'] = section_4_2
            
            if 'error' in section_4_2:
                print(f"❌ SECTION 4.2 REFERENCE COLLECTION FAILED")
                all_reference['overall_status'] = 'failed'
                all_reference['error'] = f"Section 4.2 failed: {section_4_2['error']}"
                return all_reference
            
            # Section 4.3: KSampler Setup
            print("\n" + "="*60)
            print("⚙️ COLLECTING SECTION 4.3 REFERENCE")
            print("="*60)
            
            section_4_3 = self.collect_section_4_3_reference(
                section_4_2['outputs']['fixed_latent_tensor'],
                unet_model, steps, sampler_name, scheduler, denoise
            )
            all_reference['sections']['4.3'] = section_4_3
            
            if 'error' in section_4_3:
                print(f"❌ SECTION 4.3 REFERENCE COLLECTION FAILED")
                all_reference['overall_status'] = 'failed'
                all_reference['error'] = f"Section 4.3 failed: {section_4_3['error']}"
                return all_reference
            
            # Section 4.4: Denoising Execution
            print("\n" + "="*60)
            print("🎯 COLLECTING SECTION 4.4 REFERENCE")
            print("="*60)
            
            section_4_4 = self.collect_section_4_4_reference(
                section_4_3['outputs']['ksampler_instance'],
                section_4_2['outputs']['noise_tensor'],
                positive_conditioning,
                negative_conditioning,
                section_4_2['outputs']['fixed_latent_tensor'],
                cfg, seed
            )
            all_reference['sections']['4.4'] = section_4_4
            
            if 'error' in section_4_4:
                print(f"❌ SECTION 4.4 REFERENCE COLLECTION FAILED")
                all_reference['overall_status'] = 'failed'
                all_reference['error'] = f"Section 4.4 failed: {section_4_4['error']}"
                return all_reference
            
            # Store final output
            all_reference['final_output'] = section_4_4['outputs']['denoised_latent_tensor']
            all_reference['timing']['total_time'] = time.time() - total_start
            
            print(f"\n🎉 ALL CRITICAL SECTIONS REFERENCE COLLECTED in {time.time() - total_start:.2f}s")
            print("✅ ComfyUI reference data ready for motion pipeline verification")
            return all_reference
            
        except Exception as e:
            print(f"❌ REFERENCE COLLECTION FAILED: {str(e)}")
            all_reference['overall_status'] = 'failed'
            all_reference['error'] = str(e)
            all_reference['timing']['total_time'] = time.time() - total_start
            return all_reference
    
    def save_reference_data(self, reference_data: Dict[str, Any], filename: str = None) -> str:
        """
        Save reference data to JSON file
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"comfyui_step4_reference_{timestamp}.json"
        
        # Convert tensors to serializable format
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return {
                    'tensor_data': obj.detach().cpu().numpy().tolist(),
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'device': str(obj.device)
                }
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_tensors(reference_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"📁 Reference data saved to: {filename}")
        return filename

def main():
    """
    Main function to run the reference data collector
    This should be run inside ComfyUI environment
    """
    print("🚀 ComfyUI Step 4 Reference Data Collector")
    print("=" * 60)
    
    print("This script collects reference data from ComfyUI for motion pipeline verification.")
    print("Run this inside ComfyUI environment with the following steps:")
    print()
    print("1. Load your models (UNet, CLIP, etc.)")
    print("2. Prepare your initial latent tensor")
    print("3. Prepare your positive and negative conditioning")
    print("4. Run this collector to get reference data")
    print("5. Use the reference data to verify motion pipeline")
    print()
    print("Example usage:")
    print("```python")
    print("collector = ComfyUIStep4ReferenceCollector()")
    print("reference_data = collector.collect_all_sections_reference(")
    print("    initial_latent=your_latent,")
    print("    positive_conditioning=your_positive_cond,")
    print("    negative_conditioning=your_negative_cond,")
    print("    unet_model=your_unet,")
    print("    seed=42, steps=4, cfg=7.0")
    print(")")
    print("filename = collector.save_reference_data(reference_data)")
    print("```")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
