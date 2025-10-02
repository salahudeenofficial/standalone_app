#!/usr/bin/env python3
"""
Step 4 KSampling Debugging Sections
Divides Step 4 into manageable sections for debugging with ComfyUI reference
"""

import sys
import os
from pathlib import Path
import torch
import time
from typing import Dict, Any, Optional, List, Tuple

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

class Step4DebuggingSections:
    """
    Step 4 KSampling divided into debugging sections
    Each section can be tested independently with ComfyUI reference
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.unet = pipeline.unet
        self.clip = pipeline.clip
        
        # Section results storage
        self.section_results = {}
        
    def section_4_1_input_validation(self, 
                                   initial_latent: torch.Tensor,
                                   positive_conditioning: Any,
                                   negative_conditioning: Any,
                                   seed: int = 42,
                                   steps: int = 4,
                                   cfg: float = 7.0,
                                   sampler_name: str = "euler",
                                   scheduler: str = "normal",
                                   denoise: float = 1.0,
                                   noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Section 4.1: Input Validation and Prerequisites
        Validates all inputs and prerequisites for Step 4
        """
        print("\n" + "="*80)
        print("🔍 SECTION 4.1: INPUT VALIDATION AND PREREQUISITES")
        print("="*80)
        
        section_start = time.time()
        results = {
            'section': '4.1_input_validation',
            'inputs': {},
            'validation': {},
            'prerequisites': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Validate prerequisites from previous steps
            print("1. Validating prerequisites from previous steps...")
            prerequisites = {
                'step_3_completed': self.pipeline.step_completed[3],
                'unet_loaded': self.unet is not None,
                'clip_loaded': self.clip is not None
            }
            
            for key, value in prerequisites.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    raise RuntimeError(f"Prerequisite failed: {key}")
            
            results['prerequisites'] = prerequisites
            
            # 2. Validate input parameters
            print("\n2. Validating input parameters...")
            params = {
                'seed': seed,
                'steps': steps,
                'cfg': cfg,
                'sampler_name': sampler_name,
                'scheduler': scheduler,
                'denoise': denoise,
                'noise_inds_provided': noise_inds is not None
            }
            
            for key, value in params.items():
                print(f"   {key}: {value}")
            
            results['inputs']['parameters'] = params
            
            # 3. Validate initial_latent tensor
            print("\n3. Validating initial_latent tensor...")
            latent_validation = {
                'is_tensor': isinstance(initial_latent, torch.Tensor),
                'shape': initial_latent.shape if isinstance(initial_latent, torch.Tensor) else None,
                'dtype': initial_latent.dtype if isinstance(initial_latent, torch.Tensor) else None,
                'device': initial_latent.device if isinstance(initial_latent, torch.Tensor) else None,
                'mean': initial_latent.mean().item() if isinstance(initial_latent, torch.Tensor) else None,
                'std': initial_latent.std().item() if isinstance(initial_latent, torch.Tensor) else None,
                'min': initial_latent.min().item() if isinstance(initial_latent, torch.Tensor) else None,
                'max': initial_latent.max().item() if isinstance(initial_latent, torch.Tensor) else None
            }
            
            for key, value in latent_validation.items():
                print(f"   {key}: {value}")
            
            results['inputs']['initial_latent'] = latent_validation
            
            # 4. Validate positive conditioning
            print("\n4. Validating positive conditioning...")
            pos_validation = self._validate_conditioning(positive_conditioning, "positive")
            results['inputs']['positive_conditioning'] = pos_validation
            
            # 5. Validate negative conditioning
            print("\n5. Validating negative conditioning...")
            neg_validation = self._validate_conditioning(negative_conditioning, "negative")
            results['inputs']['negative_conditioning'] = neg_validation
            
            # 6. Overall validation status
            print("\n6. Overall validation status...")
            all_valid = (
                prerequisites['step_3_completed'] and
                prerequisites['unet_loaded'] and
                prerequisites['clip_loaded'] and
                latent_validation['is_tensor']
            )
            
            status = "✅ ALL VALID" if all_valid else "❌ VALIDATION FAILED"
            print(f"   Overall status: {status}")
            
            results['validation']['overall_status'] = all_valid
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.1 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ SECTION 4.1 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def section_4_2_latent_preparation(self, 
                                     initial_latent: torch.Tensor,
                                     seed: int = 42,
                                     noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Section 4.2: Latent Preparation
        Prepares the latent tensor and generates noise
        """
        print("\n" + "="*80)
        print("🔧 SECTION 4.2: LATENT PREPARATION")
        print("="*80)
        
        section_start = time.time()
        results = {
            'section': '4.2_latent_preparation',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Fix empty latent channels (ComfyUI pattern)
            print("1. Fixing empty latent channels...")
            from sample import fix_empty_latent_channels
            
            original_latent = initial_latent.clone()
            fixed_latent = fix_empty_latent_channels(self.unet, initial_latent)
            
            print(f"   Original shape: {original_latent.shape}")
            print(f"   Fixed shape: {fixed_latent.shape}")
            print(f"   Shape changed: {'✅ YES' if original_latent.shape != fixed_latent.shape else '❌ NO'}")
            
            results['inputs']['original_latent'] = {
                'shape': original_latent.shape,
                'mean': original_latent.mean().item(),
                'std': original_latent.std().item()
            }
            
            results['outputs']['fixed_latent'] = {
                'shape': fixed_latent.shape,
                'mean': fixed_latent.mean().item(),
                'std': fixed_latent.std().item()
            }
            
            # 2. Prepare noise
            print("\n2. Preparing noise...")
            from sample import prepare_noise
            
            noise = prepare_noise(fixed_latent, seed, noise_inds)
            
            print(f"   Noise shape: {noise.shape}")
            print(f"   Noise mean: {noise.mean().item():.6f}")
            print(f"   Noise std: {noise.std().item():.6f}")
            print(f"   Noise range: [{noise.min().item():.6f}, {noise.max().item():.6f}]")
            
            results['outputs']['noise'] = {
                'shape': noise.shape,
                'mean': noise.mean().item(),
                'std': noise.std().item(),
                'min': noise.min().item(),
                'max': noise.max().item()
            }
            
            # 3. Validate noise properties
            print("\n3. Validating noise properties...")
            noise_validation = {
                'shape_match': noise.shape == fixed_latent.shape,
                'device_match': noise.device == fixed_latent.device,
                'dtype_match': noise.dtype == fixed_latent.dtype,
                'is_random': abs(noise.mean().item()) < 0.1,  # Should be close to 0
                'has_variance': noise.std().item() > 0.5  # Should have reasonable variance
            }
            
            for key, value in noise_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
            
            results['outputs']['noise_validation'] = noise_validation
            
            # 4. Store outputs for next section
            results['outputs']['fixed_latent_tensor'] = fixed_latent
            results['outputs']['noise_tensor'] = noise
            
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.2 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ SECTION 4.2 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def section_4_3_ksampler_setup(self, 
                                  fixed_latent: torch.Tensor,
                                  steps: int = 4,
                                  sampler_name: str = "euler",
                                  scheduler: str = "normal",
                                  denoise: float = 1.0) -> Dict[str, Any]:
        """
        Section 4.3: KSampler Setup
        Creates and configures the KSampler instance
        """
        print("\n" + "="*80)
        print("⚙️ SECTION 4.3: KSAMPLER SETUP")
        print("="*80)
        
        section_start = time.time()
        results = {
            'section': '4.3_ksampler_setup',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Create KSampler instance
            print("1. Creating KSampler instance...")
            from standalone_ksampler import StandaloneKSampler
            
            ksampler = StandaloneKSampler(
                model=self.unet,
                steps=steps,
                device=self.device,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise
            )
            
            print(f"   KSampler created successfully")
            print(f"   Model type: {type(ksampler.model)}")
            print(f"   Steps: {ksampler.steps}")
            print(f"   Device: {ksampler.device}")
            print(f"   Sampler: {ksampler.sampler}")
            print(f"   Scheduler: {ksampler.scheduler}")
            print(f"   Denoise: {ksampler.denoise}")
            
            results['inputs']['ksampler_config'] = {
                'steps': steps,
                'sampler_name': sampler_name,
                'scheduler': scheduler,
                'denoise': denoise,
                'device': str(self.device)
            }
            
            # 2. Validate KSampler properties
            print("\n2. Validating KSampler properties...")
            ksampler_validation = {
                'model_loaded': ksampler.model is not None,
                'steps_set': ksampler.steps == steps,
                'device_set': ksampler.device == self.device,
                'sampler_set': ksampler.sampler == sampler_name,
                'scheduler_set': ksampler.scheduler == scheduler,
                'denoise_set': ksampler.denoise == denoise
            }
            
            for key, value in ksampler_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
            
            results['outputs']['ksampler_validation'] = ksampler_validation
            
            # 3. Test KSampler methods
            print("\n3. Testing KSampler methods...")
            method_validation = {
                'has_sample_method': hasattr(ksampler, 'sample'),
                'has_get_sigmas_method': hasattr(ksampler, 'get_sigmas'),
                'has_get_noise_method': hasattr(ksampler, 'get_noise')
            }
            
            for key, value in method_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
            
            results['outputs']['method_validation'] = method_validation
            
            # 4. Store KSampler for next section
            results['outputs']['ksampler'] = ksampler
            
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.3 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ SECTION 4.3 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def section_4_4_denoising_execution(self, 
                                      ksampler,
                                      noise: torch.Tensor,
                                      positive_conditioning: Any,
                                      negative_conditioning: Any,
                                      fixed_latent: torch.Tensor,
                                      cfg: float = 7.0,
                                      seed: int = 42) -> Dict[str, Any]:
        """
        Section 4.4: Denoising Execution
        Performs the actual denoising process
        """
        print("\n" + "="*80)
        print("🎯 SECTION 4.4: DENOISING EXECUTION")
        print("="*80)
        
        section_start = time.time()
        results = {
            'section': '4.4_denoising_execution',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Clear CUDA cache before denoising
            print("1. Clearing CUDA cache...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   CUDA cache cleared")
            else:
                print("   CUDA not available, skipping cache clear")
            
            # 2. Perform denoising
            print("\n2. Performing denoising...")
            denoising_start = time.time()
            
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
            
            print(f"   Denoising completed in {denoising_time:.2f}s")
            print(f"   Denoised latent shape: {denoised_latent.shape}")
            print(f"   Denoised latent mean: {denoised_latent.mean().item():.6f}")
            print(f"   Denoised latent std: {denoised_latent.std().item():.6f}")
            print(f"   Denoised latent range: [{denoised_latent.min().item():.6f}, {denoised_latent.max().item():.6f}]")
            
            results['inputs']['denoising_config'] = {
                'cfg': cfg,
                'seed': seed,
                'noise_shape': noise.shape,
                'fixed_latent_shape': fixed_latent.shape
            }
            
            results['outputs']['denoised_latent'] = {
                'shape': denoised_latent.shape,
                'mean': denoised_latent.mean().item(),
                'std': denoised_latent.std().item(),
                'min': denoised_latent.min().item(),
                'max': denoised_latent.max().item()
            }
            
            # 3. Validate denoising results
            print("\n3. Validating denoising results...")
            denoising_validation = {
                'shape_preserved': denoised_latent.shape == fixed_latent.shape,
                'device_consistent': denoised_latent.device == fixed_latent.device,
                'dtype_consistent': denoised_latent.dtype == fixed_latent.dtype,
                'has_changes': not torch.allclose(denoised_latent, fixed_latent, atol=1e-6),
                'reasonable_values': torch.isfinite(denoised_latent).all()
            }
            
            for key, value in denoising_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
            
            results['outputs']['denoising_validation'] = denoising_validation
            
            # 4. Calculate differences
            print("\n4. Calculating differences...")
            if denoised_latent.shape == fixed_latent.shape:
                diff = torch.abs(denoised_latent - fixed_latent)
                diff_stats = {
                    'mean_absolute_diff': diff.mean().item(),
                    'max_absolute_diff': diff.max().item(),
                    'std_absolute_diff': diff.std().item()
                }
                
                for key, value in diff_stats.items():
                    print(f"   {key}: {value:.6f}")
                
                results['outputs']['difference_stats'] = diff_stats
            
            # 5. Store outputs
            results['outputs']['denoised_latent_tensor'] = denoised_latent
            results['timing']['denoising_time'] = denoising_time
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.4 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ SECTION 4.4 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def section_4_5_cleanup_and_validation(self, 
                                          denoised_latent: torch.Tensor,
                                          fixed_latent: torch.Tensor,
                                          ksampler) -> Dict[str, Any]:
        """
        Section 4.5: Cleanup and Final Validation
        Performs cleanup and final validation of results
        """
        print("\n" + "="*80)
        print("🧹 SECTION 4.5: CLEANUP AND FINAL VALIDATION")
        print("="*80)
        
        section_start = time.time()
        results = {
            'section': '4.5_cleanup_and_validation',
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Ensure device consistency
            print("1. Ensuring device consistency...")
            if fixed_latent.device != denoised_latent.device:
                print(f"   Device mismatch detected: {fixed_latent.device} vs {denoised_latent.device}")
                denoised_latent = denoised_latent.to(fixed_latent.device)
                print(f"   Moved denoised_latent to: {denoised_latent.device}")
            else:
                print(f"   Devices consistent: {denoised_latent.device}")
            
            results['inputs']['device_consistency'] = {
                'fixed_latent_device': str(fixed_latent.device),
                'denoised_latent_device': str(denoised_latent.device),
                'consistent': fixed_latent.device == denoised_latent.device
            }
            
            # 2. UNet cleanup for memory management
            print("\n2. Performing UNet cleanup...")
            cleanup_methods = []
            
            if hasattr(self.unet, 'cleanup'):
                try:
                    self.unet.cleanup()
                    cleanup_methods.append('cleanup')
                    print("   UNet cleanup() called")
                except Exception as e:
                    print(f"   UNet cleanup() failed: {e}")
            
            if hasattr(self.unet, 'unload'):
                try:
                    self.unet.unload()
                    cleanup_methods.append('unload')
                    print("   UNet unload() called")
                except Exception as e:
                    print(f"   UNet unload() failed: {e}")
            
            if not cleanup_methods:
                print("   No cleanup methods available for UNet")
            
            results['outputs']['cleanup_methods'] = cleanup_methods
            
            # 3. Clear CUDA cache
            print("\n3. Clearing CUDA cache...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   CUDA cache cleared")
            else:
                print("   CUDA not available, skipping cache clear")
            
            # 4. Final validation
            print("\n4. Final validation...")
            final_validation = {
                'denoised_latent_is_tensor': isinstance(denoised_latent, torch.Tensor),
                'shape_valid': len(denoised_latent.shape) == 5,  # [B, C, T, H, W]
                'device_valid': denoised_latent.device.type in ['cpu', 'cuda'],
                'dtype_valid': denoised_latent.dtype in [torch.float16, torch.float32],
                'values_finite': torch.isfinite(denoised_latent).all(),
                'shape_match_input': denoised_latent.shape == fixed_latent.shape
            }
            
            for key, value in final_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
            
            results['outputs']['final_validation'] = final_validation
            
            # 5. Summary statistics
            print("\n5. Summary statistics...")
            summary_stats = {
                'input_shape': fixed_latent.shape,
                'output_shape': denoised_latent.shape,
                'input_mean': fixed_latent.mean().item(),
                'output_mean': denoised_latent.mean().item(),
                'input_std': fixed_latent.std().item(),
                'output_std': denoised_latent.std().item(),
                'input_range': [fixed_latent.min().item(), fixed_latent.max().item()],
                'output_range': [denoised_latent.min().item(), denoised_latent.max().item()]
            }
            
            for key, value in summary_stats.items():
                print(f"   {key}: {value}")
            
            results['outputs']['summary_stats'] = summary_stats
            
            # 6. Store final outputs
            results['outputs']['final_denoised_latent'] = denoised_latent
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ SECTION 4.5 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ SECTION 4.5 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def _validate_conditioning(self, conditioning, name):
        """Validate conditioning structure"""
        validation = {
            'is_list': isinstance(conditioning, list),
            'length': len(conditioning) if isinstance(conditioning, list) else None,
            'has_text_tensor': False,
            'has_vace_data': False,
            'tensor_count': 0
        }
        
        if isinstance(conditioning, list):
            for i, item in enumerate(conditioning):
                if hasattr(item, 'shape'):
                    validation['tensor_count'] += 1
                    if i == 0:  # First item should be text tensor
                        validation['has_text_tensor'] = True
                        print(f"   Text tensor shape: {item.shape}")
                elif isinstance(item, dict):
                    if 'vace_frames' in item or 'vace_mask' in item:
                        validation['has_vace_data'] = True
                        print(f"   VACE data found in item {i}")
        
        for key, value in validation.items():
            print(f"   {key}: {value}")
        
        return validation
    
    def run_all_sections(self, 
                        initial_latent: torch.Tensor,
                        positive_conditioning: Any,
                        negative_conditioning: Any,
                        seed: int = 42,
                        steps: int = 4,
                        cfg: float = 7.0,
                        sampler_name: str = "euler",
                        scheduler: str = "normal",
                        denoise: float = 1.0,
                        noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Run all Step 4 sections in sequence
        """
        print("\n" + "="*80)
        print("🚀 RUNNING ALL STEP 4 SECTIONS")
        print("="*80)
        
        total_start = time.time()
        all_results = {
            'sections': {},
            'overall_status': 'success',
            'timing': {},
            'final_output': None
        }
        
        try:
            # Section 4.1: Input Validation
            section_4_1 = self.section_4_1_input_validation(
                initial_latent, positive_conditioning, negative_conditioning,
                seed, steps, cfg, sampler_name, scheduler, denoise, noise_inds
            )
            all_results['sections']['4.1'] = section_4_1
            
            if section_4_1['status'] != 'success':
                raise RuntimeError(f"Section 4.1 failed: {section_4_1.get('error', 'Unknown error')}")
            
            # Section 4.2: Latent Preparation
            section_4_2 = self.section_4_2_latent_preparation(
                initial_latent, seed, noise_inds
            )
            all_results['sections']['4.2'] = section_4_2
            
            if section_4_2['status'] != 'success':
                raise RuntimeError(f"Section 4.2 failed: {section_4_2.get('error', 'Unknown error')}")
            
            # Section 4.3: KSampler Setup
            section_4_3 = self.section_4_3_ksampler_setup(
                section_4_2['outputs']['fixed_latent_tensor'],
                steps, sampler_name, scheduler, denoise
            )
            all_results['sections']['4.3'] = section_4_3
            
            if section_4_3['status'] != 'success':
                raise RuntimeError(f"Section 4.3 failed: {section_4_3.get('error', 'Unknown error')}")
            
            # Section 4.4: Denoising Execution
            section_4_4 = self.section_4_4_denoising_execution(
                section_4_3['outputs']['ksampler'],
                section_4_2['outputs']['noise_tensor'],
                positive_conditioning,
                negative_conditioning,
                section_4_2['outputs']['fixed_latent_tensor'],
                cfg, seed
            )
            all_results['sections']['4.4'] = section_4_4
            
            if section_4_4['status'] != 'success':
                raise RuntimeError(f"Section 4.4 failed: {section_4_4.get('error', 'Unknown error')}")
            
            # Section 4.5: Cleanup and Validation
            section_4_5 = self.section_4_5_cleanup_and_validation(
                section_4_4['outputs']['denoised_latent_tensor'],
                section_4_2['outputs']['fixed_latent_tensor'],
                section_4_3['outputs']['ksampler']
            )
            all_results['sections']['4.5'] = section_4_5
            
            if section_4_5['status'] != 'success':
                raise RuntimeError(f"Section 4.5 failed: {section_4_5.get('error', 'Unknown error')}")
            
            # Store final output
            all_results['final_output'] = section_4_5['outputs']['final_denoised_latent']
            all_results['timing']['total_time'] = time.time() - total_start
            
            print(f"\n🎉 ALL STEP 4 SECTIONS COMPLETED in {time.time() - total_start:.2f}s")
            return all_results
            
        except Exception as e:
            print(f"❌ STEP 4 SECTIONS FAILED: {str(e)}")
            all_results['overall_status'] = 'failed'
            all_results['error'] = str(e)
            all_results['timing']['total_time'] = time.time() - total_start
            return all_results

def main():
    """Test the Step 4 debugging sections"""
    print("🚀 Step 4 Debugging Sections Test")
    print("=" * 60)
    
    # This would be used with actual pipeline instance
    # For testing, you would create mock data and test each section
    
    print("Step 4 debugging sections are ready for use!")
    print("Each section can be tested independently with ComfyUI reference data.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
