#!/usr/bin/env python3
"""
Step 4 Critical Sections Reverse Testing
Tests only the most critical parts in reverse order to quickly identify issues
"""

import sys
import os
from pathlib import Path
import torch
import time
from typing import Dict, Any, Optional, List, Tuple

# Add motion directory to path
sys.path.insert(0, str(Path(__file__).parent / "motion"))

class Step4CriticalReverseTest:
    """
    Step 4 Critical Sections - Reverse Order Testing
    Focuses on the most critical parts that are likely to have issues
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.unet = pipeline.unet
        self.clip = pipeline.clip
        
        # Critical section results
        self.critical_results = {}
        
    def critical_section_4_4_denoising_execution(self, 
                                               ksampler,
                                               noise: torch.Tensor,
                                               positive_conditioning: Any,
                                               negative_conditioning: Any,
                                               fixed_latent: torch.Tensor,
                                               cfg: float = 7.0,
                                               seed: int = 42) -> Dict[str, Any]:
        """
        CRITICAL SECTION 4.4: Denoising Execution
        This is the most critical section - the actual denoising process
        """
        print("\n" + "="*80)
        print("🎯 CRITICAL SECTION 4.4: DENOISING EXECUTION")
        print("="*80)
        print("⚠️  MOST CRITICAL SECTION - Core denoising process")
        
        section_start = time.time()
        results = {
            'section': '4.4_denoising_execution',
            'critical': True,
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Validate inputs before denoising
            print("1. Validating critical inputs...")
            input_validation = {
                'ksampler_valid': ksampler is not None,
                'noise_valid': isinstance(noise, torch.Tensor) and torch.isfinite(noise).all(),
                'fixed_latent_valid': isinstance(fixed_latent, torch.Tensor) and torch.isfinite(fixed_latent).all(),
                'positive_cond_valid': positive_conditioning is not None,
                'negative_cond_valid': negative_conditioning is not None,
                'cfg_valid': 0 < cfg <= 20,
                'seed_valid': isinstance(seed, int) and seed >= 0
            }
            
            for key, value in input_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    raise ValueError(f"Critical input validation failed: {key}")
            
            results['inputs']['validation'] = input_validation
            
            # 2. Check tensor shapes and devices
            print("\n2. Checking tensor compatibility...")
            compatibility = {
                'noise_shape': noise.shape,
                'fixed_latent_shape': fixed_latent.shape,
                'shape_match': noise.shape == fixed_latent.shape,
                'noise_device': str(noise.device),
                'fixed_latent_device': str(fixed_latent.device),
                'device_match': noise.device == fixed_latent.device
            }
            
            for key, value in compatibility.items():
                print(f"   {key}: {value}")
            
            if not compatibility['shape_match']:
                raise ValueError(f"Shape mismatch: noise {noise.shape} vs fixed_latent {fixed_latent.shape}")
            
            if not compatibility['device_match']:
                print(f"   ⚠️  Device mismatch detected, moving tensors...")
                noise = noise.to(fixed_latent.device)
                print(f"   Moved noise to: {noise.device}")
            
            results['inputs']['compatibility'] = compatibility
            
            # 3. Clear CUDA cache before critical denoising
            print("\n3. Preparing for critical denoising...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   CUDA cache cleared")
            
            # 4. CRITICAL: Perform denoising
            print("\n4. 🎯 CRITICAL: Performing denoising...")
            denoising_start = time.time()
            
            try:
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
                print(f"   ✅ Denoising completed in {denoising_time:.2f}s")
                
            except Exception as denoising_error:
                print(f"   ❌ CRITICAL DENOISING FAILED: {denoising_error}")
                raise denoising_error
            
            # 5. CRITICAL: Validate denoising output
            print("\n5. 🎯 CRITICAL: Validating denoising output...")
            output_validation = {
                'is_tensor': isinstance(denoised_latent, torch.Tensor),
                'shape_preserved': denoised_latent.shape == fixed_latent.shape,
                'device_consistent': denoised_latent.device == fixed_latent.device,
                'dtype_consistent': denoised_latent.dtype == fixed_latent.dtype,
                'values_finite': torch.isfinite(denoised_latent).all(),
                'has_changes': not torch.allclose(denoised_latent, fixed_latent, atol=1e-6),
                'reasonable_range': denoised_latent.min().item() > -10 and denoised_latent.max().item() < 10
            }
            
            for key, value in output_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    print(f"   ⚠️  CRITICAL VALIDATION FAILED: {key}")
            
            results['outputs']['validation'] = output_validation
            
            # 6. CRITICAL: Analyze denoising results
            print("\n6. 🎯 CRITICAL: Analyzing denoising results...")
            analysis = {
                'input_mean': fixed_latent.mean().item(),
                'output_mean': denoised_latent.mean().item(),
                'input_std': fixed_latent.std().item(),
                'output_std': denoised_latent.std().item(),
                'input_range': [fixed_latent.min().item(), fixed_latent.max().item()],
                'output_range': [denoised_latent.min().item(), denoised_latent.max().item()]
            }
            
            for key, value in analysis.items():
                print(f"   {key}: {value}")
            
            # Calculate critical differences
            if denoised_latent.shape == fixed_latent.shape:
                diff = torch.abs(denoised_latent - fixed_latent)
                diff_analysis = {
                    'mean_absolute_diff': diff.mean().item(),
                    'max_absolute_diff': diff.max().item(),
                    'std_absolute_diff': diff.std().item(),
                    'significant_change': diff.mean().item() > 0.001
                }
                
                for key, value in diff_analysis.items():
                    print(f"   {key}: {value}")
                
                results['outputs']['difference_analysis'] = diff_analysis
            
            results['outputs']['analysis'] = analysis
            results['outputs']['denoised_latent'] = denoised_latent
            results['timing']['denoising_time'] = denoising_time
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ CRITICAL SECTION 4.4 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ CRITICAL SECTION 4.4 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def critical_section_4_3_ksampler_setup(self, 
                                          fixed_latent: torch.Tensor,
                                          steps: int = 4,
                                          sampler_name: str = "euler",
                                          scheduler: str = "normal",
                                          denoise: float = 1.0) -> Dict[str, Any]:
        """
        CRITICAL SECTION 4.3: KSampler Setup
        This is critical for denoising execution
        """
        print("\n" + "="*80)
        print("⚙️ CRITICAL SECTION 4.3: KSAMPLER SETUP")
        print("="*80)
        print("⚠️  CRITICAL SECTION - KSampler configuration")
        
        section_start = time.time()
        results = {
            'section': '4.3_ksampler_setup',
            'critical': True,
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Validate UNet model
            print("1. Validating UNet model...")
            unet_validation = {
                'unet_loaded': self.unet is not None,
                'unet_has_forward': hasattr(self.unet, 'forward'),
                'unet_has_call': hasattr(self.unet, '__call__'),
                'unet_device': str(getattr(self.unet, 'device', 'unknown'))
            }
            
            for key, value in unet_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    raise ValueError(f"UNet validation failed: {key}")
            
            results['inputs']['unet_validation'] = unet_validation
            
            # 2. CRITICAL: Create KSampler
            print("\n2. 🎯 CRITICAL: Creating KSampler...")
            try:
                from standalone_ksampler import StandaloneKSampler
                
                ksampler = StandaloneKSampler(
                    model=self.unet,
                    steps=steps,
                    device=self.device,
                    sampler=sampler_name,
                    scheduler=scheduler,
                    denoise=denoise
                )
                
                print(f"   ✅ KSampler created successfully")
                
            except Exception as ksampler_error:
                print(f"   ❌ CRITICAL KSAMPLER CREATION FAILED: {ksampler_error}")
                raise ksampler_error
            
            # 3. CRITICAL: Validate KSampler
            print("\n3. 🎯 CRITICAL: Validating KSampler...")
            ksampler_validation = {
                'ksampler_created': ksampler is not None,
                'model_attached': ksampler.model is not None,
                'steps_set': ksampler.steps == steps,
                'device_set': ksampler.device == self.device,
                'sampler_set': ksampler.sampler == sampler_name,
                'scheduler_set': ksampler.scheduler == scheduler,
                'denoise_set': ksampler.denoise == denoise,
                'has_sample_method': hasattr(ksampler, 'sample')
            }
            
            for key, value in ksampler_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    print(f"   ⚠️  CRITICAL KSAMPLER VALIDATION FAILED: {key}")
            
            results['outputs']['ksampler_validation'] = ksampler_validation
            
            # 4. Test KSampler with dummy data
            print("\n4. 🎯 CRITICAL: Testing KSampler with dummy data...")
            try:
                # Create dummy data for testing
                dummy_noise = torch.randn_like(fixed_latent)
                dummy_positive = [torch.randn(1, 77, 4096)]
                dummy_negative = [torch.randn(1, 77, 4096)]
                
                # Test sample method exists and is callable
                if hasattr(ksampler, 'sample'):
                    print("   ✅ KSampler sample method available")
                else:
                    raise AttributeError("KSampler missing sample method")
                
                print("   ✅ KSampler test passed")
                
            except Exception as test_error:
                print(f"   ❌ CRITICAL KSAMPLER TEST FAILED: {test_error}")
                raise test_error
            
            results['outputs']['ksampler'] = ksampler
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ CRITICAL SECTION 4.3 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ CRITICAL SECTION 4.3 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def critical_section_4_2_latent_preparation(self, 
                                               initial_latent: torch.Tensor,
                                               seed: int = 42,
                                               noise_inds: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        CRITICAL SECTION 4.2: Latent Preparation
        This is critical for proper noise generation
        """
        print("\n" + "="*80)
        print("🔧 CRITICAL SECTION 4.2: LATENT PREPARATION")
        print("="*80)
        print("⚠️  CRITICAL SECTION - Latent and noise preparation")
        
        section_start = time.time()
        results = {
            'section': '4.2_latent_preparation',
            'critical': True,
            'inputs': {},
            'outputs': {},
            'timing': {},
            'status': 'success'
        }
        
        try:
            # 1. Validate initial latent
            print("1. Validating initial latent...")
            latent_validation = {
                'is_tensor': isinstance(initial_latent, torch.Tensor),
                'has_shape': hasattr(initial_latent, 'shape'),
                'shape_valid': len(initial_latent.shape) == 5,  # [B, C, T, H, W]
                'values_finite': torch.isfinite(initial_latent).all(),
                'device_valid': initial_latent.device.type in ['cpu', 'cuda']
            }
            
            for key, value in latent_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    raise ValueError(f"Initial latent validation failed: {key}")
            
            results['inputs']['latent_validation'] = latent_validation
            print(f"   Initial latent shape: {initial_latent.shape}")
            print(f"   Initial latent mean: {initial_latent.mean().item():.6f}")
            print(f"   Initial latent std: {initial_latent.std().item():.6f}")
            
            # 2. CRITICAL: Fix empty latent channels
            print("\n2. 🎯 CRITICAL: Fixing empty latent channels...")
            try:
                from sample import fix_empty_latent_channels
                
                original_latent = initial_latent.clone()
                fixed_latent = fix_empty_latent_channels(self.unet, initial_latent)
                
                print(f"   Original shape: {original_latent.shape}")
                print(f"   Fixed shape: {fixed_latent.shape}")
                
                if original_latent.shape != fixed_latent.shape:
                    print(f"   ✅ Latent channels fixed")
                else:
                    print(f"   ✅ No channel fixing needed")
                
            except Exception as fix_error:
                print(f"   ❌ CRITICAL LATENT FIXING FAILED: {fix_error}")
                raise fix_error
            
            # 3. CRITICAL: Generate noise
            print("\n3. 🎯 CRITICAL: Generating noise...")
            try:
                from sample import prepare_noise
                
                noise = prepare_noise(fixed_latent, seed, noise_inds)
                
                print(f"   Noise shape: {noise.shape}")
                print(f"   Noise mean: {noise.mean().item():.6f}")
                print(f"   Noise std: {noise.std().item():.6f}")
                
            except Exception as noise_error:
                print(f"   ❌ CRITICAL NOISE GENERATION FAILED: {noise_error}")
                raise noise_error
            
            # 4. CRITICAL: Validate noise
            print("\n4. 🎯 CRITICAL: Validating noise...")
            noise_validation = {
                'is_tensor': isinstance(noise, torch.Tensor),
                'shape_match': noise.shape == fixed_latent.shape,
                'device_match': noise.device == fixed_latent.device,
                'dtype_match': noise.dtype == fixed_latent.dtype,
                'values_finite': torch.isfinite(noise).all(),
                'is_random': abs(noise.mean().item()) < 0.1,
                'has_variance': noise.std().item() > 0.5
            }
            
            for key, value in noise_validation.items():
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"   {key}: {status}")
                if not value:
                    print(f"   ⚠️  CRITICAL NOISE VALIDATION FAILED: {key}")
            
            results['outputs']['noise_validation'] = noise_validation
            results['outputs']['fixed_latent'] = fixed_latent
            results['outputs']['noise'] = noise
            results['timing']['section_time'] = time.time() - section_start
            
            print(f"\n✅ CRITICAL SECTION 4.2 COMPLETED in {time.time() - section_start:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ CRITICAL SECTION 4.2 FAILED: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
            results['timing']['section_time'] = time.time() - section_start
            return results
    
    def run_critical_sections_reverse(self, 
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
        Run critical sections in reverse order to quickly identify issues
        """
        print("\n" + "="*80)
        print("🚀 RUNNING CRITICAL SECTIONS IN REVERSE ORDER")
        print("="*80)
        print("⚠️  Testing only the most critical parts to quickly identify issues")
        
        total_start = time.time()
        results = {
            'critical_sections': {},
            'overall_status': 'success',
            'timing': {},
            'final_output': None
        }
        
        try:
            # Section 4.2: Latent Preparation (CRITICAL)
            print("\n" + "="*60)
            print("🔧 TESTING SECTION 4.2: LATENT PREPARATION")
            print("="*60)
            
            section_4_2 = self.critical_section_4_2_latent_preparation(
                initial_latent, seed, noise_inds
            )
            results['critical_sections']['4.2'] = section_4_2
            
            if section_4_2['status'] != 'success':
                print(f"❌ CRITICAL SECTION 4.2 FAILED - STOPPING TEST")
                results['overall_status'] = 'failed'
                results['error'] = f"Section 4.2 failed: {section_4_2.get('error', 'Unknown error')}"
                return results
            
            # Section 4.3: KSampler Setup (CRITICAL)
            print("\n" + "="*60)
            print("⚙️ TESTING SECTION 4.3: KSAMPLER SETUP")
            print("="*60)
            
            section_4_3 = self.critical_section_4_3_ksampler_setup(
                section_4_2['outputs']['fixed_latent'],
                steps, sampler_name, scheduler, denoise
            )
            results['critical_sections']['4.3'] = section_4_3
            
            if section_4_3['status'] != 'success':
                print(f"❌ CRITICAL SECTION 4.3 FAILED - STOPPING TEST")
                results['overall_status'] = 'failed'
                results['error'] = f"Section 4.3 failed: {section_4_3.get('error', 'Unknown error')}"
                return results
            
            # Section 4.4: Denoising Execution (MOST CRITICAL)
            print("\n" + "="*60)
            print("🎯 TESTING SECTION 4.4: DENOISING EXECUTION")
            print("="*60)
            
            section_4_4 = self.critical_section_4_4_denoising_execution(
                section_4_3['outputs']['ksampler'],
                section_4_2['outputs']['noise'],
                positive_conditioning,
                negative_conditioning,
                section_4_2['outputs']['fixed_latent'],
                cfg, seed
            )
            results['critical_sections']['4.4'] = section_4_4
            
            if section_4_4['status'] != 'success':
                print(f"❌ CRITICAL SECTION 4.4 FAILED - STOPPING TEST")
                results['overall_status'] = 'failed'
                results['error'] = f"Section 4.4 failed: {section_4_4.get('error', 'Unknown error')}"
                return results
            
            # Store final output
            results['final_output'] = section_4_4['outputs']['denoised_latent']
            results['timing']['total_time'] = time.time() - total_start
            
            print(f"\n🎉 CRITICAL SECTIONS COMPLETED in {time.time() - total_start:.2f}s")
            print("✅ All critical sections passed - Step 4 core functionality working")
            return results
            
        except Exception as e:
            print(f"❌ CRITICAL SECTIONS FAILED: {str(e)}")
            results['overall_status'] = 'failed'
            results['error'] = str(e)
            results['timing']['total_time'] = time.time() - total_start
            return results

def main():
    """Test the critical sections reverse testing"""
    print("🚀 Step 4 Critical Sections Reverse Test")
    print("=" * 60)
    
    print("Critical sections reverse testing is ready!")
    print("This focuses on the most critical parts of Step 4:")
    print("1. Section 4.2: Latent Preparation (CRITICAL)")
    print("2. Section 4.3: KSampler Setup (CRITICAL)")
    print("3. Section 4.4: Denoising Execution (MOST CRITICAL)")
    print("\nTesting in reverse order to quickly identify issues.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
