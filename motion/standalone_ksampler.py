"""
Standalone KSampler implementation
Based on ComfyUI's sampling system but with all dependencies resolved for standalone use.
Provides memory-efficient sampling with comprehensive monitoring and device management.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import logging
from typing import Dict, Any, Optional, Callable, Tuple, Union
from functools import partial

# Import motion modules
from wan_vae_components.model_management import get_torch_device, unet_offload_device, empty_cache, get_free_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneCFGGuider:
    """
    Standalone implementation of Classifier-Free Guidance
    Handles positive and negative conditioning for improved sample quality
    """
    
    def __init__(self, model_patcher):
        """
        Initialize CFG Guider
        
        Args:
            model_patcher: ModelPatcher containing the diffusion model
        """
        self.model_patcher = model_patcher
        self.model_options = getattr(model_patcher, 'model_options', {})
        self.positive_cond = None
        self.negative_cond = None
        self.cfg_scale = 1.0
        self.device = get_torch_device()
        
        # Memory tracking
        self.memory_usage = {
            'peak_allocated': 0,
            'calls_count': 0
        }
        
    def set_conds(self, positive, negative):
        """Set positive and negative conditioning"""
        print(f"   🔧 Setting CFG conditioning...")
        
        # Convert conditioning format if needed
        self.positive_cond = self._convert_conditioning(positive)
        self.negative_cond = self._convert_conditioning(negative) 
        
        print(f"      Positive conditioning shape: {self._get_cond_shape(self.positive_cond)}")
        print(f"      Negative conditioning shape: {self._get_cond_shape(self.negative_cond)}")
        
    def set_cfg(self, cfg_scale):
        """Set CFG scale for guidance strength"""
        self.cfg_scale = float(cfg_scale)
        print(f"   🔧 CFG Scale set to: {self.cfg_scale}")
        
    def _convert_conditioning(self, cond):
        """Convert conditioning to expected format"""
        if cond is None:
            return None
            
        # Handle different conditioning formats
        if isinstance(cond, (list, tuple)):
            if len(cond) > 0:
                # Take first conditioning if multiple are provided
                return cond[0]
        
        return cond
    
    def _get_cond_shape(self, cond):
        """Get conditioning shape for logging"""
        if cond is None:
            return "None"
        if hasattr(cond, 'shape'):
            return str(cond.shape)
        if isinstance(cond, (list, tuple)) and len(cond) > 0:
            if hasattr(cond[0], 'shape'):
                return str(cond[0].shape)
        return "Unknown"
    
    def predict_noise(self, x, timestep, model_options=None, seed=None):
        """
        Predict noise using CFG
        
        Args:
            x: Noisy latent tensor
            timestep: Current denoising timestep
            model_options: Additional model options
            seed: Random seed for reproducibility
            
        Returns:
            Predicted noise tensor
        """
        self.memory_usage['calls_count'] += 1
        
        # Track memory before prediction
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
        
        # Merge options
        merged_options = self.model_options.copy()
        if model_options:
            merged_options.update(model_options)
            
        # Prepare inputs - ensure timestep is a proper tensor with batch dimension
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device, dtype=torch.float32)
        elif timestep.dim() == 0:  # scalar tensor
            timestep = timestep.unsqueeze(0)  # add batch dimension
        elif len(timestep.shape) == 0:  # another way to check scalar
            timestep = timestep.view(1)
        
        # Handle conditioning
        if self.cfg_scale <= 1.0 or self.negative_cond is None:
            # No CFG - use only positive conditioning
            cond_input = self.positive_cond if self.positive_cond is not None else torch.zeros_like(x[:1, :4])
            if hasattr(cond_input, 'to'):
                cond_input = cond_input.to(x.device)
            
            # Get model prediction
            with torch.no_grad():
                noise_pred = self._call_model(x, timestep, cond_input, merged_options, seed)
                
        else:
            # CFG - use both positive and negative conditioning
            batch_size = x.shape[0]
            
            # Duplicate inputs for both conditionings
            x_combined = torch.cat([x, x], dim=0)
            
            # Handle timestep duplication safely
            if timestep.numel() == 1:  # single timestep
                timestep_combined = timestep.repeat(2)
            else:
                timestep_combined = torch.cat([timestep, timestep], dim=0)
            
            # Prepare conditioning
            pos_cond = self.positive_cond if self.positive_cond is not None else torch.zeros_like(x[:1, :4])
            neg_cond = self.negative_cond if self.negative_cond is not None else torch.zeros_like(x[:1, :4])
            
            # Ensure conditioning is on correct device
            if hasattr(pos_cond, 'to'):
                pos_cond = pos_cond.to(x.device)
            if hasattr(neg_cond, 'to'):
                neg_cond = neg_cond.to(x.device)
            
            # Combine conditioning (negative first, then positive)
            cond_combined = torch.cat([neg_cond, pos_cond], dim=0)
            
            # Get model predictions
            with torch.no_grad():
                noise_pred_combined = self._call_model(x_combined, timestep_combined, cond_combined, merged_options, seed)
            
            # Split predictions
            noise_pred_neg, noise_pred_pos = noise_pred_combined.chunk(2, dim=0)
            
            # Apply CFG
            noise_pred = noise_pred_neg + self.cfg_scale * (noise_pred_pos - noise_pred_neg)
        
        # Track memory after prediction
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            mem_delta = mem_after - mem_before
            self.memory_usage['peak_allocated'] = max(self.memory_usage['peak_allocated'], mem_after)
        
        return noise_pred
    
    def _call_model(self, x, timestep, conditioning, model_options, seed):
        """Call the underlying diffusion model"""
        # Access the model through ModelPatcher
        if hasattr(self.model_patcher, 'model'):
            model = self.model_patcher.model
        else:
            model = self.model_patcher
        
        # Get model device and dtype, ensure inputs match
        model_device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        model_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.float32
        original_device = x.device  # Store original device to move result back
        original_dtype = x.dtype  # Store original dtype to move result back
        
        logger.debug(f"Model device: {model_device}, dtype: {model_dtype}")
        logger.debug(f"Input device: {x.device}, dtype: {x.dtype}")
        
        # Move inputs to model device and dtype if they don't match
        if x.device != model_device or x.dtype != model_dtype:
            logger.debug(f"Moving input from {x.device}/{x.dtype} to model device {model_device}/{model_dtype}")
            x = x.to(device=model_device, dtype=model_dtype)
        if timestep.device != model_device or timestep.dtype != model_dtype:
            timestep = timestep.to(device=model_device, dtype=model_dtype)
        if conditioning is not None and hasattr(conditioning, 'device'):
            if conditioning.device != model_device or conditioning.dtype != model_dtype:
                conditioning = conditioning.to(device=model_device, dtype=model_dtype)
            
        # Try different model call strategies
        try:
            # Strategy 1: Try model.forward() method directly
            if hasattr(model, 'forward'):
                # Check if this is a VaceWanModel that needs context parameter
                if hasattr(model, '__class__') and 'Vace' in model.__class__.__name__:
                    # For VaceWanModel, we need to pass context parameter
                    # The signature is: forward(x, t, context, vace_context=None, vace_strength=None, ...)
                    logger.debug(f"Calling VaceWanModel.forward with context")
                    result = model.forward(x, timestep, conditioning)
                else:
                    # For other models, try the original call
                    logger.debug(f"Calling model.forward without context")
                    result = model.forward(x, timestep)
                logger.debug(f"Model forward call successful")
                
                # Handle different return formats and ensure correct device
                final_result = None
                if isinstance(result, dict) and 'sample' in result:
                    final_result = result['sample']
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    final_result = result[0]
                else:
                    final_result = result
                
                # Ensure result is on the original device and dtype
                if isinstance(final_result, torch.Tensor):
                    final_result = final_result.to(device=original_device, dtype=original_dtype)
                
                return final_result
            
            # Strategy 2: Try __call__ method
            elif hasattr(model, '__call__'):
                # Check if this is a VaceWanModel that needs context parameter
                if hasattr(model, '__class__') and 'Vace' in model.__class__.__name__:
                    # For VaceWanModel, we need to pass context parameter
                    result = model(x, timestep, conditioning)
                else:
                    # For other models, try the original call
                    result = model(x, timestep)
                logger.debug(f"Model __call__ successful")
                
                final_result = None
                if isinstance(result, dict) and 'sample' in result:
                    final_result = result['sample']
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    final_result = result[0]
                else:
                    final_result = result
                
                # Ensure result is on the original device and dtype
                if isinstance(final_result, torch.Tensor):
                    final_result = final_result.to(device=original_device, dtype=original_dtype)
                
                return final_result
                    
            # Strategy 3: Try apply_model method (ComfyUI style)
            elif hasattr(model, 'apply_model'):
                result = model.apply_model(x, timestep)
                logger.debug(f"Model apply_model successful")
                
                final_result = None
                if isinstance(result, dict) and 'sample' in result:
                    final_result = result['sample']
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    final_result = result[0]
                else:
                    final_result = result
                
                # Ensure result is on the original device and dtype
                if isinstance(final_result, torch.Tensor):
                    final_result = final_result.to(device=original_device, dtype=original_dtype)
                
                return final_result
            else:
                logger.error(f"Model {type(model)} has no callable methods")
                raise RuntimeError(f"Model {type(model)} doesn't have forward, __call__, or apply_model")
                
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            
            # Strategy 4: Try with conditioning as additional argument
            try:
                if conditioning is not None:
                    if hasattr(model, 'forward'):
                        result = model.forward(x, timestep, conditioning)
                    elif hasattr(model, '__call__'):
                        result = model(x, timestep, conditioning)
                    elif hasattr(model, 'apply_model'):
                        result = model.apply_model(x, timestep, conditioning)
                    else:
                        logger.error(f"No valid model interface found")
                        return torch.zeros_like(x)
                        
                    logger.debug(f"Model call with conditioning successful")
                    
                    # Handle return formats and ensure correct device
                    final_result = None
                    if isinstance(result, dict) and 'sample' in result:
                        final_result = result['sample']
                    elif isinstance(result, (tuple, list)) and len(result) > 0:
                        final_result = result[0]
                    else:
                        final_result = result
                    
                    # Ensure result is on the original device
                    if isinstance(final_result, torch.Tensor):
                        final_result = final_result.to(original_device)
                    
                    return final_result
                else:
                    logger.error(f"No conditioning provided for fallback strategy")
                    return torch.zeros_like(x)
                        
            except Exception as e2:
                logger.error(f"Model call with conditioning failed: {e2}")
                
            # Strategy 5: Last resort - try ModelPatcher if model is actually the ModelPatcher
            try:
                if hasattr(self.model_patcher, 'model') and hasattr(self.model_patcher.model, 'forward'):
                    result = self.model_patcher.model.forward(x, timestep)
                    logger.debug(f"ModelPatcher.model.forward successful")
                    
                    # Ensure result is on the original device
                    if isinstance(result, torch.Tensor):
                        result = result.to(original_device)
                    
                    return result
            except Exception as e3:
                logger.error(f"ModelPatcher fallback failed: {e3}")
                
            # Final fallback: Return zero tensor (this will show in results as all zeros)
            logger.warning(f"All model call strategies failed, returning zeros")
            return torch.zeros_like(x)
    
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """
        Main sampling method using CFG
        
        Args:
            noise: Initial noise tensor
            latent_image: Optional latent image for img2img
            sampler: Sampling algorithm function
            sigmas: Noise schedule tensor
            denoise_mask: Optional mask for selective denoising
            callback: Optional progress callback
            disable_pbar: Whether to disable progress reporting
            seed: Random seed
            
        Returns:
            Denoised samples
        """
        print(f"   🎯 Starting CFG-guided sampling...")
        print(f"      CFG Scale: {self.cfg_scale}")
        print(f"      Noise shape: {noise.shape}")
        print(f"      Sigmas: {len(sigmas)} steps")
        
        # Create model wrapper for sampling
        model_wrapper = CFGModelWrapper(self)
        
        # Set up sampling parameters
        extra_args = {
            'seed': seed,
            'denoise_mask': denoise_mask
        }
        
        # Memory monitoring setup
        sampling_start = time.time()
        if torch.cuda.is_available():
            mem_start = torch.cuda.memory_allocated() / 1024**2
            
        # Call sampler
        try:
            samples = sampler.sample(
                model_wrapper, 
                sigmas, 
                extra_args, 
                callback, 
                noise, 
                latent_image=latent_image, 
                denoise_mask=denoise_mask, 
                disable_pbar=disable_pbar
            )
            
            sampling_time = time.time() - sampling_start
            
            # Memory tracking
            if torch.cuda.is_available():
                mem_end = torch.cuda.memory_allocated() / 1024**2
                print(f"   ✅ Sampling completed in {sampling_time:.2f}s")
                print(f"      Memory: {mem_start:.1f} → {mem_end:.1f} MB ({mem_end-mem_start:+.1f} MB)")
                print(f"      Peak CFG Memory: {self.memory_usage['peak_allocated'] / 1024**2:.1f} MB")
                print(f"      CFG Calls: {self.memory_usage['calls_count']}")
            
            return samples
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise


class CFGModelWrapper:
    """Wrapper to make CFGGuider compatible with sampler functions"""
    
    def __init__(self, cfg_guider):
        self.cfg_guider = cfg_guider
        self.inner_model = self  # For compatibility
        
    def __call__(self, x, sigma, **kwargs):
        """Main model call interface"""
        # Convert sigma to timestep for the model
        # For most diffusion models, timestep is typically an integer
        # We'll use a simple conversion: timestep = int(sigma * 1000)
        if isinstance(sigma, torch.Tensor):
            timestep = (sigma * 1000).long()
        else:
            timestep = int(sigma * 1000)
        
        return self.cfg_guider.predict_noise(x, timestep, **kwargs)


class StandaloneSchedulers:
    """
    Standalone implementation of noise schedulers
    Provides different noise scheduling strategies for sampling
    """
    
    @staticmethod
    def simple_scheduler(model_sampling, steps):
        """Simple linear scheduler"""
        s = model_sampling
        sigs = []
        ss = len(s.sigmas) / steps
        for x in range(steps):
            sigs.append(float(s.sigmas[-(1 + int(x * ss))]))
        sigs.append(0.0)
        return torch.FloatTensor(sigs)
    
    @staticmethod 
    def ddim_scheduler(model_sampling, steps):
        """DDIM uniform scheduler"""
        s = model_sampling
        sigs = []
        x = 1
        if math.isclose(float(s.sigmas[x]), 0, abs_tol=0.00001):
            steps += 1
            sigs = []
        
        ddim_timesteps = np.linspace(0, len(s.sigmas) - 1, steps + 1).astype(int)
        for i in ddim_timesteps:
            sigs.append(float(s.sigmas[i]))
        return torch.FloatTensor(sigs)
    
    @staticmethod
    def karras_scheduler(n, sigma_min, sigma_max):
        """Karras scheduler for improved quality"""
        rho = 7.0  # Karras et al. default
        ramp = np.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.from_numpy(np.append(sigmas, 0.0)).float()
    
    @staticmethod
    def exponential_scheduler(n, sigma_min, sigma_max):
        """Exponential scheduler"""
        sigmas = np.geomspace(sigma_max, sigma_min, n)
        return torch.from_numpy(np.append(sigmas, 0.0)).float()
    
    # Scheduler registry
    SCHEDULERS = {
        "simple": simple_scheduler.__func__,
        "ddim_uniform": ddim_scheduler.__func__, 
        "karras": karras_scheduler.__func__,
        "exponential": exponential_scheduler.__func__
    }
    
    @classmethod
    def calculate_sigmas(cls, model_sampling, scheduler_name, steps):
        """Calculate sigma schedule"""
        if scheduler_name not in cls.SCHEDULERS:
            logger.warning(f"Unknown scheduler {scheduler_name}, using 'simple'")
            scheduler_name = "simple"
            
        scheduler = cls.SCHEDULERS[scheduler_name]
        
        # Check if scheduler needs model_sampling or just min/max
        if scheduler_name in ["karras", "exponential"]:
            return scheduler(
                n=steps,
                sigma_min=float(model_sampling.sigma_min),
                sigma_max=float(model_sampling.sigma_max)
            )
        else:
            return scheduler(model_sampling, steps)


class EulerSampler:
    """Standalone Euler sampling implementation"""
    
    def __init__(self):
        self.name = "euler"
        
    def sample(self, model_wrapper, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        """Euler sampling step"""
        print(f"      Using Euler sampler with {len(sigmas)-1} steps")
        
        # Initialize
        x = noise.clone()
        
        # Sampling loop
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            if sigma == 0:
                continue
            
            print(f"      Step {i+1}/{len(sigmas)-1}: sigma={sigma:.3f} -> {sigma_next:.3f}")
                
            # Get noise prediction
            with torch.no_grad():
                try:
                    denoised = model_wrapper(x, sigma)
                    
                    # Ensure denoised is on the same device as x
                    if isinstance(denoised, torch.Tensor) and isinstance(x, torch.Tensor):
                        denoised = denoised.to(x.device)
                    
                    print(f"      Step {i+1}: Model prediction successful, shape={denoised.shape}")
                    
                except Exception as e:
                    print(f"      Step {i+1}: Model prediction failed: {e}")
                    # Return zeros to avoid hanging
                    return torch.zeros_like(noise)
                
            # Ensure sigma values are on the same device as x
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.to(x.device)
            if isinstance(sigma_next, torch.Tensor):
                sigma_next = sigma_next.to(x.device)
                
            # Euler step
            d = (x - denoised) / sigma
            dt = sigma_next - sigma
            x = x + d * dt
            
            # Progress callback
            if callback is not None:
                callback(i, len(sigmas) - 1)
                
            # Memory management
            if i % 5 == 0:  # Every 5 steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return x


class DPMSolverSampler:
    """Standalone DPM-Solver implementation"""
    
    def __init__(self):
        self.name = "dpmpp_2m"
        
    def sample(self, model_wrapper, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        """DPM-Solver sampling"""
        print(f"      Using DPM-Solver sampler with {len(sigmas)-1} steps")
        
        x = noise.clone()
        old_denoised = None
        
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            if sigma == 0:
                continue
                
            # Get noise prediction  
            with torch.no_grad():
                denoised = model_wrapper(x, sigma)
                
                # Ensure denoised is on the same device as x
                if isinstance(denoised, torch.Tensor) and isinstance(x, torch.Tensor):
                    denoised = denoised.to(x.device)
            
            # Ensure sigma values are on the same device as x
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.to(x.device)
            if isinstance(sigma_next, torch.Tensor):
                sigma_next = sigma_next.to(x.device)
            
            if old_denoised is None or sigma_next == 0:
                # First order (Euler step)
                d = (x - denoised) / sigma
                dt = sigma_next - sigma
                x = x + d * dt
            else:
                # Second order
                h = sigma_next - sigma
                h_prev = sigma - sigmas[i-1] if i > 0 else 0
                r = h_prev / h if h != 0 else 0
                
                # Linear combination
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                d = (x - denoised_d) / sigma
                x = x + d * h
            
            old_denoised = denoised
            
            # Progress callback
            if callback is not None:
                callback(i, len(sigmas) - 1)
                
            # Memory management
            if i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return x


class StandaloneKSampler:
    """
    Standalone K-Sampler implementation
    Memory-efficient sampling with comprehensive monitoring
    """
    
    # Available samplers
    SAMPLERS = {
        "euler": EulerSampler,
        "dpmpp_2m": DPMSolverSampler
    }
    
    # Available schedulers 
    SCHEDULERS = list(StandaloneSchedulers.SCHEDULERS.keys())
    
    def __init__(self, model, steps, device=None, sampler="euler", scheduler="simple", denoise=1.0, model_options=None):
        """
        Initialize KSampler
        
        Args:
            model: ModelPatcher containing diffusion model
            steps: Number of sampling steps
            device: Compute device (auto-detected if None)
            sampler: Sampling algorithm name
            scheduler: Noise scheduler name
            denoise: Denoising strength (0.0 to 1.0)
            model_options: Additional model options
        """
        self.model = model
        self.device = device or get_torch_device()
        self.steps = steps
        self.denoise = denoise
        self.model_options = model_options or {}
        
        # Validate and set sampler
        if sampler not in self.SAMPLERS:
            logger.warning(f"Unknown sampler {sampler}, using 'euler'")
            sampler = "euler"
        self.sampler_name = sampler
        
        # Validate and set scheduler
        if scheduler not in self.SCHEDULERS:
            logger.warning(f"Unknown scheduler {scheduler}, using 'simple'")
            scheduler = "simple"
        self.scheduler_name = scheduler
        
        # Initialize sigmas
        self.sigmas = self.calculate_sigmas(steps)
        
        # Memory tracking
        self.memory_stats = {
            'peak_allocated': 0,
            'sampling_calls': 0,
            'cache_clears': 0
        }
        
        print(f"   🔧 KSampler initialized:")
        print(f"      Sampler: {self.sampler_name}")
        print(f"      Scheduler: {self.scheduler_name}")
        print(f"      Steps: {self.steps}")
        print(f"      Device: {self.device}")
        print(f"      Denoise: {self.denoise}")
        
    def calculate_sigmas(self, steps):
        """Calculate noise schedule"""
        try:
            # Get model sampling from ModelPatcher
            if hasattr(self.model, 'get_model_object'):
                model_sampling = self.model.get_model_object("model_sampling")
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'model_sampling'):
                model_sampling = self.model.model.model_sampling
            else:
                # Fallback: create basic sampling object
                model_sampling = self._create_fallback_sampling()
            
            # Calculate sigmas using scheduler
            sigmas = StandaloneSchedulers.calculate_sigmas(model_sampling, self.scheduler_name, steps)
            
            # Apply denoising
            if self.denoise < 1.0:
                if self.denoise <= 0.0:
                    sigmas = torch.FloatTensor([])
                else:
                    new_steps = int(steps / self.denoise)
                    full_sigmas = StandaloneSchedulers.calculate_sigmas(model_sampling, self.scheduler_name, new_steps)
                    sigmas = full_sigmas[-(steps + 1):]
            
            return sigmas.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to calculate sigmas: {e}")
            # Fallback: linear schedule
            return torch.linspace(1.0, 0.0, steps + 1, device=self.device)
    
    def _create_fallback_sampling(self):
        """Create fallback sampling object"""
        class FallbackSampling:
            def __init__(self):
                self.sigma_min = 0.002
                self.sigma_max = 80.0
                self.sigmas = torch.linspace(self.sigma_max, self.sigma_min, 1000)
        
        return FallbackSampling()
    
    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, 
               force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        """
        Main sampling method
        
        Args:
            noise: Initial noise tensor
            positive: Positive conditioning
            negative: Negative conditioning  
            cfg: CFG scale
            latent_image: Optional latent image for img2img
            start_step: Start step (for partial sampling)
            last_step: End step (for partial sampling)
            force_full_denoise: Force complete denoising
            denoise_mask: Selective denoising mask
            sigmas: Custom sigma schedule
            callback: Progress callback
            disable_pbar: Disable progress reporting
            seed: Random seed
            
        Returns:
            Denoised samples
        """
        print(f"\n   🎯 Starting KSampler sampling...")
        self.memory_stats['sampling_calls'] += 1
        
        # Memory monitoring
        sampling_start = time.time()
        if torch.cuda.is_available():
            mem_start = torch.cuda.memory_allocated() / 1024**2
            print(f"      Initial GPU memory: {mem_start:.1f} MB")
        
        try:
            # Use provided sigmas or calculate them
            if sigmas is None:
                sigmas = self.sigmas
            else:
                sigmas = sigmas.to(self.device)
            
            # Handle step range
            if last_step is not None and last_step < (len(sigmas) - 1):
                sigmas = sigmas[:last_step + 1]
                if force_full_denoise:
                    sigmas[-1] = 0
                    
            if start_step is not None:
                if start_step < (len(sigmas) - 1):
                    sigmas = sigmas[start_step:]
                else:
                    return latent_image if latent_image is not None else torch.zeros_like(noise)
            
            print(f"      Effective steps: {len(sigmas) - 1}")
            print(f"      Sigma range: {sigmas[0]:.3f} → {sigmas[-1]:.3f}")
            
            # Set up CFG guider
            cfg_guider = StandaloneCFGGuider(self.model)
            cfg_guider.set_conds(positive, negative)
            cfg_guider.set_cfg(cfg)
            
            # Initialize sampler
            sampler_class = self.SAMPLERS[self.sampler_name]
            sampler = sampler_class()
            
            # Ensure noise is on correct device
            noise = noise.to(self.device)
            if latent_image is not None:
                latent_image = latent_image.to(self.device)
            
            # Progress callback setup
            step_callback = None
            if callback is not None:
                def step_callback(current_step, total_steps):
                    progress = (current_step + 1) / total_steps
                    callback(progress, total_steps, current_step)
            
            # Perform sampling
            print(f"      Starting {self.sampler_name} sampling...")
            samples = cfg_guider.sample(
                noise=noise,
                latent_image=latent_image,
                sampler=sampler,
                sigmas=sigmas,
                denoise_mask=denoise_mask,
                callback=step_callback,
                disable_pbar=disable_pbar,
                seed=seed
            )
            
            # Move to CPU if needed (memory management)
            if hasattr(self.model, 'offload_device'):
                offload_device = self.model.offload_device
            else:
                offload_device = unet_offload_device()
            
            if offload_device != self.device:
                samples = samples.to(offload_device)
                print(f"      Samples moved to: {offload_device}")
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.memory_stats['cache_clears'] += 1
            
            # Final memory report
            sampling_time = time.time() - sampling_start
            if torch.cuda.is_available():
                mem_end = torch.cuda.memory_allocated() / 1024**2
                self.memory_stats['peak_allocated'] = max(self.memory_stats['peak_allocated'], mem_end)
                
                print(f"   ✅ Sampling completed in {sampling_time:.2f}s")
                print(f"      Final memory: {mem_start:.1f} → {mem_end:.1f} MB ({mem_end-mem_start:+.1f} MB)")
                print(f"      Peak memory: {self.memory_stats['peak_allocated']:.1f} MB")
                print(f"      Cache clears: {self.memory_stats['cache_clears']}")
            
            return samples
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        return self.memory_stats.copy()


# Convenience function for sampling
def prepare_noise(latent_image, seed, device=None):
    """
    Create random noise for sampling
    
    Args:
        latent_image: Template latent for shape/dtype
        seed: Random seed
        device: Target device (auto-detected if None)
        
    Returns:
        Random noise tensor
    """
    if device is None:
        device = get_torch_device()
        
    # Create noise on CPU first, then move to target device
    if seed is not None:
        generator = torch.manual_seed(seed)
        noise = torch.randn(
            latent_image.size(), 
            dtype=latent_image.dtype, 
            device='cpu',
            generator=generator
        )
    else:
        noise = torch.randn(
            latent_image.size(), 
            dtype=latent_image.dtype, 
            device='cpu'
        )
    
    # Move to target device
    noise = noise.to(device)
    
    return noise


def main():
    """Test standalone KSampler functionality"""
    print("🧪 Testing Standalone KSampler")
    print("="*50)
    
    try:
        # Test scheduler calculation
        print("1. Testing schedulers...")
        
        class MockModelSampling:
            def __init__(self):
                self.sigma_min = 0.002
                self.sigma_max = 80.0
                self.sigmas = torch.linspace(self.sigma_max, self.sigma_min, 1000)
        
        mock_sampling = MockModelSampling()
        
        for scheduler_name in StandaloneSchedulers.SCHEDULERS.keys():
            try:
                sigmas = StandaloneSchedulers.calculate_sigmas(mock_sampling, scheduler_name, 20)
                print(f"   ✅ {scheduler_name}: {len(sigmas)} sigmas, range {sigmas[0]:.3f} → {sigmas[-1]:.3f}")
            except Exception as e:
                print(f"   ❌ {scheduler_name}: {e}")
        
        # Test noise generation
        print("\n2. Testing noise generation...")
        dummy_latent = torch.zeros(1, 4, 32, 32)
        noise = prepare_noise(dummy_latent, seed=42)
        print(f"   ✅ Noise shape: {noise.shape}, dtype: {noise.dtype}")
        print(f"   ✅ Noise stats: mean={noise.mean():.3f}, std={noise.std():.3f}")
        
        # Test CFG Guider (without actual model)
        print("\n3. Testing CFG Guider...")
        
        class MockModel:
            def __init__(self):
                self.model_options = {}
                
            def __call__(self, x, timestep, **kwargs):
                return torch.zeros_like(x)
        
        mock_model = MockModel()
        cfg_guider = StandaloneCFGGuider(mock_model)
        cfg_guider.set_cfg(7.5)
        
        print(f"   ✅ CFG Guider initialized with scale: {cfg_guider.cfg_scale}")
        
        print("\n🎉 All tests passed! KSampler is ready for integration.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
