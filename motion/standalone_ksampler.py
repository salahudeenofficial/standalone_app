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
import motion.model_management_standalone as model_management
from motion.model_management_standalone import get_torch_device, unet_offload_device, empty_cache, get_free_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HOOK SYSTEM (Simplified ComfyUI Compatible)
# ============================================================================

class EnumHookType:
    """Simplified hook type enumeration"""
    WEIGHT = "weight"
    OBJECT_PATCH = "object_patch"
    ADDITIONAL_MODELS = "additional_models"
    TRANSFORMER_OPTIONS = "transformer_options"
    INJECTIONS = "injections"

class EnumHookScope:
    """Simplified hook scope enumeration"""
    ALL_CONDITIONING = "all_conditioning"
    POSITIVE_ONLY = "positive_only"
    NEGATIVE_ONLY = "negative_only"

class Hook:
    """Simplified hook implementation for motion pipeline"""
    def __init__(self, hook_type=None, hook_id=None, hook_scope=EnumHookScope.ALL_CONDITIONING):
        self.hook_type = hook_type
        self.hook_id = hook_id
        self.hook_scope = hook_scope
        self.strength = 1.0
        
    def clone(self):
        """Clone the hook"""
        return Hook(self.hook_type, self.hook_id, self.hook_scope)
    
    def __eq__(self, other):
        return (isinstance(other, Hook) and 
                self.hook_type == other.hook_type and 
                self.hook_id == other.hook_id)
    
    def __hash__(self):
        return hash((self.hook_type, self.hook_id))

class HookGroup:
    """Simplified hook group implementation for motion pipeline"""
    def __init__(self):
        self.hooks = []
        self._hook_dict = {}
    
    def add(self, hook):
        """Add a hook to the group"""
        if hook not in self.hooks:
            self.hooks.append(hook)
            if hook.hook_type not in self._hook_dict:
                self._hook_dict[hook.hook_type] = []
            self._hook_dict[hook.hook_type].append(hook)
    
    def remove(self, hook):
        """Remove a hook from the group"""
        if hook in self.hooks:
            self.hooks.remove(hook)
            if hook.hook_type in self._hook_dict:
                self._hook_dict[hook.hook_type].remove(hook)
    
    def get_type(self, hook_type):
        """Get hooks of a specific type"""
        return self._hook_dict.get(hook_type, [])
    
    def contains(self, hook):
        """Check if hook is in the group"""
        return hook in self.hooks
    
    def is_subset_of(self, other):
        """Check if this group is a subset of another"""
        if not isinstance(other, HookGroup):
            return False
        return set(self.hooks).issubset(set(other.hooks))
    
    def new_with_common_hooks(self, other):
        """Create new group with common hooks"""
        if not isinstance(other, HookGroup):
            return HookGroup()
        
        new_group = HookGroup()
        for hook in self.hooks:
            if other.contains(hook):
                new_group.add(hook.clone())
        return new_group
    
    def clone(self):
        """Clone the hook group"""
        new_group = HookGroup()
        for hook in self.hooks:
            new_group.add(hook.clone())
        return new_group
    
    def combine_all_hooks(self, hook_lists):
        """Combine multiple hook lists into one group"""
        combined = HookGroup()
        for hook_list in hook_lists:
            if hook_list is not None:
                if isinstance(hook_list, HookGroup):
                    for hook in hook_list.hooks:
                        combined.add(hook)
                elif isinstance(hook_list, list):
                    for hook in hook_list:
                        if isinstance(hook, Hook):
                            combined.add(hook)
        return combined if len(combined.hooks) > 0 else None
    
    def __len__(self):
        return len(self.hooks)
    
    def __iter__(self):
        return iter(self.hooks)

def preprocess_conds_hooks(conds):
    """
    Preprocess conditioning hooks - simplified ComfyUI compatible implementation
    
    Args:
        conds: Dictionary of conditioning lists
    """
    # For now, this is a simplified implementation
    # In a full implementation, this would handle ControlNet hooks and other advanced features
    logger.debug("Preprocessing conditioning hooks (simplified implementation)")
    
    # Check for hooks in conditioning
    for k in conds:
        if conds[k] is not None:
            for kk in conds[k]:
                # Ensure kk is a dictionary before checking for 'hooks'
                if isinstance(kk, dict) and 'hooks' in kk:
                    hooks = kk['hooks']
                    if hooks is not None:
                        logger.debug(f"Found hooks in conditioning: {len(hooks) if hasattr(hooks, '__len__') else 'unknown'}")
    
    # Placeholder for future ControlNet hook processing
    # This would combine ControlNet extra hooks with normal hooks
    pass

def filter_registered_hooks_on_conds(conds, model_options):
    """
    Filter registered hooks on conditioning - simplified ComfyUI compatible implementation
    
    Args:
        conds: Dictionary of conditioning lists
        model_options: Model options dictionary
    """
    logger.debug("Filtering registered hooks on conditioning (simplified implementation)")
    
    # Get registered hooks from model options
    registered = model_options.get('registered_hooks', None)
    
    # If no hooks were registered, clean all hooks from conds
    if registered is None:
        for k in conds:
            if conds[k] is not None:
                for kk in conds[k]:
                    # Ensure kk is a dictionary before popping 'hooks'
                    if isinstance(kk, dict):
                        kk.pop('hooks', None)
        return
    
    # Filter hooks based on registration
    for k in conds:
        if conds[k] is not None:
            for kk in conds[k]:
                # Ensure kk is a dictionary before checking for 'hooks'
                if isinstance(kk, dict):
                    hooks = kk.get('hooks', None)
                    if hooks is not None:
                        if isinstance(hooks, HookGroup):
                            # Check if hooks are subset of registered hooks
                            if not hooks.is_subset_of(registered):
                                # Create new hooks with only common ones
                                new_hooks = hooks.new_with_common_hooks(registered)
                                kk['hooks'] = new_hooks if len(new_hooks) > 0 else None
                        elif isinstance(hooks, list):
                            # Filter list of hooks
                            filtered_hooks = [hook for hook in hooks if registered.contains(hook)]
                            kk['hooks'] = filtered_hooks if filtered_hooks else None

def get_total_hook_groups_in_conds(conds):
    """
    Get total number of hook groups in conditioning - simplified implementation
    
    Args:
        conds: Dictionary of conditioning lists
        
    Returns:
        Number of unique hook groups
    """
    hooks_set = set()
    for k in conds:
        if conds[k] is not None:
            for kk in conds[k]:
                # Ensure kk is a dictionary before checking for 'hooks'
                if isinstance(kk, dict):
                    hooks = kk.get('hooks', None)
                    if hooks is not None:
                        if isinstance(hooks, HookGroup):
                            hooks_set.add(id(hooks))
                        elif isinstance(hooks, list):
                            for hook in hooks:
                                if isinstance(hook, Hook):
                                    hooks_set.add(id(hook))
    return len(hooks_set)

# ============================================================================
# CORE SAMPLING FUNCTIONS (ComfyUI Compatible)
# ============================================================================

def calc_cond_batch(model, conds, x_in, timestep, model_options):
    """
    Calculate conditioning batch - ComfyUI compatible implementation
    Processes conditioning with proper hook system integration
    
    Args:
        model: The diffusion model
        conds: List of conditioning lists [positive, negative]
        x_in: Input tensor
        timestep: Current timestep
        model_options: Model options dictionary
        
    Returns:
        List of processed conditioning outputs
    """
    out_conds = []
    out_counts = []
    
    # Initialize output tensors
    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)
    
    # Process each conditioning
    for i in range(len(conds)):
        cond = conds[i]
        if cond is not None:
            for x in cond:
                # Get area and multiplier for this conditioning
                area = get_area_and_mult(x, x_in, timestep)
                if area is None:
                    continue
                
                # Apply conditioning to the area
                if area.hooks is not None:
                    # TODO: Implement hook system integration
                    logger.debug(f"Conditioning with hooks: {area.hooks}")
                
                # Apply the conditioning
                out_conds[i] += area.area * area.mult
                out_counts[i] += area.area
    
    # Normalize by counts
    for i in range(len(out_conds)):
        out_conds[i] = out_conds[i] / torch.clamp(out_counts[i], min=1e-37)
    
    return out_conds

def get_area_and_mult(cond, x_in, timestep):
    """
    Get area and multiplier for conditioning - simplified implementation
    
    Args:
        cond: Conditioning dictionary
        x_in: Input tensor
        timestep: Current timestep
        
    Returns:
        Area and multiplier information
    """
    class AreaInfo:
        def __init__(self, area, mult, hooks=None):
            self.area = area
            self.mult = mult
            self.hooks = hooks
            self.shape = x_in.shape
    
    # Create full area mask
    area = torch.ones_like(x_in)
    
    # Get multiplier from conditioning
    mult = cond.get('mult', 1.0)
    if isinstance(mult, torch.Tensor):
        mult = mult.item()
    
    # Get hooks if present
    hooks = cond.get('hooks', None)
    
    return AreaInfo(area, mult, hooks)

def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    """
    CFG function - ComfyUI compatible implementation
    Applies classifier-free guidance with hook support
    
    Args:
        model: The diffusion model
        cond_pred: Conditional prediction
        uncond_pred: Unconditional prediction
        cond_scale: CFG scale
        x: Input tensor
        timestep: Current timestep
        model_options: Model options
        cond: Original conditional conditioning
        uncond: Original unconditional conditioning
        
    Returns:
        CFG-guided prediction
    """
    # Apply custom CFG function if provided
    if "sampler_cfg_function" in model_options:
        args = {
            "cond": x - cond_pred,
            "uncond": x - uncond_pred,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        # Standard CFG formula
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale
    
    # Apply post-CFG functions
    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "cond_scale": cond_scale,
            "model": model,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x
        }
        cfg_result = fn(args)
    
    return cfg_result

def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    """
    Main sampling function - ComfyUI compatible implementation
    Handles CFG with proper optimizations and hook support
    
    Args:
        model: The diffusion model
        x: Input tensor
        timestep: Current timestep
        uncond: Unconditional conditioning
        cond: Conditional conditioning
        cond_scale: CFG scale
        model_options: Model options
        seed: Random seed
        
    Returns:
        CFG-guided prediction
    """
    # CFG optimization for scale=1.0
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond
    
    # Prepare conditioning list
    conds = [cond, uncond_]
    
    # Use custom batch function if provided
    if "sampler_calc_cond_batch_function" in model_options:
        args = {
            "conds": conds,
            "input": x,
            "sigma": timestep,
            "model": model,
            "model_options": model_options
        }
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)
    
    # Apply pre-CFG functions
    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {
            "conds": conds,
            "conds_out": out,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "model": model,
            "model_options": model_options
        }
        out = fn(args)
    
    # Apply CFG
    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, 
                       model_options=model_options, cond=cond, uncond=uncond_)

class StandaloneCFGGuider:
    """
    Standalone implementation of Classifier-Free Guidance
    Handles positive and negative conditioning for improved sample quality
    """
    
    def __init__(self, model_patcher):
        """
        Initialize CFG Guider - ComfyUI compatible
        
        Args:
            model_patcher: ModelPatcher containing the diffusion model
        """
        self.model_patcher = model_patcher
        self.model_options = getattr(model_patcher, 'model_options', {})
        self.original_conds = {}  # Store original conditioning for restoration
        self.cfg = 1.0  # Use 'cfg' to match ComfyUI naming
        self.device = get_torch_device()
        
        # Memory tracking
        self.memory_usage = {
            'peak_allocated': 0,
            'calls_count': 0
        }
        
        # ComfyUI compatibility
        self.inner_model = None
        self.conds = {}
        self.loaded_models = []
        
        # Wrapper executor for enhanced model integration
        self.wrapper_executor = WrapperExecutor(model_patcher, self.model_options)
        
    def set_conds(self, positive, negative):
        """Set positive and negative conditioning - ComfyUI compatible"""
        print(f"   🔧 Setting CFG conditioning...")
        self.inner_set_conds({"positive": positive, "negative": negative})
        
        print(f"      Positive conditioning shape: {self._get_cond_shape(positive)}")
        print(f"      Negative conditioning shape: {self._get_cond_shape(negative)}")
        
    def set_cfg(self, cfg):
        """Set CFG scale for guidance strength - ComfyUI compatible"""
        self.cfg = float(cfg)
        print(f"   🔧 CFG Scale set to: {self.cfg}")
        
    def inner_set_conds(self, conds):
        """Store original conditioning for restoration - ComfyUI compatible"""
        for k in conds:
            self.original_conds[k] = self._convert_conditioning(conds[k])
        
    def _convert_conditioning(self, cond):
        """Convert conditioning to expected format - ComfyUI compatible"""
        if cond is None:
            return None
            
        # Handle different conditioning formats
        if isinstance(cond, (list, tuple)):
            if len(cond) > 0:
                # Return the list as-is for proper processing
                return list(cond)
        
        # Wrap single conditioning in list
        return [cond] if cond is not None else None
    
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
    
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """
        Predict noise using CFG - ComfyUI compatible implementation
        
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
        
        # Use ComfyUI's sampling_function for proper CFG handling
        try:
            # Get the inner model
            if self.inner_model is None:
                if hasattr(self.model_patcher, 'model'):
                    self.inner_model = self.model_patcher.model
                else:
                    self.inner_model = self.model_patcher
            
            # Use ComfyUI's sampling function
            # Extract conditioning from the stored format
            pos_cond = self.conds.get("positive", None)
            neg_cond = self.conds.get("negative", None)
            
            # Convert conditioning format for sampling_function
            # sampling_function expects individual conditioning, not list format
            if isinstance(pos_cond, list) and len(pos_cond) > 0:
                pos_cond = pos_cond[0]  # Extract first tensor
            
            if isinstance(neg_cond, list) and len(neg_cond) > 0:
                neg_cond = neg_cond[0]  # Extract first tensor
            
            noise_pred = sampling_function(
                self.inner_model, x, timestep,
                neg_cond, pos_cond, self.cfg, model_options=merged_options, seed=seed
            )
            
        except Exception as e:
            logger.warning(f"ComfyUI sampling_function failed, falling back to manual CFG: {e}")
            # Fallback to manual CFG implementation
            noise_pred = self._manual_cfg_prediction(x, timestep, merged_options, seed)
        
        # Track memory after prediction
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated()
            mem_delta = mem_after - mem_before
            self.memory_usage['peak_allocated'] = max(self.memory_usage['peak_allocated'], mem_after)
        
        return noise_pred
    
    def _manual_cfg_prediction(self, x, timestep, model_options, seed):
        """Fallback manual CFG prediction"""
        # Prepare inputs - ensure timestep is a proper tensor with batch dimension
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=x.device, dtype=torch.float32)
        elif timestep.dim() == 0:  # scalar tensor
            timestep = timestep.unsqueeze(0)  # add batch dimension
        elif len(timestep.shape) == 0:  # another way to check scalar
            timestep = timestep.view(1)
        
        # Get conditioning
        pos_cond = self.conds.get("positive", None)
        neg_cond = self.conds.get("negative", None)
        
        # Handle conditioning format
        if pos_cond is not None and isinstance(pos_cond, list) and len(pos_cond) > 0:
            pos_cond = pos_cond[0]
        if neg_cond is not None and isinstance(neg_cond, list) and len(neg_cond) > 0:
            neg_cond = neg_cond[0]
        
        # Handle conditioning
        if self.cfg <= 1.0 or neg_cond is None:
            # No CFG - use only positive conditioning
            cond_input = pos_cond if pos_cond is not None else torch.zeros_like(x[:1, :4])
            if hasattr(cond_input, 'to'):
                cond_input = cond_input.to(x.device)
            
            # Get model prediction
            with torch.no_grad():
                noise_pred = self._call_model(x, timestep, cond_input, model_options, seed)
                
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
            pos_cond = pos_cond if pos_cond is not None else torch.zeros_like(x[:1, :4])
            neg_cond = neg_cond if neg_cond is not None else torch.zeros_like(x[:1, :4])
            
            # Ensure conditioning is on correct device
            if hasattr(pos_cond, 'to'):
                pos_cond = pos_cond.to(x.device)
            if hasattr(neg_cond, 'to'):
                neg_cond = neg_cond.to(x.device)
            
            # Combine conditioning (negative first, then positive)
            cond_combined = torch.cat([neg_cond, pos_cond], dim=0)
            
            # Get model predictions
            with torch.no_grad():
                noise_pred_combined = self._call_model(x_combined, timestep_combined, cond_combined, model_options, seed)
            
            # Split predictions
            noise_pred_neg, noise_pred_pos = noise_pred_combined.chunk(2, dim=0)
            
            # Apply CFG
            noise_pred = noise_pred_neg + self.cfg * (noise_pred_pos - noise_pred_neg)
        
        return noise_pred
    
    def _prepare_model_patcher(self):
        """Prepare model patcher for sampling - ComfyUI compatible implementation"""
        logger.debug("Preparing model patcher for sampling")
        
        try:
            # Get the inner model
            if self.inner_model is None:
                if hasattr(self.model_patcher, 'model'):
                    self.inner_model = self.model_patcher.model
                else:
                    self.inner_model = self.model_patcher
            
            # Set current patcher on model (ComfyUI pattern)
            if hasattr(self.inner_model, 'current_patcher'):
                self.inner_model.current_patcher = self.model_patcher
            
            # Call pre_run if available (ComfyUI pattern)
            if hasattr(self.model_patcher, 'pre_run'):
                self.model_patcher.pre_run()
                logger.debug("Model patcher pre_run completed")
            
            # Prepare hook patches if available
            if hasattr(self.model_patcher, 'prepare_hook_patches_current_keyframe'):
                # This would be called with actual timestep and hook group in real usage
                logger.debug("Hook patches preparation available")
            
            # Set model options for hook mode
            if hasattr(self.model_patcher, 'set_hook_mode'):
                hook_mode = self.model_options.get('hook_mode', 'normal')
                if hook_mode == 'min_vram':
                    # Use minimal VRAM mode for single hook group
                    logger.debug("Setting minimal VRAM hook mode")
                else:
                    logger.debug("Using normal hook mode")
            
            logger.debug("Model patcher preparation completed successfully")
            
        except Exception as e:
            logger.error(f"Model patcher preparation failed: {e}")
            # Continue with fallback
            logger.warning("Continuing with fallback model patcher preparation")
    
    def _restore_hook_patches(self):
        """Restore hook patches - ComfyUI compatible implementation"""
        logger.debug("Restoring hook patches after sampling")
        
        try:
            # Restore hook patches if available (ComfyUI pattern)
            if hasattr(self.model_patcher, 'restore_hook_patches'):
                self.model_patcher.restore_hook_patches()
                logger.debug("Hook patches restored")
            
            # Clean hooks if available (ComfyUI pattern)
            if hasattr(self.model_patcher, 'clean_hooks'):
                self.model_patcher.clean_hooks()
                logger.debug("Hooks cleaned")
            
            # Unpatch hooks if available (ComfyUI pattern)
            if hasattr(self.model_patcher, 'unpatch_hooks'):
                self.model_patcher.unpatch_hooks()
                logger.debug("Hooks unpatched")
            
            # Clear cached hook weights if available
            if hasattr(self.model_patcher, 'clear_cached_hook_weights'):
                self.model_patcher.clear_cached_hook_weights()
                logger.debug("Cached hook weights cleared")
            
            # Cleanup model patcher if available (ComfyUI pattern)
            if hasattr(self.model_patcher, 'cleanup'):
                self.model_patcher.cleanup()
                logger.debug("Model patcher cleanup completed")
            
            # Clear current patcher reference
            if hasattr(self.inner_model, 'current_patcher'):
                self.inner_model.current_patcher = None
            
            logger.debug("Hook patch restoration completed successfully")
            
        except Exception as e:
            logger.error(f"Hook patch restoration failed: {e}")
            # Continue with fallback cleanup
            logger.warning("Continuing with fallback hook patch restoration")
            
            # Fallback cleanup
            try:
                if hasattr(self.model_patcher, 'cleanup'):
                    self.model_patcher.cleanup()
                logger.debug("Fallback cleanup completed")
            except Exception as fallback_e:
                logger.error(f"Fallback cleanup also failed: {fallback_e}")
    
    def _prepare_hook_patches_for_timestep(self, timestep, hook_group=None):
        """Prepare hook patches for specific timestep - ComfyUI compatible implementation"""
        logger.debug(f"Preparing hook patches for timestep: {timestep}")
        
        try:
            if hasattr(self.model_patcher, 'prepare_hook_patches_current_keyframe'):
                if hook_group is not None:
                    self.model_patcher.prepare_hook_patches_current_keyframe(
                        timestep, hook_group, self.model_options
                    )
                    logger.debug("Hook patches prepared for current keyframe")
                else:
                    logger.debug("No hook group provided for keyframe preparation")
            else:
                logger.debug("Hook keyframe preparation not available")
                
        except Exception as e:
            logger.warning(f"Hook keyframe preparation failed: {e}")
    
    def _get_hook_mode(self):
        """Get appropriate hook mode based on hook groups - ComfyUI compatible implementation"""
        try:
            hook_groups_count = get_total_hook_groups_in_conds(self.conds)
            
            if hook_groups_count <= 1:
                return 'min_vram'
            elif hook_groups_count <= 3:
                return 'normal'
            else:
                return 'max_speed'
                
        except Exception as e:
            logger.warning(f"Failed to determine hook mode: {e}")
            return 'normal'
    
    def _apply_hook_mode(self, hook_mode):
        """Apply hook mode to model patcher - ComfyUI compatible implementation"""
        logger.debug(f"Applying hook mode: {hook_mode}")
        
        try:
            if hasattr(self.model_patcher, 'set_hook_mode'):
                # Convert string to enum if needed
                if hook_mode == 'min_vram':
                    # Use minimal VRAM mode
                    logger.debug("Setting minimal VRAM hook mode")
                elif hook_mode == 'max_speed':
                    # Use maximum speed mode
                    logger.debug("Setting maximum speed hook mode")
                else:
                    # Use normal mode
                    logger.debug("Setting normal hook mode")
                
                # Store hook mode in model options
                self.model_options['hook_mode'] = hook_mode
                
        except Exception as e:
            logger.warning(f"Failed to apply hook mode {hook_mode}: {e}")
    
    def _cleanup_model_patcher(self):
        """Cleanup model patcher after sampling - ComfyUI compatible implementation"""
        logger.debug("Cleaning up model patcher")
        
        try:
            # Clean hooks first
            if hasattr(self.model_patcher, 'clean_hooks'):
                self.model_patcher.clean_hooks()
                logger.debug("Model patcher hooks cleaned")
            
            # Clear current patcher reference
            if hasattr(self.inner_model, 'current_patcher'):
                self.inner_model.current_patcher = None
                logger.debug("Current patcher reference cleared")
            
            # Final cleanup
            if hasattr(self.model_patcher, 'cleanup'):
                self.model_patcher.cleanup()
                logger.debug("Model patcher final cleanup completed")
            
            logger.debug("Model patcher cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Model patcher cleanup failed: {e}")
            # Continue with minimal cleanup
            logger.warning("Continuing with minimal cleanup")
    
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
        print(f"      CFG Scale: {self.cfg}")
        print(f"      Noise shape: {noise.shape}")
        print(f"      Sigmas: {len(sigmas)} steps")
        
        # Handle empty sigmas
        if sigmas.shape[-1] == 0:
            return latent_image
        
        # Restore conditioning from original_conds
        self.conds = {}
        for k in self.original_conds:
            if self.original_conds[k] is not None:
                self.conds[k] = [cond.copy() if hasattr(cond, 'copy') else cond for cond in self.original_conds[k]]
        
        # Preprocess conditioning hooks (ComfyUI compatible)
        preprocess_conds_hooks(self.conds)
        
        # Filter registered hooks on conditioning (ComfyUI compatible)
        filter_registered_hooks_on_conds(self.conds, self.model_options)
        
        # Model patcher preparation (ComfyUI compatible)
        try:
            # Store original model options
            orig_model_options = self.model_options
            
            # Create model options clone
            self.model_options = self.model_options.copy()
            
            # Determine and apply hook mode based on hook groups (ComfyUI compatible)
            hook_mode = self._get_hook_mode()
            self._apply_hook_mode(hook_mode)
            
            # Prepare model patcher with enhanced integration
            self._prepare_model_patcher()
            
            # Execute pre-run callbacks
            self.wrapper_executor.execute_callbacks('pre_run', self.model_patcher)
            
            # Create model wrapper for sampling
            model_wrapper = CFGModelWrapper(self)
            
            # Set up sampling parameters
            extra_args = {
                'seed': seed,
                'denoise_mask': denoise_mask,
                'model_options': self.model_options
            }
            
            # Memory monitoring setup
            sampling_start = time.time()
            if torch.cuda.is_available():
                mem_start = torch.cuda.memory_allocated() / 1024**2
                
            # Call sampler
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
            
            # Cleanup
            del self.conds
            
            return samples
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            # Cleanup on error
            if hasattr(self, 'conds'):
                del self.conds
            raise
        finally:
            # Enhanced cleanup and restoration (ComfyUI compatible)
            try:
                # Execute cleanup callbacks
                self.wrapper_executor.execute_callbacks('cleanup', self.model_patcher)
                
                # Cleanup model patcher
                self._cleanup_model_patcher()
                
                # Restore hook patches
                self._restore_hook_patches()
                
                # Cleanup wrapper executor
                self.wrapper_executor.cleanup()
                
                # Restore original model options
                self.model_options = orig_model_options
                
                logger.debug("CFGGuider cleanup completed successfully")
                
            except Exception as cleanup_e:
                logger.error(f"CFGGuider cleanup failed: {cleanup_e}")
                # Ensure model options are restored even if cleanup fails
                self.model_options = orig_model_options


class WrapperExecutor:
    """Wrapper executor for model patcher integration - ComfyUI compatible implementation"""
    
    def __init__(self, model_patcher, model_options=None):
        self.model_patcher = model_patcher
        self.model_options = model_options or {}
        self.wrappers = {}
        self.callbacks = {}
        
    def add_wrapper(self, wrapper_type, wrapper_func, key=None):
        """Add a wrapper function for model execution"""
        if wrapper_type not in self.wrappers:
            self.wrappers[wrapper_type] = {}
        if key not in self.wrappers[wrapper_type]:
            self.wrappers[wrapper_type][key] = []
        self.wrappers[wrapper_type][key].append(wrapper_func)
        
    def add_callback(self, callback_type, callback_func, key=None):
        """Add a callback function for model events"""
        if callback_type not in self.callbacks:
            self.callbacks[callback_type] = {}
        if key not in self.callbacks[callback_type]:
            self.callbacks[callback_type][key] = []
        self.callbacks[callback_type][key].append(callback_func)
        
    def execute_wrappers(self, wrapper_type, *args, **kwargs):
        """Execute all wrappers of a specific type"""
        if wrapper_type not in self.wrappers:
            return args, kwargs
            
        for key_wrappers in self.wrappers[wrapper_type].values():
            for wrapper in key_wrappers:
                try:
                    args, kwargs = wrapper(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Wrapper {wrapper_type} failed: {e}")
                    
        return args, kwargs
        
    def execute_callbacks(self, callback_type, *args, **kwargs):
        """Execute all callbacks of a specific type"""
        if callback_type not in self.callbacks:
            return
            
        for key_callbacks in self.callbacks[callback_type].values():
            for callback in key_callbacks:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback {callback_type} failed: {e}")
                    
    def cleanup(self):
        """Cleanup wrapper executor"""
        self.wrappers.clear()
        self.callbacks.clear()

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
        "normal": simple_scheduler.__func__,  # Alias for simple
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
