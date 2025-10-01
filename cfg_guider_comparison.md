# CFGGuider Class Comparison: Motion Pipeline vs ComfyUI

## 🎯 **OVERVIEW**

This document compares the `CFGGuider` implementation in the motion pipeline's `standalone_ksampler.py` with ComfyUI's `CFGGuider` in `comfy/samplers.py` to identify key differences and potential issues.

---

## 📋 **CLASS STRUCTURE COMPARISON**

### **Motion Pipeline: StandaloneCFGGuider**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/standalone_ksampler.py`
- **Lines**: 23-950
- **Class Name**: `StandaloneCFGGuider`

### **ComfyUI: CFGGuider**
- **File**: `/home/fashionx/v_pipe/standalone_app/comfy/samplers.py`
- **Lines**: 931-1027
- **Class Name**: `CFGGuider`

---

## 🔧 **INITIALIZATION COMPARISON**

### **Motion Pipeline:**
```python
def __init__(self, model_patcher):
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
```

### **ComfyUI:**
```python
def __init__(self, model_patcher: ModelPatcher):
    self.model_patcher = model_patcher
    self.model_options = model_patcher.model_options
    self.original_conds = {}
    self.cfg = 1.0
```

**Key Differences:**
- ✅ **Motion**: Has memory tracking
- ❌ **Motion**: Missing `original_conds` storage
- ❌ **Motion**: Uses `cfg_scale` vs ComfyUI's `cfg`
- ✅ **Motion**: Has device management

---

## 🎯 **CONDITIONING SETUP COMPARISON**

### **Motion Pipeline:**
```python
def set_conds(self, positive, negative):
    """Set positive and negative conditioning"""
    print(f"   🔧 Setting CFG conditioning...")
    
    # Convert conditioning format if needed
    self.positive_cond = self._convert_conditioning(positive)
    self.negative_cond = self._convert_conditioning(negative) 
    
    print(f"      Positive conditioning shape: {self._get_cond_shape(self.positive_cond)}")
    print(f"      Negative conditioning shape: {self._get_cond_shape(self.negative_cond)}")

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
```

### **ComfyUI:**
```python
def set_conds(self, positive, negative):
    self.inner_set_conds({"positive": positive, "negative": negative})

def inner_set_conds(self, conds):
    for k in conds:
        self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])
```

**Key Differences:**
- ❌ **Motion**: Simple conditioning conversion vs ComfyUI's `convert_cond`
- ❌ **Motion**: No `original_conds` storage for restoration
- ❌ **Motion**: No proper conditioning preprocessing

---

## 🚀 **NOISE PREDICTION COMPARISON**

### **Motion Pipeline:**
```python
def predict_noise(self, x, timestep, model_options=None, seed=None):
    """Predict noise using CFG"""
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
    
    return noise_pred
```

### **ComfyUI:**
```python
def predict_noise(self, x, timestep, model_options={}, seed=None):
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
```

**Key Differences:**
- ❌ **Motion**: Manual CFG implementation vs ComfyUI's `sampling_function`
- ❌ **Motion**: No `inner_model` abstraction
- ❌ **Motion**: No proper conditioning processing
- ❌ **Motion**: Manual tensor concatenation vs ComfyUI's batch processing

---

## 🎯 **SAMPLING FUNCTION COMPARISON**

### **ComfyUI's `sampling_function`:**
```python
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out  = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)
```

### **ComfyUI's `cfg_function`:**
```python
def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result
```

**Key Differences:**
- ❌ **Motion**: Missing `calc_cond_batch` for proper conditioning
- ❌ **Motion**: Missing hook system integration
- ❌ **Motion**: Missing pre/post CFG function hooks
- ❌ **Motion**: Missing CFG optimization for scale=1.0

---

## 🔧 **MODEL CALLING COMPARISON**

### **Motion Pipeline:**
```python
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
    
    # Try different model call strategies
    try:
        # Strategy 1: Try model.forward() method directly
        if hasattr(model, 'forward'):
            # Check if this is a VaceWanModel that needs context parameter
            if hasattr(model, '__class__') and 'Vace' in model.__class__.__name__:
                # For VaceWanModel, we need to pass context parameter
                result = model.forward(x, timestep, conditioning)
            else:
                # For other models, try the original call
                result = model.forward(x, timestep)
            
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
        # Strategy 3: Try apply_model method (ComfyUI style)
        # Strategy 4: Try with conditioning as additional argument
        # Strategy 5: Last resort - try ModelPatcher
        
    except Exception as e:
        # Final fallback: Return zero tensor
        logger.warning(f"All model call strategies failed, returning zeros")
        return torch.zeros_like(x)
```

### **ComfyUI:**
```python
# Uses calc_cond_batch which handles:
# - Hook system integration
# - Proper conditioning processing
# - Batch optimization
# - Model patcher integration
```

**Key Differences:**
- ❌ **Motion**: Manual model calling vs ComfyUI's integrated system
- ❌ **Motion**: No hook system support
- ❌ **Motion**: No proper conditioning processing
- ❌ **Motion**: Fallback to zeros vs proper error handling

---

## 📊 **SAMPLING METHOD COMPARISON**

### **Motion Pipeline:**
```python
def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    """Main sampling method using CFG"""
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
    
    # Use the sampler
    samples = sampler.sample(model_wrapper, noise, sigmas, extra_args, callback, disable_pbar, seed)
    
    return samples
```

### **ComfyUI:**
```python
def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    if sigmas.shape[-1] == 0:
        return latent_image

    self.conds = {}
    for k in self.original_conds:
        self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
    preprocess_conds_hooks(self.conds)

    try:
        orig_model_options = self.model_options
        self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
        orig_hook_mode = self.model_patcher.hook_mode
        if get_total_hook_groups_in_conds(self.conds) <= 1:
            self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
        filter_registered_hooks_on_conds(self.conds, self.model_options)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.outer_sample,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
        )
        output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
    finally:
        cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
        self.model_options = orig_model_options
        self.model_patcher.hook_mode = orig_hook_mode
        self.model_patcher.restore_hook_patches()

    del self.conds
    return output
```

**Key Differences:**
- ❌ **Motion**: No hook system integration
- ❌ **Motion**: No proper model patcher preparation
- ❌ **Motion**: No wrapper executor system
- ❌ **Motion**: No proper cleanup and restoration

---

## 🚨 **CRITICAL DIFFERENCES SUMMARY**

### **1. Missing Core Components:**
- ❌ **No `calc_cond_batch`**: ComfyUI's sophisticated conditioning processing
- ❌ **No hook system**: ComfyUI's extensible hook architecture
- ❌ **No wrapper executor**: ComfyUI's execution wrapper system
- ❌ **No proper model patcher integration**: ComfyUI's model management

### **2. Simplified vs Sophisticated:**
- ❌ **Motion**: Manual CFG implementation
- ✅ **ComfyUI**: Integrated `sampling_function` with optimizations
- ❌ **Motion**: Basic conditioning handling
- ✅ **ComfyUI**: Advanced conditioning processing with hooks

### **3. Error Handling:**
- ❌ **Motion**: Falls back to zeros on model call failure
- ✅ **ComfyUI**: Proper error handling and recovery
- ❌ **Motion**: No proper cleanup
- ✅ **ComfyUI**: Comprehensive cleanup and restoration

### **4. Performance Optimizations:**
- ❌ **Motion**: No CFG optimization for scale=1.0
- ✅ **ComfyUI**: Optimized CFG handling
- ❌ **Motion**: No batch optimization
- ✅ **ComfyUI**: Advanced batch processing

---

## 🎯 **RECOMMENDATIONS**

### **1. Implement Missing Core Functions:**
```python
# Need to implement:
def calc_cond_batch(model, conds, x_in, timestep, model_options):
    # ComfyUI's sophisticated conditioning processing
    
def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    # ComfyUI's CFG function with hooks
```

### **2. Add Hook System Support:**
```python
# Need to add:
- Hook system integration
- Pre/post CFG function hooks
- Sampler function hooks
```

### **3. Improve Model Integration:**
```python
# Need to improve:
- Proper model patcher integration
- Wrapper executor system
- Better error handling
```

### **4. Add Performance Optimizations:**
```python
# Need to add:
- CFG optimization for scale=1.0
- Batch processing optimizations
- Memory management improvements
```

---

## 📝 **CONCLUSION**

The motion pipeline's `StandaloneCFGGuider` is a **simplified version** of ComfyUI's `CFGGuider` that **lacks critical functionality**:

1. **Missing sophisticated conditioning processing**
2. **No hook system integration**
3. **Manual CFG implementation vs ComfyUI's optimized system**
4. **No proper model patcher integration**
5. **Basic error handling vs ComfyUI's robust system**

**This explains why the motion pipeline's KSampling may not produce the same results as ComfyUI** - the CFG implementation is fundamentally different and missing key components that ComfyUI uses for proper guidance.

**Next Steps:**
1. Implement ComfyUI's `calc_cond_batch` function
2. Add hook system support
3. Implement proper CFG function with optimizations
4. Improve model integration and error handling
5. Add performance optimizations
