# STEP 4 & STEP 5 DEBUGGING GUIDE
## Complete Manual Debugging Guide for KSampling and Latent Trimming

### 🎯 **OVERVIEW**

This guide provides a comprehensive roadmap for manually debugging Step 4 (KSampling) and Step 5 (Latent Trimming) in the motion pipeline.

---

## 📋 **STEP 4: KSAMPLER DENOISING - DEBUGGING GUIDE**

### **Core Files to Explore:**

#### **1. Main Pipeline File**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/pipeline.py`
- **Lines**: 1221-1404
- **Function**: `step_4_ksampler_denoising()`
- **Purpose**: Main Step 4 implementation with input/output analysis

#### **2. KSampler Implementation**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/standalone_ksampler.py`
- **Classes**: `StandaloneKSampler`, `StandaloneCFGGuider`
- **Purpose**: Core sampling logic and CFG guidance

#### **3. Analysis Functions**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/pipeline.py`
- **Lines**: 37-216
- **Functions**: `analyze_ksampler_inputs()`, `_analyze_conditioning()`, `_print_tensor_info()`
- **Purpose**: Detailed tensor analysis and debugging

#### **4. Test Files**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/test_pipeline_step4.py`
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/debug_ksampler_step_4.py`
- **Purpose**: Standalone testing and debugging

#### **5. Documentation**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/STEP4_CRITICAL_FIXES.md`
- **Purpose**: Known issues and fixes

### **Step 4 Debugging Checklist:**

#### **A. Input Analysis (Lines 1275-1285)**
```python
# 1. Check positive conditioning structure
analyze_ksampler_inputs(positive_conditioning, negative_conditioning, initial_latent)

# 2. Verify tensor types and shapes:
# - CLIP text embeddings: [B, 77, 4096]
# - VACE frames: [B, 32, T, H, W] 
# - VACE masks: [B, 64, T, H, W]
# - Initial latent: [B, 16, T, H, W]
```

#### **B. Noise Preparation (Lines 1278-1285)**
```python
# 1. Check ComfyUI integration
try:
    from comfy.sample import fix_empty_latent_channels
    initial_latent = fix_empty_latent_channels(self.unet, initial_latent)
except ImportError:
    pass  # Use original latent

# 2. Verify noise generation
noise = prepare_noise(initial_latent, seed, noise_inds)
# Check: noise.shape == initial_latent.shape
```

#### **C. KSampler Setup (Lines 1287-1295)**
```python
# 1. Verify KSampler parameters
ksampler = StandaloneKSampler(
    model=self.unet,        # Check: self.unet is loaded
    steps=steps,            # Check: steps > 0
    device=self.device,     # Check: device is correct
    sampler=sampler_name,   # Check: valid sampler name
    scheduler=scheduler,    # Check: valid scheduler
    denoise=denoise         # Check: 0.0 <= denoise <= 1.0
)
```

#### **D. Sampling Execution (Lines 1300-1294)**
```python
# 1. Monitor sampling process
denoised_latent = ksampler.sample(
    noise=noise,
    positive=positive_conditioning,
    negative=negative_conditioning,
    cfg=cfg,
    latent_image=initial_latent,
    # ... other parameters
)

# 2. Verify output
# Check: denoised_latent.shape == initial_latent.shape
# Check: denoised_latent is not all zeros
# Check: denoised_latent has reasonable value range
```

#### **E. Output Analysis (Lines 1332-1365)**
```python
# 1. Compare input vs output
print(f"Input shape: {initial_latent.shape}")
print(f"Output shape: {denoised_latent.shape}")
print(f"Input mean: {initial_latent.mean().item():.6f}")
print(f"Output mean: {denoised_latent.mean().item():.6f}")

# 2. Check for significant changes
diff = torch.abs(denoised_latent - initial_latent)
print(f"Mean absolute difference: {diff.mean().item():.6f}")
print(f"Significant change: {'YES' if diff.mean().item() > 0.001 else 'NO'}")
```

---

## 📋 **STEP 5: TRIM VIDEO LATENT - DEBUGGING GUIDE**

### **Core Files to Explore:**

#### **1. Main Pipeline File**
- **File**: `/home/fashionx/v_pipe/standalone_app/motion/pipeline.py`
- **Lines**: 1415-1499
- **Function**: `step_5_trim_latent()`
- **Purpose**: Main Step 5 implementation

#### **2. Video Processor Component**
- **File**: `/home/fashionx/v_pipe/standalone_app/components/video_processor.py`
- **Class**: `TrimVideoLatent`
- **Purpose**: Core trimming logic

### **Step 5 Debugging Checklist:**

#### **A. Input Validation (Lines 1447-1449)**
```python
# 1. Check input latent
print(f"Input latent shape: {denoised_latent.shape}")
print(f"Trim amount: {trim_amount} frames")

# 2. Verify tensor format
# Expected: [B, C, T, H, W] where T is temporal dimension
# Check: len(denoised_latent.shape) == 5
# Check: denoised_latent.shape[2] > trim_amount
```

#### **B. Component Import (Lines 1439-1442)**
```python
# 1. Check component import
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.video_processor import TrimVideoLatent

# 2. Verify component availability
trim_processor = TrimVideoLatent()
# Check: trim_processor is not None
```

#### **C. Trimming Process (Lines 1454-1461)**
```python
# 1. Prepare input format
latent_dict = {"samples": denoised_latent}

# 2. Execute trimming
trimmed_latent_dict = trim_processor.op(latent_dict, trim_amount)

# 3. Extract result
trimmed_latent = trimmed_latent_dict["samples"]

# 4. Verify output
# Check: trimmed_latent.shape[2] == denoised_latent.shape[2] - trim_amount
# Check: trimmed_latent.shape[0:2] == denoised_latent.shape[0:2]  # B, C unchanged
# Check: trimmed_latent.shape[3:5] == denoised_latent.shape[3:5]  # H, W unchanged
```

#### **D. Core Trimming Logic (video_processor.py)**
```python
# 1. Examine the actual trimming operation
def op(self, samples, trim_amount):
    samples_out = samples.copy()
    s1 = samples["samples"]
    samples_out["samples"] = s1[:, :, trim_amount:]  # Remove first trim_amount frames
    return samples_out

# 2. Verify the slicing operation
# s1[:, :, trim_amount:] means:
# - Keep all batch dimensions (:,)
# - Keep all channel dimensions (:,)
# - Remove first trim_amount temporal frames (trim_amount:)
# - Keep all height and width dimensions (implicit)
```

#### **E. Output Analysis (Lines 1465-1472)**
```python
# 1. Calculate frame statistics
original_frames = denoised_latent.shape[2]
trimmed_frames = trimmed_latent.shape[2]
frames_removed = original_frames - trimmed_frames

# 2. Verify calculations
print(f"Original frames: {original_frames}")
print(f"Trimmed frames: {trimmed_frames}")
print(f"Frames removed: {frames_removed}")
print(f"Expected frames removed: {trim_amount}")
print(f"Calculation correct: {'YES' if frames_removed == trim_amount else 'NO'}")
```

---

## 🔧 **MANUAL DEBUGGING STEPS**

### **Step 4 Manual Debugging:**

1. **Add Debug Prints in `standalone_ksampler.py`:**
   ```python
   # In StandaloneKSampler.sample() method
   print(f"🔍 DEBUG: Input noise shape: {noise.shape}")
   print(f"🔍 DEBUG: Input noise range: [{noise.min():.6f}, {noise.max():.6f}]")
   print(f"🔍 DEBUG: Input noise mean: {noise.mean():.6f}")
   
   # In each sampling step
   print(f"🔍 DEBUG: Step {i}: x shape: {x.shape}, x range: [{x.min():.6f}, {x.max():.6f}]")
   ```

2. **Add Debug Prints in `pipeline.py` Step 4:**
   ```python
   # Before sampling
   print(f"🔍 DEBUG: UNet model type: {type(self.unet)}")
   print(f"🔍 DEBUG: UNet device: {next(self.unet.parameters()).device}")
   print(f"🔍 DEBUG: CFG value: {cfg}")
   
   # After sampling
   print(f"🔍 DEBUG: Denoised latent range: [{denoised_latent.min():.6f}, {denoised_latent.max():.6f}]")
   print(f"🔍 DEBUG: Denoised latent mean: {denoised_latent.mean():.6f}")
   ```

### **Step 5 Manual Debugging:**

1. **Add Debug Prints in `video_processor.py`:**
   ```python
   def op(self, samples, trim_amount):
       print(f"🔍 DEBUG: Input samples keys: {samples.keys()}")
       print(f"🔍 DEBUG: Input samples shape: {samples['samples'].shape}")
       print(f"🔍 DEBUG: Trim amount: {trim_amount}")
       
       samples_out = samples.copy()
       s1 = samples["samples"]
       print(f"🔍 DEBUG: Before trim - s1 shape: {s1.shape}")
       
       samples_out["samples"] = s1[:, :, trim_amount:]
       print(f"🔍 DEBUG: After trim - output shape: {samples_out['samples'].shape}")
       
       return samples_out
   ```

2. **Add Debug Prints in `pipeline.py` Step 5:**
   ```python
   # Before trimming
   print(f"🔍 DEBUG: Input latent dtype: {denoised_latent.dtype}")
   print(f"🔍 DEBUG: Input latent device: {denoised_latent.device}")
   print(f"🔍 DEBUG: Input latent memory: {denoised_latent.numel() * denoised_latent.element_size() / (1024**2):.2f} MB")
   
   # After trimming
   print(f"🔍 DEBUG: Output latent dtype: {trimmed_latent.dtype}")
   print(f"🔍 DEBUG: Output latent device: {trimmed_latent.device}")
   print(f"🔍 DEBUG: Memory saved: {(denoised_latent.numel() - trimmed_latent.numel()) * denoised_latent.element_size() / (1024**2):.2f} MB")
   ```

---

## 🎯 **KEY DEBUGGING POINTS**

### **Step 4 Critical Points:**
1. **Model Interface**: Verify UNet model has correct `apply_model` method
2. **CFG Guidance**: Check that CFG is properly applied
3. **Noise Generation**: Verify noise has correct shape and distribution
4. **Sampling Steps**: Monitor each denoising step
5. **Output Validation**: Ensure output is not dummy/zero data

### **Step 5 Critical Points:**
1. **Input Validation**: Verify input tensor has 5 dimensions
2. **Trim Amount**: Check trim_amount doesn't exceed available frames
3. **Slicing Operation**: Verify `[:, :, trim_amount:]` works correctly
4. **Output Shape**: Ensure output shape is as expected
5. **Memory Management**: Check memory usage before/after trimming

---

## 📊 **EXPECTED OUTPUTS**

### **Step 4 Expected Output:**
- **Input**: `[1, 16, 11, 104, 60]` (zero latent)
- **Output**: `[1, 16, 11, 104, 60]` (denoised latent)
- **Significant Change**: YES (mean absolute difference > 0.001)
- **Value Range**: Reasonable (not all zeros, not extreme values)

### **Step 5 Expected Output:**
- **Input**: `[1, 16, 11, 104, 60]` (denoised latent)
- **Output**: `[1, 16, 11-trim_amount, 104, 60]` (trimmed latent)
- **Frames Removed**: Exactly `trim_amount`
- **Memory Reduction**: Proportional to frames removed

---

## 🚨 **COMMON ISSUES TO WATCH FOR**

### **Step 4 Issues:**
1. **Dummy Output**: KSampler returns zero/constant values
2. **Shape Mismatch**: Output shape differs from input
3. **CFG Not Applied**: No difference between positive/negative conditioning
4. **Model Interface**: UNet model doesn't have correct methods
5. **Memory Issues**: OOM during sampling

### **Step 5 Issues:**
1. **Import Errors**: `TrimVideoLatent` not found
2. **Shape Errors**: Input tensor doesn't have 5 dimensions
3. **Trim Amount**: trim_amount exceeds available frames
4. **Slicing Errors**: Incorrect tensor slicing
5. **Memory Leaks**: Tensors not properly cleaned up

---

## 🔍 **DEBUGGING TOOLS**

### **Tensor Analysis:**
```python
def debug_tensor(tensor, name):
    print(f"🔍 {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    print(f"   Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"   Mean: {tensor.mean():.6f}")
    print(f"   Std: {tensor.std():.6f}")
    print(f"   Memory: {tensor.numel() * tensor.element_size() / (1024**2):.2f} MB")
```

### **Memory Monitoring:**
```python
def debug_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"🔍 Memory {stage}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
```

This guide provides a comprehensive roadmap for manually debugging both Step 4 and Step 5. Follow the checklists and add the suggested debug prints to trace through each component systematically.
