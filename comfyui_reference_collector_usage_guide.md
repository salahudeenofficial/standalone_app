# ComfyUI Step 4 Reference Data Collector Usage Guide
## Collect essential data from ComfyUI for motion pipeline verification

### 🎯 **OVERVIEW**

This guide explains how to use the ComfyUI Step 4 Reference Data Collector to gather essential data from ComfyUI for verifying the motion pipeline's Step 4 critical sections.

---

## 🚀 **QUICK START**

### **1. Setup in ComfyUI Environment**
```python
# Inside ComfyUI environment
import sys
sys.path.append('/path/to/your/motion/pipeline')

from comfyui_step4_reference_data_collector import ComfyUIStep4ReferenceCollector
```

### **2. Basic Usage**
```python
# Initialize collector
collector = ComfyUIStep4ReferenceCollector()

# Collect all critical sections reference data
reference_data = collector.collect_all_sections_reference(
    initial_latent=your_latent,
    positive_conditioning=your_positive_cond,
    negative_conditioning=your_negative_cond,
    unet_model=your_unet,
    seed=42,
    steps=4,
    cfg=7.0,
    sampler_name="euler",
    scheduler="normal",
    denoise=1.0
)

# Save reference data
filename = collector.save_reference_data(reference_data)
print(f"Reference data saved to: {filename}")
```

---

## 📊 **COLLECTED REFERENCE DATA**

### **Section 4.2: Latent Preparation**
```python
{
    'section': '4.2_latent_preparation',
    'inputs': {
        'initial_latent_shape': [1, 16, 9, 104, 60],
        'initial_latent_dtype': 'torch.float32',
        'initial_latent_device': 'cuda:0',
        'initial_latent_mean': 0.123456,
        'initial_latent_std': 0.789012,
        'seed': 42
    },
    'outputs': {
        'fix_empty_channels': {
            'original_shape': [1, 16, 9, 104, 60],
            'fixed_shape': [1, 16, 9, 104, 60],
            'shape_changed': False
        },
        'noise': {
            'noise_shape': [1, 16, 9, 104, 60],
            'noise_mean': 0.001234,
            'noise_std': 0.987654
        }
    }
}
```

### **Section 4.3: KSampler Setup**
```python
{
    'section': '4.3_ksampler_setup',
    'inputs': {
        'steps': 4,
        'sampler_name': 'euler',
        'scheduler': 'normal',
        'denoise': 1.0
    },
    'outputs': {
        'unet_validation': {
            'unet_loaded': True,
            'unet_type': 'UNetModel',
            'unet_device': 'cuda:0'
        },
        'ksampler': {
            'ksampler_created': True,
            'steps_set': True,
            'device_set': 'cuda:0'
        }
    }
}
```

### **Section 4.4: Denoising Execution**
```python
{
    'section': '4.4_denoising_execution',
    'inputs': {
        'noise_shape': [1, 16, 9, 104, 60],
        'fixed_latent_shape': [1, 16, 9, 104, 60],
        'cfg': 7.0,
        'seed': 42
    },
    'outputs': {
        'denoising': {
            'denoised_latent_shape': [1, 16, 9, 104, 60],
            'denoised_latent_mean': 0.234567,
            'denoised_latent_std': 0.654321,
            'shape_preserved': True,
            'denoising_time': 2.34
        }
    }
}
```

---

## 🔧 **DETAILED USAGE PATTERNS**

### **1. Collect Individual Section Reference**
```python
# Collect only Section 4.2 reference
section_4_2 = collector.collect_section_4_2_reference(
    initial_latent=your_latent,
    unet_model=your_unet,
    seed=42
)

# Collect only Section 4.3 reference
section_4_3 = collector.collect_section_4_3_reference(
    fixed_latent=section_4_2['outputs']['fixed_latent_tensor'],
    unet_model=your_unet,
    steps=4
)

# Collect only Section 4.4 reference
section_4_4 = collector.collect_section_4_4_reference(
    ksampler=section_4_3['outputs']['ksampler_instance'],
    noise=section_4_2['outputs']['noise_tensor'],
    positive_conditioning=your_positive_cond,
    negative_conditioning=your_negative_cond,
    fixed_latent=section_4_2['outputs']['fixed_latent_tensor'],
    cfg=7.0
)
```

### **2. Custom Parameters**
```python
# Custom parameters for different test scenarios
reference_data = collector.collect_all_sections_reference(
    initial_latent=your_latent,
    positive_conditioning=your_positive_cond,
    negative_conditioning=your_negative_cond,
    unet_model=your_unet,
    seed=123,                    # Different seed
    steps=8,                     # More steps
    cfg=12.0,                    # Higher CFG
    sampler_name="dpmpp_2m",     # Different sampler
    scheduler="karras",          # Different scheduler
    denoise=0.8                  # Partial denoising
)
```

### **3. Save with Custom Filename**
```python
# Save with descriptive filename
filename = collector.save_reference_data(
    reference_data, 
    "comfyui_step4_reference_custom_test.json"
)
```

---

## 📋 **REQUIREMENTS**

### **ComfyUI Environment Setup**
```python
# Required imports in ComfyUI
from comfy.sample import fix_empty_latent_channels, prepare_noise
from comfy.samplers import KSampler
import torch
```

### **Required Data**
- **initial_latent**: 5D tensor [B, C, T, H, W]
- **positive_conditioning**: List of conditioning tensors
- **negative_conditioning**: List of conditioning tensors
- **unet_model**: Loaded UNet model
- **Optional**: noise_inds tensor for reproducible noise

---

## 🎯 **VERIFICATION WORKFLOW**

### **1. Collect ComfyUI Reference**
```python
# In ComfyUI environment
collector = ComfyUIStep4ReferenceCollector()
comfyui_reference = collector.collect_all_sections_reference(
    initial_latent, positive_conditioning, negative_conditioning,
    unet_model, seed=42, steps=4, cfg=7.0
)
filename = collector.save_reference_data(comfyui_reference)
```

### **2. Load Reference in Motion Pipeline**
```python
# In motion pipeline environment
import json
import torch

# Load ComfyUI reference
with open('comfyui_step4_reference_20241201_143022.json', 'r') as f:
    comfyui_reference = json.load(f)

# Convert tensor data back to tensors
def load_tensor_from_data(tensor_data):
    if isinstance(tensor_data, dict) and 'tensor_data' in tensor_data:
        return torch.tensor(tensor_data['tensor_data']).to(tensor_data['device'])
    return tensor_data

# Load tensors
comfyui_fixed_latent = load_tensor_from_data(comfyui_reference['sections']['4.2']['outputs']['fixed_latent_tensor'])
comfyui_noise = load_tensor_from_data(comfyui_reference['sections']['4.2']['outputs']['noise_tensor'])
comfyui_denoised = load_tensor_from_data(comfyui_reference['sections']['4.4']['outputs']['denoised_latent_tensor'])
```

### **3. Compare with Motion Pipeline**
```python
# Run motion pipeline critical sections
from step4_critical_sections_reverse_test import Step4CriticalReverseTest

critical_tester = Step4CriticalReverseTest(pipeline)
motion_results = critical_tester.run_critical_sections_reverse(
    initial_latent, positive_conditioning, negative_conditioning,
    seed=42, steps=4, cfg=7.0
)

# Compare results
def compare_section_results(comfyui_section, motion_section, section_name):
    print(f"\n🔍 COMPARING {section_name}:")
    
    # Compare key metrics
    if 'outputs' in comfyui_section and 'outputs' in motion_section:
        comfyui_outputs = comfyui_section['outputs']
        motion_outputs = motion_section['outputs']
        
        # Compare shapes, means, stds, etc.
        for key in ['shape', 'mean', 'std']:
            if key in comfyui_outputs and key in motion_outputs:
                comfyui_val = comfyui_outputs[key]
                motion_val = motion_outputs[key]
                match = abs(comfyui_val - motion_val) < 1e-6
                print(f"   {key}: ComfyUI={comfyui_val}, Motion={motion_val}, Match={match}")

# Compare each section
compare_section_results(
    comfyui_reference['sections']['4.2'], 
    motion_results['critical_sections']['4.2'], 
    'Section 4.2'
)

compare_section_results(
    comfyui_reference['sections']['4.4'], 
    motion_results['critical_sections']['4.4'], 
    'Section 4.4'
)
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

#### **1. Import Errors**
```python
# If ComfyUI imports fail
try:
    from comfy.sample import fix_empty_latent_channels, prepare_noise
    from comfy.samplers import KSampler
except ImportError as e:
    print(f"ComfyUI import failed: {e}")
    print("Make sure you're running in ComfyUI environment")
```

#### **2. Tensor Device Mismatches**
```python
# Ensure all tensors are on the same device
def ensure_device_consistency(tensor, target_device):
    if tensor.device != target_device:
        return tensor.to(target_device)
    return tensor

# Apply to all tensors
initial_latent = ensure_device_consistency(initial_latent, unet_model.device)
```

#### **3. Memory Issues**
```python
# Clear CUDA cache before collection
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared")
```

---

## 📊 **OUTPUT FORMAT**

### **Complete Reference Data Structure**
```python
{
    'sections': {
        '4.2': {
            'section': '4.2_latent_preparation',
            'inputs': {...},
            'outputs': {...},
            'timing': {...}
        },
        '4.3': {
            'section': '4.3_ksampler_setup',
            'inputs': {...},
            'outputs': {...},
            'timing': {...}
        },
        '4.4': {
            'section': '4.4_denoising_execution',
            'inputs': {...},
            'outputs': {...},
            'timing': {...}
        }
    },
    'overall_status': 'success',
    'timing': {
        'total_time': 5.67
    },
    'comfyui_version': 'reference',
    'collection_info': {
        'timestamp': '2024-12-01 14:30:22',
        'purpose': 'motion_pipeline_verification'
    }
}
```

---

## 🎯 **BEST PRACTICES**

### **1. Consistent Parameters**
- Use the same seed, steps, CFG, and other parameters in both ComfyUI and motion pipeline
- Ensure tensor shapes and devices match

### **2. Multiple Test Cases**
- Collect reference data for different scenarios (different seeds, CFG values, etc.)
- Test edge cases (empty latents, extreme values, etc.)

### **3. Validation**
- Always validate that ComfyUI reference collection succeeded
- Check that all critical sections completed without errors
- Verify tensor properties (shape, dtype, device, values)

### **4. Documentation**
- Save reference data with descriptive filenames
- Include metadata about the test scenario
- Document any special conditions or parameters

---

## 🚀 **NEXT STEPS**

1. **Setup ComfyUI Environment**: Ensure all required imports and models are available
2. **Collect Reference Data**: Run the collector with your specific parameters
3. **Save Reference Data**: Store the collected data for motion pipeline verification
4. **Load in Motion Pipeline**: Import the reference data into your motion pipeline environment
5. **Compare Results**: Use the comparison functions to verify motion pipeline accuracy
6. **Debug Differences**: Focus on any discrepancies found during comparison
7. **Iterate**: Repeat the process until motion pipeline matches ComfyUI reference

---

## 📝 **CONCLUSION**

The ComfyUI Step 4 Reference Data Collector provides:

- ✅ **Complete Reference Data**: All critical sections covered
- ✅ **Detailed Metrics**: Shapes, means, stds, timing, etc.
- ✅ **Serializable Format**: JSON format for easy transfer
- ✅ **Comprehensive Validation**: Input/output validation for each section
- ✅ **Easy Integration**: Simple API for collecting and saving data

**This tool enables systematic verification of the motion pipeline against ComfyUI reference data, ensuring accuracy and compatibility.**
