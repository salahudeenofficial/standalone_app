# 📋 **ComfyUI WAN Model Dependencies Analysis**

## **🎯 Target File Analysis**
**Source:** `/home/fashionx/comfy/ComfyUI/comfy/ldm/wan/model.py`  
**Target:** Implement equivalent classes in motion pipeline

---

## **🔍 DEPENDENCIES IDENTIFIED**

### **🚨 CRITICAL DEPENDENCIES (Must Implement)**

#### **1. comfy.ldm.modules.attention**
```python
from comfy.ldm.modules.attention import optimized_attention

# USAGE: optimized_attention(q, k, v, heads=self.num_heads)
# Used in: WanSelfAttention, WanT2VCrossAttention, WanI2VCrossAttention
```
**Motion Status:** ❌ **MISSING** - Need to create standalone version

#### **2. comfy.ldm.flux.layers**
```python
from comfy.ldm.flux.layers import EmbedND

# USAGE: EmbedND class for embeddings
# Used in: May be used in embedding layers
```
**Motion Status:** ❌ **MISSING** - Need to check usage and create if needed

#### **3. comfy.ldm.flux.math**
```python
from comfy.ldm.flux.math import apply_rope

# USAGE: q, k = apply_rope(q, k, freqs)  
# Used in: WanSelfAttention.forward() for RoPE embeddings
```
**Motion Status:** ❌ **MISSING** - Essential for attention mechanisms

#### **4. comfy.ldm.common_dit**
```python
import comfy.ldm.common_dit

# USAGE: Likely contains common DiT (Diffusion Transformer) utilities
# Used in: May be imported for utility functions
```
**Motion Status:** ❌ **MISSING** - Need to investigate usage

#### **5. comfy.model_management**
```python
import comfy.model_management

# USAGE: comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device)
# Used in: WanAttentionBlock.forward() for dtype/device management
```
**Motion Status:** ⚠️ **PARTIAL** - Motion has basic model_management, need cast_to function

---

### **✅ AVAILABLE DEPENDENCIES**

#### **6. einops**
```python
from einops import repeat

# USAGE: repeat_e function implementation  
# Status: ✅ Motion has einops support
```

#### **7. Pure PyTorch**
```python
import torch
import torch.nn as nn
import math

# Status: ✅ All available
```

---

## **📊 CLASS IMPLEMENTATION PRIORITY**

### **🔴 HIGH PRIORITY (Core Classes)**

| **Class** | **Motion Status** | **ComfyUI Priority** | **Action Required** |
|-----------|-------------------|---------------------|-------------------|
| **WanSelfAttention** | ⚠️ Different implementation | 🔴 Critical | Update with ComfyUI pattern |
| **WanT2VCrossAttention** | ❌ Missing | 🔴 Critical | Implement complete |
| **WanI2VCrossAttention** | ❌ Missing | 🔴 Critical | Implement complete |
| **WanAttentionBlock** | ⚠️ Different implementation | 🔴 Critical | Update with ComfyUI pattern |
| **VaceWanAttentionBlock** | ✅ Implemented | 🟡 Medium | Validate consistency |
| **VaceWanModel** | ⚠️ Different implementation | 🔴 Critical | Update with ComfyUI pattern |

### **🟡 MEDIUM PRIORITY (Support Classes)**

| **Class** | **Motion Status** | **ComfyUI Priority** | **Action Required** |
|-----------|-------------------|---------------------|-------------------|
| **WanModel** | ✅ Implemented | 🟡 Medium | Validate base class |
| **Head** | ✅ Implemented | 🟡 Medium | May need updates |
| **EmbedND** | ❌ Missing | 🟡 Medium | Investigate usage |

---

## **🛠️ IMPLEMENTATION STRATEGY**

### **Phase 1: Dependency Modules**
```bash
# Create standalone modules for ComfyUI dependencies:

1. motion/ldm/modules/attention.py    → optimized_attention function
2. motion/ldm/flux/layers.py          → EmbedND class  
3. motion/ldm/flux/math.py           → apply_rope function
4. motion/ldm/common_dit.py           → Common DiT utilities
5. motion/model_management.py        → Add cast_to function
```

### **Phase 2: Core Classes**
```bash
# Implement ComfyUI model classes in motion/models.py:

1. WanSelfAttention      → RoPE + optimized attention
2. WanT2VCrossAttention  → Text-to-video cross attention  
3. WanI2VCrossAttention  → Image-to-video cross attention
4. WanAttentionBlock     → Complete attention block with modulation
5. VaceWanAttentionBlock → VACE-specific attention (update existing)
6. VaceWanModel         → Complete ComfyUI-compatible model (update existing)
```

### **Phase 3: Integration**
```bash
# Test integration with existing motion pipeline:

1. Test standalone model creation
2. Test forward pass compatibility  
3. Test memory management
4. Test with existing motion pipeline
```

---

## **🔍 DETAILED DEPENDENCY MODULES**

### **1. optimized_attention Function**
```python
# Need to implement:
def optimized_attention(q, k, v, heads, mask=None):
    """
    Optimized multi-head attention
    Args:
        q: [B, L, C] query tensor
        k: [B, L, C] key tensor  
        v: [B, L, C] value tensor
        heads: number of attention heads
        mask: optional attention mask
    Returns:
        [B, L, C] attention output
    """
```

### **2. apply_rope Function**
```python
# Need to implement:
def apply_rope(q, k, freqs):
    """
    Apply Rotary Position Embeddings (RoPE)
    Args:
        q: [B, L, num_heads, dim] query tensor
        k: [B, L, num_heads, dim] key tensor
        freqs: [1024, dim/2] frequency tensor
    Returns:
        q, k with RoPE applied
    """
```

### **3. EmbedND Class**
```python
# Need to implement:
class EmbedND(nn.Module):
    """N-dimensional embedding layer"""
    # Usage dependent on investigation
```

### **4. cast_to Function**
```python
# Need to add to motion/model_management:
def cast_to(tensor, dtype, device):
    """Cast tensor to specific dtype and device"""
    return tensor.to(dtype=dtype, device=device)
```

---

## **🎯 NEXT STEPS**

### **Immediate Actions (Next 1-2 hours)**
1. **🔴 Create optimized_attention.py** - Standalone attention implementation
2. **🔴 Create apply_rope.py** - RoPE implementation  
3. **🔴 Create cast_to function** - Add to existing model_management
4. **🔴 Update WanSelfAttention** - Use ComfyUI operations pattern

### **Secondary Actions (Next 3-4 hours)**
5. **🟡 Create WanT2VCrossAttention** - Text-to-video attention
6. **🟡 Create WanI2VCrossAttention** - Image-to-video attention  
7. **🟡 Update WanAttentionBlock** - Complete ComfyUI pattern
8. **🟡 Update VaceWanModel** - Ensure full compatibility

### **Testing Phase (Next 2-3 hours)**
9. **🟢 Test standalone models** - Individual class testing
10. **🟢 Test integration** - With existing motion pipeline
11. **🟢 Performance validation** - Memory usage and speed

---

## **💡 QUICK START COMMANDS**

```bash
# Navigate to motion directory
cd /home/fashionx/v_pipe/standalone_app/motion

# Create directory structure
mkdir -p ldm/modules ldm/flux ldm/common

# Create dependency files
touch ldm/modules/attention.py ldm/flux/layers.py ldm/flux/math.py ldm/common_dit.py

# Start implementing optimized_attention first
vim ldm/modules/attention.py
```

**Estimated Implementation Time:** 6-8 hours for complete ComfyUI compatibility  
**Most Critical:** optimized_attention and apply_rope (needed for basic functionality)
