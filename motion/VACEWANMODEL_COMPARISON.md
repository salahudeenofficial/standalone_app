# 🔍 **VaceWanModel: Motion vs ComfyUI Deep Comparison**

## 📋 **Critical Architectural Differences**

### **🏗️ Motion Pipeline VaceWanModel**
```python
# motion/models.py - Lines 407-515
class VaceWanModel(nn.Module):
    def __init__(self, model_type='vace', patch_size=(1, 1, 1), in_dim=16, 
                 dim=2048, ffn_dim=8192, out_dim=16, num_heads=16, 
                 num_layers=32, device=None):
        
        # BASIC ARCHITECTURE ONLY
        self.patch_embedding = nn.Conv3d(...)
        self.blocks = [-generic transformer blocks-]
        self.head = nn.Linear(self.dim, self.out_dim * 4)
        
        # SIMPLIFIED VACE COMPONENTS
        self.vace_patch_embedding = nn.Conv3d(...)  # Duplicate of patch_embedding!
        self.vace_blocks = [generic attention blocks]  # NOT VaceWanAttentionBlock!
```

### **🏗️ ComfyUI VaceWanModel**  
```python
# comfy/ldm/wan/model.py - Lines 626-740
class VaceWanModel(WanModel):  # Inherits from WanModel!
    def __init__(self, model_type='vace', patch_size=(1, 2, 2), text_len=512,
                 in_dim=16, dim=2048, ffn_dim=8192, freq_dim=256, text_dim=4096,
                 out_dim=16, num_heads=16, num_layers=32, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=True, eps=1e-6,
                 vace_layers=None, vace_in_dim=None, operations=None):
        
        super().__init__(model_type='t2v', ...)  # Full WanModel!
        
        # PROFESSIONAL VACE ARCHITECTURE
        if vace_layers is not None:
            self.vace_layers = vace_layers
            self.vace_in_dim = vace_in_dim
            
            # REAL VACE BLOCKS
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, 
                                    self.num_heads, self.window_size, 
                                    self.qk_norm, self.cross_attn_norm, self.eps, 
                                    block_id=i, operation_settings=operation_settings)
                for i in range(self.vace_layers)
            ])
            
            # VACE LAYERS MAPPING
            self.vace_layers_mapping = {
                i: n for n, i in enumerate(
                    range(0, self.num_layers, self.num_layers // self.vace_layers)
                )
            }
            
            # OPERATIONS-BASED CONV3D
            self.vace_patch_embedding = operations.Conv3d(
                self.vace_in_dim, self.dim, 
                kernel_size=self.patch_size, stride=self.patch_size, 
                device=device, dtype=torch.float32
            )
```

---

## 🚨 **Critical Missing Components in Motion**

### **1. ❌ Missing Inheritance from WanModel**
**Motion**: Standalone class
```python
class VaceWanModel(nn.Module):  # ❌ Standalone
```

**ComfyUI**: Proper inheritance
```python
class VaceWanModel(WanModel):  # ✅ Inherits full WAN architecture
    super().__init__(model_type='t2v', ...)  # ✅ Gets complete WanModel
```

### **2. ❌ Missing Real VaceWanAttentionBlock**
**Motion**: Generic attention blocks
```python
def _create_vace_block(self):
    return nn.ModuleDict({
        'norm1': nn.LayerNorm(self.dim),
        'vace_attn': nn.MultiheadAttention(...),  # ❌ Generic MultiheadAttention!
        'norm2': nn.LayerNorm(self.dim),
        'mlp': nn.Sequential(...)
    })
```

**ComfyUI**: Specialized VaceWanAttentionBlock
```python
self.vace_blocks = nn.ModuleList([
    VaceWanAttentionBlock('t2v_cross_attn', self.dim, self.ffn_dim, 
                         self.num_heads, self.window_size, 
                         self.qk_norm, self.cross_attn_norm, self.eps, 
                         block_id=i, operation_settings=operation_settings)
    for i in range(self.vace_layers)
])
```

### **3. ❌ Missing Operations System**
**Motion**: Direct PyTorch operations
```python
self.vace_patch_embedding = nn.Conv3d(...)  # ❌ Direct PyTorch
```

**ComfyUI**: Operations-based with optimizations
```python
self.vace_patch_embedding = operations.Conv3d(...)  # ✅ Operations system
```

### **4. ❌ Missing VACE Layers Mapping**
**Motion**: None!

**ComfyUI**: Sophisticated mapping
```python
self.vace_layers_mapping = {
    i: n for n, i in enumerate(
        range(0, self.num_layers, self.num_layers // self.vace_layers)
    )
}
```

### **5. ❌ Missing Sophisticated Forward**
**Motion**: Simplified forward
```python
def forward(self, x, t=None, context=None, vace_context=None, vace_strength=None, **kwargs):
    # ❌ SIMPLIFIED PLACEHOLDER IMPLEMENTATION
    x = self.patch_embedding(x)
    for block in self.blocks:
        # Generic transformer processing
    return x
```

**ComfyUI**: Complete VACE forward
```python
def forward_orig(self, x, t, context, vace_context, vace_strength, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
    # ✅ COMPLETE VACE PROCESSING
    
    # 1. Patch embedding with operations
    x = self.patch_embedding(x.float()).to(x.dtype)
    
    # 2. Time embeddings with sinusoidal embedding
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    
    # 3. Text context processing
    context = self.text_embedding(context)
    
    # 4. VACE CONTEXT PROCESSING ❗
    orig_shape = list(vace_context.shape)
    vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
    c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
    c = c.flatten(2).transpose(1, 2)
    c = list(c.split(orig_shape[0], dim=0))
    
    # 5. Main transformer blocks with VACE integration
    for i, block in enumerate(self.blocks):
        x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
        
        # 6. VACE BLOCK INTEGRATION ❗
        ii = self.vace_layers_mapping.get(i, None)
        if ii is not None:
            for iii in range(len(c)):
                c_skip, c[iii] = self.vace_blocks[ii](
                    c[iii], x=x_orig, e=e0, freqs=freqs, 
                    context=context, context_img_len=context_img_len
                )
                x += c_skip * vace_strength[iii]  # VACE STRENGTH APPLIED!
            del c_skip
    
    # 7. Head and unpatchify
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)
    return x
```

---

## 🔬 **VaceWanAttentionBlock Analysis**

### **ComfyUI's VaceWanAttentionBlock**
```python
class VaceWanAttentionBlock(WanAttentionBlock):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, 
                 window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, 
                 eps=1e-6, block_id=0, operation_settings={}):
        
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, 
                        window_size, qk_norm, cross_attn_norm, eps, 
                        operation_settings=operation_settings)
        
        self.block_id = block_id
        
        # VACE-SPECIFIC PROJECTIONS ❗
        if block_id == 0:
            self.before_proj = operations.Linear(self.dim, self.dim)
        self.after_proj = operations.Linear(self.dim, self.dim)
    
    def forward(self, c, x, **kwargs):
        # VACE-SPECIFIC PROCESSING ❗
        if self.block_id == 0:
            c = self.before_proj(c) + x  # Integrate context
        c = super().forward(c, **kwargs)  # Full attention processing
        c_skip = self.after_proj(c)      # Project for skip connection
        return c_skip, c  # Return skip connection + output
```

**Key Features Missing in Motion:**
- ❌ **No Projection Layers**: Missing `before_proj` and `after_proj`
- ❌ **No Skip Connection Logic**: Missing `c_skip` mechanism
- ❌ **Generic Attention**: Using `MultiheadAttention` instead of specialized attention
- ❌ **No Block ID Logic**: Missing first-block special handling

---

## 🎯 **Parameter Differences**

| **Parameter** | **Motion** | **ComfyUI** | **Impact** |
|---------------|------------|-------------|------------|
| `patch_size` | `(1, 1, 1)` ❌ | `(1, 2, 2)` ✅ | **Critical** - Wrong patch size |
| `text_len` | Missing ❌ | `512` ✅ | **Critical** - Missing text handling |
| `freq_dim` | Missing ❌ | `256` ✅ | **Critical** - Missing time embedding |
| `text_dim` | Missing ❌ | `4096` ✅ | **Critical** - Missing text embedding |
| `window_size` | Missing ❌ | `(-1, -1)` ✅ | **Important** - Missing attention windows |
| `qk_norm` | Missing ❌ | `True` ✅ | **Important** - Missing query/key normalization |
| `cross_attn_norm` | Missing ❌ | `True` ✅ | **Important** - Missing cross attention normalization |
| `eps` | Missing ❌ | `1e-6` ✅ | **Minor** - Missing epsilon values |
| `operations` | Missing ❌ | Complex ✅ | **Critical** - Missing operations system |
| `vace_layers` | Fixed `4` ❌ | Configurable ✅ | **Critical** - Missing VACE layer flexibility |
| `vace_in_dim` | Same as `in_dim` ❌ | Separate ✅ | **Critical** - Wrong VACE input dimensions |

---

## 🚨 **Critical Issues Summary**

### **🏗️ Architecture Issues**
1. **Missing WanModel inheritance** - No base WAN architecture
2. **Missing VaceWanAttentionBlock** - Generic attention instead of VACE-specific
3. **Missing operations system** - No optimizations or FP8 support
4. **Missing VACE layers mapping** - No sophisticated layer routing
5. **Missing time/text embeddings** - Core components absent

### **⚙️ Implementation Issues**
6. **Wrong patch size** - `(1,1,1)` vs `(1,2,2)`
7. **Missing parameters** - No freq_dim, text_dim, window_size, etc.
8. **Simplified forward** - Placeholder vs complete VACE processing
9. **Missing VACE integration** - No VACE strength application
10. **Wrong VACE input dim** - Should be separate from main model

### **🎯 Fix Priority**
1. **MOST CRITICAL**: Inherit from WanModel instead of nn.Module
2. **CRITICAL**: Implement real VaceWanAttentionBlock with projections
3. **CRITICAL**: Implement complete VACE forward with context processing
4. **IMPORTANT**: Add operations system
5. **IMPORTANT**: Fix parameter mismatches (patch_size, etc.)

---

## 💡 **Recommendation**

**The motion VaceWanModel is essentially a placeholder** - it lacks the sophisticated VACE architecture that ComfyUI implements. To fix this, motion needs to:

1. **Copy ComfyUI's WanModel** as base class
2. **Implement real VaceWanAttentionBlock** with projections
3. **Implement complete VACE forward** with time/text/VACE processing
4. **Add operations system** for optimizations
5. **Fix all parameter mismatches**

Without these fixes, the VACE model will not produce correct diffusion results! 🚨
