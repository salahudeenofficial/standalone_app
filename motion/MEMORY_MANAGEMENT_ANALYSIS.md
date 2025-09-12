# Memory Management Analysis: WAN Standalone Pipeline

## Overview

The standalone pipeline implements a comprehensive memory management system designed to handle large-scale diffusion models efficiently. The system is built around the `ModelPatcher` class and supporting utilities, providing multiple layers of memory optimization.

## Core Components

### 1. ModelPatcher Memory Management (`standalone_model_patcher.py`)

The `ModelPatcher` class serves as the central hub for memory management with several key features:

#### Memory Tracking
```python
class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        # Core memory tracking
        self.size = size
        self.model_size()  # Calculate total model size
        
        # Device management
        self.load_device = load_device      # Target inference device (GPU)
        self.offload_device = offload_device # Storage device (CPU/disk)
```

**Key Functions:**
- `model_size()`: Calculates total model memory footprint
- `loaded_size()`: Tracks currently loaded memory usage
- `memory_required(input_shape)`: Estimates memory needed for inference

#### Low VRAM Operation System

The pipeline implements a sophisticated low-VRAM system for models that exceed available GPU memory:

```python
class LowVramPatch:
    """Memory-efficient weight modifications"""
    def __init__(self, key, patches):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        # Apply patches in memory-efficient manner
        # Uses intermediate dtype conversion to save memory
        intermediate_dtype = weight.dtype
        if intermediate_dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            intermediate_dtype = torch.float32
        return calculate_weight(self.patches[self.key], weight.to(intermediate_dtype), self.key)
```

**Low-VRAM Loading Strategy:**
1. **Partial Loading**: Only loads essential model parts to GPU
2. **On-Demand Weight Loading**: Weights loaded just before computation
3. **Immediate Offloading**: Weights moved back to CPU after use
4. **Memory Threshold Management**: Uses `lowvram_model_memory` parameter to control GPU usage

#### Memory Counter System

```python
class MemoryCounter:
    """Memory tracking for available GPU memory"""
    def __init__(self, initial: int, minimum=0):
        self.value = initial
        self.minimum = minimum

    def use(self, weight: torch.Tensor):
        weight_size = weight.nelement() * weight.element_size()
        if self.is_useable(weight_size):
            self.decrement(weight_size)
            return True
        return False
```

### 2. Device and Model Management (`wan_vae_components/model_management.py`)

Provides device-agnostic memory utilities:

#### Memory Information Gathering
```python
def get_device_memory_info(device: torch.device) -> dict:
    """Get comprehensive device memory information"""
    if device.type == 'cuda':
        return {
            'total': torch.cuda.get_device_properties(device.index).total_memory,
            'allocated': torch.cuda.memory_allocated(device.index),
            'cached': torch.cuda.memory_reserved(device.index)
        }
```

#### Memory Optimization Functions
- `get_free_memory(device)`: Real-time available memory calculation
- `empty_cache(device)`: Force GPU memory cleanup
- `cast_to_device()`: Efficient tensor device/dtype transfers
- `device_supports_non_blocking()`: Async transfer capability detection

### 3. Attention Memory Optimization (`wan_vae_components/modules/diffusionmodules/model.py`)

Implements memory-aware attention computation:

```python
def slice_attention(q, k, v):
    """Memory-efficient attention with adaptive slicing"""
    mem_free_total = get_free_memory(q.device)
    
    # Calculate memory requirements
    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    modifier = 3 if q.element_size() == 2 else 2.5
    mem_required = tensor_size * modifier
    
    # Adaptive slicing based on available memory
    steps = 1
    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    while True:
        try:
            # Process attention in memory-efficient slices
            slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
            for i in range(0, q.shape[1], slice_size):
                # Compute attention slice by slice
                # Immediate cleanup of intermediate tensors
                pass
        except OOM_EXCEPTION:
            soft_empty_cache(True)
            steps *= 2  # Increase slicing granularity
```

## Memory Management Strategies

### 1. Hierarchical Loading System

The pipeline uses a three-tier loading strategy:

**Tier 1: Full Load**
- All model weights on GPU
- Fastest inference
- Requires GPU memory ≥ model size

**Tier 2: Partial Load**
- Critical layers on GPU
- Non-critical layers on CPU with on-demand loading
- Balanced performance/memory trade-off

**Tier 3: Low-VRAM Mode**
- Minimal GPU footprint
- Most weights on CPU
- Slowest but most memory-efficient

### 2. Patch-Based Weight Management

The system implements sophisticated patching for LoRA and other weight modifications:

```python
def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    """Load model with memory optimization"""
    mem_counter = 0
    patch_counter = 0
    lowvram_counter = 0
    
    loading = self._load_list()  # Get modules sorted by size
    loading.sort(reverse=True)   # Load largest modules first
    
    for module_mem, name, module, params in loading:
        if mem_counter + module_mem >= lowvram_model_memory:
            # Use low-VRAM patches instead of full loading
            lowvram_weight = True
            if weight_key in self.patches:
                module.weight_function = [LowVramPatch(weight_key, self.patches)]
```

### 3. Memory Monitoring and Recovery

**Proactive Monitoring:**
- Real-time memory usage tracking
- Predictive memory requirement calculation
- Adaptive loading strategy adjustment

**Recovery Mechanisms:**
- Automatic cache clearing on OOM
- Gradual model unloading
- Emergency fallback to CPU computation

### 4. VAE Memory Optimization

The VAE implementation includes specialized memory management:

```python
class VAE:
    def __init__(self):
        # Memory usage calculators for different operations
        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * dtype_size(dtype)
        
        # Processing optimizations
        self.working_dtypes = [torch.bfloat16, torch.float32]
        self.disable_offload = False  # Allow offloading for memory efficiency
```

## Advanced Features

### 1. Partial Loading/Unloading

```python
def partially_unload(self, device_to, memory_to_free=0):
    """Intelligently unload model parts to free specific amount of memory"""
    memory_freed = 0
    unload_list = self._load_list()
    unload_list.sort()  # Unload smallest modules first
    
    for module_mem, name, module, params in unload_list:
        if memory_to_free < memory_freed:
            break
        # Selectively unload modules to reach target memory
```

### 2. Injection System

Provides temporary model modifications without permanent changes:

```python
class AutoPatcherEjector:
    """Context manager for temporary model modifications"""
    def __enter__(self):
        self.model.eject_model()
        return self
    
    def __exit__(self, *args):
        self.model.inject_model()
```

### 3. Weight Function System

Allows dynamic weight computation for memory efficiency:

```python
def move_weight_functions(module, device):
    """Move weight computation functions to appropriate device"""
    # Enables on-demand weight calculation instead of storage
```

## Performance Characteristics

### Memory Efficiency Metrics

1. **Base Model Storage**: CPU-based with selective GPU loading
2. **Patch Overhead**: Minimal - patches stored separately from base weights
3. **Inference Memory**: Dynamically allocated based on batch size and sequence length
4. **Memory Recovery**: Aggressive cleanup after each operation

### Scalability Features

1. **Large Model Support**: Handles models >50GB through chunked loading
2. **Multi-GPU Awareness**: Device-specific memory management
3. **Mixed Precision**: Automatic dtype optimization based on hardware
4. **Streaming Support**: On-demand loading for extremely large models

## Usage Patterns

### Standard Usage (GPU Available)
```python
model_patcher = create_model_patcher(model, load_device="cuda", offload_device="cpu")
model_patcher.load(device_to="cuda", lowvram_model_memory=8*1024**3)  # 8GB limit
```

### Low-VRAM Usage (Limited GPU Memory)
```python
model_patcher.load(device_to="cuda", lowvram_model_memory=2*1024**3)  # 2GB limit
# Automatically enables chunked loading and aggressive offloading
```

### Memory Monitoring
```python
print(f"Model size: {model_patcher.model_size() / (1024**3):.2f} GB")
print(f"Loaded size: {model_patcher.loaded_size() / (1024**3):.2f} GB")
memory_freed = model_patcher.partially_unload(device_to="cpu", memory_to_free=1024**3)
```

## Summary

The memory management system provides:

1. **Flexibility**: Adapts to available hardware resources
2. **Efficiency**: Minimal memory overhead with maximum utilization
3. **Reliability**: Robust error handling and recovery mechanisms
4. **Scalability**: Supports models from small to extremely large scales
5. **Transparency**: Clear monitoring and control interfaces

This implementation enables running large WAN models on consumer hardware while maintaining reasonable inference performance through intelligent memory management and optimization strategies.
