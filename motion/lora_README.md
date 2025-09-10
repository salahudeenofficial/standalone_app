# Standalone LoRA Implementation - WAN Focused

## Overview

This is a complete standalone implementation of ComfyUI's LoRA (Low-Rank Adaptation) functionality, specifically optimized for WAN (Wan 2.1 VACE) models. It provides all the essential LoRA features without requiring ComfyUI dependencies.

## Features

### ✅ Implemented Features

1. **Basic LoRA Support**
   - Standard LoRA format (`lora_up.weight`, `lora_down.weight`)
   - Diffusers format (`_lora.up.weight`, `_lora.down.weight`)
   - Diffusers2 format (`lora_B.weight`, `lora_A.weight`)
   - Multiple other formats (Mochi, Transformers, Qwen)

2. **Advanced LoRA Features**
   - **DoRA (Weight-Decomposed Low-Rank Adaptation)** - Complete implementation
   - **LoCon (LoRA + Convolution)** - Mid-weight support for convolutional layers
   - **Alpha scaling** - Proper alpha parameter handling
   - **Reshape support** - Tensor reshaping for different architectures

3. **Patch Types**
   - `diff` - Direct weight differences
   - `set` - Direct weight replacement
   - `model_as_lora` - Model-based LoRA patches
   - Weight normalization (`w_norm`, `b_norm`)

4. **Device and Memory Management**
   - Proper device casting with `cast_to_device`
   - Intermediate dtype handling
   - Memory usage calculation
   - CUDA stream support

5. **WAN-Specific Optimizations**
   - WAN model key mappings
   - WAN Fun LoRA format conversion
   - T5-XXL CLIP key mappings
   - Optimized for WAN2.1 VACE architecture

6. **Utility Functions**
   - `weight_decompose` - DoRA implementation
   - `pad_tensor_to_shape` - Tensor reshaping
   - `tucker_weight_from_conv` - LoCon mid-weight handling
   - Multiple conversion functions

## Usage

### Basic Usage

```python
from lora import load_lora_for_models, load_lora_from_file
from standalone_sd import load_state_dict_guess_config

# Load base model
model, clip, _, _ = load_state_dict_guess_config(
    unet_state_dict,
    output_vae=False,
    output_clip=True,
    output_model=True
)

# Load LoRA from file
new_model, new_clip = load_lora_from_file(
    'path/to/lora.safetensors',
    model, clip,
    strength_model=1.0,
    strength_clip=1.0
)

# Or load LoRA from state dict
lora_sd = load_torch_file('path/to/lora.safetensors')
new_model, new_clip = load_lora_for_models(
    model, clip, lora_sd,
    strength_model=1.0,
    strength_clip=1.0
)
```

### Advanced Usage

```python
from lora import (
    LoRAAdapter, weight_decompose, pad_tensor_to_shape,
    model_lora_keys_unet, model_lora_keys_clip,
    convert_lora_wan, convert_lora_bfl_control
)

# Create custom LoRA adapter
adapter = LoRAAdapter(set(), (
    torch.randn(64, 16),  # lora_up
    torch.randn(16, 32),   # lora_down
    1.0,                  # alpha
    None,                  # mid (for LoCon)
    None,                  # dora_scale (for DoRA)
    None                   # reshape
))

# Generate key mappings
key_map = {}
key_map = model_lora_keys_unet(model.model, key_map)
key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

# Convert LoRA formats
wan_lora = convert_lora_wan(lora_state_dict)
bfl_lora = convert_lora_bfl_control(lora_state_dict)
```

## Supported LoRA Formats

### Standard Formats
- **ComfyUI**: `layer.lora_up.weight`, `layer.lora_down.weight`
- **Diffusers**: `layer_lora.up.weight`, `layer_lora.down.weight`
- **Diffusers2**: `layer.lora_B.weight`, `layer.lora_A.weight`
- **Mochi**: `layer.lora_B`, `layer.lora_A`
- **Transformers**: `layer.lora_linear_layer.up.weight`
- **Qwen**: `layer.lora_B.default.weight`

### Advanced Formats
- **DoRA**: Includes `dora_scale` parameter
- **LoCon**: Includes `lora_mid.weight` for convolutional layers
- **Reshape**: Includes `reshape_weight` for tensor reshaping

## Integration with Standalone Components

The LoRA implementation integrates seamlessly with:

- `standalone_sd.py` - Model loading
- `standalone_model_patcher.py` - Model patching
- `wan_vae_components/model_management.py` - Device management
- `utils.py` - Utility functions

## Dependencies Resolved

All ComfyUI dependencies have been replaced with standalone implementations:

- `comfy.model_management` → `wan_vae_components.model_management`
- `comfy.utils` → `utils.py`
- `comfy.weight_adapter` → Built-in adapter classes
- `comfy.lora` → Complete standalone implementation

## Testing

Run the comprehensive test suite:

```bash
python3 test_lora_integration.py
```

This tests:
- Individual LoRA features
- Integration with standalone components
- Multiple LoRA formats
- WAN model compatibility
- DoRA and LoCon support

## Performance

The implementation is optimized for:
- **Memory efficiency** - Proper device management
- **Speed** - Optimized tensor operations
- **Compatibility** - Multiple format support
- **WAN models** - Specific optimizations

## Limitations

Currently focused on WAN models. For full ComfyUI compatibility, additional adapter types would need to be implemented:
- LoHa (Low-Rank Hadamard)
- LoKr (Low-Rank Kronecker)  
- OFT (Orthogonal Fine-Tuning)
- BOFT (Block-wise OFT)
- GLoRA (Generalized LoRA)

## Future Enhancements

1. **Additional Adapter Types** - Implement remaining ComfyUI adapters
2. **Training Support** - Add training mode functionality
3. **Hook System** - Implement weight hooks for advanced use cases
4. **Memory Optimization** - Add low VRAM patches
5. **Batch Processing** - Support for multiple LoRA files

## Conclusion

This standalone LoRA implementation provides complete functionality for WAN models with:
- ✅ Full LoRA support
- ✅ DoRA and LoCon support  
- ✅ Multiple format compatibility
- ✅ WAN-specific optimizations
- ✅ No ComfyUI dependencies
- ✅ Production-ready code

The implementation is ready for integration into production pipelines and provides a solid foundation for further enhancements.
