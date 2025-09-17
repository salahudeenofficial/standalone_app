"""
ComfyUI-style operations for dynamic weight loading
Based on ComfyUI's ops.py but simplified for our standalone use case.
"""

import torch
import torch.nn.functional as F
import logging

def cast_bias_weight(module, input_tensor):
    """
    ComfyUI's cast_bias_weight function - loads weights on-demand via weight_function
    """
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Handle bias
    bias = None
    if hasattr(module, 'bias') and module.bias is not None:
        bias = module.bias
        if hasattr(module, 'bias_function') and len(module.bias_function) > 0:
            # Apply bias functions (LowVramPatch calls)
            for f in module.bias_function:
                bias = f(bias)
    
    # Handle weight
    weight = module.weight
    if hasattr(module, 'weight_function') and len(module.weight_function) > 0:
        # Apply weight functions (LowVramPatch calls) - this loads weights to GPU!
        for f in module.weight_function:
            weight = f(weight)
    
    # Cast to appropriate device and dtype
    weight = weight.to(device=device, dtype=dtype)
    if bias is not None:
        bias = bias.to(device=device, dtype=dtype)
    
    return weight, bias

class ComfyUILinear(torch.nn.Linear):
    """
    ComfyUI-style Linear layer with weight_function support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_function = []
        self.bias_function = []
    
    def forward(self, input):
        if len(self.weight_function) > 0 or len(self.bias_function) > 0:
            # Use ComfyUI's dynamic loading approach
            weight, bias = cast_bias_weight(self, input)
            return F.linear(input, weight, bias)
        else:
            # Standard forward pass
            return super().forward(input)

class ComfyUIConv2d(torch.nn.Conv2d):
    """
    ComfyUI-style Conv2d layer with weight_function support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_function = []
        self.bias_function = []
    
    def forward(self, input):
        if len(self.weight_function) > 0 or len(self.bias_function) > 0:
            # Use ComfyUI's dynamic loading approach
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        else:
            # Standard forward pass
            return super().forward(input)

class ComfyUIConv3d(torch.nn.Conv3d):
    """
    ComfyUI-style Conv3d layer with weight_function support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_function = []
        self.bias_function = []
    
    def forward(self, input):
        if len(self.weight_function) > 0 or len(self.bias_function) > 0:
            # Use ComfyUI's dynamic loading approach
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)
        else:
            # Standard forward pass
            return super().forward(input)

class ComfyUIGroupNorm(torch.nn.GroupNorm):
    """
    ComfyUI-style GroupNorm layer with weight_function support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_function = []
        self.bias_function = []
    
    def forward(self, input):
        if len(self.weight_function) > 0 or len(self.bias_function) > 0:
            # Use ComfyUI's dynamic loading approach
            weight, bias = cast_bias_weight(self, input)
            return F.group_norm(input, self.num_groups, weight, bias, self.eps)
        else:
            # Standard forward pass
            return super().forward(input)

class ComfyUILayerNorm(torch.nn.LayerNorm):
    """
    ComfyUI-style LayerNorm layer with weight_function support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_function = []
        self.bias_function = []
    
    def forward(self, input):
        if len(self.weight_function) > 0 or len(self.bias_function) > 0:
            # Use ComfyUI's dynamic loading approach
            weight, bias = cast_bias_weight(self, input)
            return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
        else:
            # Standard forward pass
            return super().forward(input)

def patch_model_with_comfyui_ops(model):
    """
    Patch a model to use ComfyUI-style operations for dynamic weight loading
    """
    logging.info("🔧 Patching model with ComfyUI-style operations...")
    
    patched_count = 0
    
    for name, module in model.named_modules():
        # Patch Linear layers
        if isinstance(module, torch.nn.Linear):
            # Create new ComfyUI-style Linear layer
            new_module = ComfyUILinear(
                module.in_features,
                module.out_features,
                module.bias is not None
            )
            # Copy weights and bias
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            
            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                attr_name = name.split('.')[-1]
                setattr(parent_module, attr_name, new_module)
            else:
                # Root module
                model = new_module
            
            patched_count += 1
            logging.debug(f"  ✅ Patched Linear layer: {name}")
        
        # Patch Conv2d layers
        elif isinstance(module, torch.nn.Conv2d):
            new_module = ComfyUIConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                attr_name = name.split('.')[-1]
                setattr(parent_module, attr_name, new_module)
            else:
                model = new_module
            
            patched_count += 1
            logging.debug(f"  ✅ Patched Conv2d layer: {name}")
        
        # Patch Conv3d layers
        elif isinstance(module, torch.nn.Conv3d):
            new_module = ComfyUIConv3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                module.padding_mode
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                attr_name = name.split('.')[-1]
                setattr(parent_module, attr_name, new_module)
            else:
                model = new_module
            
            patched_count += 1
            logging.debug(f"  ✅ Patched Conv3d layer: {name}")
        
        # Patch GroupNorm layers
        elif isinstance(module, torch.nn.GroupNorm):
            new_module = ComfyUIGroupNorm(
                module.num_groups,
                module.num_channels,
                module.eps,
                module.affine
            )
            if module.weight is not None:
                new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                attr_name = name.split('.')[-1]
                setattr(parent_module, attr_name, new_module)
            else:
                model = new_module
            
            patched_count += 1
            logging.debug(f"  ✅ Patched GroupNorm layer: {name}")
        
        # Patch LayerNorm layers
        elif isinstance(module, torch.nn.LayerNorm):
            new_module = ComfyUILayerNorm(
                module.normalized_shape,
                module.eps,
                module.elementwise_affine
            )
            if module.weight is not None:
                new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
                attr_name = name.split('.')[-1]
                setattr(parent_module, attr_name, new_module)
            else:
                model = new_module
            
            patched_count += 1
            logging.debug(f"  ✅ Patched LayerNorm layer: {name}")
    
    logging.info(f"✅ Model patching complete: {patched_count} layers patched")
    return model
