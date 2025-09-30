"""
Standalone Ops Module
Replaces comfy.ops functionality without external dependencies
"""

import torch
import torch.nn as nn
import contextlib
from typing import Optional, List, Callable
from .model_management import cast_to, get_offload_stream, device_supports_non_blocking, sync_stream, supports_fp8_compute
from .float import stochastic_rounding
from .rmsnorm import RMSNorm


class CastWeightBiasOp:
    """Base class for operations that cast weights and biases."""
    comfy_cast_weights = False
    weight_function: List[Callable] = []
    bias_function: List[Callable] = []


def cast_to_input(weight: torch.Tensor, input_tensor: torch.Tensor, 
                  non_blocking: bool = False, copy: bool = True) -> torch.Tensor:
    """
    Cast weight to match input tensor's dtype and device.
    
    Args:
        weight: Weight tensor
        input_tensor: Input tensor to match
        non_blocking: Whether to use non-blocking transfer
        copy: Whether to copy the tensor
        
    Returns:
        Casted weight tensor
    """
    return cast_to(weight, input_tensor.dtype, input_tensor.device, 
                   non_blocking=non_blocking, copy=copy)


def cast_bias_weight(s: nn.Module, input_tensor: Optional[torch.Tensor] = None, 
                    dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, 
                    bias_dtype: Optional[torch.dtype] = None) -> tuple:
    """
    Cast bias and weight tensors.
    
    Args:
        s: Module containing weight and bias
        input_tensor: Input tensor for reference
        dtype: Target dtype for weight
        device: Target device
        bias_dtype: Target dtype for bias
        
    Returns:
        Tuple of (weight, bias) tensors
    """
    if input_tensor is not None:
        if dtype is None:
            dtype = input_tensor.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input_tensor.device

    offload_stream = get_offload_stream(device)
    if offload_stream is not None:
        wf_context = offload_stream
    else:
        wf_context = contextlib.nullcontext()

    bias = None
    non_blocking = device_supports_non_blocking(device)
    
    if hasattr(s, 'bias') and s.bias is not None:
        has_function = len(s.bias_function) > 0
        bias = cast_to(s.bias, bias_dtype, device, non_blocking=non_blocking, 
                      copy=has_function, stream=offload_stream)

        if has_function:
            with wf_context:
                for f in s.bias_function:
                    bias = f(bias)

    has_function = len(s.weight_function) > 0
    weight = cast_to(s.weight, dtype, device, non_blocking=non_blocking, 
                    copy=has_function, stream=offload_stream)
    if has_function:
        with wf_context:
            for f in s.weight_function:
                weight = f(weight)

    sync_stream(device, offload_stream)
    return weight, bias


class disable_weight_init:
    """Disable weight initialization for layers."""
    
    class Linear(torch.nn.Linear, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_function = []
            self.bias_function = []

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_function = []
            self.bias_function = []

    class RMSNorm(RMSNorm, CastWeightBiasOp):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True):
            super().__init__(dim, eps, elementwise_affine)
            self.weight_function = []
            self.bias_function = []

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_function = []
            self.bias_function = []

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_function = []
            self.bias_function = []

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_function = []
            self.bias_function = []


class manual_cast(disable_weight_init):
    """Manual casting operations."""
    
    class Linear(disable_weight_init.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Conv1d(disable_weight_init.Conv1d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Conv2d(disable_weight_init.Conv2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Conv3d(disable_weight_init.Conv3d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class GroupNorm(disable_weight_init.GroupNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class LayerNorm(disable_weight_init.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class RMSNorm(disable_weight_init.RMSNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Embedding(disable_weight_init.Embedding):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)


class fp8_ops(manual_cast):
    """FP8 operations."""
    
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *args, **kwargs):
            if hasattr(self, 'weight') and self.weight is not None:
                # Apply stochastic rounding for FP8
                seed = torch.randint(0, 2**32, (1,)).item()
                weight = stochastic_rounding(
                    self.weight / self.scale_weight.to(device=self.weight.device, dtype=self.weight.dtype),
                    self.weight.dtype,
                    seed=seed
                )
                self.weight.data = weight
            return super().forward(*args, **kwargs)

    class scaled_fp8_op(manual_cast):
        class Linear(manual_cast.Linear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.scale_weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, *args, **kwargs):
                if hasattr(self, 'weight') and self.weight is not None:
                    seed = torch.randint(0, 2**32, (1,)).item()
                    weight = stochastic_rounding(
                        self.weight / self.scale_weight.to(device=self.weight.device, dtype=self.weight.dtype),
                        self.weight.dtype,
                        seed=seed
                    )
                    self.weight.data = weight
                return super().forward(*args, **kwargs)


def pick_operations(weight_dtype, compute_dtype, load_device=None, 
                   disable_fast_fp8=False, fp8_optimizations=False, scaled_fp8=None):
    """
    Pick appropriate operations based on device capabilities.
    
    Args:
        weight_dtype: Weight data type
        compute_dtype: Compute data type
        load_device: Device to load on
        disable_fast_fp8: Whether to disable FP8
        fp8_optimizations: Whether to enable FP8 optimizations
        scaled_fp8: Scaled FP8 configuration
        
    Returns:
        Appropriate operations class
    """
    fp8_compute = supports_fp8_compute(load_device)
    
    if scaled_fp8 is not None:
        return fp8_ops.scaled_fp8_op(fp8_matrix_mult=fp8_compute and fp8_optimizations, 
                                     scale_input=fp8_optimizations, override_dtype=scaled_fp8)

    if fp8_compute and fp8_optimizations and not disable_fast_fp8:
        return fp8_ops

    return manual_cast
