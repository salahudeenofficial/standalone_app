"""
Standalone RMSNorm Module
Replaces comfy.rmsnorm functionality - borrowed from ComfyUI
"""

import torch
import torch.nn as nn
import math
import numbers
from typing import Optional

# Try to use PyTorch's native RMSNorm if available (borrowed from ComfyUI)
RMSNorm = None
try:
    rms_norm_torch = torch.nn.functional.rms_norm
    RMSNorm = torch.nn.RMSNorm
except:
    rms_norm_torch = None

def rms_norm(x, weight=None, eps=1e-6):
    """RMS normalization function (borrowed from ComfyUI)."""
    if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if weight is None:
            return rms_norm_torch(x, (x.shape[-1],), eps=eps)
        else:
            # Use our local cast_to function instead of comfy.model_management.cast_to
            from .model_management import cast_to
            return rms_norm_torch(x, weight.shape, weight=cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
    else:
        r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        else:
            from .model_management import cast_to
            return r * cast_to(weight, dtype=x.dtype, device=x.device)

if RMSNorm is None:
    class RMSNorm(torch.nn.Module):
        """RMSNorm implementation (borrowed from ComfyUI)."""
        def __init__(
            self,
            normalized_shape,
            eps=1e-6,
            elementwise_affine=True,
            device=None,
            dtype=None,
        ):
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = torch.nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("weight", None)
            self.bias = None

        def forward(self, x):
            return rms_norm(x, self.weight, self.eps)


class RMSNorm2D(nn.Module):
    """
    2D RMS Normalization for image tensors.
    
    Args:
        dim: Number of channels
        eps: Small value to avoid division by zero
        elementwise_affine: Whether to use learnable parameters
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 2D RMSNorm.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS over spatial dimensions
        rms = torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + self.eps)
        
        # Normalize
        x_normalized = x / rms
        
        # Apply learnable scale if enabled
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
        
        return x_normalized


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Functional RMS normalization.
    
    Args:
        x: Input tensor
        weight: Scale parameter
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


def rms_norm_2d(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Functional 2D RMS normalization.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        weight: Scale parameter of shape (C, 1, 1)
        eps: Small value to avoid division by zero
        
    Returns:
        Normalized tensor
    """
    rms = torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + eps)
    return (x / rms) * weight
