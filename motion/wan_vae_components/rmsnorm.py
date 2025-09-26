"""
Standalone RMSNorm Module
Replaces comfy.rmsnorm functionality
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Args:
        dim: Dimension of the input tensor
        eps: Small value to avoid division by zero
        elementwise_affine: Whether to use learnable parameters
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS over the channel dimension (dim=1 for 4D tensors, dim=-1 for others)
        if x.dim() == 4:  # 4D tensor (B, C, H, W)
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        else:  # Other dimensions, normalize over last dimension
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x_normalized = x / rms
        
        # Apply learnable scale if enabled
        if self.elementwise_affine:
            # Reshape weight to match input dimensions
            if x.dim() == 4:  # 4D tensor (B, C, H, W)
                weight_shape = [1, self.dim, 1, 1]
            else:  # Other dimensions
                weight_shape = [1] * (x.dim() - 1) + [self.dim]
            weight = self.weight.view(weight_shape)
            x_normalized = x_normalized * weight
        
        return x_normalized


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
