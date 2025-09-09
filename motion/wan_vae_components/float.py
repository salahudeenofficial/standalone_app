"""
Standalone Float Module
Replaces comfy.float functionality
"""

import torch
import random
from typing import Optional


def stochastic_rounding(tensor: torch.Tensor, target_dtype: torch.dtype, seed: Optional[int] = None) -> torch.Tensor:
    """
    Apply stochastic rounding to tensor.
    
    Args:
        tensor: Input tensor
        target_dtype: Target dtype for rounding
        seed: Random seed for reproducibility
        
    Returns:
        Stochastically rounded tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Convert to float32 for rounding calculations
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # Get integer and fractional parts
    integer_part = torch.floor(tensor)
    fractional_part = tensor - integer_part
    
    # Generate random values for stochastic rounding
    random_values = torch.rand_like(fractional_part)
    
    # Stochastic rounding: round up if random value < fractional part
    rounded = integer_part + (random_values < fractional_part).float()
    
    # Convert to target dtype
    return rounded.to(target_dtype)


def quantize_to_fp8(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Quantize tensor to FP8 format.
    
    Args:
        tensor: Input tensor
        scale: Scaling factor
        
    Returns:
        FP8 quantized tensor
    """
    # Simple FP8 quantization (E4M3 format)
    scaled = tensor * scale
    
    # Clamp to FP8 range
    fp8_min = -448.0
    fp8_max = 448.0
    clamped = torch.clamp(scaled, fp8_min, fp8_max)
    
    # Round to nearest integer
    rounded = torch.round(clamped)
    
    return rounded / scale


def dequantize_from_fp8(tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Dequantize tensor from FP8 format.
    
    Args:
        tensor: FP8 quantized tensor
        scale: Scaling factor used during quantization
        
    Returns:
        Dequantized tensor
    """
    return tensor * scale


def get_fp8_scale(tensor: torch.Tensor) -> float:
    """
    Calculate optimal scale for FP8 quantization.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Optimal scale factor
    """
    # Calculate scale to fit tensor values in FP8 range
    max_val = torch.max(torch.abs(tensor)).item()
    if max_val == 0:
        return 1.0
    
    fp8_max = 448.0
    scale = fp8_max / max_val
    
    return scale
