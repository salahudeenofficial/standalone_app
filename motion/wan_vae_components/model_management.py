"""
Standalone Model Management Module
Replaces comfy.model_management functionality
"""

import torch
import contextlib
from typing import Optional, Union


def cast_to(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device, 
            non_blocking: bool = False, copy: bool = True, stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
    """
    Cast tensor to specified dtype and device.
    
    Args:
        tensor: Input tensor
        dtype: Target dtype
        device: Target device
        non_blocking: Whether to use non-blocking transfer
        copy: Whether to copy the tensor
        stream: CUDA stream for async operations
    
    Returns:
        Casted tensor
    """
    if stream is not None and device.type == 'cuda':
        with torch.cuda.stream(stream):
            return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)
    else:
        return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


def get_offload_stream(device: torch.device) -> Optional[torch.cuda.Stream]:
    """
    Get CUDA stream for offloading operations.
    
    Args:
        device: Target device
        
    Returns:
        CUDA stream if device is CUDA, None otherwise
    """
    if device.type == 'cuda':
        return torch.cuda.Stream(device=device)
    return None


def device_supports_non_blocking(device: torch.device) -> bool:
    """
    Check if device supports non-blocking transfers.
    
    Args:
        device: Target device
        
    Returns:
        True if device supports non-blocking transfers
    """
    if device.type == 'cuda':
        return True
    elif device.type == 'cpu':
        return False
    else:
        # For other devices, assume they don't support non-blocking
        return False


def sync_stream(device: torch.device, stream: Optional[torch.cuda.Stream]) -> None:
    """
    Synchronize CUDA stream.
    
    Args:
        device: Target device
        stream: CUDA stream to synchronize
    """
    if stream is not None and device.type == 'cuda':
        stream.synchronize()


def supports_fp8_compute(device: torch.device) -> bool:
    """
    Check if device supports FP8 compute.
    
    Args:
        device: Target device
        
    Returns:
        True if device supports FP8 compute
    """
    if device.type == 'cuda':
        # Check if CUDA device supports FP8 (requires compute capability 8.9+)
        try:
            capability = torch.cuda.get_device_capability(device.index)
            return capability[0] > 8 or (capability[0] == 8 and capability[1] >= 9)
        except:
            return False
    return False


# Additional utility functions that might be needed
def get_device_memory_info(device: torch.device) -> dict:
    """
    Get device memory information.
    
    Args:
        device: Target device
        
    Returns:
        Dictionary with memory info
    """
    if device.type == 'cuda':
        return {
            'total': torch.cuda.get_device_properties(device.index).total_memory,
            'allocated': torch.cuda.memory_allocated(device.index),
            'cached': torch.cuda.memory_reserved(device.index)
        }
    else:
        return {'total': 0, 'allocated': 0, 'cached': 0}


def empty_cache(device: torch.device) -> None:
    """
    Empty device cache.
    
    Args:
        device: Target device
    """
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# Additional functions needed by model.py
def xformers_enabled_vae() -> bool:
    """Check if xformers is enabled for VAE."""
    try:
        import xformers
        return True
    except ImportError:
        return False


def pytorch_attention_enabled_vae() -> bool:
    """Check if PyTorch attention is enabled for VAE."""
    return True  # PyTorch attention is always available


def get_free_memory(device: torch.device) -> int:
    """Get free memory on device."""
    if device.type == 'cuda':
        return torch.cuda.get_device_properties(device.index).total_memory - torch.cuda.memory_allocated(device.index)
    else:
        return 1024 * 1024 * 1024  # Return 1GB for non-CUDA devices


class OOM_EXCEPTION(Exception):
    """Out of Memory exception."""
    pass


def soft_empty_cache(force: bool = False) -> None:
    """Soft empty cache."""
    if force:
        empty_cache(torch.device('cuda'))
