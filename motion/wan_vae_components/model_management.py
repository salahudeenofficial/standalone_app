import logging
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


# Global variables for xformers management (borrowed from ComfyUI)
XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
XFORMERS_IS_AVAILABLE = False

# Initialize xformers availability (borrowed from ComfyUI logic)
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    try:
        XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
    except:
        pass
    try:
        XFORMERS_VERSION = xformers.version.__version__
        logging.info("xformers version: {}".format(XFORMERS_VERSION))
        if XFORMERS_VERSION.startswith("0.0.18"):
            logging.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
            logging.warning("Please downgrade or upgrade xformers to a different version.\n")
            XFORMERS_ENABLED_VAE = False
    except:
        pass
except:
    XFORMERS_IS_AVAILABLE = False

def xformers_enabled() -> bool:
    """Check if xformers is enabled (borrowed from ComfyUI)."""
    return XFORMERS_IS_AVAILABLE

def xformers_enabled_vae() -> bool:
    """Check if xformers is enabled for VAE (borrowed from ComfyUI)."""
    enabled = xformers_enabled()
    if not enabled:
        return False
    return XFORMERS_ENABLED_VAE


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


# Additional functions needed for standalone_sd.py
def get_torch_device():
    """Get the best available torch device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def unet_offload_device():
    """Get UNet offload device"""
    return torch.device("cpu")


def unet_dtype(model_params, supported_dtypes, weight_dtype):
    """Determine UNet dtype"""
    device = get_torch_device()
    if device.type == "cuda" and torch.float16 in supported_dtypes:
        return torch.float16
    elif torch.float32 in supported_dtypes:
        return torch.float32
    else:
        return supported_dtypes[0]


def unet_manual_cast(unet_dtype, load_device, supported_dtypes):
    """Determine manual cast dtype"""
    return unet_dtype


def unet_inital_load_device(parameters, unet_dtype):
    """Determine initial load device"""
    device = get_torch_device()
    # Simple logic: use CPU for very large models, GPU for smaller ones
    if parameters > 1e9:  # 1B parameters
        return torch.device("cpu")
    else:
        return device


def load_models_gpu(model_patchers, force_full_load=False):
    """Load models to GPU"""
    for patcher in model_patchers:
        if hasattr(patcher, 'load_device') and patcher.load_device.type == "cuda":
            logging.info("Model loaded to GPU")


def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, 
                   copy: bool = False) -> torch.Tensor:
    """
    Cast tensor to specified device and dtype.
    
    Args:
        tensor: Input tensor
        device: Target device
        dtype: Target dtype
        copy: Whether to copy the tensor
    
    Returns:
        Casted tensor
    """
    return tensor.to(device=device, dtype=dtype, copy=copy)
