#!/usr/bin/env python3
"""
Memory Management Utilities for Large Model Loading
"""

import torch
import logging
import gc

def clear_cuda_memory():
    """Clear CUDA memory cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logging.info("CUDA memory cache cleared")

def get_memory_info():
    """Get current memory usage information"""
    info = {}
    
    if torch.cuda.is_available():
        info['cuda_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['cuda_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
        info['cuda_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        # Get device properties
        device_props = torch.cuda.get_device_properties(0)
        info['cuda_total'] = device_props.total_memory / 1024**3  # GB
        info['cuda_free'] = info['cuda_total'] - info['cuda_allocated']
    
    return info

def log_memory_usage(stage=""):
    """Log current memory usage"""
    info = get_memory_info()
    
    if torch.cuda.is_available():
        logging.info(f"Memory Usage {stage}:")
        logging.info(f"  CUDA Allocated: {info['cuda_allocated']:.2f} GB")
        logging.info(f"  CUDA Reserved: {info['cuda_reserved']:.2f} GB")
        logging.info(f"  CUDA Free: {info['cuda_free']:.2f} GB")
        logging.info(f"  CUDA Total: {info['cuda_total']:.2f} GB")
        logging.info(f"  CUDA Max Allocated: {info['cuda_max_allocated']:.2f} GB")
    else:
        logging.info(f"CUDA not available - using CPU only")

def estimate_model_memory(model):
    """Estimate memory usage of a model"""
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    return {
        'parameters': total_params,
        'size_bytes': total_size,
        'size_gb': total_size / 1024**3
    }

def safe_model_to_device(model, device, min_free_gb=2.0):
    """Safely move model to device with memory checking"""
    if device.type == 'cuda' and torch.cuda.is_available():
        # Clear cache first
        clear_cuda_memory()
        
        # Check available memory
        info = get_memory_info()
        model_info = estimate_model_memory(model)
        
        logging.info(f"Attempting to move model to {device}")
        logging.info(f"  Model size: {model_info['size_gb']:.2f} GB")
        logging.info(f"  Available GPU memory: {info['cuda_free']:.2f} GB")
        logging.info(f"  Required buffer: {min_free_gb:.2f} GB")
        
        if info['cuda_free'] > model_info['size_gb'] + min_free_gb:
            try:
                model = model.to(device)
                logging.info(f"✅ Model successfully moved to {device}")
                return model, device
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"❌ CUDA OOM when moving model: {e}")
                logging.info("🔄 Falling back to CPU")
                return model, torch.device('cpu')
        else:
            logging.warning(f"❌ Insufficient GPU memory")
            logging.info(f"   Need: {model_info['size_gb'] + min_free_gb:.2f} GB")
            logging.info(f"   Have: {info['cuda_free']:.2f} GB")
            logging.info("🔄 Using CPU instead")
            return model, torch.device('cpu')
    else:
        logging.info(f"Using CPU device: {device}")
        return model, torch.device('cpu')

if __name__ == "__main__":
    print("Memory Management Utilities")
    log_memory_usage("Startup")
